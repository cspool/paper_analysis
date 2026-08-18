# 知识库_kernel调度

## SpMM / SDDMM（稀疏-稠密矩阵乘 / 采样稠密-稠密矩阵乘）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SpMM（Sparse-Dense Matrix Multiplication）= 稀疏矩阵 × 稠密矩阵：C = S·D（或 D·S），结果稠密；SDDMM（Sampled Dense-Dense Matrix Multiplication）= 按稀疏模板对两个稠密矩阵做"部分乘法"：C = (A·B) ∘ mask，即只计算 mask 非零位置对应的点积，结果保持稀疏模式。二者互为对偶，是图神经网络消息传递的骨架：SDDMM 生成边消息（沿边对源/目标特征做点积），SpMM 聚合消息到顶点（FusedMM 等工作把两者融合以避免物化中间消息）。二者都是高稀疏、访存不规则 kernel，在 graph analytics、GNN（PyG/DGL 底层依赖 MKL/cuSPARSE）、ML 中流行。ATX 论文选它们作为 NCA 的头号评测负载，理由正是其"CPU 边遍历稀疏模式边动态生成任务、与加速器细粒度交错"的访存特征最能体现 NCA 相对 ICA/OCA 的优势。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CPU 侧 SpMM 通用骨架（CSR 格式，与论文图 9 的流式任务同构）：
```
for r in rows:                      # 每个稀疏行一个任务（论文每任务 16 行）
    for j in rowptr[r]: rowptr[r+1]:   # 本行非零列
        for k in K:                     # 稠密列维（tile 化）
            C[r][k] += vals[j] * D[colidx[j]][k]
```
调度要点：外层切任务（行块）、内层对稠密维分 tile 以匹配向量/矩阵单元宽度；CPU 负责"观察稀疏模式 → 定任务边界 → 发任务"，加速器负责 tile 内的乘加。ATX 版本把每个任务的所有访存编码为流：S1 根流取 rowptr 边界、S2 子流间接取 vals、S3 流取稠密 D tile，全部由 UTE 流引擎异步供给 NCA。SDDMM 对偶地按非零位置 mask 逐元素取两个稠密向量做点积后写回稀疏位置。论文基准矩阵：asia_osm、com-LiveJournal、delaunay_n24、packing-500x100x100、Serena（SuiteSparse），双精度、2MB 大页降低 TLB 影响。结果：ATX NCA 较纯 CPU 2.8×（SpMM）/2.7×（SDDMM）、较 ICA 2.3×/2.0×、较 L2 OCA（含预取）2.1×/2.0×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CPU 实现：MKL 的 SpMM（`mkl_sparse_?_mm`）、SDDMM 缺标准接口故论文用 TACO 生成的优化 kernel；均 AVX512 向量化。GPU 实现：cuSPARSE、GE-SpMM、HP-SpMM-SDDMM（Ampere/Hopper 调优）；编译生成：TVM/FeatGraph。硬件实现：SPADE（论文 NCA 算术单元的建模原型，ISCA'23，tile ISA + 复用 CPU 内存系统，相对 GPU 43× 以上）。性能关键：稀疏格式选择（CSR/COO）、行内负载均衡、tile 尺寸与缓存/scratchpad 匹配、间接访存（vals 索引）的 MLP——最后一环正是 UTE 父-子流 + 任务预取所针对的问题。使用注意：任务划分要保证不溢出 scratchpad（论文每 NCA 2×32KB）且输出 ≤2KB 寄存器容量；TACO/MKL 路径在 SDDMM 上差距大，跨实现比较时需标明后端。

涉及论文标题：
- ATX: Accelerator Task Extensions
- Harmonia: A Unified Hierarchical Scheduling Framework for Sparse Matrix Multiplication
- TensorPrism: Rethinking Sparse High-order Tensor Acceleration via Co-occurrence Graph

> **SpMSpM 补充（源自 Harmonia）**：SpMSpM（Sparse Matrix-Sparse Matrix Multiplication，稀疏-稀疏矩阵乘）C=A×B 是 Harmonia 的评估负载，A∈R^(M×K)、B∈R^(K×N)，与 SpMM 不同处在于 B 也是稀疏的，非零遍历与索引对齐（intersection）不规则度更高，数据流选择（三个嵌套循环的执行顺序）同时决定数据复用与控制复杂度。Harmonia 用三种数据流执行 SpMSpM：Inner Product（每个输出元素 C_m,n 做行-列点积，强输出复用、弱输入复用，B 列需反复取）、Outer Product（每个 k 用 A 的一列×B 的一行生成 psum 矩阵，最大化输入复用但需大量 psum 归并）、Row-based（A 的非零 A_m,k × B 整行 B_k,:，复用中等、归并开销小）。tile 形状 (T_M,T_K,T_N) 与 tile 内数据流的耦合决定性能：同一 16×16 PE 阵列、1MB SRAM 上，K 小 N 大时 OutP 最优、K 大时 InP/Row 因复用与 PE 并行度受益，形状 (64,128,64) 使 Row 优于 OutP。评估矩阵来自 SuiteSparse Matrix Collection（bcsstk10.mtx、email.mtx、orani678、rajat19 等），并含 DNN 剪枝权重（LLaMA-7B/OPT-1.3B 经 SparseGPT 剪到 0.2/0.4/0.6 密度，ResNet-50 经 STR 剪到 0.1/0.2，VGG-16 幅度剪枝到 0.1/0.32）。

> **TensorPrism 视角（ISCA'26）**：TensorPrism 把高阶张量收缩（$C_{f_1,f_2}=\sum_c A_{f_1,c}B_{c,f_2}$）展开成 2D SpMM（$C_{M,L}=A_{M,K}B_{K,L}$，M=自由模式合并 IJ）作为 unfold 路线 baseline（SPADE/HotTiles）的底层 kernel，同时指出 SpMM 与图计算数学等价（A=邻接矩阵、非零=边 j→i、聚合=稠密行累加）——这正是把图抽象推广到高阶张量的起点。三个经典 SpMM 数据流（Inner/Outer/Gustavson）在 TensorPrism 语境下被扩展到图原生 push/pull 数据流：Inner（输出点积复用、输入重复取）↔ PULL（自由顶点拉取源特征累加）、Outer（$Partial\_C_{M,L}=\sum_k A_{:,k}B_{k,:}$ 稠密行广播复用但部分和存储/同步贵）↔ PUSH（contraction 顶点广播稠密行 B[K,:] 给目标顶点集）、Gustavson（row-wise 流式）↔ 图遍历顺序的 push/pull 交替。unfold 到 SpMM 的代价（论文 §III）：元数据 O(I+J+K)→O(IJ+K)、复用距离 I+J→I×J、相邻非零邻居 6→4，导致 SPADE/HotTiles 展开后稠密行重复取数（uber 上 SPADE 91% 开销、2.09× 超额执行时间）。TensorPrism 的收缩引擎以标量-向量乘+向量累加执行等价计算但不展开（8 个 MAC 共享 feed unit 广播的稀疏输入、寄存器堆供 32 FP32 稠密向量、多累加器免冲突累加），最高 128× 复用/取数；PUSH 写不同输出地址免同步（消除 GSpTC 归约串行化，chcr 上其归约竞争占 73% 执行时间）。

## Expert Parallelism（EP，专家并行）与 Dispatch / Combine 通信算子

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Parallelism（专家并行）是把 MoE 的专家分布到不同 GPU 的并行策略：每 GPU 持有 N/E 个专家（或按层/流水段混合放置）。由于每个 token 只激活 topk 个专家且选择动态，EP 引入两类跨 GPU 通信算子：Dispatch——源 GPU（token 所在）把 token 发往被激活专家所在的一个或多个目标 GPU；Combine——各专家 GPU 把输出发回源 GPU 并聚合。二者的通信模式是"动态不规则"的：目标集随 token 变化（varying targets）、各 GPU 内存分配独立（asymmetric addressing），与静态集合通信（AllGather/Reduce-Scatter）有本质区别。通信是 MoE 训练的主要瓶颈：占 MoE 层执行时间 50-80%（本论文在模拟 GH200 NVL32 上测得 DeepSeek-V3 为 70.4%）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# baseline（DeepEP 风格）Dispatch kernel：目标集动态导致链路重复传输
for token in local_batch:
    for e in topk_experts(token):          # 目标集随 token 变化
        dst = expert_device(e)
        send(token_hidden, dst, addr[e])   # 同一 token 多目标时源链路重复传
# Combine kernel：可聚合输出被分开回传
for token in local_batch:
    out = zeros(hidden)
    for e in topk_experts(token):
        out += recv(expert_device(e))      # 同一 token 的输出多次独立回传
```
Annotations：Dispatch 冗余 = 同一 token 对多个目标的重复链路传输；Combine 冗余 = 同一 token 的可聚合输出多次独立回传，二者合计约占总流量 50%。DySHARP 的 kernel 层映射：Dispatch → dymultimem.st（单发、交换机多播），Combine → dymultimem.ld_reduce（交换机内归约），通信 kernel 以任务形式并入 megakernel（见 token-centric kernel fusion 条目）。DeepEP（https://github.com/deepseek-ai/DeepEP）是 SOTA 开源 EP 通信库：NVLink/RDMA 上的 token 分组-路由-重排实现，无 in-switch 计算，是本论文最主要 baseline。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：通信库（DeepEP、Tutel、FasterMoE）+ 布局管理（token 排序、按专家重排、recv 计数与元数据）。常见优化：分级通信（intra-node NVLink + inter-node RDMA 分层）、低精度通信（FP8）、计算-通信重叠（FasterMoE/Tutel 粗粒度流水、CCFuser/COMET 细粒度 Dispatch-GEMM / GEMM-Combine 重叠）、跨层 pipeline（DualPipe）。DySHARP 的做法：把 Dispatch 换成交换机多播、Combine 换成交换机归约 + GEMM-2 epilogue 乘权，同时消除软件维护远端内存状态（token 到达计数等元数据）的开销。推理侧用法（Approaching Shannon Bound 论文）：EP 作为多卡 serving 部署方式——Mixtral-176B（8x22B）以 EP 分布到 4×A100（320 GB 总预算），配合权重无损压缩把 batch 上限从 20 提到 95（4.8×）、吞吐 1.6×；该论文不优化 Dispatch/Combine 通信本身，只利用 EP 放大压缩带来的显存收益。

MoE-Hub 补充视角（ISCA'26）：EP 的 dispatch/combine 在算法上是"动态、不规则"的（每轮 token→expert 映射变化、每个 expert 入站 token 集/负载动态），而 GPU 的 UVA 通信模型是"静态、address-centric"（生产者必须知道远程确切地址）——这个语义失配迫使软件在通信前做昂贵的地址解析中介阶段（所有 token 同步、shuffle、CPU/GPU 协调 per-token 偏移），这正是软件重叠系统的开销根源（Fig.3 显示调度+暴露通信合计占 MoE 层 >24%）。MoE-Hub 把 dispatch/combine 的 kernel 侧实现改为：路由 kernel 用新 ISA 指令 `st.rowsp`（逻辑目的地 = 目标 expert 的 MallocID + token 的 RowID）立即发起 dispatch 传输（调度代码 0 行、通信指令 <10 条 vs Comet 调度 6347/5589 行、DeepEP 通信 498/1899 行，Table I）；combine 方向反向传输，源信息作为 expert 输入激活张量的一列额外数据、用 `st.rowsp.nop`（非关键路径）随 token 一起经逻辑地址翻译到达，专家最终 GEMM 末尾读源元数据用常规 store 发起 combine。地址分配、包整形与数据就绪信号全部由硬件（AAU/RPM/DAM）接管，dispatch/combine 通信与专家 GEMM 实现无缝透明重叠。

EP 的复制专家动态分发补充视角（ISCA'26，Patterns behind Chaos，Case Study 2）：在多 GPU MoE serving 中，除常规 EP（每专家单副本、专家按连续块分配到 GPU）外，还可为热门专家放置多副本（replication）以均衡负载。实现依赖 DeepEP 通信库的 ep_dispatch_algorithm="dynamic" 模式：被复制的专家（同时存在于多个 GPU）收到的 token 在各副本间均匀分配（动态 dispatch），即把每个 token 分发到持有该专家的一个（而非全部）副本 GPU，避免副本间负载不均。SGLang 侧通过 init_expert_location 接口把算法（Remap/Dup）算出的专家放置写入，DeepEP 作为 MoE 后端执行 dispatch/combine。dup_based 放置中复制专家被"token 均分到所有副本"（Algorithm 2），配合 dynamic dispatch 使每个副本处理近似等量 token，把负载从"单热点专家 GPU 拥塞"转为"多副本分摊"。该机制与 dispatch/combine 常规语义一致：dispatch 目标集 = 该 token 所选专家的某个副本所在 GPU（由 dynamic 策略选定），combine 收集各副本输出。效果：Dup（R=1）在 Qwen3-235B/8×H100 上 MoE 计算提速 12.5%（相对默认连续放置），Remap 15.5%。

  - SHyLA 补充：MoE（Mixtral 8×22B、Grok1 314B）中专家并行 pe 嵌套在 pt 维内——每个专家组内单个专家再经本地张量并行（pt/pe 个 chiplet）子切分，遵循与 Dense FFN（FFN1/FFN2）相同的分区逻辑，保证两种模型结构硬件利用率一致。MoE 稀疏性为粗粒度（专家级），激活专家的 Weight 以大的连续块读取，保持 NVM 带宽利用与数据流效率（对 DRAM-only 增益仍随 MoE 容量需求扩展）。
STEP 补充视角（ISCA'26，EP 之上的轻量运行时层）：STEP 的预取/选举机制与 EP 正交——每个 expert-parallel group 独立维护本组的热专家本地缓存并运行 token 感知自适应预取，在不改 EP 执行模型（不要求修改 dispatch/combine 语义）的前提下复用高频专家、减少组间 all-to-all 交换。高带宽互联（NVLink/NVSwitch）下，STEP 把 peer GPU HBM 用作二级缓存：若某专家已在相邻 GPU，经 peer-to-peer 传输直接取用而非回 CPU 主存，形成"组内本地缓存 + 组间 peer-HBM 缓存"的分层缓存；预取操作与专家计算重叠。论文单卡评估刻意不用 NVLink peer-GPU 共享以保证与 baseline 公平（全部走 PCIe host offloading），peer-HBM 缓存是其 EP 扩展方向而非单卡实验配置。

- STAGE 补充视角（ISCA'26）：STAGE 将 EP 建模为 MoE 层张量的图级/张量级分布组合：每个 token 经 gating 路由到目标 expert，通过 AllToAll 在设备间交换 token；通信匹配器把 producer 分布 [B/dp,S,H@1/tp] 到 consumer 分布 [B,S/dp,H@1/tp] 匹配为 AllToAll，把 [B/dp,S,H@1/tp]→[B/tp,S,H/dp] 匹配为 ReduceScatter+AllToAll（Table IV）。MoE 专家激活用逐层专家激活直方图建模（默认均匀分布，用户可覆盖为自定义统计）；真实训练中 micro-batch>1 时所有专家通常都被激活，而 micro-batch=1 时部分专家不激活造成与默认假设的通信量偏差（论文 Table VII 注释）。DeepSeek-R1 推理案例（Table VIII）：prefill 阶段 compute-bound、偏好低 EP 度减少 AllToAll 开销；decode 阶段短序列受益于更大有效 batch，高 EP 度大集群吞吐更高。

涉及论文标题：
- Accelerating MoE with Dynamic In-Switch Computing on Multi-GPUs
- Scalable Synthesis of Distributed LLM Workloads Through Symbolic Tensor Graphs
- STEP: Adaptive Spatio-Temporal Expert Prefetching for Low-Latency and Memory-Efficient MoE Inference
- Approaching Shannon Bound with Lossless LLM Weight Compression
- MoE-Hub Taming Software Complexity for Seamless MoE Overlap with Hardware-Accelerated Communication on Multi-GPU Systems
- Patterns behind Chaos: Forecasting Data Movement for Efficient Large-Scale MoE LLM Inference
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity

## Token-Centric Kernel Fusion（token 中心 kernel 融合）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
本论文提出的通信感知调度技术：把 MoE 层的 Dispatch→GEMM-1→GEMM-2→Combine 四个算子重组为 token-paced pipeline。核心洞察：算子间依赖可以在 token/tile 粒度判定，因此某 token（或 tsize=128 个 token 的 tile）的输入一就绪即可执行对应操作，无需等待算子级全局完成。通过显式跟踪 token 级依赖并在 readiness 边界调度，Dispatch（GPU→交换机方向为主）与 Combine（交换机→GPU 方向为主）并发执行，把 in-switch 多播/归约造成的互补非对称流量合并，双向带宽利用率提升——从而把 dynamic multimem addressing 削减的流量转化为真实加速（消融：仅流量削减因方向不对称不加速；仅 kernel 融合不超过 COMET；二者缺一不可）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# megakernel 内持久 TB 循环（SM 分 4 组：A=Dispatch, B=GEMM-1, C=GEMM-2, D=Combine）
while (task = fetch_task(task_list)) != NULL:
    switch task.type:
      case DISPATCH: dymultimem.st(token_tile)                      # SM 组 A
      case GEMM1:    if TS_Table.row_ready(tile): gemm1(tile)       # SM 组 B
      case GEMM2:    if TS_Table.row_ready(tile): gemm2(tile)       # SM 组 C（可共享）
      case COMBINE:  if OR_Table.nReady(token) == topk:
                         dymultimem.ld_reduce(token)                # SM 组 D
```
Annotations：GEMM-1/GEMM-2 无就绪 TB 时可共享 SM；ready 检查用专用 load 指令 spin-poll tracker 表；tile 尺寸 128 = GEMM tile 尺寸。时间轴效果：同一时刻 tile i 的 Dispatch 在发、tile i-2 在 GEMM、tile i-4 的 Combine 在归约——双向链路同时被使用，互补的非对称流量互相填补。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现 = megakernel + persistent TB（Rammer rTasks 机制）+ token tracker 三表（TS/TID/OR）+ 软件 readiness 轮询。适用场景：任何"动态多目标通信 + 依赖可 token 化"的 MoE 训练/推理；必须与 in-switch computing 绑定使用（单独 kernel fusion 不带来加速）。对比：FasterMoE/Tutel 是算子级粗粒度重叠，CCFuser/COMET 是 Dispatch-GEMM / GEMM-Combine 细粒度重叠但两通信算子仍隔离，DySHARP 是四个算子全链路 token 级流水。

涉及论文标题：
- Accelerating MoE with Dynamic In-Switch Computing on Multi-GPUs

## Megakernel 与 Persistent Thread Blocks（持久线程块调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Megakernel（巨核）是把多个算子合并进一次 kernel launch 的持久化执行模式：固定数量的持久线程块（persistent TB）常驻 SM，从任务列表循环取"任务"执行，用软件调度替代硬件 TB 调度器。机制源自 Rammer（OSDI'20，NNFusion 编译器）：rOperator 分解为 rTask（逻辑调度单元，原 TB 粒度），vDevice 抽象为 vEU（GPU 上映射到 SM），rTask 内用逻辑 rtask_id 替代 blockIdx 寻址。优势：消除 kernel launch/上下文准备开销、支持跨算子的细粒度并行与动态依赖门控（Rammer 为编译期静态计划，DySHARP 为动态 readiness 门控）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
__global__ void megakernel(task_list_t *tasks):
    while (tid = next_task(tasks)) != -1:      # 持久 TB 循环取任务
        switch tasks[tid].type:                # 原 TB → 任务
            case GEMM1_TB:   gemm1_row(tasks[tid])
            case DISPATCH_TB: dymultimem.st(...)
            case COMBINE_TB:  dymultimem.ld_reduce(...)
```
Annotations：持久 TB 数量 = 各 SM 组内可驻留 TB 数；"向 SM 发射 TB"被模拟为持久 TB 取任务。DySHARP 变体：取任务前先查 token tracker 表的 readiness 位（TS/OR 表），未就绪则用专用 load 指令 spin-wait，形成 readiness-gated 调度；SM 分 4 组（Dispatch/GEMM-1/GEMM-2/Combine），GEMM-1/GEMM-2 可共享 SM。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Rammer（OSDI'20，约 52k 行 C++，输入 TF frozen graph/TorchScript/ONNX）在编译期生成 rTask 调度计划，运行期按计划静态映射执行；CUDA 层以 persistent kernel + 软件任务队列实现。DySHARP 的用法：megakernel 内嵌 token tracker 轮询，实现 token 级动态依赖流水（区别于 Rammer 的静态计划），用于 MoE 层四算子融合。适用：算子多、粒度细、依赖动态的负载；代价是失去硬件 TB 调度器的抢占/负载均衡，需要软件分组与容量管理。

涉及论文标题：
- Accelerating MoE with Dynamic In-Switch Computing on Multi-GPUs

## Off-chip Queue Spill 与 FIFO 预取（队列溢写调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Morpha Core 处理"队列超过片上 scratchpad 容量"的运行时数据调度机制：队列按 FIFO 顺序溢写到 off-chip 内存，片上仅保留 head slice，slice 末字复用为 off-chip 指针；利用队列访问的严格 FIFO 顺序性做自动 prefetch 与 double-buffering，从而不需要 cache 层次（tag 比较、替换、一致性等复杂逻辑）。公平性：Q_ID 指令中一个字段设置每队列最大片上 slice 数，超过即强制 spill。
- 从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 过程伪代码（队列读循环 + spill，按论文 §III-C 描述整理）：
    ```
    while queue not empty:
        if head slice on-chip:
            elem = sub_bank[head]        # head slice 驻留片上
            head += 1
        else:
            DMA_fetch_next_slices(queue, prefetch_depth=2)  # 按 FIFO 顺序预取下一批 slice
            # 双缓冲：当前 slice 被消费时，下一个 slice 已在途
        if tail slice full and onchip_slices >= max_onchip_slices:
            DMA_spill(oldest_slice, offchip_addr)            # 最老 slice 换出
            oldest_slice.last_word = offchip_ptr             # 末字改存 off-chip 指针
    ```
  - 评估（论文 Fig. 8c）：对输入/输出超过片上容量的通用向量 kernel，扫数据量 1.5×–20× 片上容量与算术强度 5/10/20 FLOP/B，per-element latency 归一化到"理想 cache + prefetch"。最坏情形（1.5× 容量、AI=5）溢写开销 12.7%，3× 容量降至 6.6%；AI=10 时仅 6.2%（memory-bound 时固定 spill 成本被摊薄）。对照：scratchpad 相比 cache 省 ~30% 面积与能耗（论文引 [106,107]），再加省掉的 cache 控制逻辑。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：溢写决策在 Queue Manager（片上 slice 计数 vs Q_ID 上限字段），数据面走 Data Movement Engine 的 DMA；预取依据是"队列必然按 FIFO 消费"这一确定性，等同于软件流式引擎（如 SDF/streaming accelerator）的确定性预取。使用场景：动态尺寸数据结构超过片上存储的加速器负载（大点云过滤、长 frontier 队列）；与通用 cache 相比牺牲随机访问友好性换取简单性与能效。论文未明确说明该机制的开源实现（Web 未找到）。

涉及论文标题：
- Accelerator Polymorphism: Transcending Domain-Specific Architectures with Robotics

## Fused rANS 解压与 GEMM Tile 计算（融合解压-GEMM 内核）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
把熵解码直接融入 GEMM 内核的"压缩权重执行原语"：权重以压缩 bitstream 常驻全局内存，解码 warp 按 GEMM tiling 序把 tile 解压进 shared memory，GEMM warp 立即用 tensor core 消费，解码与矩阵乘在同一线程块内以生产者-消费者方式流水重叠，解压后的权重从不写回全局内存。对比层粒度 decompress-store-compute（NeuZip/DFloat11：整层解压写回全局内存再 GEMM），融合内核消除层同步屏障、冗余全局内存流量与额外解压缓冲。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Algorithm V.1 伪代码（论文）：
```
Shared: T~ (decode table, 从全局拷贝), A(0),A(1) in R^{M×K} (双缓冲), ready[2] (原子标志)
for k = 0 .. K_tiles-1:
    b = k % 2; p = 1 - b
    # 生产者：warp 0 只跑 rANS 解码
    Warp0: RansDecodeTile(A(b), stream[k], T~); ready[b] = 1
    # 消费者：warp 1..W 只跑 tensor-core GEMM
    if k == 0: wait(ready[b]==1); GemmTile(A(b), B(k), C); ready[b] = 0
    else:      wait(ready[p]==1); GemmTile(A(p), B(k), C); ready[p] = 0
    __syncthreads()
```
Annotations：每个权重只解码一次（解码与计算共享 on-chip footprint，不再从 HBM 重读）；ready 用 cuda::atomic_ref<int, thread_scope_block> 实现；权重全局内存流量从 V_B = (M/M_t)·K·N 降为 (M/M_t + α − 1)·K·N（每权重只取一次压缩形式）。性能：tile 对齐（vs DietGPU 解码 + CUTLASS GEMM 分离两段）3.3–8.2×，再加双缓冲总计 4.0–10.1×（Qwen-1.5B 4.41×、DeepSeek-67B 6.71×、Llama-405B 10.06×）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：以 CUTLASS plugin 式投影算子 override 库级 GEMM 实现 + DietGPU ANS 内核扩展 + PyTorch wrapper；tile 几何由 profiling 选定（128×32/256×64/128×128；A100 32×128、H200 64×256）。批大小行为：小 batch 时 tensor 管线先耗尽、内核 decoder-bound（退化为 decode-then-GEMM）；大 batch 时 GEMM 主导、解码完全隐藏。效果：A100 大 batch 达 CUTLASS 的 1.0–1.1× 内、H200 最高 1.2× 超越 CUTLASS；vs NeuZip 最高 ~10×、vs DFloat11 ~6–7×。使用：压缩权重模型推理、显存受限的 serving（batch 放大 1.3–4.8×）。

涉及论文标题：
- Approaching Shannon Bound with Lossless LLM Weight Compression

## Warp-Cooperative rANS 解码（warp 协作并行解码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
把串行的 rANS 状态机用 warp 级并行化的技术：rANS 状态转移依赖前序符号，单流无法并行；解法是把每 tile 的压缩位流划分为 R 个独立 substream（ANS 状态可独立初始化），每 warp lane 维护一个 rANS 状态、各解一条 substream，一个 warp 并发推进 R 条流。本论文扩展 DietGPU 的做法，让解码结果按 GEMM 所需 swizzle 布局直写 shared memory。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Algorithm V.2 伪代码（论文）：
```
RansDecodeTile(A, stream, T~):
  init rANS state s per lane                      # 每 lane 一条 substream
  for i = 0 .. S_lane-1:
      x = s.value mod R                           # 取状态低 b 位 → 表槽
      (σ, f, c) = T~[x]                           # shared memory 查解码表
      w = DecodeSymbol(σ)                         # 符号
      (r, c) = symbol index → write A[r,c] = w    # 直写 shared memory tile
      s.value = f * floor(s.value/R) + (x - c)    # rANS 状态回溯
      while s.value < renorm_thresh:
          u = load_32bit(stream)                  # 跨 lane 交错 → coalesced
          s.value = (s.value << 32) | u
```
Annotations：R = 2^b（b=12 概率精度）；T~ 从全局 codebook 拷贝进 shared memory 实现低延迟高带宽查表；重归一化 load 因压缩流跨 lane 交错而天然 coalesced，保持全局内存吞吐；写 A 的 (r,c) 与交错解码序锁步，无需二次布局变换。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
基座 DietGPU（Meta，A100 解码 250–410 GB/s、H100 约 592 GB/s）已提供 warp-cooperative rANS；本论文在此之上做三点改造：tile 粒度 substream 划分（对齐 GEMM tile）、直写 shared memory（不落全局内存）、与 tensor-core GEMM 融合流水。使用：GPU 上近熵解码、数据搬运压缩（all-gather/reduce-scatter 前压缩）、LLM 权重无损压缩后端。实现要点：每 lane 状态的独立重归一化、解码表常驻 shared memory 降低查表延迟。

涉及论文标题：
- Approaching Shannon Bound with Lossless LLM Weight Compression

## 共享内存双缓冲与生产者-消费者管线重叠（producer-consumer pipeline overlap）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GPU kernel 内用两块（或多块）shared memory buffer 交替：生产者（解码/加载）填充 buffer A 的同时消费者（tensor core）计算 buffer B，经原子标志/barrier 同步，从而把生产延迟藏在计算之后。CUTLASS 的 multi-stage 流水（num_stages）、cuBLASDx 的 pipeline、Hopper 的 TMA + mbarrier 都是同一思想的工业化形态。本论文在双缓冲之上再叠加四阶段 shared memory ring buffer，让解码领先若干 sub-tile，消费者每步都能拿到已就绪操作数。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
论文的流水时序（Section V-D）：生产者（rANS 解码）与消费者（MMA）运行在 SM 内物理分立的管线上——解码 warp 只用整数（查表/状态更新）+ LSU（coalesced 补位、shared store）管线，GEMM warp 只用 shared load + tensor pipeline；warp scheduler 同周期 co-issue，互不占用对方资源。代价模型：解码成本/sub-tile 近似常数（概率表访问 + 重归一化读），MMA 成本/sub-tile 随 M-rows 增长——故小 batch 时 tensor 管线先耗尽（decoder-bound，退化为 decode-then-GEMM 行为），大 batch 时消费者永不 stall、解码完全隐藏、近完全重叠。效果：加双缓冲较单缓冲 tile 对齐版本进一步提升，总计 4.0–10.1× vs 分离两段。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：`As[2][...]` 双缓冲 + cp.async/`__pipeline_memcpy_async`；或 CUTLASS num_stages=N 多级流水；Hopper 上 TMA descriptor + mbarrier 的 producer_acquire/consumer_acquire；本文用 cuda::atomic_ref<int, thread_scope_block> 的 ready 标志。注意实测边界：当瓶颈是寄存器压力/occupancy 而非访存延迟时，双缓冲收益有限（开源 benchmark 显示 4096 规模 GEMM 上双缓冲仅 1.03×）。使用：任何"内存加载/解码 vs 计算"重叠场景（GEMM 主循环、attention 流水、解压-计算融合内核）。

  - SHyLA 补充：解析模型假设 GEMM/GEMV 以双缓冲把 Weight/KVCache 加载与 MAC 计算重叠（Fig. 8），即"加载下个 tile 的 Weight 的同时计算当前 tile"；因 LLM 线性层顺序访问命中 row buffer、NVM 读利用率可达 70%（> 一般工作负载），该重叠有效。数据布局层面，NVM plane 存 NTile 与同组 DTile 的 Weight 切块、DRAM plane 存 IA 行，tile group 内专用高速链路供跨内存交换，避免 plane starvation；这些缓冲/加载调度在 GPGPU-Sim 中以 CUDA 双缓冲 + plane-aware tile 映射实现（论文未开源）。
涉及论文标题：
- Approaching Shannon Bound with Lossless LLM Weight Compression
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity

## Tile Swizzle 与确定性驱逐（swizzle 布局 + 静态替换序）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Swizzle 是 GEMM kernel 对 shared memory 中 tile 的布局重排（XOR 布局、K-strided interleave 等），用于消除 bank conflict、使 tensor core 从 shared memory 无冲突取片段；threadblock swizzle 则把 blockIdx 映射重排以改善 L2 locality。本论文扩展出一个"确定性驱逐"机制：大维度 GEMM 按固定 tile-swizzle 遍历序取 tile，当活跃 tile 工作集暂时超过 shared memory 容量时，按该静态序确定驱逐顺序（recency 由 swizzle 访问序静态定义，无需运行时记账），被逐 tile 的解压形式暂存小 decompression buffer 以避免近期重复解压，压缩形式保留在全局内存。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
访问序 = fixed_swizzle_order(tile grid)        # 由 GEMM tiling schedule 静态确定
for tile in 访问序:
    if tile not in shared_mem and shared_mem full:
        victim = next_to_evict(访问序)          # 按静态序选 victim（无 LRU 记账）
        stash(victim, decompress_buffer)        # 暂存解压形式防近期重访
    decode(tile -> shared_mem)                  # 压缩形式常驻全局内存
    gemm(shared_mem tile)
```
Annotations：驱逐顺序离线可知 → 零额外 bookkeeping；与"解压一次、不再从 HBM 重读"原则结合：重复访问命中 shared memory 或 decompression buffer，避免重复 rANS 解码。论文用于压缩 tile 工作集管理（decoded tile 超 shared memory 容量时）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CUTLASS 提供 hgemm_swizzle.h / igemm_swizzle.h（16-bit/8-bit tile 转置布局）与 identity_block_swizzle.h（threadblock→GEMM 分块映射，L2 局部性）；CuTe 用 Layout 代数表达 swizzle。本论文的确定性驱逐是 swizzle 序的延伸应用（静态替换而非 LRU）。使用：大维度 GEMM 的 shared memory 管理、压缩/稀疏 tile 的替换策略；与 double buffering 正交（双缓冲管流水、驱逐管容量）。

涉及论文标题：
- Approaching Shannon Bound with Lossless LLM Weight Compression

## CUTLASS（CUDA Templates for Linear Algebra Subroutines）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NVIDIA 开源的 header-only CUDA C++ 模板库，用于构建高性能线性代数（主要为 GEMM）CUDA kernel。核心抽象：把 GEMM 分解为 thread / warp / block / device 四个作用域的模板组件；支持 tensor core（mma 指令、Hopper wgmma）与混合精度（int8/fp8/bf16/fp16/tf32/fp64）；3.x 引入 CuTe（Layout/Tensor/Shape/Stride 布局代数 + Mma_Atom/TiledMma/Copy_Atom），支撑多级流水（num_stages）、swizzled shared memory 与 collective 算子。性能与 cuBLAS 相当。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
本论文的用法：以 CUTLASS 为基座实现"plugin 式投影算子"——override 库级 GEMM 实现、把 rANS 解压塞进 GEMM 主循环，并提供 PyTorch wrapper。关键依赖：CUTLASS 暴露可编程的 tensor-core tiling、warp 调度与内存布局，使压缩 tile 几何（128×32/256×64/128×128；A100 32×128、H200 64×256）能与 GEMM tile 精确对齐——"压缩 tile 对齐 GEMM tile"是 tile 级随机访问解码的前提。基线对比也用 CUTLASS（"with the CUTLASS-based GEMM baselines across platforms to ensure fair comparison"）。论文声明设计不绑定 CUTLASS。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
github.com/NVIDIA/cutlass；C++17 + CUDA 12.4+；用 CuTe 定义 Mma_Atom/TiledMma/Copy_Atom，加流水 stage 与 swizzle 布局。使用：定制 GEMM/Conv kernel、混合精度与量化 GEMM 研究、vLLM/SGLang 后端基座（本论文即替换 SGLang 默认 GEMM 后端）。局限：模板展开编译慢、tile 配置手工调优——本论文用 profiling 定 tile 几何。

MXFFP 补充视角（ISCA'26）：MXFFP 论文用 CUTLASS 生成 MXFP/MXFFP 4/6/8-bit 量化格式的 GEMM kernel 并提取指令 trace，作为 Accel-Sim（RTX 5090 派生配置）硬件性能模拟的输入，评估相对 BF16 的 GEMM 延迟/加速比（矩阵 256/512/1024）与端到端 LLM prefill 推理（1024 token、7 个 LLM）。CUTLASS 在此承担"可配置低位宽 GEMM kernel 生成器"角色：模板化 mma/tiling 允许 4/6/8-bit 操作数打包（6-bit 装 8-bit 容器、4-bit 走更窄数据通路）与 block-shared exponent 元数据访存建模；MXFFP 的配置位/共享指数额外访存流量在模拟器中单独建模。MXFFP 论文未修改 CUTLASS 本身。

MoE-Hub 补充视角（ISCA'26）：CUTLASS 承担 MoE-Hub 模拟评估中的"专家 GEMM kernel 实现"角色——三个工作负载（Mixtral 8x7B、Qwen2-MoE-2.7B、Phi-3.5-MoE）的专家前向 GEMM（Top-K 专家各做两次 GEMM：hidden→FFN hidden 与 FFN hidden→hidden，中间夹激活）都用 CUTLASS kernel 在 Accel-Sim 扩展模拟器中执行；由于通信-计算重叠由 hub 硬件（AAU/RPM/DAM）透明实现，专家 GEMM 本身无需修改，CUTLASS kernel 以标准方式消费 DAM 派发的 TB（数据一就绪即被调度执行）。MoE-Hub 与 CUTLASS 的关系：硬件把"数据何时可用"的编排接走，kernel 侧只保留纯计算（与 MXFFP 把 CUTLASS 当 trace 生成器、DySHARP 把 Dispatch/Combine 并入 megakernel 的做法都不同）。

XtraMAC 补充视角（ISCA'26，GPU GEMV 对比基线）：CUTLASS 承担 XtraMAC 论文中"GPU 混合精度 GEMV kernel baseline"角色——在 NVIDIA H100 PCIe（2 TB/s HBM）上用 CUTLASS 官方 GEMV kernel 测量 1×4096×4096 与 1×4096×12288 GEMV 执行时间（0.0294/0.0879 ms）与功耗（nvidia-smi，135 W），与 XtraMAC 的 FPGA 版 GEMV kernel（U55c：0.0246/0.0743 ms、85 W、xbutil 测量）对比得出 1.2× 时延与 1.9× 能量效率优势；尽管 H100 带宽 4× 于 U55c，带宽受限 GEMV 下 FPGA 因 2× lane 打包 + 无格式转换开销 + ~74% HBM 利用率反超。CUTLASS kernel 未修改、作为官方实现基线（与 MoE-Hub/MXFFP 的"生成器/执行体"角色不同）。

涉及论文标题：
- Approaching Shannon Bound with Lossless LLM Weight Compression
- MXFFP Microscaling Flexible Floating Point Format for Large-Scale AI Model Acceleration
- MoE-Hub Taming Software Complexity for Seamless MoE Overlap with Hardware-Accelerated Communication on Multi-GPU Systems
- XtraMAC An Efficient MAC Architecture for Mixed-Precision LLM Inference on FPGA

## DietGPU（GPU ANS 熵编解码库）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Meta（Jeff Johnson，FAISS 作者）开源的 GPU 端 ANS（rANS）熵编解码库（MIT license，github.com/facebookresearch/dietgpu）：byte-oriented rANS codec + float 编解码扩展（float16/bfloat16 无损），A100 上 ANS 吞吐约 250–410 GB/s、float codec 250–600 GB/s；以 4 KiB segment 为压缩单元、batch-oriented API（C++ device pointer + PyTorch tensor）。设计目标：在 PCIe/NVLink/网络传输前压缩数据（牺牲一点压缩率换速度），用于分布式 collective（all-to-all/all-gather/reduce-scatter/all-reduce）加速——首个公开 GPU ANS 实现，GPU 版 FSE。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
本论文的用法：把 DietGPU 的 ANS kernel 作为解压后端，扩展为 tile 粒度解码 + tensor-core GEMM 集成。DietGPU 原设计面向数据搬运（segment 批量压缩/解压，无 tile 随机访问、无 GEMM 融合）；论文保留其 warp-cooperative 解码结构（每 lane 一条 substream、coalesced 重归一化），在其上加：tile 对齐 substream 划分、shared memory 直写、与 GEMM 生产者-消费者融合（naive 基线即"DietGPU ANS decode + CUTLASS GEMM 分离两段"）。第三方评测：H100 上 DietGPU ANS 解码约 592 GB/s，接近论文目标的权重输入速率。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：rANS 状态 + per-symbol 频率表 + segment 并行（batch 处理 4 KiB 段）；C++/PyTorch 双 API。使用：GPU 集群数据搬运压缩（训练 all-gather 前压缩可提速 ~10% wall-clock）、LLM 权重无损压缩的基座（本论文、ZipServ 等）。局限：无随机访问/tile 语义——需上层扩展（本论文的 offset 表 + tile substream）。

ENEC 补充视角（Ascend NPU 侧对比）：ENEC 论文把 DietGPU 作为 GPU 侧主要 baseline（Diet_ANS 与 Diet_Float 两种模式）。ENEC 对 DietGPU 的评价：其 Diet_Float 模式虽然做了指数-尾数分离（只压指数），但指数压缩仍依赖 ANS 变长编码——在 Ascend NPU 上因不规则访存与控制流而效率极低（Ascend AIV 无条件分支、无 gather）。ENEC 用"分支无关整数变换 + 定长位打包"替代 DietGPU 的变长 ANS 编码，结果是：压缩吞吐 3.43× 高于 DietGPU、压缩比 1.12× 优于 nvCOMP；Diet_Float 在 BF16 上的压缩比（1.47-1.48）仍略高于 ENEC（1.35-1.37），ENEC 以少量压缩率换取 NPU 端 2 个数量级吞吐（BF16 压缩 372 GB/s vs ZipNN 0.4 GB/s 级）。跨平台对照（Table VII）：ENEC-GPU-V1（CUB 前缀和 + warp 内建）在 A800 达 419.2/421.0 GB/s，接近 Diet-Float 的 271.9/271.3 GB/s 两倍。

涉及论文标题：
- Approaching Shannon Bound with Lossless LLM Weight Compression
- ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs

## AP 位串行/位并行存内算术算子（SIMD 模式：ap_add/ap_mul 微码展开）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
把 SIMD 算术用 CAM 搜索/更新序列实现，一次作用于子阵列列方向的全部元素（论文表 IV）：位并行 ap_and/ap_or 3 周期、ap_xor 4 周期；位串行 ap_add/ap_sub 8n+2 周期、ap_mul 4n²+4n 周期（n=位宽，按位从 LSB 到 MSB 传播进位/部分积）、ap_redsum n 周期归约、ap_eq n+4、谓词 ap_merge 4 周期。全部由 ASU 展开成微码，位切片布局下第 i 个子阵列存所有元素第 i 位。吞吐模型 P_op = VL/c_op × f（75% 配置、350MHz、VL=320：位运算 ≈37 GOPS、32-bit add ≈0.43 GOPS、mul ≈0.027 GOPS；对照 UPMEM 单 DPU roofline ≈0.35 GOPS）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# SIMD 模式，位切片布局，tag 为每列 1 位的匹配结果锁存
ap_xor(vdst, vs1, vs2):                 # 4 周期
  tag  = search(vs1==0 AND vs2==1)      # 真值表第一类，1 周期
  tag |= searchacc(vs1==1 AND vs2==0)   # 反例 OR 累积进 tag，2 周期
  update(vdst, tag)                     # 按 tag 掩码位线批量写回，1 周期

ap_add(vdst, vs1, vs2):                 # 8n+2 周期，位串行进位传播
  carry = 0
  for i in 0..n-1:                      # 每位：进位扩展 2 次搜索 + 加法真值表 2 次
    s     = vs1[i] XOR vs2[i] XOR carry
    carry = majority(vs1[i], vs2[i], carry)
    update(vdst[i], s)                  # 第 i 子阵列写回
```
调度要点：长位串行乘（4n²+4n）必须与 DMA 重叠（VPU 预取），否则 bank 带宽闲置；VA（向量加）在 BAAP 上收益有限，因为 UPMEM 标量加已单周期。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
与 CAPE/PUMICE 的向量 ISA 同源（web：https://par.nsf.gov/biblio/10225228-cape-content-addressable-processing-engine），BAAP 首次把它放进 DRAM 工艺约束（降频 350MHz）。使用：PrIM 的 GEMV/MLP/TS 靠位串行乘并行化 + VPU 重叠；SEL/SCAN/REDSUM 靠流式比较/归约达到带宽饱和。权衡：位串行省面积但延迟长（乘法数百~数千周期），依赖重叠与宽向量（VL 96→384 使 decode GEMV 的 VPU 收益从 39% 降至 20.9%）摊薄。

涉及论文标题：
- BAAP: Coupling Compute-in-SRAM with DRAM Banks for Near-Memory Processing

## DirectAP 模式与 ap_regex（CAM 搜索-更新算子与模式匹配）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DirectAP 模式把 AP 的 CAM 语义直接暴露给程序员/编译器（论文表 IV）：ap_search(field_mask, imm) 1 周期按键置 tag、ap_searchacc 2 周期非破坏 OR 累积、ap_update(field_mask, imm) 1 周期按 tag 掩码位线写回、ap_set_tags 1 周期；field_mask 表达 don't-care（X）；伪指令 ap_regex(rdst, imm) 对长度 m 文本穷举长度 k 模式（<m−k 周期、支持通配符、模式超出单子阵列时溢出到下一子阵列、损失子阵列级并行）。数据用列连续布局，整词 CAM 匹配 1 周期完成（位切片 SIMD 需逐位 n 周期）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
论文算法 1（DirectAP BFS，边按列装载、FROM/TO/PROC/CURR 各占一段行）：
```
AP-BFS(ap, start_node):
  traversal_order = [start_node]
  ap.search(TO, start_node); ap.searchacc(FROM, start_node)
  ap.update(CURR, true)                    # 初始前沿
  repeat:
    ap.search({CURR,PROC},{true,false})    # 当前前沿未处理边，1 周期
    if ap.tag_popcount() == 0: break       # 前沿空，遍历结束
    ap.update(PROC, true)                  # 批量标记已处理
    tagged = ap.read_tags(); ap.set_tags(0)
    for e in tagged:
      if e.TO not in visited:
        ap.searchacc({FROM,PROC},{e.TO,false})   # 累积新邻居边
    ap.write_tags_to(CURR, true)           # 单次批量更新物化新前沿
    for n in new_nodes: ap.search(TO,n); ap.update(PROC,true)
```
关键思想：tag 位紧凑编码当前前沿，一次批量 update 物化下一前沿——把 frontier 型图遍历变成关联查询。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
适用形态（论文 §IV-E 结论）：计算可表述为对大而稀疏状态的重复 membership/邻域查询——文本模式匹配、直方图计数、图遍历。基因组案例：k-mer 计数用 bank 内穷举 CAM 匹配替代扫 DRAM+哈希表更新，对 UPMEM 2–38×；de Bruijn 图遍历 1.1–2.8×，直到跨 DPU 前沿交换经 host 中转成为瓶颈；k>21 时搜索空间 2^k 爆炸、回退多核 host。HST（直方图）类 kernel 同样受益。

涉及论文标题：
- BAAP: Coupling Compute-in-SRAM with DRAM Banks for Near-Memory Processing

## VPU 向量预取与 DMA-计算重叠（Vector Prefetching Unit）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VPU 是 BAAP 放在 DMA 引擎与 AP 数据通路之间的微结构：AP 计算当前向量时，非投机地预发射下一向量 load/store 到 DMA 引擎（操作数就绪才发起、命中五级流水线的访存级），用额外一个向量大小的缓冲吸收预取数据。动机：位串行算术每指令数百至数千周期（ap_mul 4n²+4n），若不重叠，DMA 空闲、流水线干等内存——把"位串行长延迟计算"与"bank 级快速 DMA"这两个本来串行的环节并行化。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
周期三分类（论文 §IV-C）：每轮执行周期分为 Overlap（AP 忙 ∧ bank 忙）、Idle（等内存）、Compute-only（AP 忙 ∧ 内存闲）；VPU 把 Idle 转为 Overlap。
```
# VPU 生效的执行流水（decode GEMV，每 token 一个激活向量）
loop:
  VPU.preissue(DMA.load(vec_next))     # 非投机预发射下一向量
  ASU.run(ap_mul(vacc, vw, vec_cur))   # 当前向量位串行乘（与 DMA 重叠）
  vec_cur = VPU.buffer                 # 就绪后切换缓冲
```
消融结果（GPT-2 Large，VL 96–384）：prefill GEMM（计算受限）关 VPU 仅 −5.9~−7.9%；decode GEMV（访存受限）关 VPU −39.0%（VL=96）→ −20.9%（VL=384）——宽向量每指令处理更多数据、每指令访存停顿更少，VPU 可隐藏窗口变小。VPU 把约 1/4 的 decode 执行时间从内存停顿转为有效计算。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：非投机（不污染状态、操作数就绪检查）、单级向量缓冲、复用 DMA 引擎既有能力（无新增数据通路）。同类参照：PUMICE（DAC 2023）的访存-计算重叠。使用场景：乘法密集 + 流式 DMA 的 kernel（PrIM 的 GEMV/TS/MLP、LLM decode GEMV）；计算受限的 GEMM 收益小，应靠 AP 原始吞吐而非重叠。

涉及论文标题：
- BAAP: Coupling Compute-in-SRAM with DRAM Banks for Near-Memory Processing

## PIM 运行时数据布局管理（bit-sliced / column-contiguous / baap_set_mode drain-flip-reload）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BAAP 三模式各有"自然布局"（论文 §III-B2）：Scratchpad 模式字节寻址；SIMD 模式位切片（bit-sliced：第 i 子阵列存所有向量元素第 i 位，利于位并行/位串行算术）；DirectAP 模式列连续（column-contiguous：整词单周期 CAM 匹配）。布局责任分四层：① host 用扩展的 scatter/gather/broadcast（带 layout 参数选 stride）做初始摆放；② DPU 运行时库 baap_set_mode(mode, m) 把 m 个向量寄存器工作集 drain（若脏则排回 DRAM bank）→ flip → reload 成新布局，纯 DPU 侧 DMA strided 访问实现、无新硬件；③ kernel 主模式编译期静态声明，ISA 允许隐式换模（跨模式指令）或显式换模；④ 硬件写 CSR 切换灵敏放大器（控制级、无微架构状态排空）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
成本模型（论文）：最坏重排 t = 2×6KiB ÷ 72.58MB/s ≈ 0.171ms ≈ 6×10⁴ cycles@350MHz（72.58MB/s 为最坏细粒度访问的每 bank 有效带宽 [21]），相当于 2–15 次 SIMD 乘法。
```
baap_set_mode(new_mode, m):        # DPU 侧运行时例程
  for v in working_set[0..m-1]:
    if dirty(v): dma.store(bank_addr(v), v)   # 排空到 bank
  flip_layout()                      # CSR 切换 + 布局元数据更新
  for v in working_set[0..m-1]:
    dma.load_strided(v, bank_addr(v), new_mode.stride)  # 按新布局重载
```
决策规则：频繁换模不摊平 → 应单模式映射 + 特定阶段 host 回退/协同（同固定功能 PIM 与 UPMEM 基线的做法）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
类比 Castle/CAPE 的 mode reconfiguration。使用：BAAP 评估的 PrIM/Phoenix kernel 均为固定模式（免换模）；基因组 k-mer 计数与图遍历两阶段都可用 DirectAP 布局连续执行。布局选择即算子选择：位切片适合算术、列连续适合匹配——编译器/程序员在 kernel 粒度静态声明，运行时按需 drain-flip-reload。

涉及论文标题：
- BAAP: Coupling Compute-in-SRAM with DRAM Banks for Near-Memory Processing

## Time Dilation（时间膨胀调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Time dilation（时间膨胀）是 BULLETTIME（ISCA 2026）提出的运行时线程节奏均衡机制，名字借自《黑客帝国》"子弹时间"（时间放慢、视角正常移动）的比喻。其要解决的问题是：tracing 框架把 trace 落盘的 I/O 延迟是 bursty 且跨线程不对称的（只插桩内存指令时，访存密集线程被反复 I/O 停顿拖慢、计算密集线程几乎不受影响，内核守护线程则完全不受影响），导致应用线程之间、应用与系统守护线程之间的操作重排，破坏被研究行为（内存连续性、同步）相对 untraced 执行的保真度。
- 核心思想：把"tracing 延迟 : 执行进度"的比值在所有相关线程间拉平到最慢线程的水平——对"较快"（tracing 延迟少）的线程注入额外延迟，使其节奏与最慢线程一致，从而恢复 key operation 的原始顺序（正确性条件 C2）。窗口长度 L=1 即 lockstep 强制（任一线程发生 tracing 延迟时所有线程同时停等）；L>1 的窗口化把可能的重排限制在单个窗口内，以有界近似换取远低于逐操作 lockstep 的开销。
- BULLETTIME 的工程实现分两部分：对应用线程用 Buffer-Driven Delay Module 在每线程 trace buffer 填满的 I/O 事件上注入延迟；对内核守护线程（khugepaged 等，不被插桩、无 I/O 事件可用）用 Sleep Dilation Kernel Module 放大其睡眠时长。可选 zstd -7 在线压缩（BT-Comp）以空闲 CPU 换 I/O，抵消延迟注入带来的运行时间增长。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 理论算法（论文 Algorithm 1，窗口化时间膨胀）：输入为 traced 执行 Ops^t 按每线程 L 个操作切成的窗口流。
```
for each window of L operations per thread in Ops^t:
    tracingDelays = [#tracing operations in this window, per thread]
    maxDelay = max over threads of tracingDelays
    for each thread thd:
        injectedDelay[thd] = maxDelay - tracingDelays[thd]
        execute first (L - injectedDelay[thd]) operations of thd's window,
        then stall thd for injectedDelay[thd] time units
        defer last injectedDelay[thd] operations to next window
```
  每窗口把各线程的比值 tracingDelays[thd] : (L − tracingDelays[thd]) 拉平；L=1 退化为 lockstep（精确恢复顺序），L>1 时重排只可能发生在窗口内、跨窗口顺序由窗口粒度上的 lockstep 保证。
- BULLETTIME 的实际调度循环（论文 §IV-C，图 9）：应用线程执行指令 → Pin 把访存 trace 写进 2MB 线程内 buffer → buffer 满触发 I/O 事件（= 一次"窗口"边界）→ Controller 进程接管落盘（O_DIRECT 直写 SSD）并用 EWMA（窗口 5s、衰减率 0.5）更新每线程 progress（= 指令数，用户态/内核态按各自 IPC 加权）与 tracing delay → 计算各线程 progress-to-delay 比值、识别最慢线程 → 对较快线程计算出 injectedDelay，在其下一次 buffer 填满事件时注入（图 9 例子：Thread2 第 2 个 I/O 事件前 progress 4:delay 4，落后于 Thread1 的 2:2，注入延迟把二者拉平；已平衡时注入 0）。内核线程侧：Controller 周期性把最慢线程的减速因子（≈ 应用 CPU 时间 / 等待落盘时间）经内核模块放大 khugepaged 等守护线程的睡眠长度。
- 与 COZ causal profiler 互为镜像：COZ 按比例放慢其它代码段来估计某段代码的潜在加速收益，BULLETTIME 则按比例放慢其它线程来抵消某线程被 tracing 拖慢的效应。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 论文实现（Intel Pin 3.30，Linux 6.13，i7-8700 + SATA SSD）：Pin 内新增 BULLETTIME Controller（拦截 buffer 满事件、算延迟、管落盘）；Pin 外两个模块——Buffer-Driven Delay Module（向应用线程注入延迟）与 Sleep Dilation Kernel Module（kprobes 插桩 schedule_timeout 与 hrtimer_nanosleep、仅对 TASK_INTERRUPTIBLE 任务延长睡眠）。落地要点：(1) 不必跟踪单个 key operation，只需等比例拉平各"key thread"的进度/延迟比；(2) 进度用 EWMA 估计（无法前瞻未来窗口）；(3) O_DIRECT + hugetlbfs 内部 buffer 使框架自身不破坏内存连续性（条件 C1）。效果：Misplaced Memory 平均 8.89% vs Disk-Traced 35.49%、DynamoRIO 56.77%；同步研究 GET:UPDATE 比值保持在 untraced 的 10% 内（Disk-Traced 最高偏离 25×）；代价是运行时间较 Disk-Traced 平均 +35%（最高约 58–60%），BT-Comp 压缩后改善 >2× 且精度无损。代码开源：https://github.com/ysarch-lab/BulletTime。
- 使用方式：用户通过提供一组函数标识 key threads（线程执行到这些函数即被纳入膨胀）；适用于任何数据生成率显著超过存储带宽、且需要无损（完整）trace 的场景。

涉及论文标题：
- BULLETTIME: Time Dilation for High-Fidelity Tracing

## 输出切分/输入切分映射（Output-split / Input-split）与 GeMV/GeMM 形态

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
把一层矩阵乘法 W·X 切分到多个 PIM bank 的两种策略：output-split = 按输出维把 W 的行切给各 bank，各 bank 算自己的输出段，输入 X 需广播到所有 bank（无归约、有广播）；input-split = 按输入维切 W 的列与 X 的段，各 bank 算部分和，需跨 bank 归约（有归约、无广播）。GeMV（广义矩阵向量乘，batch=1 形态）与 GeMM（batch>1）是同一算子在不同 batch 下的两种形态：GeMV 内存受限（每字节权重只做一次乘加）、GeMM 计算受限。DRAM-PIM 传统回避 input-split——global buffer 的归约带宽有限且需串行访问 bank——被迫用 output-split，但 output-split 造成形状失衡：CompAir 中 Llama2-13B Q/K/V 每 bank 权重为 5120×10（输入输出比 >17:1），输入广播代价大。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CompAir 的分析流程：① 对每个 FC 算子判断形态——batch 增大时 Q/K/V 投影从 GeMV 转 GeMM，SRAM-PIM 收益出现（batch=32 6.3×、batch=1 无收益）；② SRAM-PIM 偏好平衡映射（均值不等式：输入输出维相近时带宽需求最小），(512,8) output-split 形状失衡、(256,16) input-split 降带宽压力；③ CompAir-NoC 归约树消除 global buffer 串行归约后，input-split 可行甚至更优（2560×20 一致优于纯 output-split）。伪代码（bank b 视角）：
```
# output-split：X 广播给所有 bank，无归约
Y[b] = W[b] @ X          # 每 bank 输出段
# input-split：X 分段，部分和经 Reduce 树归约
Y = Reduce('+', [W[:, b] @ X[b] for b in banks])
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GPU 上对应概念是权重/输入切分 + all-reduce；PIM 上归约经 global buffer（串行、慢）或 NoC 树（并行、快）完成。选择规则（CompAir）：按算子形态选硬件（GeMM→SRAM-PIM、GeMV→DRAM-PIM）；切分维按归约成本与广播成本权衡；TP 沿 seqlen 切 K^T/V 时 seqlen 映射为 SRAM-PIM 的 batch 维、输出维对齐 GQA group size。Qwen 8K 采用 input-split 使本地指令 +27%、但经紧凑的 NoC_Reduce 稀释为系统级 +2%。

涉及论文标题：
- Bridging Efficiency and Scalability in LLM System via 3D Hybrid PIM with 2D In-Transit Computation

## 归约树 / 广播树（树形集合通信：reduce/broadcast tree）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
在互连网络上以二叉树并行完成 N 路归约：16 宽归约 = 4 层二叉树，2^N 个节点恰好需要 2^N−1 个中间节点（每个中间节点都被利用）、每非叶节点累加两个子节点结果，深度 log2 N。广播是归约在树结构上的逆操作（根分发到叶）。LLM 里 Softmax 的 max/sum、input-split 的部分和、TP 的跨设备汇总都需要归约/广播；传统 PIM 靠 global buffer 做串行归约，带宽受限且需串行访问 bank，成为 input-split 的阻碍。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CompAir 的实现：bank 为归约粒度，ArgReg 作每非叶节点的累加器；行级指令 NoC_Reduce(OP, Addr, Addr, Mask, DstBank) 由编译期按固定树模板 + bank id 实例化为各 bank 的 packet 序列；支持 4 棵并行树（64-bit Mask 决定宏参与）。Softmax 例子（16 bank）：
```
# 每 bank 本地 exp 部分和
for b in 16 banks: part[b] = sum(exp(x)) over local slice
# 4 层归约树（每层 NoC_Reduce packet，ArgReg += 子节点）
for level in 1..4: tree_reduce('+', part)      # 深度 log2(16)=4
# 根 bank 得总和 → NoC_BCast 沿树反向下发
for b in banks: softmax[b] = part[b] / total
```
相对集中式 NLU：数据无需搬出 bank 再搬回，通信与计算合流、避免根节点拥塞。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
硬件：NoC/交换机内归约树（CompAir-NoC、NVLink SHARP 交换机内归约）；软件：ring/tree all-reduce。使用方式：input-split 映射的配套机制——归约树效率直接决定 input-split 是否可行；中间节点复用（2^N−1 个中间节点对应 2^N 叶）保证满利用率；广播与归约共享同一套树硬件（互为逆）。

Raptor 补充视角（ISCA'26，层级 collectives）：Raptor 的 all-reduce 用层级分解（hierarchical decomposition）实现——reduce-scatter + all-gather 两阶段：数据归约的主体在 chiplet 内经 on-chip NoC 局部完成，越往 MCM（D2D）与卡（PCIe）层级交换的消息越小；allgather 实现为广播，利用源侧多播（source-side multicast）能力降低高层拥塞。每 transformer 层触发两次 TP all-reduce（attention 投影后一次、FFN 后一次），各传输 O(h·b)（h=hidden dim、b=micro-batch）。collective 数据量由并行度决定：3D-DRAM 的高每卡容量使部署用更少卡/更低 TP（Llama-70B TP=4 或 1 vs SRAM 的 TP=8），参与者少、传输量小 → 对网络延迟/带宽不敏感（0.01-10µs 扫描中 SRAM 因高 TP/PP 下降最快，3D-DRAM 在 4K 上下文现实网络 0.5µs/1TB/s 下 4.38× vs HBM、3.15× vs SRAM）。MoE disaggregated 部署每层最多四种 collective：attention 组内 all-to-all 交换部分注意力输出与 log-sum-exp（~16KB/card @TP=4）、post-attention all-reduce（∝h·b）、dispatch many-to-many（数百 KB/card）与 combine many-to-many（MB/card），dispatch/combine 随激活专家数与 EP 度缩放。

涉及论文标题：
- Bridging Efficiency and Scalability in LLM System via 3D Hybrid PIM with 2D In-Transit Computation
- Early Silicon of Raptor: The First 3D-DRAM Accelerator for Generative Inference
- PipeComm Maximizing Link Utilization through Pipeline-Aware Collective Communication Synthesis

PipeComm 补充视角（ISCA'26，拓扑感知综合中的树形 pattern）：PipeComm 的通信 pattern 构建以"有向 spanning tree"为基本单元——每个 pattern 是一棵从根节点广播/归约的有向树，树的边（链路）选择由 MILP 决策变量 x_{s,e} 决定、深度由 l_{s,v} 建模（Eq.4）；广播与归约利用对偶性（reverse 原语：交换 reduce/broadcast 方向、翻转边）互为逆操作。关键差异：①相比固定二叉树模板（log2 N 层），PipeComm 的树由求解器按拓扑异构带宽最优选边（不假设 uniform/对称）；②多个 pattern 在 II 容量约束（Σx≤II/w）下并行共存并跨迭代重叠（流水线化），而不是单棵树顺序执行；③AllReduce 用"多棵广播树 + 多棵归约树交错（interleave）"完成（3×3 2D mesh 上 II=1 可容纳 2 broadcast + 1 reduce 三个 pattern），而非经典的 ReduceScatter+AllGather 对称分解——这使 reduce 与 broadcast 相位在同一 pipeline 内重叠，比单相位 AllGather 有效提速 1.45×/1.16×。真实 GPU（16×L20）上 Pipe-Sol 的树形 schedule 平均 1.24× over NCCL。

## Tensor Parallelism（TP，张量并行）与 Pipeline Parallelism（PP，流水并行）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TP = 把单层权重按输出/头维度切分到多个设备，激活广播、输出经归约汇总，通信频繁但同步细粒度；PP = 按层把模型切成多段串成流水，通信少但流水气泡多。CompAir 的发现：CENT 原实验用全 PP（单 token 延迟显著增大），CompAir 改用 8 设备 TP=8 的均衡配置；扫描 TP=1..32 发现高 TP 下 bank 利用率骤降（每 bank 分得的 batch 复用机会减少）、吞吐退化，TP≤8 为最优范围（Llama2-13B 在此范围端到端 1.5–2.14×）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CompAir 的 TP 映射例子（GQA attention）：
```
# TP 沿 seqlen 维切分 K^T/V：每 bank 处理 seqlen/TP 段
for bank b: Q_b = Q[:, b*L/TP:(b+1)*L/TP]   # L=seqlen
# SRAM-PIM 视角：seqlen 映射为 batch 维、输出维对齐 GQA group size(8)
scores_b = Q_b @ K_b^T     # QK^T 依 TP/seqlen 决定 DRAM 或 SRAM
out_b = softmax(scores_b) @ V_b
```
权衡：TP 增大 → 每 bank 的 batch 变小 → SRAM-PIM 权重复用优势被稀释，同时数据搬移增多 → CompAir-NoC 归约/在途计算收益上升。结论：TP≤8；SRAM-PIM 的性能优势来自数据复用，过度并行削弱复用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Megatron-LM 式 f/g 切分 + all-reduce；PIM 场景用 NoC 归约树替代 all-reduce（见归约树条目）。使用方式：按设备数与模型大小选 TP/PP 组合；注意 bank 利用率与 batch 复用（SRAM-PIM 权重驻留场景对 TP 尤其敏感——TP 提高直接削薄每 bank 的有效 batch）。AttAcc 基线（4×A100 + 4×HBM3-PIM）与 CENT 均以 TP/PP 做设备级并行，CompAir 的 96 设备配置在该均衡策略下与 AttAcc 吞吐相当、能耗 3.52× 更低。

ConServe 补充视角（ISCA'26）：Yi-34B-200K（Hq=56、Hkv=8、L=60）以 TP=2 跨 2×A100 NVLink 部署。TP 切分下每卡只持有部分层的 KV，ConServe 的 VA slice 按分片内层数 L_shard 分层分段（每 token KV 字节 B_tok=2·L_shard·H_shard·d_head·b），slice 大小与 resize 触发频率都随 TP 分片减小；跨卡通信由框架负责，allocator 本身不涉及 TP 通信。

DynoPipe 补充视角（ISCA'26）：PP 用于边云异构域（边缘 RTX 3090 ∥ 云端 A40），与同构数据中心 PP 的关键差异是显式建模边界 stage 开销 T_boundary（Eq.1 的 I_boundary·T_boundary，含激活序列化/反序列化、跨域传输、状态同步、格式转换）与动态 split point（LRP 按遥测切换）。TP 跨边云被判定不可行——AllReduce 开销从 NVLink 0.8ms 增至边云链路 25ms（94% 效率退化，LLaMA2-70B attention），故约束"每请求单一边云边界"（跨域延迟比域内高 10-50×）。切分只在完整 transformer block 之间（残差块内解析、跨域只传 fully-resolved hidden state、无额外 buffering/重算），数值一致性 <10⁻⁶ 相对误差。

HybridSpec 补充视角（ISCA'26，HB 栈 logic block 间 TP 的细粒度计算-通信重叠）：权重按输出/输入通道交替切分到 4 个 logic block（常规 TP 切分），但执行方式改进——不先算完所有计算再发起集合通信（图 7(a)），而是把计算分解为 tile、每个 tile 与 ring-based 通信流水重叠（图 7(b)），大部分通信延迟被计算掩盖（图 7(c) 时间线）。本质是 kernel 层的"通信计算 overlap + 环形通信"优化，把 TP 的 all-reduce 瓶颈从"每层一次同步"细化为"逐 tile 流水"。设备级 TP 只用于 HB 栈内部（draft 模型）；更大模型（Qwen3-32B/OPT-66B）在 HybridSpec 中走数据并行（每设备独立完整请求、无跨设备通信），与 GPU baseline 需 TP+all-reduce 形成对比——这是 HybridSpec 相对 GPU 的优势来源之一。

从kernel调度角度拆解（伪代码示意）：
```
# 权重沿输出通道切分到 4 个 block；计算按 tile 流水
for tile_t in tiles:                        # 沿序列/输出维分 tile
    partial = GEMM_tile(block_i, tile_t)    # 本地计算
    ring_send(block_i, partial, next)       # 与下一 tile 计算重叠
    acc = ring_recv(block_i, prev)          # 环形聚合部分和
```
每个 block 边算边把部分和沿 ring 传给邻居，末轮聚合出完整输出；通信延迟被后续 tile 计算隐藏。

实现与使用：device 级 TP 的标准实现是 Megatron-LM 式 f/g 切分 + all-reduce；HybridSpec 在 block 级用 tile+ring 流水替代整层同步 all-reduce；评估经事件驱动模拟器（silicon-derived 参数）验证通信隐藏效果。

BusyBarn 补充视角（ISCA'26，wafer-scale 上的混合并行映射）：BusyBarn 把混合并行用作 wafer-scale 部署的核心手段——把 die 划分为 die 组，die 组之间做 Pipeline Parallelism（PP），每个 die 组内部联合应用 Sequence Parallelism（SP）、Context Parallelism（CP）与 Tensor Parallelism（TP）。分层依据：TP/CP 相对 PP 的通信比更高 [41]，故把高频通信的并行放在组内（die 内 mesh 近距、低延迟），把低频的 PP 放组间（跨 D2D 链路）。inter-die 映射用 SA 把 transformer block 层按 Hamiltonian Loop 排到 die 组（适配自回归 PP 的递归数据依赖，见"层次化 SA 映射（Hamiltonian Loop / ZigZag）"条目）；intra-die 映射在组内用第二个 SA 把 SP/CP/TP 算子的数据切片分配到 core。端到端模型（GPT-NeoX-20B、OPT-30B、Qwen3-32B、Llama-3-70B、Qwen3-MoE-30B、Qwen2-MoE-57B）以三种 wafer 拓扑（HW1 5×5/HW2 7×12/HW3 8×8）评估，混合并行 + BALD 通信使端到端相对 Tangram ZigZag+Gemini+XY-YX-FT baseline 加速 1.08–2.14×（几何均值 1.40×）。

- PIPEWEAVE: Synergizing Analytical and Learning Models for Unified GPU Performance Prediction

PIPEWEAVE 补充视角（ISCA'26，把 TP/PP 的 E2E 延迟预测当作 kernel 序列建模问题）：在分布式 E2E 推理中，TP 与 PP 不仅改变 kernel 切分方式，还引入通信 kernel——TP 产生 All-Reduce（activation 归约），PP 产生 Send/Recv 原语。PIPEWEAVE 对计算 kernel 逐个用解析特征+MLP 预测（Workload Generator 按 SGLang/vLLM 的 kernel 调用逻辑生成串行 kernel 序列、假定无重叠、求和得 E2E 延迟）；对通信 kernel 用简化方法：跨不同网络拓扑与通信量 profiling 建性能基线库，再用 Random Forest 回归估计通信延迟。评估覆盖 SGLang（Qwen3-32B TP=2、Llama3.1-70B TP=4/8）与 vLLM（Llama3.1-70B TP=4&PP=2），20 种配置平均 MAPE 6.6% vs Neusight 34.7%。论文还观察到 E2E 误差可能低于 kernel 级误差（如某 baseline E2E 0.5%）：E2E 聚合大量 kernel 造成系统性误差抵消，且 E2E workload 维度更窄、常落在 baseline 预测"甜蜜点"——因此 
ShadowUpdate 补充视角（ISCA'26，UVM 评估用的 column-wise TP + output-tiled GEMM 微 kernel）：ShadowUpdate 为评估 LLM 在 multi-GPU UVM 下的表现，构造 QKV projection 与 FFN 微 kernel：GEMM 核心用 output-tiled kernel（输出矩阵沿 M 与 N 维分 tile、迭代 K 维累加，仿 cuBLAS 结构），并按输出维做 column-wise tensor parallelism——把权重矩阵沿输出维切分到各 GPU，每个 GPU 算部分和、输出跨 GPU 拼接，从而在 MGPUSim 上模拟多 GPU 执行。五个模型（Llama3-8B、Llama2-7B、Mistral-7B-v0.3、Deepseek-llm-7b-chat、Qwen-14B）的 QKV 平均 1.37×、FFN 平均 1.42×；因 GEMM 主导且 tile 执行 + 相同 job 分配策略下页级 UVM 行为一致，各模型收益稳定。另外评估了三种 CTA/线程块调度方案对 ShadowUpdate 的影响：分布式 CTA 调度（baseline，1.40×）、CTA Clustering（相邻 CTA 共调度提 L1 局部性，降迁移频率，仍 1.24×）、LADM（联合优化线程块与数据放置，1.26×）——调度改善共享/远程访问但无法消除共享页迁移引发的 re-fault。

RoCC 补充视角（ISCA'26，Column-Linear/RowLinear 张量并行的 CC 卸载）：RoCC 论文用两种线性张量并行构造评估 workload：Column-Linear（权重按输出列切分，各 rank 算自己分片后 AllGather 收集）与 RowLinear（权重按输入行切分，各 rank 出部分和后 AllReduce 归约），外加 Expert Parallelism 的 AllToAll。关键量化：PyTorch distributed+NCCL 实测 CC 占 tensor 并行执行时间 40%-70%（输入尺寸不同而变），GEMM 与 CC 顺序执行时 CC 是主要瓶颈；RoCC 把 TP 产生的 AllGather/AllReduce/AllToAll 全部卸载到 ROP，SM 全容量算 GEMM，CC-only 延迟（大消息）AllReduce 35%、AllGather 11%、AllToAll 25% 加速，RoCC-Overlap 相对顺序 baseline 平均 51% 加速。
  - SHyLA 补充（inter-chiplet 空间切分）：16-chiplet MCM 采用 Megatron 式分区——QKV Generation/FFN1 的 Weight 按宽切分（本地算完整 IA 和）、Attention Output/FFN2 按高切分（需 all-reduce 累加）；ATTN 沿 head 维（MHA）或 attention group（GQA，g<pt 时 sequence parallelism）；MoE 把专家并行 pe 嵌套在 pt 维内。pipeline 并行只在 stage 最后 block 末尾通信且可与 tensor 通信重叠。片间通信（每 block 2 次 all-reduce，经 ICNT_BW 429GB/s）相对片内访存可忽略。DSE 部署空间含 (pp, pt, pe) 与 prefill/decode 各自的并行度；PD aggregation 下 (pt,pp)=(8,2)。
- STAGE 补充视角（ISCA'26）：STAGE 用符号张量表示把 TP/PP 编码进算子形状，自动推导通信。TP 分 Row/Column 两种切分：Column TP 把权重沿输出维切 w[H,4H/tp]、每设备算 y[B,4H/tp] 局部输出、需 AllReduce 汇聚 partial sum；Row TP 切 w[H/tp,4H@1/tp]、输入侧先 AllGather 恢复完整 x、输出 y[B,4H@1/tp] 是 partial sum 再 ReduceScatter。生产者/消费者分布不一致时（如 producer 输出 [a,c@1/tp] 而 consumer 期望 [a,c]），通信匹配器自动插入 AllReduce；TP 与 SP 组合用 AllGather+ReduceScatter 替代 AllReduce。PP 用图级分布实现：把计算图按层均分为 PP 段（规则脚本按 num_stacks 划分），在跨图边上按源/目的 rank 插入 send/recv 对，不产生张量级通信。STAGE 的 Table III 用符号记法（如 x[B,tp]→w[H,4H/tp]→y[B,4H/tp]）系统枚举了 dp/tp/fsdp/hp 等分布的张量表示，并在 Table VI/VII 中验证 TP/PP 配置下算子时间与通信量误差（如 GPT-3-5B TP8-w/SP 总误差 6.7%、通信误差 0.237%）。


- Symbiotic MLLM Serving: Dynamically Balancing Parallelism Across GPUs and Resources Within GPUs
RESONATOR 补充视角（ISCA'26，encoder 的 DP/TP 动态选择 + logical sharding）：RESONATOR 把 TP/DP 用于 MLLM 的 vision encoder（ViT-675M/MoonViT，权重小、并行选择灵活），关键差异是"TP 度逐 batch 运行时选择而非 AoT 固定"——PRISM 把请求排队建模为 MCKP 按 GPU 预算选每请求 TP 度（低分辨率 1 GPU/DP 最优、高分辨率 4-TP 最优，Figure 5：encoder compute 随序列长 ~二次增长、TP 通信随 ~线性增长）；TP 切换用 logical sharding（strided GEMM 只改 ld 参数，不搬权重）实现近零开销。TP 的通信是 NVLink all-reduce 类，其等待间隙的 SM 被 Intra-GPU 引擎回收给 co-located decode（TPOT 收益来源之一）。Data Parallelism（DP，各 GPU 处理不同请求、无跨设备通信）在低分辨率/高并发下最优——DP 与 TP 的价值随负载在 latency（TP）与 concurrency（DP）间权衡，是"无单一静态并行度最优"（Figure 10 landscape：低 RPS 8TP 最优、高 RPS 低分辨率 8DP 最优、高分辨率 2DP-4TP 最优）的实证。
涉及论文标题：
- Reducing Page Faults via Invalidation-based Mapping Propagation in Multi-GPU Systems
- Scalable Synthesis of Distributed LLM Workloads Through Symbolic Tensor Graphs
- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding
- Bridging Efficiency and Scalability in LLM System via 3D Hybrid PIM with 2D In-Transit Computation
- ConServe: Contiguity-Preserving Memory Management for Multi-Turn LLM Serving
- DynoPipe: Heterogeneous Edge-Cloud LLM Serving with Dynamically Orchestrated Pipeline Boundaries
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
- Mapping and Communication Optimizations with Fault Tolerance for Wafer-Scale LLM Inference
- Rearchitecting the Datacenter Lifecycle for AI
- RoCC Harnessing Raster Operations Pipeline for Efficient Tensor Collective Communication
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity

MERIDIAN 补充视角（ISCA'26，PIM 设备上的 TP/PP/Hybrid 模型映射）：MERIDIAN 的调度器在 32 个 CXL Type-3 PIM 设备上做模型映射——(1) tensor 并行：CEC 的 FC 层跨设备分片（每设备存部分权重、出部分输出再聚合），轻量算子（attention/激活）集中到主设备（主设备持 query/生成 token 的 KV cache）以减通信；DAC 按 attention head 切分文档 KV、每 head 分配到最少设备（避免广播、支持 head 级并行），设备内 KV 张量均匀分布到各 DRAM bank。(2) pipeline 并行：decoder 分成多 stage、DAC/CEC 分级匹配，batch 拆 micro-batch 顺序穿流水（两 stage 时两个 micro-batch 并行）。(3) hybrid：stage 跨多设备时内部用 tensor 并行，接口可配组合与粒度。与 GPU TP（Megatron 式列/行切分 + all-reduce）的差异：通信对象不同——CEC 聚合部分和（tensor 并行同步 partial results）、DAC 只传紧凑注意力统计量；pipeline 只传轻量激活，因此扩展性更好（32 设备 pipeline 4.19× vs tensor 3.68×）。ICE 交错调度进一步让 DAC/CEC 并发推进（见"Interleaved Cluster Execution"条目）。

MTIA 300 补充视角（ISCA'26，LLM 推理的 TP8-TP8 与 DP8-EP8 配置）：MTIA 300 评估 DeepSeek-R1（8 加速器，InferenceMax + vLLM）用了两种分片策略：(1) **TP8-TP8**——每层权重矩阵切分到全部 8 设备，每设备算每操作的一个切片、每个并行区域后 AllReduce 同步（对应本条目标准 TP）；(2) **DP8-EP8**——稠密层（attention + 共享 MLP）8 设备复制、各自处理独立 batch（DP），MoE 部分每设备拥有 1/8 专家、token 经 AllToAll 在设备间路由（对应"Expert Parallelism"条目）。BF16 做 attention/KV cache、FP8 做 MoE 计算（两平台原生支持）。结果：decode 主导下靠高 HBM 带宽在高并发（>64）优于 H200；低并发差距小（MTIA 300 小 batch 通信开销 vs H200 NVLink、部分 kernel 未按 batch/token 维充分并行致 PE 未充分利用）。

## FlashInfer（native 与 paged 注意力 kernel 库）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FlashInfer 是 LLM serving 的开源 kernel 库与生成器（github.com/flashinfer-ai/flashinfer，论文 arXiv:2501.01005），提供 prefill/decode/append attention、采样等 CUDA kernel，同时支持 paged 与 ragged 两种 KV 布局：paged 实现即 block-sparse attention（paged_kv_t 用 indptr/indices 数组做页索引，page_size 为块列数，page_size=1 即向量稀疏）；native 实现按 base+offset 在单连续区域访问 KV。wrapper（BatchPrefillWithPagedKVCacheWrapper 等）走 plan()→run() 两阶段：plan 调度变长输入、构建可复用辅助结构，run 跨 transformer 层复用 kernel。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ConServe 的动机实验（同配置只换 KV 放置，Llama-3-8B、8K prefill+1K decode、A100、batch 1–16）：
```
# paged：每 access 查 vLLM block table、散页 gather
ptr = block_table[logical_block]; kv = ptr + intra_block_offset
if cross_block_boundary: ptr = block_table[next_block]   # 边界检查+查表
# native / ConServe：base+offset 连续流式
VA(t,l) = base + seg_off[l] + t * B_layer + delta         # 纯算术
```
结果：FlashInfer-paged prefill kernel 慢 12–24%；Nsight Compute 显示长 scoreboard stall 84.64% vs 79.37%、eligible warps/cycle 0.718 vs 0.825、SM/L2/DRAM 吞吐 −22.4%/−16.7%/−21.1%；多轮（每轮 512 输入+64 decode）paged/native 比从 1.2× 升至 1.75×。ConServe 用其 variable-length 模式 + 紧凑描述符（每序列 KV base 指针 + live 长度）承载连续寻址。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SGLang/vLLM/MLC-LLM/TRT-LLM/TGI 等默认集成；支持 FA2/FA3、CUTLASS/cuDNN 后端自动选择（Turing–Blackwell）、CUDA Graph、cascade attention（共享前缀层级 KV）、POD-Attention、稀疏注意力与 FP8/FP4 量化。使用：动态 batch serving 中替换自研 attention kernel；页大小可配（page_size=1 用于 SGLang 的 token 级 KV 裁剪）。Web 证据：https://github.com/flashinfer-ai/flashinfer 与 https://docs.flashinfer.ai/api/attention.html 确认 wrapper 与 paged KV API。

涉及论文标题：
- ConServe: Contiguity-Preserving Memory Management for Multi-Turn LLM Serving

## 长 Scoreboard Stall 与 Eligible Warps（GPU kernel 访存/翻译延迟度量）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Nsight Compute 的 SM 调度度量：long-scoreboard stall = warp 等待长延迟内存操作（TLB miss、页走查、全局访存）的 stall 占比，值越高说明可用算术不足以掩盖访存延迟；eligible warps per active cycle = 每活跃周期可发射的 warp 数，衡量调度器可用并行度。两者配合定位 kernel 是"延迟受限"还是"吞吐受限"：stall 高 + eligible 低 ⇒ 内存系统（含地址翻译）是关键路径；stall 低 + 吞吐低 ⇒ 指令供应/计算问题。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ConServe 的用法（A100，Llama-3-8B，8K prefill+1K decode，batch=8）：FlashInfer-paged 长 scoreboard stall 84.64%、eligible warps 0.718；native 79.37%、0.825。解释链：散页 KV 布局 → TLB 工作集超容量、页走查增多、页走查缓存（PTE cache）复用下降 → 更多 warp 挂在翻译上 → 可发射 warp 变少 → 每周期发出的访存/计算指令变少 → SM/L2/DRAM 吞吐 −22.4%/−16.7%/−21.1%。即用这两个指标把"布局差异"归因到"地址翻译延迟"，支撑 contiguity-preserving 设计动机。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Nsight Compute（ncu）profiler 采集（论文引用 docs.nvidia.com/nsight-compute/NsightCompute/index.html）。使用：对比 kernel 实现、验证访存布局优化（如 KV 连续化）的微架构效果；配合 SM/L2/DRAM 吞吐、TLB/页走查统计使用。局限：百分比指标必须在同配置下对比才有意义——ConServe 强调两组配置同上下文长度、同逐 token 算数、仅 KV 放置不同。
- HyperDrive 补充视角（ISCA'26，FHE NTT kernel 的 warp stall 分类与应用）：用 Nsight Compute 的 warp state 分类定位 FHE kernel 的访存瓶颈，六类状态——Stall Long Scoreboard（长延迟数据依赖，如 GMEM 访问）、Stall LG Throttle（global/local 访存指令队列满，GMEM 访问密集）、Stall Short Scoreboard（短延迟 I/O 依赖，如 SMEM 访问）、Stall MIO Throttle（内存 I/O 指令队列满，SMEM 访问密集）、Compute（等待功能单元完成）、Eligible（就绪等待发射）。量化结果（N=2^16、36 limbs）：baseline FP64-TCU NTT 的 Inner-NTT stage1/2 中 SMEM 相关 stall 占 32%/48%、总 stall 占比 77%/72%；Outer-NTT stage1/2 的 GMEM 相关 stall 占 45.9%/22.6%；跨多项式 kernel 的 stall long scoreboard 占 BConv 60.6%、IP 74.6%。TLMOP 使 SMEM stall -33.2%、scheduler stall cycle -44.2%（加 TransOP 累计 -50%）、occupancy 55.0%→76.8%；TFOP+RowMaj 使 GMEM stall -59.5%、scheduler stall -39.3%。用法：对每个 kernel 分别统计 warp state 占比并叠加消融配置，把"stall 种类"归因到"具体内存优化"。
- MNEMOS 补充视角（ISCA'26，TFHE PBS kernel 的 stall 分解与优化归因）：对 PBS 最耗时 kernel（盲旋转）的 stall 统计显示 baseline 中 stall_long_scoreboard 占 >50%（多数参数集），与 stall_MIO_throttle（共享内存/L1 操作争用）合计超 60%，证明 PBS 为 memory-bound；根因是 BSK 作为热数据被多 SM 并发访问、仅 L2 级复用（GMEM→L2 与 L2→SM 流量失衡）。优化后（BSK 分块复用 + Tensor Core FFT + 跨迭代融合 + 片上 Fourier 矩阵生成 + swizzle）stall_long_scoreboard 降至约 20%（Para-E 下 <15%），主导 stall 变为 Stall Math Pipe（compute-bound）；FFT kernel 的 stall_MIO_throttle 延迟降 3.2×/2.9×/1.8×（N̄=256/512/1024）；GMEM→L2 流量平均降 15.7%、L2→SM 降 69.4%。用法：以"stall 种类占比变化"作为内存优化是否生效的判据——memory stall 让位于 math pipe stall 即说明访存瓶颈已解除。

涉及论文标题：
- ConServe: Contiguity-Preserving Memory Management for Multi-Turn LLM Serving
- HyperDrive: Hierarchical Exploitation of Memory Efficiency for GPU-Based FHE Acceleration
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization

## 去量化隐藏（Dequantization-Hiding：指令重排利用 DRAM 行切换空闲窗口）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
去量化隐藏是 FlexQ-NDP 在 NDP 指令调度层消除低比特 FP 去量化开销的技术：把与 QGroup 绑定的高精度 dequant 指令（部分和 × 激活 scale × 权重 scale，复用 PU 的乘法器执行）从原始位置"前移"到 PU 空闲窗口（free slot）内执行。观察依据：一次 DRAM 行切换（precharge t_RP + activate t_RCD = 48 cycle，DRAMSim3 参数）期间 PU 空闲，而一次 dequant 仅 8 cycle（2·t_CCDL），放入该窗口即零额外延迟。动机实验：QGroup 增大到 128 时 dequant 恰好都落在行切换窗口、贡献 0 额外延迟；小组尺寸下仅约 10% 的 dequant 天然落入 free slot，而 W-A 量化 dequant 占总延迟最高 40%。三大约束：① 数据依赖——只能前移、不能越过后续 dequant、不能越过 scale 缓冲 refill 触发的 DRAM 读（否则所需 scale 被丢弃）；② partial-sum 缓冲容量——越过 Compute 指令要扣减其产生的 partial sum 数（extra_buf 预算）；③ 空闲窗口容量——slot 已隐藏的 dequant 填满空闲窗口即失效。只用于 weight-activation 量化（weight-only 中 dequant 输出是权重值、越过计算需缓存权重，移动范围过小）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
伪代码（论文 Alg.1，逆序贪心前移）：
```
Input : 指令列表 I；未隐藏 dequant 列表 D；partial 缓冲上限 Bmax
1  S ← {}                       # 扫描全部 free slot（行切换/DRAM 读的 PU 空闲窗口）及剩余 idle cycle
2  foreach (pos, inst) in reverse(D) do          # 逆序：后到先占，保前面指令的移动弹性
3      extra_buf ← Bmax − #PartialSum(inst)
4      pos_tmp ← pos; candidate ← None
5      while pos_tmp < |I| − 1 do
6          if pos_tmp ∈ S and S[pos_tmp] > 0: candidate ← pos_tmp
7          next ← I[pos_tmp + 1]
8          if next.type == Compute:   extra_buf −= #PartialSum(next)
9          elif next.type == ReadData: extra_buf −= 0
10         else: return                # ReadScale/Dequant/WriteBack 阻断前移
11         if extra_buf < 0: return    # 缓冲容量耗尽，不可再移
12         pos_tmp += 1
13     Move(inst, candidate); 更新 S[candidate]；窗口耗尽则标记失效
```
计算过程例子（LLaMA2-7B MVM、W4A4S8、QGroup(1,16)）：PU 处理完 DRAM row-1 的权重块、切换 row-2（48 cycle 空闲）→ 把本块末尾产生的 dequant 指令前移进该窗口执行（8 cycle，仍余 40 cycle）→ 后续多个小组的 dequant 继续合并填充同一/后续窗口 → dequant 与行切换、值读取完全重叠。效果：消融实验累计贡献 ×1.18；DRAM 行切换开销平均降约 2×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：编译期指令重排 pass——对单个权重矩阵迭代切片后的指令列表做一次扫描（限制在单次迭代内、控制重排开销），先扫描记录 free slot 位置与空闲 cycle 数，再逆序移动 dequant 指令。使用：作为 NDP 编译流程 code generation 阶段的后置优化，配合"scale-value 交织布局"（布局决定空闲窗口周期）与缓冲分配（决定 Bmax）一起工作；仅在 W-A 量化场景启用。通用化洞察：任何"短计算指令"都可利用 DRAM 行切换的固定空闲窗口隐藏——窗口周期由数据布局决定、指令周期由缓冲容量决定，二者解耦是重排（而非强制对齐）能生效的关键。

P3-LLM 补充视角（ISCA'26，量化算子融合 Fusion，非指令重排）：与 FlexQ-NDP 的"指令重排把 dequant 塞进 DRAM 行切换空闲窗口"不同，P3-LLM 通过 operator fusion 从源头消除运行时 dequant 的粒度——把量化缩放因子折叠进另一个操作数的量化过程：线性层（Y=X@W_q）的 dequant 缩放放在矩阵乘法之后统一执行一次；Q·K^T 把 post-RoPE key cache 的 per-channel smoothing factor（SSF）元素乘进 query（先于 FP8-E4M3 量化），从而量化 key 无需解量化即可与量化 query 在 PIM 上相乘；P·V 把 per-value-head 缩放因子 S^V 融合进 attention-score（除以 S^V_max 二级缩放防 FP8-S0E4M4 越界，P·V 结果乘回 S^V_max）。效果：NPU 只需对整层输出做一次高精度 dequant（而非每个量化张量），配合 8-bit attention-score 使 attention 全模块在低精度 PIM 上执行；架构消融显示 W4A8KV4 + TEP 后加 8-bit attention-score（实现全 attention PIM 化）再获 1.2×。

涉及论文标题：
- Bringing Near Data Processing into the Low-Bit Floating-Point Era
- P3-LLM An Integrated NPU-PIM Accelerator for Edge LLM Inference Using Hybrid Numerical Formats

## Bubble-free pipelining（无气泡流水与 head 映射）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CHIME-PIM 的 attention kernel 流水化设计，目标是让跨芯片数据传输完全被 bank PU 计算覆盖（无气泡）。两个基础观察：(1) 内部内存总线（bank PU↔bank）与外部内存总线（rank PU↔bank PU/shared buffer）可解耦，允许 bank PU 计算与 rank PU 传输同时进行；(2) 借鉴 FlashAttention 的 chunked tile 融合，attention 按 chunk 计算并流水。具体：bank PU 每算出一个 token 的 score 输出 O^s 暂存本地 result buffer，rank PU 立即经空闲外部总线取回，adder 累加后 softmax 单元做 per-chunk softmax；全部 token 处理完后 streaming chunk-wise 做跨 chunk 归一化得到全局正确 S，S 元素写回 DRAM 与后续 context（S×V）计算再流水。无气泡条件 T_comm ≤ T_comp：跨芯片传输时间 T_comm = L_t×N_gqa×N_chips/B_rk（DIMM 多芯片协作使传输放大 N_chips 倍、GQA 进一步放大），bank PU 计算时间 T_comp = L_t×E_h×⌈N_gqa/N_cmr⌉/(B_bk×N_bk×N_hc)，由此推出 head 映射约束 N_hc ≤ E_h×B_rk×⌈N_gqa/N_cmr⌉/(B_bk×N_bk×N_gqa×N_chips)——MHA（N_gqa=1）取 N_hc=8、GQA-8 取 N_hc=1。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
score 阶段流水（一个 head、N_hc 个 chips）：
```
for chunk in chunks:                          # chunk = 单 head 跨多个 bank PU 并行产生的数据
    O_s[chunk] = bank_PUs.MAC(Q, K[chunk])    # bank PU 从 DRAM cell 读 K、从 shared buffer 读 Q
    rank_PU.fetch(O_s[chunk])                 # 经外部总线，与下一 chunk 的 MAC 重叠
    S[chunk] = softmax_unit(adder.accum(O_s[chunk]))   # per-chunk softmax
# 所有 token 完成后：
S = normalize(S_chunks)                       # streaming 跨 chunk 归一化（online softmax 修正）
# S 逐元素写回 DRAM 与 context 计算 S×V 流水
```
无气泡判据推导：T_comm = N_comm/B_comm，N_comm = L_t×N_gqa×N_hc、B_comm = B_rk×N_hc/N_chips → T_comm = L_t×N_gqa×N_chips/B_rk；T_comp = L_t×E_h×⌈N_gqa/N_cmr⌉/(B_bk×N_bk×N_hc)；T_comm ≤ T_comp 即式 (1)，N_hc 是唯一可调变量。效果：MHA 延迟 -27.9%、GQA -74.4%（对 baseline = 跨芯片映射无重叠的 bank-level CHIME-PIM）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：bank PU 与 rank PU 异步执行（rank PU 在 buffer chip 上以逻辑工艺实现，独立于 DRAM 阵列运行）；PIM 命令流 PIM_MAC/PIM_RD_RB 交替驱动；chunk 粒度与中间结果缓冲（rank PU 片上 SRAM）约束 head 足迹。使用方式：MHA 多 head 可映射多 chips（N_hc=8）利于 rank 级负载均衡，GQA-8 大 group 只能 N_hc=1；计算访存比 N_cmr 按 GQA-n 配置保证带宽利用率。类比：与 GPU 上 FlashAttention 的 tiling 目标一致（减少中间物化），但重叠对象是跨芯片数据传输而非 HBM 访存。

涉及论文标题：
- CHIME: A Case for Efficient Long-Context Attention-FC Disaggregated Inference with DIMM-PIM

## Hybrid-grained re-layout（混合粒度重布局）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CHIME-PIM 在 rank PU 上以 re-layout 单元完成的 in-flight 数据布局变换，解决 DIMM 存储布局与 PIM 计算布局不匹配的问题：DIMM 多 chip 交织下单个元素跨多个 ×8 chip（如 FP16 的 16 bit 跨 2 个 chip），而 PIM 计算要求元素完整位于单一 chip，同时计算需要按 head 映射到指定 chips。两级变换：fine-grained（元素级/位级）——把单元素各位连续排进同一 chip 的 burst beat（元素不跨 chip）；coarse-grained（head 级）——把每个 head 的元素映射到 N_hc 个 chip。数据 offload 时 QKV 先缓存在 rank PU 片上 SRAM，重排后再写入 DRAM chips；onload 反向。对比 CPU 辅助 re-layout（读旧布局+写新布局的往返访存），CHIME 把变换融合进传输路径，消除每层每 token 的累积开销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Fig.8 例子（8×8 chips、E0-E127 为 head 内元素 0-127、C0-C7 为 chips）：
```
# offload QKV: rank PU SRAM → re-layout unit → DRAM chips
for element e in head:                    # fine-grained：元素不跨 chip
    put bits(e) into contiguous burst beats of chip(e % N_hc)
# coarse-grained：按 N_hc 组织 burst beat 内容
if N_hc == 8:  # 每个 burst beat 只含单 head 元素 → 一个 head 分布到 8 个 chips
    beat = elements_of_single_head()
else:          # N_hc == 1：8 个 head 的元素混排一个 burst beat → 每 head 落单 chip
    beat = elements_of_8_heads()
# onload：reverse 过程，从 chips 读回并恢复原布局
```
效果：最多 -17% attention 延迟（随 token 长度增大占比下降，因计算主导，但 re-layout 开销每层每 token 累积，仍需消除）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：re-layout 单元做于 buffer chip（rank PU 内、逻辑工艺），数据路径上动态重排；与 bubble-free pipelining 的传输流水协同。使用方式：所有 DIMM-PIM/多 chip PIM 的必答题——UPMEM 系用 CPU/专用引擎重排、Facil 用灵活地址映射、PIM-MMU 用 MMU 级变换；CHIME 的贡献是把两级粒度（位级元素对齐 + head 级映射）统一到传输中完成。推广到其他数据（激活、权重）的布局变换同样适用。

涉及论文标题：
- CHIME: A Case for Efficient Long-Context Attention-FC Disaggregated Inference with DIMM-PIM

## Rankset-granular communication-computation overlapping（rankset 粒度通信计算重叠）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CHIME-sys 隐藏 GPU↔CHIME-PIM 之间 PCIe 通信开销的机制。问题：GPU-PIM 数据通信与 PIM attention 计算共享内存总线，粗粒度地"全 rank 要么通信要么计算"会互相阻塞，而 PCIe 带宽比 CHIME-PIM 低数个数量级，通信不可忽略。关键观察：一个通道内同一时刻只能访问一个 rank（总线共享），其余 rank 空闲。据此定义 rankset = 从每个通道取一个 rank 组成的最小独立通信/计算单元（同时用满全部通道的最小集合）：对一个 rankset 做通信时，其余 rankset 可并行做计算——DGX-A100 上 4 ranksets 时通信期间保留 3/4 计算能力（每通道 3 ranks 的例子则保留 2/3）。负载均衡：利用各层 KV cache 大小相同的特性，把每个请求的 KV cache 按 layer 粒度 interleaved 存放到各 rank，保证各 rankset 传输量相等。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
ranksets = [ {rank_i from each channel} for i in 0..R-1 ]   # R = 每通道 rank 数
# 异步流水：通信 rankset i 与计算 rankset (i+1) mod R 并行
for t in timeline:
    PCIe.write(QKV_next, rankset = t % R)      # 通信：占该 rankset 的总线
    PIM.compute_attention(rankset = (t-1) % R) # 计算：其余 rankset 不受阻塞
```
数据流：prefill K/V（∝输入长度）、decode QKV（每请求每步 1 token）、attention 输出（1 token/请求）三类数据按 rankset 轮转传输，与上一轮 attention 计算重叠。效果：PCIe 开销最多降 75.08%（不同 batch 配置，4 ranksets 的 DGX-A100）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：调度器按 rankset 分配请求的 KV 存储与传输任务，硬件侧每个 rankset 的通信与计算独立可切换。使用方式：任何"共享总线的多单元异构系统"的通信隐藏——类同 GPU 上 streams/双缓冲，但粒度由内存通道组织（channel/rank）决定；需要配合负载均衡（layer 粒度 interleaved）避免最慢 rankset 拖慢整体。

涉及论文标题：
- CHIME: A Case for Efficient Long-Context Attention-FC Disaggregated Inference with DIMM-PIM

## 空闲感知内存调度与 IWE（Idleness-aware Scheduling, Idle Window Estimator）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
内存控制器在 CPU 优先原则下，分析 CPU 请求队列预测每个 DRAM bank 与内存总线的空闲时间窗口，把 PIM 命令精确插入窗口——既保证 CPU 延迟又最大化 PIM 吞吐。IWE（空闲窗口估计器）输出 window_bank[b]（每 bank 最早服务周期）与 window_bus（全部请求的最小服务周期）。窗口有两类来源：(1) 应用级请求稀疏造成的 bank 队列空闲间隔（Chopim 的 CPU-first 已利用）；(2) 多 bank 命令在共享总线串行化导致的"ACT 已发、数据访问未到"间隔——bank 行已打开但总线忙，内部带宽浪费（Fig.3c③，CPU-first 未利用）。COSM 把第二类窗口通过"推迟过早的 ACT"转化为 PIM 执行时间。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Algorithm 1（IWE 最早访问周期估计）：
```
REQ[] = 每 bank 最早到达请求
ready_cycles = [get_ready_cycle(r) for r in REQ]  # bank 三态：Row-Closed 需 ACT(+tRCD)、
                                                  # Opened-to-target-row、Opened-to-different-row 需 PRE+ACT(+tRP+tRCD)
t = cur_tick(); cr = cur_rank(); service_time = {}
while REQ.size():
    if 存在 r.rank==cr 且 r.ready<=t:      # FR-FCFS 特性1：同 rank 行命中请求连续处理
        r = 最早就绪(同 rank 请求); t += tBL
    else:                                   # 特性2：无就绪才跨 rank 切换（防 tRT_RS 惩罚）
        r = 最早就绪(全部); t = max(r.ready, t); cr = r.rank
    service_time[r] = t; REQ.remove(r)
window_bank[b] = service_time[该 bank 请求]; window_bus = min(service_time)
```
Annotations：利用 FR-FCFS 的两个特性（行命中连续处理、同 rank 分组）使估计近真实调度序且开销小（可适配其他调度策略，改估计逻辑即可）。Command Arbiter 的用法：CPU 调度器发 ACT 时，若该 bank 窗口 ≥ tRP+tRCD+至少一列 PIM 执行则推迟 ACT 先跑 PIM；PIM 执行中 CPU 队列非空时不立即 Pause，延迟到"不推迟下次 CPU 访问的最后周期"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：内存控制器内硬件模块（COSM 面积 0.0085 mm²，占 LPDDR5 控制器 7.4% 开销的一部分）；效果较 CPU-first：平均 PIM 性能 1.21×、多利用 37.0% 可用带宽（剩余 <1% 未用）。使用：任何共享内存 CPU-PIM 并发调度；对 SIMD PIM 需扩展为"多 bank 同时空闲"的窗口预测（未来工作）。与可抢占命令、解耦传输正交组合（消融的 All 配置）。

涉及论文标题：
- COSM: A Cooperative Scheduling Framework for Concurrent PIM and CPU Execution on Mobile Devices

## 带宽解耦 CPU-mediated 数据传输（PIM_RdBuf/PIM_WrBuf + PIM_LdBuf/PIM_StBuf）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CPU-mediated transfer = 主机 CPU 为 PIM 任务搬运输入/收集结果的必要传输（写权重/激活进 bank、读结果出），与 CPU 访问共享外部总线。问题（COSM §3.3）：传输呈突发高行缓冲局部性，在 FR-FCFS 下独占控制器、饿死需行冲突的 CPU 请求——CPU 任务降速 >80%；传输占解码注意力推理时间 50–60%（KV 长度 64 时 >60%、4k 时约 50%），折合整体 CPU 性能 -40%。带宽解耦 = 把一次传输拆成两段命令：外部总线段 PIM_WrBuf（控制器→bank buffer）/PIM_RdBuf（buffer→控制器）+ 内部带宽段 PIM_StBuf（buffer→bank）/PIM_LdBuf（bank→buffer），配每 bank 1kB SRAM buffer；两段可分别利用 bus/bank 空闲窗口，不再要求内/外带宽同时空闲。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
一次写传输的调度流程（写序列 = PIM_WrBuf & PIM_StBuf，读序列 = PIM_LdBuf & PIM_RdBuf）：
```
CPU 发起 PIM 写 → 编译器分解为命令对 (PIM_WrBuf, PIM_StBuf) 入 PRWQ
PIM scheduler 每周期按优先级取命令：
    1) PIM_RdBuf/PIM_WrBuf   # 独占外部总线（瓶颈），优先利用 window_bus 前的 bus 空闲
    2) PIM_LdBuf/PIM_StBuf   # 清空/填充 buffer，防阻塞后续总线命令
    3) PIM_Exec              # 匹配当前 bank 空闲窗口
```
Annotations：内存序——同一传输派生的命令对必须按程序序执行（调度器对同 bank PRWQ 保到达序），其他命令（PIM 执行/CPU 访问）不与传输流共享专用 buffer 区域、可任意交错。时序：RdBuf/WrBuf 与同通道 Read/Write 间 tBL；LdBuf/StBuf 与 PIM_Exec(Ld/St) 镜像同一执行模型（nPTL、可抢占）→ 统一调度逻辑。buffer 容量只需覆盖单条 nPTL 长度命令的传输量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：编译器辅助翻译（读写序列自动分解为命令对，无需改用户程序与 OS）+ 内存控制器 PRWQ 队列。效果（COSM 消融）：Overlapped+Decoupled 使 CPU 性能较 Base +11.5%，叠加 IWE（All 配置）后全 workload CPU 降速 <5%。使用：PIM 系统中所有主机侧数据搬运；把"外部总线"与"内部带宽"解耦是消除传输干扰的关键（对比：baseline 全部用标准读/写命令，无专门调度）。

涉及论文标题：
- COSM: A Cooperative Scheduling Framework for Concurrent PIM and CPU Execution on Mobile Devices

## 重叠调度与 PIM_Barrier（Overlapped Scheduling，tile 级传输/执行重叠）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
把 PIM workload 切成无数据依赖的 tile（如矩阵乘子矩阵，reduction 由 CPU 做），跨 tile 并行发不同阶段的命令——收集 tile T_i 结果的同时执行 T_{i+1}，打破传统 CPU 软件串行控制的"输入传输→PIM 执行→结果收集"三阶段（Fig.7a：串行使内部/外部带宽交替空闲）。PIM_Barrier 命令插在重叠阶段边界，保证上一阶段全部操作完成后才进入下一阶段。与 GPU 双缓冲/软件流水同思想，但作用在内存通道的 PIM 命令流层面，且 CPU 仍是 reduction 的执行者。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
重叠执行伪代码（Fig.7b 时间轴：阶段内同时存在"收集 T_i 结果 + 执行 T_{i+1} + 写入 T_{i+2} 输入"）：
```
for phase in 全部 tile:
    并行发：PRWQ[PIM_RdBuf(T_{i-1} 收集) / PIM_WrBuf(T_{i+1} 输入)]   # 外部总线
         与  PEQ[PIM_Exec(T_i 计算)]                                  # 内部带宽
    PIM_Barrier    # 本 phase 全部操作完成后才进入下一 phase
```
Annotations：跨 tile 无依赖（reduction 归 CPU），故传输/执行可任意交错；PIM_Barrier 由编译器在 tile 边界自动插入。效果（COSM 消融）：Overlapped 使 CPU 性能较 Base +3.7%——但高行命中率的 PolyBench 内核因激进重叠引入额外行切换而受损；须与带宽解耦（+11.5%）和 IWE（保持 <5%）组合使用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：编译器插入 PIM_Barrier + 调度器维护 PEQ/PRWQ 双队列跨 tile 取命令（优先级：总线命令 > buffer 命令 > 执行命令）。使用：PIM 上任何"传输-计算-收集"型 workload（GEMV、attention 层）；与 GPU 侧软件流水（如 TMA+num_stages 多级流水）的区别是重叠发生在内存控制器命令级、由 IWE 窗口驱动。局限：重叠带来额外行切换开销，需按 row-hit rate 权衡。

涉及论文标题：
- COSM: A Cooperative Scheduling Framework for Concurrent PIM and CPU Execution on Mobile Devices

## FR-FCFS 与 PIM 并发调度基线（CPU-first Chopim / row-hit-aware AsyncDIMM·F3FS / All-Bank 命令）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FR-FCFS（First-Ready, First-Come-First-Serve）= 经典内存调度：行命中（ready）请求优先、其余按到达序——最大化行缓冲局部性，但突发高局部性流量会饿死需行冲突的请求（内存性能攻击，[55–57]）。PIM 并发调度的三大基线：(1) All-Bank 命令（HBM2 FiM [42]）：一条命令触发全部 PIM 单元，执行期无 bank 可供 CPU → 时间片轮转（COSM 设 95%CPU/5%PIM + 理想化零切换开销），PIM 阶段外部带宽空转、PIM 命令须等全部 bank 就绪；(2) CPU-first（Chopim [39]，ISCA'20）：bank 的 CPU 队列非空即阻塞 PIM——CPU 延迟最优但浪费 ACT→数据访问间的内部带宽，命令长=tBL 使命令总线饱和（PIM 吞吐仅 1.9×）；（Web: arXiv 1908.06362）Chopim 还有随机 NDA 写节流与 next-rank 预测抑制读改写干扰、bank 分区（host-reserved/shared）；(3) row-hit-aware（AsyncDIMM [38]，HPCA'25：PRE 触发 CPU/PIM 队列切换 + rank 内 relay 控制器；F3FS [40]，ISPASS'25：FR-FCFS 前加 mode 仲裁 + 跨 mode 请求 CAP 防饿死）——行命中率最优但随机访问 CPU 任务延迟恶化（AsyncDIMM-Bank 实测使 CPU -89.9%）。三者共同缺陷：无 CPU-mediated 传输的专门调度（与 CPU 访问同走 FR-FCFS）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
并发时序对比（COSM Fig.3）：FR-FCFS CPU-only：行命中连续服务、per-bank 队列不均；All-Bank：PIM 命令插入 CPU 流但全 bank 被占 → 外部带宽空闲① + PIM 等全部 bank 就绪②；CPU-first：请求到 Row1/Bank1 立即停 PIM → ACT 与数据访问间内部带宽浪费③；row-hit-aware：PRE 强制切队列 → CPU 延迟④。COSM 的取舍：保留 CPU-first 原则（CPU/refresh > PIM_Pause > PIM 命令的严格仲裁），用 IWE 补足窗口利用、可抢占命令补足长命令吞吐——PIM 吞吐较 All-Bank 6.0×、较 Chopim 2.8×，CPU 降速 <2.0%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：FR-FCFS 实现于各内存控制器（COSM 保留为 CPU 侧调度器，PIM 侧另设 PIM scheduler，Command Arbiter 三源仲裁）；Chopim/AsyncDIMM/F3FS 构成并发调度设计空间的两个极端（偏 CPU / 偏 PIM）与折中。使用：CPU-PIM 并发系统的 baseline 选择与消融对照；F3FS 的 mode CAP 与 COSM 的空闲窗口插入是两种正交的防饿死思路；All-Bank 时间片是"无并发能力接口"的对照基线。ComPASS [41] 的批量调度是另一路线（粗粒度时间片、非真正并发）。

涉及论文标题：
- COSM: A Cooperative Scheduling Framework for Concurrent PIM and CPU Execution on Mobile Devices

## NCCL 通信原语（Broadcast / Reduce / AllReduce / AllGather / ReduceScatter）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NCCL（NVIDIA Collective Communications Library）是多 GPU 集合通信库，提供 5 种原语：AllReduce（各 GPU 归约后结果广播给所有参与者）、Broadcast（一对 N：单 GPU 缓冲复制给 N 个接收者）、Reduce（N 对一归约）、AllGather（各 GPU 数据拼接后全员可见）、ReduceScatter（归约结果按块分散到各 GPU）。底层按拓扑（NVLink 环、NVSwitch 树等）选 ring/tree 算法并做 chunk 流水。论文用其原语特征化 NVLink：Broadcast 测一对 N 带宽与链路内争用、AllReduce 测 N 对 N 聚合带宽与 crossbar 争用。
- MSDP 语境（DisDP）：MSDP 每层前向/反向各 1×AllGather + 梯度 1×ReduceScatter（ZeRO-3 一层共 2×AG+1×RS），peer-based 实现下每 worker 收/发 (N-1)S/N 流量，每方向总流量 3(N-1)S/N；algorithm bandwidth（算法带宽）= 集合数据量 S / 执行时间 t（BWalg=S/t），是衡量集合性能的标准指标。SmartSwitch 无法加速单独 AG/RS：AG 只能把发送降到 S/N、接收仍 (N-1)S/N，RS 反之——集合时间由未减小的一侧主导。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
论文实测（DGX A100，NVLink 3.0）：单个 Broadcast 仅 ≥1GB 传输接近峰值（约 262 GB/s vs 300 GB/s 理论峰值）；7 个接收 GPU 并发时延迟仅比单接收者 +13.27%（链路内争用可忽略）；AllReduce 8 GPU 比 2 GPU 平均延迟 −9.47%、每 GPU 带宽 +12.41%（NVLink crossbar 调度增益），聚合带宽最高 1878 GB/s；而 4KB/64KB 小传输仅 1.12/17.12 GB/s——量化了"细粒度页传输严重浪费 NVLink 带宽"，是 CDFD 32MB 粗粒度复制的直接依据。NVLink 4.0（DGX H100，450 GB/s 理论峰值）呈同样结论，7 接收者延迟仅 +0.11%。伪代码（Broadcast 一对 N 概念）：
for chunk in split(buffer, chunk_size):       # 大缓冲分块流水
    dst = 0
    while dst < N:                             # 逐接收者复制（或树形转发）
        memcpy_peer(gpu[dst], src_chunk)       # NVLink 远端写
        dst += 1
    fence_across_gpus()                        # 跨 GPU 可见性

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
API：ncclBroadcast/ncclReduce/ncclAllReduce/ncclAllGather/ncclReduceScatter，各 rank 持 device buffer 调用，配合 ncclCommInitRank 建通信域；内部按节点内/节点间拓扑自动选 ring/tree 算法与分段。论文实测平台：DGX A100（8×A100 80GB SXM4，NVLink 3.0，驱动 570.148.08，CUDA 12.8）与 DGX H100（8×H100 80GB SXM5，NVLink 4.0，驱动 570.195.03，CUDA 12.8），1–8 GPU、4KB–32MB 传输多轮取平均。
- DisDP 实测与大规模算法对比：并发 GEMM 下 NCCL AllReduce 的 algorithm bandwidth 降 30%（SM + 内存带宽争用）；DisDP 的 SmartNIC push/pull 相比 NCCL 在 2/4/8 GPU 上带宽高 2%/35%/44%（并发 GEMM）、4/8 GPU 上高 8%/20%（无并发）——push/pull 相对 AllReduce 流量最多减半。大规模下 NCCL ring 集合依赖链长（DP>16 扩展差、易受干扰），PAT（层次化树形 RS/AG）更好；DisDP 在 DP=256 时吞吐 2.0×（vs ZeRO-Infinity+PAT）、15.1×（vs ring）。

Lit Silicon 补充视角（ISCA'26，FSDP 训练中 AG/RS 与 C3 重叠的行为）：FSDP 前向层与层之间用 AG 收集下一层权重分片，反向用 RS 归约上一层梯度，均与 GEMM 并发执行（C3）。但重叠并非免费——计算/通信 kernel 共享 GPU 计算与内存资源互相干扰，计算 kernel 运行时被拖慢最多 40%，且重叠率跨 GPU 不同：straggler GPU 的通信 kernel 起始更晚、重叠率恒定最低（29.6%），leader 重叠率动态增长（最高 52.7%），重叠率与 kernel 时长强相关——这是节点级性能波动的直接来源。AMD 平台（MI300X）上通信集合由 RCCL/amd-smi 生态提供（论文用 Chopper 工具解析 PyTorch trace 得到各 GPU 上 AG/RS 的起始时间与重叠率，用于 lead value 检测）。

MTIA 300 补充视角（ISCA'26，HCCL 与 NCCL 的对照）：MTIA 300 的集体通信由 HCCL 执行（非 NCCL）——API 类似（AlltoAll/AllReduce/ReduceScatter/AllGather + 点对点 send/recv，经 PyTorch Distributed/torchcomms 接口暴露），但执行路径不同：HCCL 把通信编成 work packets/subgraphs/WQEs（SEND/RECV/WRITE/WAIT/SET/REDUCE + 流控字段）卸载到 16 个 ME（RDMA verbs 控制路径），主机不参与数据面。性能对照（vs H100/NCCL）：AllGather/AllReduce/AllToAll 在 16+ 加速器或 >16 MB 消息时显著更优（16 节点 scale-up 域 + 2.2× 带宽），40 卡 DLRM 训练整体通信超 H100 3.9×；小消息 HCCL 弱于 NCCL（未优化、占比小）。

RoCC 补充视角（ISCA'26，ROP 上执行 NCCL 原语的对照）：RoCC 论文以 NCCL/RCCL 为 baseline 软件库（NCCL 把 CC 编译成独立 CC kernel 在 SM 上执行），但把 CC 的执行引擎从 SM 换成 GPU 的 ROP 硬件：新增每条 CC 一个 intrinsic（rocc_allreduce 等）+ 一条 ISA 指令（ROP_AR/ROP_AG/ROP_A2A），并遵循 NCCL 的 CC 算法设计把 collective 分解为 primitive（send/recv/recvReduceSend/recvReduceCopySend/recvCopySend）再译成 ROP μOp。对比结论：CC 在 SM 上执行占 tensor 并行执行时间 40%-70%（PyTorch distributed + NCCL 实测），因 SM 距内存远、CC 网络/内存 bound 浪费算力；RoCC 把 CC 卸载到近内存 ROP 后 CC-only 延迟大消息下 AllReduce 快 35%、AllGather 快 11%、AllToAll 快 25%。
- STAGE 补充视角（ISCA'26）：STAGE 生成的执行图在验证时与 NCCL 实际行为对齐——NCCL 实现 AllToAll 时将其分解为多次 Send/Recv，Kineto 只记录分解后的原语，因此 STAGE 也把 AllToAll 通信量按 Send/Recv 分解后再与真实 trace 对比（Table VII，总通信量误差 0.000%~2.980%）；通信算子估时交给 ASTRA-Sim 模拟，端到端 runtime 平均误差 3.53%。通信匹配器按 producer/consumer 张量分布匹配出 NCCL 集合原语（AllReduce/AllGather/ReduceScatter/AllToAll 及其组合）。

涉及论文标题：
- Coarse-Grained Duplication First, Fine-Grained Deduplication Later: Duplication-Centric Multi-GPU Memory Management
- Scalable Synthesis of Distributed LLM Workloads Through Symbolic Tensor Graphs
- DisDP: Disaggregating Compute, Network, and Storage for Model-Sharded Data-Parallel Training
- Lit Silicon: A Case Where Thermal Imbalance Couples Concurrent Execution in Multiple GPUs
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
- PipeComm Maximizing Link Utilization through Pipeline-Aware Collective Communication Synthesis
- RoCC Harnessing Raster Operations Pipeline for Efficient Tensor Collective Communication

PipeComm 补充视角（ISCA'26，NCCL 作为 baseline 的对照）：PipeComm 在真实两节点 ×8 NVIDIA L20 GPU（节点内 PCIe switch、节点间 InfiniBand+RDMA 的分层异构带宽）上，把 PipeComm 合成的 AllReduce schedule 与 NCCL v2.20.3 对比（2MB–2GB 消息）：NCCL 按 GPU 拓扑在 ring/tree 算法间动态选择，但未显式建模底层链路的详细特征（分层异构下 PCIe 高带宽 + IB 低带宽的静态启发式无法充分榨取非对称带宽），Pipe-Sol 平均取得 1.24× speedup over NCCL、1.18× over partitioned TACOS、1.19× over 非分区 TACOS。这佐证了 NCCL 对均匀/对称拓扑（NVLink 环、NVSwitch 树）近似最优、但在异构/非对称带宽层次下留有提升空间，也说明拓扑感知合成（PipeComm/TACOS 类）可作为 NCCL 的补充后端。

Tetris 补充视角（ISCA'26，NCCL 作为 serving 中 KV cache 传输与并发 communicator 的使用）：Tetris 用 NCCL 实现两类跨实例通信——(1) CDSP cache balancing（chunk 间把前序 KV cache 均匀重分布到当前实例组，复用 ring 通信器与下一层 prefill 跨层重叠）；(2) prefill→decoding 的 KV cache 流式传输（handshake backend 分配后由 send/receive engine 执行）。关键依赖：NCCL 自 v2.26 支持并发 communicator 执行，使多组 cache transfer 可并行、与计算重叠；Tetris 预留专用 buffer 与 CUDA stream 提升带宽利用率。论文量化：CDSP balancing 额外开销 ≤1.8%，handshake 传输 0.6%-11.8%（平均 2.1%），backend 减半压力测试下 RPC 开销 1.5%-5.4%。与 ring attention 的 NVSHMEM one-sided 传输分工：ring 内 K/V 轮转用 NVSHMEM（kernel 内 fine-grained），跨实例/跨阶段的 cache 汇聚用 NCCL（collective/异步传输）。
涉及论文标题：
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism

## Fat GEMM（瘦批 GEMM 与方形脉动阵列的映射失配）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GEMM 记为 (M,K)×(K,N)。长上下文 agentic LLM 推理中，KV cache 容量墙限制 batch，使 M（batch 相关维）远小于 K（隐藏维：LLaMA-3-8B 为 4096、LLaMA-3-70B 为 8192），形成"fat GEMM"（宽而扁：K 长、M 短）。方形 systolic array / Tensor Core 的 tile 假设 M≈N≈K，M 小时只用到阵列窄条，乘法器利用率骤降（图 2：同乘法器数下 8×512 扁平阵列 vs 64×64 方阵可达 FLOPs 差距显著）。FlashAttention 的 per-head GEMM（head_dim 小，如 LLaMA-3-70B 的 128；GQA 一个 K 头对多个 Q 头）是第二种 fat GEMM——计算维度小导致大阵列低利用。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# FFN fat GEMM：batch M 小、隐藏维 K 长（weight-activation GEMM 沿 K 归约）
Y[M, N] = X[M, K] @ W[N, K]^T      # M=4（容量墙压小的 batch）、K=8192
# 方形阵列 128x128：M 方向只占 4/128 → 利用率约 3%
# PLENA 扁平阵列 (BLEN, MLEN) = (32, 2048) 输出驻留：
#   BLEN=32 对齐 M，K 沿 MLEN 流式推进，PE 部分和驻留，全流水无气泡
#   M_SUM 加法树在 K 归约完成后做一次跨 sub-arr 部分和求和
# FlashAttention per-head fat GEMM（头级分解）：
for head in range(MLEN // HLEN):                  # 多个 Q 头并行
    S[BLEN, BLEN] += Q_head[BLEN, HLEN] @ K_head^T[HLEN, BLEN]
```
- 硬件对齐要素：FFN 阵列在 BLEN 对齐 batch 时利用率最优（Figure 12）；FlashAttention 计算模式与 batch 无关（每 head 固定 (BLEN,HLEN)），头级分解后利用率与有效 batch 解耦——decode 长上下文（有效 batch 小）仍满利用。预填充阶段 FFN 与 FA 都接近满利用，故扁平化收益集中在 decode。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：PLENA flattened systolic array + output-stationary 数据流，把长 K 作为流水维、短 M 映射为 BLEN；跨 sub-arr 部分和用结果加法树一次归约（专用 M_SUM 指令，避免逐 tile 气泡）。Scale-Sim 支持矩形/扁平阵列仿真（阵列纵横比影响利用率的非线性结论），SARA 探索可重构阵列形状，但均未针对 autoregressive Transformer 的 fat GEMM + per-head GEMM 组合设计。使用：为 LLM FFN/attention GEMM 选择阵列形状与数据流——长上下文下方形阵列利用率受 M 限制，扁平阵列是 workload 驱动的替代；评估时按"同乘法器数、同 HBM 配置"比较可达 FLOPs 与利用率。

涉及论文标题：
- Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference

## Bin-Aware Workload Mapping（bin 感知负载映射，MLCC/LCC）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
把负载依赖图 WG=(W,D)（节点 = 计算任务如 GEMM/attention 层，边 d_{i→j} = 带权重 p_{i→j} 的数据依赖/通信任务）单射映射到修复后拓扑 TG=(V,E)（Γ(w_i)≠Γ(w_j)，Eqn.5）。目标不是最小化通信距离（hop），而是最小化期望最大链路 contention：MLCC_exp = max_e LCC_exp(e)，LCC_exp(e) = Σ_{d∈D, e∈R(v_i,v_j)} p_{i→j}（Eqn.6，沿 turn-prohibition 路由路径累加传输频率权重）。与 Si-Kintsugi（hop 数代价函数）的本质区别：故障下 contention 是主要延迟源（hop 延迟 5–6 跳后饱和、contention 近线性增长）。优化器：SEGA（Strengthen Elitist GA，Geatpy，种群 100、≤100 代、10 代停滞早停），适应度围绕 bin 目标 MLCC_exp^target（来自 pre-binning），达标后自适应升级到高一 bin（≤2 次，30 代内未达标则采纳上一有效目标最优解）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
伪代码（单个体评估）：
```
LCC = zeros(|E|)                       # 每条修复后拓扑链路的 contention 计数
for d_ij in D:                         # 负载图每条通信边
    path = turn_prohibition_route(TG, Gamma(w_i), Gamma(w_j))
    for e in path: LCC[e] += p_ij      # 沿路径累加传输频率权重
MLCC_exp = max(LCC)                    # 期望最大链路 contention
fitness = -|MLCC_exp - MLCC_exp_target|  # 逼近 bin 目标而非全局最小
# SEGA 进化：选择/交叉/变异 + 精英保留；达标 → target 升级到高一 bin
```
复杂度 O(|D|·h)（h 为平均路径长度），128×136 下 ~18.36 min；空间 O(|E|)。效果：CB*+Ours-SW* 方差降幅 ≥9.16%，Ours-WM（ConBin 硬件 + 仅映射）>45.83%，speedup 至 1.85×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
以 bin 目标（而非全局最优）为适应度中心——芯片只需"够到"本 bin 目标，达标后允许向上一档升级：弱芯片不被强求（避免无效优化）、强芯片不浪费潜力（可晋升 premium bin），这是性能收敛的关键机制。与 CUPOKer（核数贪心放置）和 Si-Kintsugi（hop 代价 + 通信距离建模）对比，bin 感知映射把优化方向从"最大化单芯片加速"改为"收敛到 bin 目标"。Web 证据：NoC 应用映射的 GA 路线成熟（能量/contention 多目标，Hu&Marculescu 分支定界、eMesh GA、GAMR），ConBin 的增量是 contention 代价 + bin 目标自适应的组合。

涉及论文标题：
- ConBin: A Performance-Convergence Framework for Wafer-Scale Chip Binning

## Communication Sequence Scheduling（通信序列调度：CAP / 三粒度批处理 / μ 历史传播）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
负载映射固定后，进一步静态决定各 core 的通信序列（传输执行顺序）以缓解时变 contention。三个机制：(a) CAP（Contention Analysis Phases）——把通信时间线切成 contention 分布准稳定的区间，每 CAP 分析并发任务（含上一 CAP 的在途任务）；(b) 三粒度批处理——早期 batch 用细粒度（≤b1）、后期用中/粗粒度（≤b2/≤b3）控制复杂度（batch = 以第 i 个目的地为目标的全部传输）；(c) 历史感知传播因子 μ∈(0,1)——未解决的高 contention 任务跨 CAP 保留，平衡质量与开销。目标：min Φ = min{φ^0, φ^1, ..., φ^{K-1}}（Eqn.7，φ^k = 该 CAP 期望最大链路/目的端 contention），以 bin 目标 Φ^target 自适应升级。优化器：多染色体 NSGA-III（Geatpy，种群 120、≤100 代、10 代停滞早停）。负载图与依赖固定 → 通信序列可静态预排，无运行时开销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
伪代码：
```
batches = group_transfers_by_destination(CSS_0)   # 第 i 批 = 所有"第 i 目的地"传输
carry = {}                                        # μ 历史传播：未解决高 contention 任务
for k in 0..K-1:                                  # K 个 CAP（三粒度：b1/b2/b3 截断）
    tasks_k = batches[k] ∪ carry
    phi_k = max_link_dest_contention(concurrent(tasks_k))
    NSGA-III 调序 tasks_k 使 phi_k 逼近 Phi_target
    carry = top_contended(tasks_k, mu)            # μ∈(0,1) 保留进下一 CAP
```
时间 O(|D|·h)，128×136 下 ~28.19 min；空间 O(n_link + n_dest)。效果：Ours-WM（仅映射）方差降幅 >45.83%，Ours-ALL*（+调度）达 79.21%——细粒度通信调序能更精确地把芯片推向目标 contention 水平；Ours-ALL* 端到端 speedup 至 2.39×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
映射与调序解耦的两级优化（先空间后时间），每级都以 bin 目标为中心自适应升级；CAP 把"全局时序优化"降为"分区间局部调序"，三粒度批处理 + μ 传播使开销对 wafer 规模可控。Web 证据：ICAPS 2020 的 contention-aware mapping & scheduling 同样"顺序化可能争用的数据传输"；Kalray MPPA 用离线全局通信调度（软件流水 + 传输抢占）。区别：这些方法最小化 makespan/能量，ConBin 最小化"与 bin 目标的距离"。

涉及论文标题：
- ConBin: A Performance-Convergence Framework for Wafer-Scale Chip Binning

## Pre-Binning（预分档）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
正式分 bin 前的软件阶段：把修复后的芯片按硬件指标预分组，并用轻量采样为每组生成 bin 级优化目标（MLCC_exp^target / Φ^target），供后续映射与调度使用。流程：芯片按硬件指标 F（冗余设计适应度，Sec.V-C）排序 → 分 B+1 个分位组（初步 bin 估计）→ 每组取 top 5–15% 芯片、用缩减种群/代数的轻量映射 + 调度各跑一遍 → 组内平均性能定义为该 bin 的目标。作用 = 把"性能收敛"变成可执行目标：同 bin 芯片追求同一目标（消除"强者更强、弱者更弱"的方差放大），又通过升级机制保留晋升空间。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
目标生成伪代码：
```
chips.sort(by=F)                       # 按硬件指标 F 排序
groups = quantile_split(chips, B+1)    # B+1 个分位组
for g in groups:
    samples = top(g, 5%..15%)          # 组内轻量采样
    for chip in samples:
        run_reduced_SEGA_mapping(chip)   # 缩减种群/代数
        run_reduced_NSGAIII_sched(chip)
    MLCC_target[g]  = mean(samples.MLCC_exp)
    Phi_target[g]   = mean(samples.Phi)
```
之后全量优化阶段，每芯片以所属 bin 目标为中心优化、达标后升级到高一 bin（≤2 次）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
轻量采样避免全量仿真的成本（每 bin 只跑 5–15% 芯片的缩减版优化）；目标以 contention 指标（而非最终 latency）表达，与映射/调度代价函数同构（MLCC_exp ↔ LCC、Φ ↔ φ^k）；B+1 组而非 B 组预留边界裕量。pre-binning 是 ConBin 软件栈三阶段（pre-binning → 映射 → 调度）的第一环，为后两级提供统一优化方向。Web 证据：未发现同名独立方法（论文自述新机制），其思想与"分层/分批生成优化目标"的产线测试流程（speed binning 前测、分档测试）同构。

涉及论文标题：
- ConBin: A Performance-Convergence Framework for Wafer-Scale Chip Binning

## PIM 数据重排（Data Rearrangement，Host 布局 ↔ PIM bank 局部布局变换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GPU-PIM 异构系统中，Host 与 PIM core 需要相反的数据布局，因此在 PIM kernel 执行前后必须做数据重排：Host 侧把连续元素跨 DRAM bank 分布（cache line 粒度访问、利用 bank/channel 级并行），PIM core 只能访问本地 bank、需要连续元素落在同一 bank 内以最大化本地带宽。由此 PIM kernel 执行固定为三步：①输入重排（把连续元素搬进各 PIM core 的本地 bank）→②PIM 计算→③输出重排（在 PIM core 间合并部分结果，或把连续元素重新跨 bank 分布以还给 Host 访问）。重排通常经 Host 内存总线完成（PIM 设备之外），开销巨大——DCC 动机实验中 TVM 式方案的数据重排占 kernel 端到端时间的 64.68%，是 PIM 编程与性能的最大痛点。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
核心矛盾（DCC §4.3 I）：(i) Host 侧顺序连续读利用 channel 并行，但写入时只能用一个 PIM channel → PIM 带宽利用不足；(ii) 多 PIM channel 并行写需要从 Host 内存同时读非连续地址 → Host channel 冲突、读性能下降。DCC 的解法（片上内存 staging）：
```
N = PIM 通道数; B = on_chip_memory_size / N   # 如 GPU shared memory 作 staging
for block in range(N):
    read(host_mem, block*B, on_chip[block])   # 串行 N 次 Host 读，读侧 channel 并行
for block in range(N):                         # 之后并行写
    parallel_write(PIM_channel[block], on_chip[block])  # N 路并行写 PIM
```
效果：数据搬移变成 N 次串行 Host 读 + N 路并行 PIM 写，两侧都拿到 channel 级并行。无可控片上内存的系统（如 CPU）退化为 PIM 后端默认 copy/DMA 接口；有对齐/交织等布局约束的后端再叠加额外 layout transformation pass。重排方向由 IR 的 `rearrange %t→%tt [to_PIM_core/to_host]` 显式编码，输入输出两条路径分别调度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：编译器生成重排指令序列（LD/ST 或 DMA），与计算命令一起注入时序仿真器（DCC 用 Ramulator 2.0 的 LD/ST 命令仿真搬移成本）。使用：DCC 把重排成本纳入 draft 的端到端时间联合优化——同一 kernel 的不同数据分区对应不同重排成本，预测器按"重排+计算"总和选优；实测 DCC 同时优化计算（1.58×）与数据重排（1.65×），重排主导的 VA/RED/RELU 提速 1.67×、compute-heavy 的 GEMV/ATTN 仅 1.18×。对比：ATiM 式 compute-centric 流程先定计算模板再补重排，无法发现"稍差计算换便宜重排"的组合。

Taking Analytic Databases to the Bank 补充数据库视角（BLIMP OLAP relayout）：论文把该布局变换称为 relayout，并系统量化其代价与规避策略——(1) 代价：host 侧软件 relayout 平均吞吐 29GBps（vs 90GBps 峰值带宽）；把一个 64-bit 字写入特定 bank 需要 8 次内存写（读取 8 次读），因为标准 DDR 地址映射把字按字节 striping 到 8 个 chip；(2) 规避：提出 PIMDT（PIM Data Type）列式存储格式——把查询常用列以 PIM 友好布局常驻（整字单 bank），host 加载时无需查询时 relayout，只对少数"host 列"保留原布局；PIMDT 只支持定长类型（可变长字符串/blob 不兼容）且更新/插入需按字节重排；(3) 调度影响：查询规划必须最小化 relayout——end-to-end 评估显示隔离算子外推（每算子把输出 relayout 回 host）导致平均 22% 查询时间在 relayout、整体比 PIM-optimal 计划慢 3.2×；物化策略（Early/Hybrid/Late）决定何时发生 compute domain 转换与 relayout（bitvector 定长 vs value array 随选择性线性增长）；bushy join 树需多次重建哈希表（重复 relayout+build+broadcast），与最小化 relayout 目标相悖。

涉及论文标题：
- DCC: Data-Centric Compilation of Machine Learning Kernels for Processing-In-Memory Architectures
- Taking Analytic Databases to the Bank

## GPU-PIM 异构 Kernel 执行（memory-intensive kernel offload 到 PIM、compute-intensive 留 GPU）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
异构 xPU-PIM 系统中按 kernel 访存特征划分执行位置：compute-intensive kernel（GEMM、卷积，如 LLM 的 FFN 主体）跑在 Host GPU；memory-intensive kernel（GEMV、element-wise、attention 的 QK^T/SV）offload 到 PIM core，利用 bank 级聚合带宽。PIM 设备有两种工作模式：作为标准 DRAM 供 Host load/store（Host 执行 kernel），或由 Host 把 kernel offload 给 PIM core 独占访问本地 bank 执行。选择依据：GEMV 的算术强度低（每元素约 2 FLOP、受外部带宽主导），PIM 内部带宽远高于外部 I/O，offload 收益大；GEMM 权重复用高、GPU 张量核算力强，留在 GPU。DCC 的适用边界即"适合 PIM 的 memory-intensive kernel"（GEMV/RED/ATTN/VA/RELU）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LLM 解码一层的执行划分（DCC + AttAcc_Full）：
```
# GPU 侧：QKV 生成与 projection 在 AttAcc 默认实现中留 GPU（固定 tiling 落后 GPU 1.25x）
# DCC 侧：QKV/attention/projection 的 GEMV 形态全部移到 PIM
for b in batch, i in seqlen:            # GEMV：weight 常驻 PIM bank
    PIM_group_broadcast_MAC(weight[i], x[b], acc[b])   # 每 bank GEMV 单元并行
PIM_softmax_unit(score)                 # AttAcc 每 channel softmax 单元
for b in batch, i in seqlen:            # SV
    PIM_group_broadcast_MAC(v[b], p[b], out[b])
# GPU 侧：FFN 的大部分 GEMM 仍在 GPU（compute-intensive）
```
关键调度决策（DCC 自动化）：每 kernel 选多少 PIM 组/核、张量维如何切（data-tile）、循环如何映射（compute-tile）、重排如何做——同一 GEMV 在不同 batch/tensor 尺寸下最优分区不同（batch 1→8 时 DCC 对 HBM-PIM 增益 1.33×→1.58×、AttAcc 1.14×→1.31×）。AttAcc 默认实现用固定分布（batch/head→16 pCH、第一维→16 bank group、第二维→4 bank）应对所有尺寸，是 QKV/projection 层被迫留 GPU 的原因；DCC 联合搜索后把这两层移上 PIM，分别 2.58×/2.91× 对 GPU。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：编译器生成"重排指令 + group/bank 级 DRAM 计算命令 + 结果回收"三段调度（见 PIM 数据重排、Group/Bank 控制命令条目）；DCC API：@DCC_kernel 定义 PIM kernel → DCC.Layer 替换模型层 → DCC.Kernel.preLoad 预载权重到 PIM → forward() 执行并返回 torch.Tensor。使用：memory-bound 层（attention/GEMV 形态）放 PIM、compute-bound 层（GEMM/FFN）放 GPU；DCC 与 GPU 侧 ML 编译器（TVM 等）经 PyTorch 协同（PIM 层标为 non-fusible）。效果：kernel 级对 GPU 至多 13.17×（AttAcc）/7.68×（HBM-PIM），LLM 端到端平均 4.52×。

P3-LLM 补充视角（ISCA'26，NPU-PIM 异构算子映射）：P3-LLM 把异构执行扩展到"NPU（主机）+ HBM-PIM"且量化感知的算子映射——decode 阶段 NPU 只保留高精度元素级算子（RoPE、Softmax），线性层 GEMM/GEMV 与注意力矩阵乘（Q·K^T、P·V）offload 到低精度 PCU；Q·K^T 的映射取决于 key cache 的量化位置：pre-RoPE 量化（Llama-1/2 短序列）的 key 缺位置信息，需每轮在线 RoPE（元素级、开销可忽略）并留在 NPU 高精度计算；post-RoPE 量化（Llama-3/Mistral 长序列）的 key 可直接与 query 相乘、Q·K^T 上 PCU。算子映射伴随量化算子融合（见去量化隐藏条目）：把 per-channel 平滑因子 SSF 融合进 query 的 FP8 量化缩放、把 per-value-head 缩放 S^V/S^V_max 融合进 attention-score、线性层 dequant 后置到 GEMM 之后，使 PIM 侧全程消费量化操作数、NPU 侧不出现逐操作数的在线 dequant。batch≥8 时线性层在 PIM 变 compute-bound，P3-LLM 将线性层 offload 回 NPU、attention 层（GQA 复用低）继续留 PIM——异构映射是动态的（Fig.16 大规模解码实验）。

涉及论文标题：
- DCC: Data-Centric Compilation of Machine Learning Kernels for Processing-In-Memory Architectures
- P3-LLM An Integrated NPU-PIM Accelerator for Edge LLM Inference Using Hybrid Numerical Formats

## Mask-Based In-NAND Computing（mask 门控的 OU 映射与专家放置调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mask-based in-NAND computing 是 DIAMoND 针对"固定 NAND 阵列尺寸 vs MoE 多样化矩阵尺寸"失配提出的灵活计算机制，由三部分组成：① Operation Unit（OU）——把单个 NAND 平面层划分为规则矩形计算单元，尺寸 H=min{ρ_in,d_min}、W=min{ρ_out,d_min·QB}（ρ_in/ρ_out 为硬件可用输入/输出维、d_min 为模型最小矩阵维、QB 为量化位数）；Mixtral-8x7B@512 并行度 → 每 OU 512 TSG × 4096×8 BL、每平面 4×4 OU 阵列。② Mask 设计——利用 NAND string 串接结构：两层 WL 同选（一层权重、一层 mask），只有两层 cell 都导通才形成电流通路，等效权重与 mask 的 AND 门控；对角线/反对角线循环 mask 使 4×4 OU 阵列只需 4 个 read cycle 并行跑完 16 个 OU（无 mask 需 16 个）；Alg.1 mask 生成：方阵用对角/反对角循环模式，C<R 补列成方阵后忽略补列，C>R 拆成多个方阵再合并；mask 开销 4~64 层、1.7%~27.6% 存储（比 sparse mapping 省 2~4× 且专家组合更灵活）。③ 映射与部署——Alg.2 Round-Robin + Mask-Guided：子矩阵轮转分配到多平面 OU 阵列，OU 内按输入竞争分组、依 mask 模式放置；再以 List Scheduling 调度 Transformer 数据流图（顶点=权重矩阵、边=依赖）确定同 cycle 联合部署，同 plane 专家组成 Expert Group。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Alg.2 简化伪代码（子矩阵集合 S、OU 阵列 {OU_1..OU_N}、WL 层集 {wl_1..wl_J}、mask 集 {M_1..M_K}）：
```
Group S by input contention            # 共享输入的矩阵分组
i <- 1; T <- {}
for s in S:
    (j, k) <- (1, 1)
    while s not assigned:
        if OU_i has free OU at wl_j under mask M_k:
            assign s to it; T.update; break
        else:
            k <- (k+1) mod K            # 换下一个 mask
            if k wraps to 1: j <- j+1   # mask 耗尽换下一 WL 层
    i <- (i+1) mod N                    # 轮转下一个 OU 阵列
```
计算过程例子（Mixtral 单个专家的 Down-Projection 矩阵，2 die × 4 plane）：@512 并行度 → 矩阵切 28 个子矩阵 → Round-Robin 分配到各 plane 的 4×4 OU 阵列 → mask 引导把同输入子矩阵对齐到对角 mask 位置 → 一次 read cycle 内 16 个 OU 并行输出、28 子矩阵 2 cycle 完成；@1024/2048 时 OU 阵列布局与切分自适应变化（4×2、2×2 OU）。List Scheduling 部署：把去掉非 in-NAND 算子（W_Q 等）的 Transformer 数据流图按拓扑序贪心调度——依赖满足的矩阵进 ready list、只要 OU 资源够就同 cycle 调度，同 cycle 矩阵子矩阵拼接后整体映射。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：映射与调度离线完成（部署即权重编程布局），运行期只需按 AES 结果选 mask 组合执行 read cycle；mask 状态由硬件 Dynamic Mask Selector（Priority Queue/Conflict FIFO/Mask Pattern RAM/Pattern State Handler）管理。使用方式：把任意形状权重矩阵对齐到固定 NAND 阵列——Mask 设计使解码加速至多 1.73×（vs 无 mask 的 Base 异构架构）；与 AES 联合后 FFN 层固定 3 cycles（Up/Gate/Down 各 1 read cycle）、合计加速 1.95×；专家冲突率从 Mask-only 的 10.2%~93.5% 降至接近 0。评估：SSDsim 基座 cycle-accurate 模拟器。局限：mask 层占用 NAND 存储（1.7%~27.6%），矩形 OU 阵列的 mask 归约（补列/拆方阵）有映射复杂度。

涉及论文标题：
- DIAMoND Dynamic Inference for Adaptive Edge MoE with Heterogeneous In-NAND and Near-DRAM Compute Architecture

## QKV-Attention Joint Caching Pipeline（QKV 投影与自注意力联合缓存流水）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QKV-Attention Joint Caching Pipeline 是 DIAMoND 提出的跨模块（in-NAND ↔ near-DRAM）算子流水重叠设计：把 QKV 投影与 self-attention 的执行顺序联合优化，利用"Q 投影结果既与新生成的 K 向量、又与历史缓存 K 矩阵相乘"这一依赖结构，把 W_Q 放在 DRAM、W_K/W_V 留在 NAND，使 attention 主体计算与 K/V 投影并行。动机：naive 流程把 Q、K、V 投影全放 in-NAND，则 near-DRAM 等待投影结果时空闲；而 Q/K/V 权重矩阵较小，全部在 in-NAND 算本身也低效（in-NAND 优势在巨大专家矩阵）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
跨模块时间线（每层 self-attention，token t）：
```
near-DRAM:  Q = W_Q @ h_t                 ┐
            S_partial = Softmax(Q K_hist^T)├─ K_hist 已缓存于 DRAM
            (等待最终 K)                   ┘
in-NAND:    K = W_K @ h_t; V = W_V @ h_t  ← 与上面并行
汇合:       attention = S_partial × V     ← 仅 K 到达后补齐尾部
```
伪代码逻辑：naive = 顺序执行 [K,V 投影(NAND) → Q 投影(DRAM) → Softmax → O 投影]；pipeline = [Q 投影与 K_hist 部分 attention 先行(DRAM) ‖ K,V 投影(NAND) → Softmax 收尾 → O 投影]，关键路径只依赖最终 K。效果：self-attention 阶段延迟至多降低 13.5%（DIAMoND 实测）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现依赖两件事：权重布局上 W_Q 驻 near-DRAM、W_K/W_V 驻 in-NAND（部署期决定，见 Mask-Based In-NAND Computing 条目）；运行时由 AES/mask 硬件按序发出 in-NAND 的 K/V read cycle 与 near-DRAM 的 attention 计算，二者经 2.5D 封装的独立 SSD 通道无冲突并行。使用方式：任何"小矩阵投影 + 大缓存 attention"的异构存储计算系统都可复用该重叠思想（把依赖后置的算子提前、与不相关投影重叠）。相关概念：kernel pipeline overlap（共享内存双缓冲、bubble-free pipelining 见本库其他条目），区别是本流水跨两个物理模块（存储计算与逻辑计算），而非同核内多级流水。

涉及论文标题：
- DIAMoND Dynamic Inference for Adaptive Edge MoE with Heterogeneous In-NAND and Near-DRAM Compute Architecture

## DSA Memory Move 运行时操作与延迟信号（time-slot 调制 + 中位数阈值解码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Memory Move 是 Intel DSA 最基础的数据搬移操作（一次读 + 一次写），由 64B descriptor 指定源/目标地址与长度提交到 WQ、engine 异步执行。DarkStream 把它当作运行时信号原语：在 9 种 Source×Sink 操作组合（Memory Move / Fill / Compare Pattern）系统测量中，Move-Move 产生最大且最稳定的争用延迟差（Move 双向搬移、负载最重；Fill 单写、延迟最短但信号弱；Compare 依赖 DRAM 读、带宽利用低），因此选 Memory Move 作为隐蔽信道与侧信道的统一原语。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
隐蔽信道运行时循环（Source 与 Sink 分属不同 CPU core、不同 group 的 DWQ+engine，仅共享 DSA 设备）：

```
# Source（发送方）：每 time slot 编码 1 bit
for bit in msg:                      # 128-bit 帧 + 10101010b 前导码
    if bit == 1:
        while slot_not_over:         # active 态：异步饱和提交
            enqcmd(movemove_desc(1B))
    else:
        sleep(slot)                  # idle 态：不提交
# Sink（接收方）：持续提交并逐请求计时
latencies = []
while True:
    t0 = rdtsc(); enqcmd(move(1B)); wait_completion(); t1 = rdtsc()
    latencies.append(t1 - t0)        # idle≈1400 cycles, active 2000-4000 cycles
# 解码：每 slot 取中位数与阈值比较
for slot in slots:
    bit = 1 if median(latencies[slot]) > threshold else 0
```

侧信道变体：攻击者持续提交 1 MB Memory Move——大传输延长共享数据通路占用时间，使受害者干扰在长操作内累积放大（1 B/1 KB 探测下延迟方差大、无法区分受害传输尺寸，1 MB 下延迟与受害传输尺寸强正相关），逐请求记录延迟得到指纹 trace。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现 = 用户态进程经 Linux idxd 驱动 write syscall / ENQCMD 提交 descriptor，无需 root（只要管理员分配了 WQ 权限）。带宽 = 信道容量 × 传输频率，容量按 Binary Asymmetric Channel 模型由 0→1/1→0 bit-flip 概率 ε0/ε1 计算（式见论文 Eq.2），传输频率即 slot 时长倒数。扫描 40–256 KHz：低频容量≈1，147 KHz 处带宽峰值 129 Kbps，之后容量下降主导、带宽回落；128-bit 帧 + 8-bit 前导码同步。抗干扰：CPU 90% 负载下仍 >100 Kbps（DSA 执行与 CPU 解耦）；1 MB 静态 DSA 噪声下约 70 Kbps、随机噪声（每操作随机选 4 KB/64 KB/1 MB）下至多 49 Kbps。跨处理器：双路 Xeon Gold 6554S 上 Local-Local 92 Kbps、Local-Remote 78 Kbps、Remote-Local 92 Kbps。

涉及论文标题：
- DarkStream: Exploiting Internal Throughput Contention in Data Streaming Accelerator for Timing Attacks

## XClock 与 XData（软件原生时序模型：软件声明时钟 + 边沿对齐数据）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
XClock 是 UCV 定义的软件声明时钟：host 与模拟器共享的规范时间基（频率/时钟边沿/相位），由软件（而非模拟器）规定 commit/sample 时间点并约束模拟器何时可推进、I/O 何时有效；它实现为从既有事件循环（asyncio、Boost.Asio）调用的普通库，不改模拟器/RTL/第三方库。XData 是绑定 XClock 的时序感知数据类型，双层抽象：下层 C/C++ 提供模拟器控制与不含时序语义的信号访问（经 SWIG 绑定暴露给 HLL），上层用语言原生并发模型做事件编排与边沿对齐——软件不用为每次读/写标注显式边沿等待。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
软件迭代循环（本文 Fig.4 的软硬件交互调度）：
```
loop:                          # 每轮软件迭代
  1. 处理 T0 时刻全部 pending 软件事件
  2. 缓冲写提交到模拟器（T0 边沿 commit）
  3. HWStep(T1 >= T0)          # 控制权进入模拟器
  4. 返回后按需读信号；值/时间变化的回调派发新事件
  5. 新事件入队，继续循环
# HWStep 内部：每个周期处理当前时刻全部事件 → 时间+1 → 直到 T0==T1 返回；
# T1==T0 时只推进零时相位至静止（观察组合逻辑与 δ-cycle）
```
XData 默认只暴露时钟边沿点（上升沿，可选下降沿）作为传输调度点，可声明额外边沿做周期内细观测。该调度同时具备：事件级表达力（重叠 in-flight 交互各占独立异步流）+ 周期精确语义（边沿对齐 commit、静止后采样）——对比之下 step-peek 只有整周期步进、cocotb 回调可能早于信号稳定触发。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
API（本文 Table II）：XClock(pin,thigh,tlow)、XEvent(conds,args,sclk,reactors)、XEvent(conds,sclk).await、XTrigger(events/reactors,sclk)、XReactor(name,cb,sclk)、@XReactor 装饰器。应用形态：Python asyncio 中 await 时序点推进、pytest 编写测试；跨语言（C++/Python/Java/Go）经 SWIG 绑定同一 backend adapter。开销数据（Fig.13）：XiangShan（3.45M LOC）吞吐损失 ≤3%（大设计时间几乎都在模拟器执行）；CoupledL2/RocketChip 损失 14%–55%（小设计每秒周期数高、软件事件调度占比大）；峰值内存主要随语言变化（JVM/解释器开销）。Web 无外部来源（UCV 平台特有机制，见 https://github.com/XS-MLVP/picker 生态）。

涉及论文标题：
- Democratizing and Accelerating Hardware Verification with Software-Native Optimization

## XEvent 与 XSocket（跨域事件同步注册表 + 有界线程池事务传输）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
XEvent 是 UCV 透明软硬件映射层的事件同步机制：通过 Event Registry 把 HVL（SystemVerilog/UVM）事件映射为软件事件对象——(i) 以字符串标识符注册跨语言事件、(ii) 镜像事件状态与参数缓冲区、(iii) 跨语言代理回调。XSocket 是事务传输机制：把 TLM 风格事务 transport 派发到有界线程池，将同步阻塞等待转换为异步阻塞，避免"模拟器等软件线程、软件等模拟器推进时间"的相互等待死锁；硬件侧保留传统 TLM 实现，软件侧暴露 socket 风格 API。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
事件同步流程（Fig.5）：
```
await HVL事件 → 注册表登记字符串标识符+参数缓冲
→ HVL trigger 更新注册表事件状态与参数
→ 同步点（Fig.4 数据流处）跨环境传播更新，恢复调度器可见性
# 无时序场景：注册表按标识符解析目标函数，经函数指针+绑定层解码同步派发（动态跨语言调用）
```
事务传输死锁与规避（Fig.6）：若软件扩展在步骤 2/3 强制软硬件异步上下文切换，先前的线程上下文不保留 → HW Transport 无法把控制权交还调度器，与模拟器形成相互等待死锁；XSocket 让 transport 代码运行在有界 worker 线程上、调度器在步骤 7 投递异步通知 → 模拟器不再等待软件线程，同时线程开销有界。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
性能（本文 Table IV，NoC 13,036 LOC / ICache 5,163 LOC）：UCV+（UCV 启用 UVM 支持）执行快约 16.6%、验证代码量少 12%（NoC：24.06s/15.41s vs UVM 15.32s/18.47s；ICache：13.69s/94.36s vs 19.14s/106.12s）——UVM 流程集成软件激励与 VIP 需进程间通信（共享内存等待/同步/序列化），XSocket 换为进程内直接传输后消除该开销。典型使用：软件测试任务与模拟器内 UVM VIP 协调（BPU 验证 Step 3），VIP 事件经 XEvent 注册表镜像、事务经 XSocket 线程池异步化。Web 无外部来源（UCV 平台特有机制）。

涉及论文标题：
- Democratizing and Accelerating Hardware Verification with Software-Native Optimization

## 列稀疏 S×V 跳过调度（Column Sparsity Skip in S×V Attention）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
action 模态的 attention score 矩阵具有列稀疏性：注意力集中在少数列，约 52% 的列近零（论文在机器人任务集上统计的平均列稀疏度）。与 token 剪枝不同（action token 数仅 10^1–10^2，剪枝空间小），列稀疏跳过直接针对 S×V 计算：近零列对输出贡献可忽略，跳过对应列的多头矩阵乘，并进一步旁路产生这些列的 V 投影计算。这是 DiTPA 模型级冗余消除在运行时 kernel 层面的执行形态。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
运行时调度伪代码（对应 multimodal scheduler 的 column sparsity controller）：
```
for head h in heads:                          # 逐注意力头并行
    col_mask[h] = OR_over_rows(|S[h][:, c]| < eps for c in cols)   # 列比较器 + bit 寄存器 + OR 归约
queue[h] = [c for c in cols if col_mask[h][c] == 0]   # 非零列索引入队
# dispatcher：均衡各头负载
for h in heads:
    queue[h] = rebalance(queue, avg(len(queue[longest]), len(queue[shortest])))
for c in queue[h]:                            # 按队列顺序出列计算
    O[:, c] += S[:, c] * V[c, :]              # 仅非稀疏列参与 S×V
# 稀疏列对应行在 V 投影阶段直接旁路（如最后一个 head 的 V projection 跳过）
```
关键点：零列判定发生在 attention score 计算后、SoftMax 与 V 相乘前，因此 SoftMax 输出零列与 V 对应行一同省略；V 投影旁路属于跨算子级联收益（列稀疏从 S×V 反向传播到 V 的权重矩阵行）。负载均衡：不同头稀疏分布不同会拉长 straggler，dispatcher 平均最长/最短序列实现均衡。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：硬件列比较器直接对 S 的每个元素做阈值比较（近零判定），结果以 bit 位图存储，OR 树归约出列掩码，索引队列 + dispatcher 完成调度——全部在 multimodal scheduler 内完成，数据操作时延可忽略（GPU 端对应开销占 35.4%）。使用：与校准多模态近似组合后消除 91.74% 冗余 token 计算、其中动作模态贡献列稀疏跳过部分；能效维度 DRAM 权重访问保持不变时（从 16.67% 升至 67.68% 总能耗），列跳过是继续压缩片上计算的关键。通用性：列稀疏模式在 vision/language token 上不存在（其冗余来自重复而非稀疏），因此该调度仅施加于 action 模态注意力。

涉及论文标题：
- DiTPA A DiT-based Action Planner Accelerator Exploiting Action–Denoising–Multimodality Redundancy for Embodied Artificial Intelligence

## 计算-通信干扰（Compute-Communication Interference：SM 争用、非抢占调度与内存带宽争用）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 在 GPU 上同时跑 GEMM 计算 kernel 与 NCCL 集合 kernel 时互相拖慢的现象，两个来源：(1) 计算单元争用——GPU 非抢占式调度，占满所有 SMs 的 GEMM kernel 会阻塞后续集合 kernel 启动（即使集合 kernel 优先级更高、只需 <10% SM）；(2) 内存带宽/L2 争用——并发 GEMM 使 NCCL AllReduce 的 algorithm bandwidth 降 30%（用 CUDA MPS 静态隔离 SM 后依然存在，证明是内存子系统争用）。
- 后果：MSDP（ZeRO-Infinity）训 OPT-175B（8×1-GPU、100Gbps）时 65% 集合时间无法与 GEMM 重叠，迭代时间比理想重叠多 41%，MFU 仅 15%。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 调度序列例子（MSDP 反向，同一 GPU 两个流）：
```
stream_compute: launch(GEMM1); launch(GEMM2)      # GEMM2 依赖 AG2 的数据
stream_coll:    launch(AG1); launch(AG2); launch(RS1); launch(RS2)
# 非抢占：GEMM1 占满所有 SM 期间 AG2 排队等待；AG2 完成前 GEMM2 无数据可算
overlap_ideal = max(T_GEMM, T_coll)
overlap_actual = T_GEMM + T_coll_unhidden          # 65% 集合时间不可隐藏
```
- 缓解选项与局限：CUDA MPS 只能静态分区 SM，与 dynamic parallelism、CUDA Graph 等 LLM 训练重度依赖特性不兼容，且不解决内存带宽争用；DisDP 的解法是把集合整体搬到 SmartNIC——实测 PCIe DMA 与 GEMM 并发时双向几乎零干扰，因此完全解耦优于 GPU 内调度优化（仿真中 ZeRO-Inf+MPS / ZeRO-Inf+Preemp 仅达 DisDP 32.7%/32.8% 吞吐，二者均只解决 SM 争用）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 判别方法：并发跑 GEMM 与 NCCL AllReduce（用 MPS 隔离 SM 作控制变量）测 algorithm bandwidth 降幅；或并发跑 PCIe 设备 DMA 与 GEMM 测双向干扰。使用要点：追求计算-通信重叠时应优先 copy engine/DMA 类传输（不占 SM）或 SmartNIC 卸载，而非 GPU 内核级重叠或 MPS 分区。信息缺口：论文未给出内存带宽争用中 L2 与 DRAM 分量的拆解。

MTIA 300 补充视角（ISCA'26，硬件级消除计算-通信干扰）：MTIA 300 以专用 ME + NMC 把 collective 完全卸载到网格边缘的独立引擎，从硬件上消除 GPU 式"SM 做归约与 GEMM 争用"的干扰。重叠微基准：1000 次 TF32 4K×4K×4K GEMM 与代表消息大小的 collective 并发（16 加速器），MTIA 300 计算与通信双 ~100% 效率（等于单独运行性能），H100 因 SM 争用明显退化（见 C3/计算-通信干扰通用条目）。40 卡 DLRM 训练实测通信超 H100 3.9×，体现"计算-通信重叠 + 无干扰"的端到端收益（Perf/TCO 1.42×）。

涉及论文标题：
- DisDP: Disaggregating Compute, Network, and Storage for Model-Sharded Data-Parallel Training
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines

## Step-Centric Optimizer Pipelining（步骤中心优化器流水）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PS 上把一次参数更新拆成固定顺序的若干「步骤」，CPU 线程按步骤分配（而非按层分配）的软件流水技术。反向阶段 6 步：①从 SmartNIC pull 2B 梯度到 CPU 内存 ②SSD 读 12B 模型状态（Adam m/v 各 4B + 主权重 4B）③CPU Adam（读 2B 梯度 + 12B 状态、17 次浮点运算/参数、写 12B 更新状态 + 2B 参数副本）④12B 状态 + 2B 参数写回 SSD ⑤push 2B 参数给 worker；前向 2 步（SSD 读 2B 参数 + push）。
- 对比 layer-centric（按层分配线程）：每层需 32 线程才够 CPU Adam 线速，并行层数受总线程数限制 → 流水气泡；step-centric 给算力密集的 Adam 步骤 32 线程、其余步骤各 1 线程，共 37 线程即满流水（layer-centric 需 104 线程）。
- 资源账（100Gbps 网络，DisDP 表 III）：前向 0 FLOPs、23.3 GB/s 内存带宽、11.6 GB/s SSD 带宽；反向 99 GFLOPS、349 GB/s 内存带宽、81.4 GB/s SSD 带宽。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 流水调度：每步骤同时只处理一层，层按 L1→L6 依次从步骤①旋转到步骤⑥；线程按步骤静态分配。伪代码：
```
threads = allocate_by_step()          # step3 (CPU Adam) = 32 线程，其余步骤各 1
for layer in layers:                  # 层顺序旋转经过各步骤
    step1.pull_grad(layer)            # 1 线程：SmartNIC -> 内存
    step2.read_state(layer)           # 1 线程：SSD -> 内存
    step3.adam(layer)                 # 32 线程：17 FLOPs/参数
    step4.write_state(layer)          # 1 线程：内存 -> SSD
    step5.push_param(layer)           # 1 线程：内存 -> SmartNIC
# 各步骤同时服务不同层 = 深度流水；吞吐 = 瓶颈步骤吞吐（Adam 32 线程达线速）
```
- 性能目标：PS 线速消费 100Gbps 聚合梯度，使吞吐瓶颈在网络而非优化器。消融：DisDP vs DisDP-LC（layer-centric 变体）1.10~1.17×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：PS 进程（双 Xeon Gold 5320：1.83 TFLOPS、375 GB/s 内存带宽、PCIe Gen4）多线程 + SIMD + 循环展开的 CPU Adam（沿 ZeRO-Offload 做法）；SSD 读、CPU Adam、网卡收发三者重叠（Web 证据：DeepSpeed ZenFlow 同思路的原生进程重叠 CPU 优化器）。扩展：6730P（Gen5：2.56 TFLOPS、819 GB/s）可支撑 200Gbps。使用场景：所有「单机吃线速流式状态更新」场景（PS 优化器、流式 KV 后端）。信息缺口：论文未给出步骤间缓冲队列的实现（ring buffer/同步原语）。

涉及论文标题：
- DisDP: Disaggregating Compute, Network, and Storage for Model-Sharded Data-Parallel Training

## Shift Automorphism Gates（移位自同构门：BB 码上的容错逻辑置换门）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- automorphism 门是"对物理 qubit 做置换、在逻辑 qubit 上诱导出 CNOT 电路"的容错逻辑门；shift automorphism 是其中保持 X/Z 类型（不混基）的 36 元素子群，源于 BB 码的循环平移对称（qubit 与单项式 $x^i y^j$ 对应，平移即移位）。实现方式：沿度 6 连通图在数据 qubit 与 check qubit 之间做连续 swap 操作（先绿后红两段，图 3）——与 syndrome extraction 共用同一稀疏连接，因此 12 个生成元可容错实现，且任意 shift automorphism 至多用 2 个生成元合成。成本（[27] Table I）：τ=14 个时间步；错误率 10⁻⁶·⁴（gross）/10⁻¹⁴·⁵（two-gross）@ p_phys=10⁻³、10⁻¹²·²/10⁻³⁷ @10⁻⁴——比 LPU 测量快一个数量级且错误低得多，故是调度中的"便宜"原语。
- 从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 本论文中 automorphism 作为"测量重定向 kernel"调度：
  ```
  测量逻辑 Pauli P 的 kernel 流程:
  1. 查表选 automorphism 序列 g_1..g_r（r≤2 生成元）使 g(P) 落入 LPU 原生集
  2. for g in g_1..g_r: 沿连接图执行 g 的 swap 序列（绿段→红段），
     逻辑算子随之置换（X_L0/X_L1/X_L2 的支撑被搬动，改变与 pivot Z_L0 的重叠）
  3. LPU 测量 g(P)（单次原生测量，占 τ=120/216）
  4. 逆置换恢复，byproduct Pauli 记录到跟踪表
  ```
  调度优化：①TSP 排序使相邻旋转间重定向所需 automorphism 轮数最少；②双轨模式（dual-track）——ZX-duality 使 automorphism 群在两个 6-qubit 块上同步作用，当两轨旋转标签一致且为纯 X/Z 型时，一条 automorphism 序列 + 一次同时 LPU 测量完成两轨旋转，吞吐近 2×（pivot Y 测量占双模块时串行）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 代数基础：BB/generalized bicycle 码的 automorphism 结构（arXiv:2412.04181 的 shift automorphism 与 ZX-duality；arXiv:2606.05044 Davenport-Blue-Chuang 给出 cyclic submodule 框架下 block-separable automorphism 的充要条件与 fold-transversal CX 判据）。注意 automorphism 保持码距（弱意义），但电路级 fault distance 可能因单 CZ 对齐逻辑算子而 -1。工业用法：自行车架构指令集中的标准指令之一，编译策略是"多 automorphism、少测量"（本论文因 automorphism 便宜且快而把 LPU 测量次数作为首要最小化目标）。
- 涉及论文标题：
- Distilling Magic States in the Bicycle Architecture

## FlashAttention-3（FA3，H100 融合注意力 kernel / 静态 warp-specialized 流水基线）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- FA3 是 H100/Hopper 上高度手调的融合注意力 kernel（论文：Shah et al., NeurIPS 2024，"Fast and Accurate Attention with Asynchrony and Low-precision"，arXiv:2407.08608），把 QK^T 乘、缩放/掩码、softmax、V 投影融合进单个 kernel，利用 Hopper 的 TMA（Tensor Memory Accelerator，硬件异步张量搬移）与 WGMMA（Warpgroup MMA，warpgroup 级异步张量核指令）实现生产者-消费者 warp 特化流水。FA2 在 H100 上只有 ~35% 理论利用率，FA3 通过异步重叠达到 FP16/BF16 约 740 TFLOPS（~75% 利用率）、FP8 近 1.2 PFLOPS，FP16/BF16 比 FA2 快 1.5–2.0×。
- 在本论文中的角色：作为"静态 tile 级流水调度"的代表性手调 kernel 基线。论文把 FA3 融合注意力分解为三类异构 tile：M0（QK^T GEMM，tensor 单元）、S（softmax，vector 单元）、M1（AV GEMM，tensor 单元），每迭代垂直一组 M0/S/M1，迭代间由静态屏障强制同步。FA3 依赖 8 处显式同步（warpgroup_fence_producer、wgmma::wait、warpgroup_barrier_arrive/wait、warpgroup_commit_batch 等）锁定静态双阶段/三阶段重叠模板，TMA 停顿会使依赖消费者序列无条件阻塞。
- Web 佐证：FA3 三个核心技术——生产者-消费者异步（warp 特化重叠数据搬移与计算）、异步块式 GEMM 下重叠 softmax（warpgroup 间 pingpong + warpgroup 内双级流水）、FP8 硬件加速 GEMM（块量化 + incoherent processing）。生产者 warps 发 TMA load 到循环共享内存 buffer 并写命名屏障，消费者 warpgroups 回收寄存器（setmaxnreg）等屏障后 WGMMA 计算。开源在 https://github.com/Dao-AILab/flash-attention（hopper/ 目录）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- CUDA 版调度过程（论文 Fig.5 左伪代码）：tma_load_q/tma_load_k_transpose 载入 s_Q/s_K → warpgroup_fence_producer → wgmma::mma_sync(s_P,s_Q,s_K)（M0）→ wgmma::wait + softmax_warpgroup(s_S,s_P,state)（S）→ 循环 j=0..Tc-1：tma_load_k_transpose(s_K_next) + warpgroup_barrier_arrive + wgmma::mma_async(s_S_next,s_Q,s_K_next)（预取下一 K tile）→ tma_load_v(s_V,V) + warpgroup_barrier_wait → wgmma::mma_sync(s_R,s_S,s_V)（M1）→ wgmma::wait + softmax_warpgroup(s_S_next,...) → wgmma::wait + rescale_warpgroup(s_O,s_R,state,state_next) → warpgroup_commit_batch + update_carousel_index → tma_store_o。要点：8 处显式同步把 S^i 与 M0^{i+1}（数据独立、ME/VE 资源不冲突）序列化，迭代间形成隐式屏障。
- TISA 版（Epoch）对比：同一循环以 tisa::gemm<me>/tisa::softmax<ve>/tisa::load<de>/tisa::store<de> 声明式表达、零屏障；硬件调度器按依赖就绪乱序发射，S_i 与 M0_{i+1}、M1_i 与 S_{i+1} 并发（图 2c/e）。TISA kernel 编译器自动生成：代码量 -30%、同步频率 -50%、性能在手调基线 5% 内。
- 调度/重叠效果：静态双阶段（M0+S | M1）或三阶段（M0 | S | M1）模板只能节省固定量（E0/E1），动态调度实现更紧凑跨迭代重叠（E0+E2 或 E1+E3）。Accumulated Overlap Score 中 DMV 三单元同时激活类重叠在静态下为 0（表 VII）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现与使用：FA3 开源（https://github.com/Dao-AILab/flash-attention，Hopper SM90 实现，新版本用 CuTe DSL），手调 CUDA/CUTLASS 风格：thread-block 分解 + 共享内存 staging + warp 级 collectives + 手调预取；生产者/消费者 warp 特化 + 命名屏障 + pingpong 调度。论文将其作为 H100 基线原样运行。
- 实验用法：论文对比 Epoch(TISA FA3, BF16) vs H100(FA3) 的持续 BF16 吞吐，seq 长度 512–16K、带/不带 causal mask；硬件利用率 = Achieved GFLOPs / Peak GFLOPs；Epoch 在向量:矩阵计算比 1:8（H100 原生比）下全序列长利用率高 >10%，head dim 128 主流配置高 26.4%，1:16 比仍高 15.7%，1:32 比多配置相当——尽管 Epoch 带宽仅 H100 的 1/3.35（1.0 vs 3.35 TB/s），证明增益来自调度（TISA 消除静态 per-iteration 同步）而非算力。
- 结论性用法：作为"静态同步固定的 SOTA 手调 kernel"与"动态调度 kernel"的对照锚点，说明静态 barrier 无法适应运行时变动（TMA 停顿即阻塞依赖序列），而 TISA 按精确就绪轨迹调整发射序。

涉及论文标题：
- Dynamic Scheduling for AI Accelerators via TISA
- TAGT: An Efficient Graph Transformer Accelerator with Topology-aware Sparsification and Merging（TAGT 把 FA 系列 IO-aware 注意力 kernel 作为对照：tiling+online softmax 只能降低稠密注意力的显存流量，不能消除 O(N²) 顶点对交互次数；且 GT 的结构编码使注意力矩阵不规则、块矩阵优化失效。TAGT 以 TDS 稀疏注意力 + FAU 流式分数/SCU 块级异步 softmax 替代）

## PIMBench（PIM 基准测试套件与 bank-PIM kernel 评估）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PIMBench（Siddique et al., IISWC'24，UVA 团队）是数字 DRAM PIM 的架构建模与基准测试套件，提供面向 bank 级 PIM 的通用 kernel 集合，开源于 PIMeval-PIMbench（https://github.com/UVA-LavaLab/PIMeval-PIMbench）。论文用它评估 reliable bank-PIM 在 GEMV 之外的通用 PIM kernel 上的表现，只取原作者报告在 all-bank PIM 上有正加速的 benchmark，并按公开实现移植进 Ramulator2 模拟器。为支持这些应用，论文给模拟架构补充了几条与乘加同延迟的简单指令（绝对值、小于、clamp、位操作），跨 bank 通信指令不用（reduction 在 host 做）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
kernel 调度视角的分层：读为主 kernel（结果存 PIM SRAM、host 后读）与写密集 kernel（PIM 直接写 DRAM）在 reliable bank-PIM 下行为不同。读为主例子（Linear Regression，论文实测 7.9× vs rank-PIM，接近理想）：host 把长输入向量流式分布到各 bank → 每个 bank 旁 PIM 单元算部分和（4 FPU 乘加）→ 结果写 PIM 本地 SRAM（仅 4 个归约和、输出大小与向量长度无关）→ host 读回做最终归约。写密集例子（K-means，1.6× vs rank-PIM）：论文加了 K-means Optimized 优化——每个 PIM 单元用本地 SRAM 缓冲跟踪最小距离与质心，每样本只写质心归属，跨 bank 归约由 host 在 kernel 结束时做；未优化的 K-means 因每样本都要 host 读回（KNN 1.6× 最低即此因，数据复用少）而降速。写路径代价：reliable bank-PIM 每次写要更新 rank 级 ECC（等效 CPU 写、约 8× 慢于非可靠 bank-PIM），写密集 kernel（vector add/AXPY，0.7×）反而不如 rank-PIM；读-执行比高的 kernel（K-means、Image Downscaling 1.3×）仍胜出。VRT 纠错开销经 Codeword Flip + 硬件纠错控制 <2.1%（Filter by Key 最大，因读 PIM 操作占比高）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：PIMeval-PIMbench 开源（https://github.com/UVA-LavaLab/PIMeval-PIMbench），论文按公开实现把 kernel 移植到 Ramulator2（含 PIM 单元指令与 SRAM 缓冲建模），对每 kernel 收集 PIM 命令 trace 喂给模拟器估执行时间。使用：评估 bank-PIM 架构（性能、写开销、纠错开销）时作为 GEMV 之外的通用 workload 补充；比较可靠/非可靠 bank-PIM 时按读-执行比到写比分组看差异（读为主近乎无差、写密集是可靠性的主要代价）。

涉及论文标题：
- ECC Enabled Reliable and Performant Processing-in-Memory

## Mini-batch Spiking Gustavson-Product（迷你批脉冲 Gustavson 积数据流）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Gustavson 积（Gustavson's algorithm，F.G. Gustavson 1978, ACM TOMS）是稀疏矩阵乘（SpGEMM）的经典行式（row-wise）公式：逐输出行计算 C(i,*) ← Σ_j A(i,j)·B(j,*)，即"对 A 的第 i 行每个非零元，把 B 对应行的缩放累加进输出行"。与内积式（inner-product，逐元素点积累加，需读满稠密权重）和外积式（outer-product，每次只算单点积、反复读写输出）相比，Gustavson 的行式累加使每个输出行（SNN 中即膜电位行）只读/写一次，显著降低访存。
- ELSA 把它适配到 SNN：SNN 里权重 4-bit、spike 1-bit、膜电位 12-bit（膜比权重大得多），因此减少膜访问收益最大。但 SNN 是异步事件驱动——spike 生成即前传、按行无规律到达；直接套用 ANN 的 Gustavson 会因行切换频繁而丧失行驻留收益。ELSA 的解决方案是 mini-batch 化：利用 Bundled AER（BAER）提供的行对齐，把同一膜行的 spike 捆成 mini-batch，每批只读一次膜行、并行累加多权重行、写回一次，在"不打同步屏障、维持 spine/token 流水"的前提下恢复 Gustavson 的低访存优势（Fig.7、Fig.23）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 单批（一个 BAER flit）的 PE 内计算流程（ELSA Fig.10）：
```
# 输入：一个 BAER mini-batch = 同行 spike { (x, y_i, q_i) }，共享行地址 x
1) 控制模块解析 spike 编码位置：x → 膜地址；{y_i, q_i} → N-way 权重 buffer 地址
2) N-way 权重 buffer 读 N 行权重 w[y_i]（负 spike 取二补码）
3) 16 输入加法树并行累加 → 膜电位行 V[x]
4) fire 组件读 spike tracer S[x]，判激发 y_t = Θ(V[x], V_thr, S[x])
5) update 组件写回膜 V[x] 与 tracer S[x+1]
# 关键：同一 x 行的全部 spike 一次读膜、一次写膜（行驻留），而非逐 spike 反复切换
```
- 例（ELSA Fig.10c）：mini-batch (0,1),(0,3) 触发读 W 第 2 行 [2,2,3,3] 与第 4 行 [1,3,1,1] → 加法树得膜行 [3,5,4,4] → fire+update。Fig.23 能量分解：inner-product 权重 buffer 能耗占 76.2%（ResNet34），outer-product 膜 buffer 占 70.3%，Gustavson 把二者合并压到 43.1%，平均比 inner 省 2.7×、比 outer 省 1.9×。
- Annotations：x 是 spike 的行地址（膜行号）、y_i 是列地址（权重行号）、q_i 极性位；N=每 mini-batch spike 数（≤PE 的加法树并行度 16）；"行驻留"= 该行膜只被读/写一次；无同步屏障是相对 TrueNorth 1kHz 全局 tick 的关键差异（ELSA 借此把吞吐从 58 GOPS 提到 4135 GOPS）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 硬件侧：ELSA 的 PE 控制模块 + N-way 权重 buffer + 膜 buffer + tracer buffer + 16 输入加法树实现该数据流；BAER 提供行对齐的 mini-batch；每 PE 128 个 ST-BIF 电路、每周期 1024 次加法。软件/系统侧：映射算法按"层内不跨核"（partition 阶段把整层放同核）避免 spike 广播与跨 PE 归约，使 mini-batch 能保持行对齐。Web 证据：Gustavson 数据流在 GPU/加速器领域是主流稀疏方案——ZeD（ASPLOS）用 row-wise product + 稀疏累加器（SPA）工作空间、Opal（16nm CGRA）以"unioner + accumulation loop"链式实现 Gustavson 模式（最多 -79% 运行时间 vs inner-product）、RELL-STC（SIGMETRICS 2026）把 Tensor Core 改为 Gustavson 数据流（vs cuSPARSE 平均 3.54×）。区别：ELSA 针对 SNN 的"异步 + 行不对齐"问题，用 BAER 行捆绑产生 mini-batch，这是 ANN 侧没有的适配。

涉及论文标题：
- ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing

## MM-sc 列切分 tiling（Column-wise Weight-Membrane Tiling）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MM-sc 的 tiling 决定权重矩阵与膜电位矩阵如何在多个 PE（或神经核）间划分。传统加速器按块切分（block-wise，行×列块都切），需要跨 PE 的 spike 归约/部分和同步。ELSA 采用列切分（column-wise）：把权重矩阵的列与膜电位的列按列分组分配到不同 PE（如 Fig.10d：第 1、2 列给 PE1，第 3、4 列给 PE2），而 spike 广播给所有 PE。
- 该切分的收益：① spike 只需广播（所有 PE 收到同一批 spike），消除跨 PE 的部分和归约（Local Input Reducer 的 spike reduction 不再需要）；② 每个 PE 只存自己的权重/膜分片，无重叠，面积利用率高；③ 与 Gustavson 行式累加天然契合——同一 spike 触发各 PE 各自累加自己的权重列分片。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 列切分下的执行（2 个 PE、权重 4 列）：
```
# 广播 spike (x=0, y∈{1,3}) 到 PE1、PE2
PE1 持有 W 列 1-2（行 2、4 的对应片段）→ 累加膜行 V[0] 的列 1-2
PE2 持有 W 列 3-4 → 累加膜行 V[0] 的列 3-4
# 无跨 PE 归约：每个输出列只在一个 PE 内完成
```
- ELSA 论文 Fig.10d 明确："column-wise divide the synaptic weight and membrane（1st and 2nd column to PE1 and 3rd and 4th column to PE2）rather than dividing them block-wise in traditional accelerators"。配套的映射原则：partition 阶段优先层内（layer-wise）划分——把整层放进同一神经核，从而"spike broadcast in tiling strategy and spike reduction between PEs in Local Input Reducer can be avoided"。
- Annotations：列切分维度 = 输出特征维（膜列）；spike 广播 = 所有 PE 收到相同输入脉冲集；归约消除 = 每输出元素由唯一 PE 产出，无部分和合并。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：ELSA 的 PE 控制模块按 spike 的行/列编码从本地权重 buffer 取对应列分片；贪心分区算法（Algorithm 2）决定哪些层放进同一神经核（约束：核内存 A 与核神经元电路数 D），层内 MM-sc 再用列切分在 4 个 PE 间展开。效果：Tab.IV 中 ELSA 面积效率 41.26 GOPS/mm² 为弹性 SNN 加速器最高（TrueNorth 0.134、PAICORE 19.78），部分归功于无重叠、无归约的列切分；Fig.14 显示映射三阶段（partition→mapping→routing）的目标之一就是"minimize NoC traffic"（层内划分直接省掉 spike 广播跨核流量）。

涉及论文标题：
- ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing

## IDD-Scan（Intra-Segment Dependency Decoupled Scan，段内依赖解耦扫描）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ENEC 提出的、专为 Ascend NPU 设计的并行前缀和（scan）实现。动机：AscendC 要求操作数 32 字节对齐，half 类型（2 字节）每行 M=16 个元素恰好 32 字节，架构禁止对同一 32 字节硬件段内相邻元素直接做 SIMD 运算——即 row[i] += row[i-1] 这类"段内依赖"被硬件锁定、无法直接计算；同时 Ascend 无 CUDA 式轻量线程同步（每 AI core 是单一重线程）。IDD-Scan 用"转置 + 列方向扫描 + 层级行扫描"把禁止的段内依赖改写成硬件友好的跨行向量加法，把计算受限问题转化为一系列向量化操作（多阶段算法）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
目标：对 N×M 局部张量（M=16）做全前缀和。两阶段（以 8×16 half 张量为例，Figure 8）：
```
# Stage 1: 行内前缀和 via 转置（把行内依赖变行间）
T = transpose(M_local)              # 8×16 → 16×8：原每行元素散到 16 个新行
R_T = column_prefix_sum(T)          # 对 16 行的每列做 log2(M)=4 步向量加法
R = transpose(R_T)                  # 16×8 → 8×16：R 每行含正确的行内局部前缀和
# Stage 2: 行间传播与最终修正
C = R                               # 保存局部结果副本
for k in 1..log2(N)=3:              # 层级行扫描
    C[i] += C[i - 2^k]              # 行间元素级加法
offset = exclusive_scan(C[:, last]) # 末列含包含式行偏移 → 去尾补 0 得排他偏移
result = R + broadcast(offset)      # 偏移矩阵广播加到 R
```
Annotations：转置把"段内（同行 32B）相邻依赖"变成"跨段列方向依赖"，列方向扫描每步只做整列向量加法（合法）；行间传播用 log2(N) 步，无需原子/同步；最终排他偏移广播回加。代价是多几次转置/内存搬运，但把不可行的操作变成全向量化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：AscendC 的向量加/转置（half 张量）在 AIV 上执行，是 ENEC 解压 kernel 的核心原语——把 bit mask 转 0/1 整数后求前缀和，得到逆 gather 的偏移。效果：V3 相比 V2（朴素 scan）解压吞吐提升近 100%（IDD-Scan 直接贡献）。通用性：任何 Ascend 上需要 scan/前缀和的 kernel（如 softmax、GEMV 累加、压缩位重排）都可复用；GPU 移植版直接改用 CUB 库并行前缀和（ENEC-GPU-V1，吞吐 419.2 GB/s），说明该算法是 Ascend 对齐约束下的专用替代品。

涉及论文标题：
- ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs

## 前缀和（Prefix Sum / 并行 Scan）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
并行计算中的基础原语：给定数组 a[0..n-1]，输出 b[i] = Σ_{j≤i} a[j]（包含式 inclusive）或 Σ_{j<i} a[j]（排他式 exclusive）。高效并行实现用工作高效的并行扫描算法（Blelloch 的 up-sweep/down-sweep 树形两阶段：O(n) 工作、O(log n) 深度）。在现代加速器上，GPU 用 CUB/Thrust 的 warp/block scan（含 Kogge-Stone、Brent-Kung 等变体 + 跨 block 的 chunk 扫描与流水）；在 Ascend NPU 上则受 32 字节段内禁止 SIMD 的约束，ENEC 用 IDD-Scan 绕行（见 IDD-Scan 条目）。ENEC 论文中前缀和占解压 kernel 30% 的开销（基础版），是解压性能的关键瓶颈之一。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ENEC 中前缀和的用途：解压时 bit mask（0/1）→ 前缀和 → 每个元素在压缩流中的起始偏移，供逆 gather 取数：
```
mask = [1,0,1,1,0,1,0,1]          # 8 个组是否"超出 m 位需要额外字节"
offset = prefix_sum(mask)          # [1,1,2,3,3,4,4,5]（排他）
for i in range(N):
    low[i] = stream[offset[i]]     # 按偏移逆 gather，偏移=前面需要额外字节的组数
```
Annotations：mask 中 1 的数量决定每个组相对起始位置的偏移；前缀和把"动态偏移计算"向量化。GPU 侧（ENEC-GPU-V1）用 CUB 的 parallel prefix sum + warp 内建快速通信；CPU 侧用 AVX2/BMI2 PEXT 处理非字节对齐。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：GPU 用 CUB cub::DeviceScan/BlockScan（Blelloch 或 decoupled look-back 变体）；Ascend 用 IDD-Scan（转置+列扫描+行扫描）或朴素逐元素（低效）；CPU 用 std::partial_sum 或 AVX2 向量化。使用：除解压偏移计算外，广泛用于 softmax 分母、注意力位置编码、内存压缩、基数排序、负载均衡等。ENEC 的经验：在 Ascend 上扫描是"没有硬件直接支持"的原语，性能取决于能否绕开段内依赖约束。

涉及论文标题：
- ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs

## AscendC（Ascend 算子编程模型与执行流水）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
华为 CANN（Compute Architecture for Neural Networks）框架内的 C++ 算子编程模型，用于在 Ascend NPU 上开发高性能自定义算子（2023 年 CANN 7.0 起替代旧 TBE 框架）。核心抽象：tensor（封装将由 AIC/AIV 操作的数据）与 queue（同步机制——操作完成 EnQue，依赖操作 DeQue），配合 Pipe 管理资源；标准三段式执行范式 CopyIn → Compute → CopyOut（数据经 MTE 搬进片上缓冲 → 计算单元执行 → 结果搬出），多流水（vector/cube/mte）可重叠。Ascend 把每个 AI core 当作单一重线程（无 CUDA 式轻量线程间同步），靠任务队列驱动的流水数据流重叠数据搬运与计算。ENEC 论文用 AscendC（C++17）实现压缩/解压算子，并指出其限制：32 字节对齐约束、无 gather/scatter、无条件分支、整数算术指令少。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ENEC 压缩 kernel 的 AscendC 流水骨架（每 AIV 线程处理 16384 元素块，循环分发）：
```
// CopyIn: MTE 把 HBM 权重块搬进 UB
Copy(ub_in, gm_weights, block);
// Compute（AIV 向量单元，全逐元素指令）:
  E = ExtractExponent(ub_in);            // 拆分 BF16 指数
  y = (E - b) * -1 & mask;               // 分支无关整数变换
  packed = HierarchicalHalvingPack(y, m, L);  // 位宽量化 + lane folding 打包
  EnQue(packed_q, packed);               // 结果入队
// CopyOut: 满 32KB buffer 后输出低 16 位 + bit mask 到压缩流，右移继续
DeQue(next_q);                            // 取下一块数据
```
Annotations：EnQue/DeQue 队列抽象让搬运-计算-搬运三段重叠；无分支意味着所有"如果超过 m 位"的判断都提前用 bit mask 表达；每 core 单线程但 48 个 AIV 并行处理不同块。编译：CANN toolkit（AscendC 编译器）把 C++ 编成 AI core 指令，产物为 .so 算子。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：AscendC 语言 + CANN 工具链（Ascend-cann-toolkit 8.2.RC1.alpha002 + kernels-910b 同版本），Ubuntu aarch64 + Python 3.9 + torch/torch_npu；build_csrc.sh 编译 csrc/ 目录 NPU kernel 为 .so；msprof 做 kernel 级 profiling。使用：任何 Ascend 自定义算子开发（ENEC 压缩/解压算子、模型算子）；开发者用 tensor/queue/pipe 表达数据流，靠 tiling 把数据切进 UB（ENEC 选 16384 元素块，32K 会超 UB 192KB）。局限（ENEC 论文强调）：无分支/无 gather/无变长内存操作/无轻量同步——这正是传统无损压缩算法在 Ascend 上"根本性不兼容"的根源。

涉及论文标题：
- ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs

## VQ-GEMM tiling 与 GEMM-Epilogue 流水重叠（流式/静止数据调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
EVA 的 tiling 与调度策略：VQ-GEMM 涉及的张量（input、WI、WC、output）预先在 DRAM 布置，其中仅 input 与 WI 分块流式（streaming）载入——input 按 v×d（v=32, d=8）tile 载入片上 buffer 参与一轮计算，WI 按 v×N 流式流入（v×N 大，靠流式平衡吞吐与 buffer 占用）；WC 与 output 在层内静止（stationary）于片上 SRAM——WC 只读全程复用，output tile 逐轮更新、全部部分和累加后写回 DRAM。片上 buffer 分两个区：GEMM 计算区（input+WC）与 epilogue 区（WI+output）。调度上 GEMM 与 Epilogue 重叠：VQ-GEMM（256 cycles）产出 OC 直接片内送 EU（4096 cycles, N=4096），无片外往返、无带宽争用，EU 为关键路径、近峰值利用率；多 batch 时不同请求复用同一 weight tile 降带宽。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# decode 一层 FC（N=4096）：
for tile in tiles:                          # tile 高 v=32
    load X_tile(32×8), stream WI_tile(32×N) # 流式（DRAM→片上）
    O_tile = X_tile @ B                     # 32×8 阵列 VQ-GEMM，256 cycles（WC 静止）
    y_tile = EU(O_tile, WI_tile)            # 查找+加法树，4096 cycles（与下一 tile GEMM 重叠）
    writeback y_tile                        # output 静止累加后写回
```
作用：把 decode 从"带宽受限的 GEMV"变成"加法受限的 GEMM+Epilogue"；EU 为瓶颈时可加 EU 数（DSE：4 EU 匹配 64GB/s 带宽），GEMM 单元部分空闲但扩展代价极小。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：EVA 片上 buffer 528KB（16KB WC + 256KB weight + 32KB input + 192KB OC + 32KB output）；decode tiling m=M、k=4·v·d/M、n=N；prefill tiling m=1024、k=32、n=1024。使用方式：作为加速器设计参数做 DSE（Table III/Fig. 8）；PE:EU 比例 = 2^n:N 决定瓶颈——2^n<N 时 EU-bound（可加 EU），2^n>N 时 PE-bound 且 spurious 乘法增多；batch scaling 下 EU 阶段多请求复用 weight tile（Fig. 7c）。

涉及论文标题：
- EVA: Accelerating LLM Decoding via an Efficient Vector Quantization Architecture

## Performance Signature Vector（PSV，性能签名向量）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PSV（Performance Signature Vector）是 TEA/IPU 的 PICS 生成中用到的核心数据结构：一个 bit-mask，表示"某个 PC 的某次动态指令实例上，发生了哪些性能事件"（如 DTLB miss、DCache miss、LLC miss、Drain-SQ full、Misspeculation 等事件的组合位图）。背景逻辑链：要生成逐指令周期栈（PICS），需要知道每个动态指令实例把时间花在哪些事件组合上；直接用 (PC, 事件组合) 的展开表存储爆炸，用 PSV 压缩为"每位一个事件"的紧凑位向量（TEA 用约 9 个事件、IPU 的 PICS 演示用 17 个信号覆盖事件+PC 控制），每个 PC 的动态实例维护一份 PSV。TEA 论文的设计值为每 400,000 cycles 采样/归并一次 PICS。论文的 IPU 版：IPU_lite 顺序核每 cycle 更新 PSV，事件发生时经 load-modify-store 序列把对应事件位置 '1'，flush 时把 PC 存进 IPU 内存供 commit 后引用；每 400,000 cycles 扫描活动 PSV 列表，把 (PC + 事件签名) 归并成 PICS 条目经 FIFO 发主机。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
IPU 的 PICS introspection kernel 中 PSV 更新的计算过程（215 bits/cycle 输入，17 个信号）：
```
// 每 cycle（数据驱动：新 IORegs 数据到达即运行 _main）
loop:
  if itlb_miss(io_reg_x0):   # 事件检测：if-else-if 事件链
      psv = load(psv_table[pc])
      psv.bit0 = 1           # load-modify-store 置位
      store(psv_table[pc], psv)
  if icache_miss(io_reg_x1):
      psv = load(psv_table[pc])
      psv.bit1 = 1
      store(psv_table[pc], psv)
  ...                        # 其余事件（17 信号逐一检测）
  if flush_occurred(pc):     # flush 时保存 PC 供 commit 后引用
      store(flush_pc, pc)
  if cycle_count == 400000:  # TEA 设计值，定期归并
      for each active psv in psv_table:      # 扫描活动 PSV 列表
          fifo_send(PC + signature(psv))     # 归并成 PICS 条目发主机
      cycle_count = 0
```
事件输入到达率分析：215 bits 中 >75% 的 cycle 无事件（ROB 采样时未 stall/drain），~25% 出现单个长延迟事件，罕见 2-3 个事件；ROB 采样之外通常每 cycle 1-3 个事件触发 3-9 条指令执行。近似误差来源：事件在上一事件 PSV 生成窗口内到达会被丢弃（IORegs 保持旧数据丢弃新数据）——单 cycle 模拟 vs 每 cycle 模拟的平均相对误差 <3%（3 个应用 10-14%），PC 排序始终正确，丢 PC 覆盖 ≤0.37% cycles。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：TEA 硬件版在 BOOM 核内为每个 in-flight 指令维护 PSV（约 9 事件、249B 存储、3.2mW、1.1% 性能开销、2.1% 平均误差）；IPU 软件版把 PSV 表放 32KB scratchpad，事件检测与置位用 RISC-V load-modify-store 序列（直方图/hash 指令作 intrinsics 优化）。使用方式：PSV 是"时间比例归因"的数据载体——主机后处理把每 PC 的 PSV 周期计数汇总成 PICS 栈（表 II 格式：PC 0x7912d0 DTLB miss+DCache Miss=50000000 cycles），开发者据此优化热点指令；IPU 版无需 BOOM RTL 改动、事件集合可软件扩展（对比 TEA 固定 9 事件）。验证：3 个 DARCHR microbenchmark（https://github.com/darchr/microbench）各只期望一个 PC 入栈，结果吻合。

涉及论文标题：
- Enabling Continuous, In-Field Introspection: The Programmable IPU Architecture

## Introspection Binary（内省二进制）与数据驱动执行模型（data-driven execution，丢数据语义）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Introspection Binary（内省二进制）是运行在 IPU 上的小签名分析程序：相对"用户程序"（跑在芯片主处理器上的应用，如浏览器、DL 推理负载），内省二进制专门分析用户程序的硬件级行为。它与普通内核/算子程序的关键差异：①输入不是内存数据而是 HIT 的微架构信号（每次执行 32 个命名输入寄存器，各带 valid 位指示新数据到达）；②按数据驱动执行模型运行——新数据到达且 IPU 处于 ACTIVE-PAUSED 状态时调用 _main，处理期间处于 ACTIVE-RUNNING，新数据被丢弃；③输出经逻辑 FIFO 或主机内存映射区域发出（IPU 可发简单访存指令透明路由到主机内存区域）；④受代码签名与策略模式（closed/restrictive/permissive）约束，经 app-store 式分发部署。程序用 RISC-V 工具链 C 编写（直方图/hash/循环指令作 intrinsics），IPU_pro 程序附带 Verilog 软逻辑；三段式结构（init 固定 0x0、_main、end 固定 0x7F0），8KB 指令内存=2048 条指令。丢数据是核心语义：IPU 不能 stall HIT，introspection 程序处理速率 < 数据到达率时（Little's Law 下必然）丢数据，程序必须用采样、聚合或事件稀疏性适配（随机化采样窗口长度可打破与周期硬件行为的病理性对齐）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
内省二进制的执行模型与丢数据语义（示意）：
```
// 配置阶段（主机 API 经 MMIO）
IPU_CONFIG_IMAGE("PICS-generation")   # 加载签名二进制
IPU_CONFIG_START(ROI_BEGIN)           # TS：区域开始 PC
IPU_CONFIG_STOP(ROI_END)              # TE：区域结束 PC

// 数据驱动执行（每新 IORegs 数据触发一次 _main）
状态机: PAUSED --(配置完成)--> ACTIVE-PAUSED
        ACTIVE-PAUSED --(新数据到达 valid=1)--> 调用 _main --> ACTIVE-RUNNING
        ACTIVE-RUNNING --(新数据到达)--> 丢弃（IORegs 保持旧值）
        _main 返回 --> ACTIVE-PAUSED（等待下一数据）
        FINALIZE 状态 --> 执行 end 清理代码 --> PAUSED

// 示例（PICS generation 的 itlb miss 处理，论文汇编片段）
_main:
    regtimer 50000, psv_loop        # 50000-cycle 定时器驱动 psv_loop
psv_loop:
    beq x0, 1, itlbm_m               # x0=itlb-miss 信号
    beq x1, 1, icache_miss           # x1=icache-miss 信号
    ...
itlbm_m:
    hash r1, x12                     # x12=PC 硬件输入，hash 得表索引
    ld   r2, r1, 0                   # load PSV
    addi r2, r2, 0x40                # 置事件位
    sw   r2, r1, 0                   # store PSV
    ret
```
数据率适配三模式（IPU 2GHz vs HIT 3-4GHz 不同步）：①亚采样（1-in-2 cycles）接受降保真；②快 buffer 异步窗口处理；③只限低频相或优化 IPU 速度。输出节奏：PICS 每 400k cycles 发几字节 PSV 数据；预取器 emulation 每 2³¹ cycles 发统计避免计数器溢出；GPU 直方图每 256 cycles 发 3 字节。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：标准 RISC-V 工具链交叉编译 C 代码 + 专用指令 intrinsics（histogram/hash/loop）；开发环境含 IPU emulator（RISC-V 功能模拟 + IPUpro Verilog co-sim）+ HIT traffic injector 测试台（按 ABI Spec 的时序/数据率生成信号流，忠实复现丢数据行为，验证近似误差并迭代采样窗口长度/分析粒度）；部署经签名+策略模式。使用方式：开发者（或芯片设计者）把分析算法写成内省二进制（如 PICS 75 行、Gaze 根因分类 50 行、GPU 直方图 2 行、预取器 emulation 300 行 Verilog），按 ABI Spec 引用信号输入寄存器；用户程序侧用 API 配置区域与触发；运行结束后主机后处理输出。局限：处理速率不足会丢数据（需采样/聚合），不能注入 HIT（只能观测+本地计算），二进制受 8KB 指令内存与 32 输入限制。

涉及论文标题：
- Enabling Continuous, In-Field Introspection: The Programmable IPU Architecture

## 逐 Miss 根因分类（Real-Time Component-Level Diagnosis，实时组件级逐事件根因诊断）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逐 Miss 根因分类是论文演示的 IPU 第四大能力（Real-Time Component-Level Diagnosis）：在事件发生的时刻，通过订阅组件内部决策点信号，把每个微架构失败事件（如 L1D demand miss）当场分类到根因类别，输出"每 miss、每 PC 的失败模式分布"。背景逻辑链：聚合计数器（PMU）只报告"组件失败了多少次"，无法解释"为什么失败"；采样工具（如 PEBS）缺乏预取器内部状态可见性；两者都区分不了"cold region（冷区，无预取覆盖）"、"no learned pattern（预取器没学到模式）"、"late prefetch（预取太晚）"、"prefetch failure（预取被逐出/失败）"这四类根因——而它们指向完全不同的微架构投资方向（PHT 容量 vs 训练触发 vs 访存调度/lookahead 距离 vs cache 容量/替换策略/预取节流）。IPU 通过"tapping 组件内部决策点信号"解决：不注入、不采样，直接看预取器决策路径上的信号（AT hit、PHT 的 missed_in_pt、MSHR in-flight 等），把诊断带到部署硅片上的真实负载，无需重新 tape-out，且同一二进制可部署到 fleet 上发现生产环境与设计期基准的根因分布差异。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
逐 miss 根因分类 introspection kernel 的计算过程（HIT=L1D cache+prefetcher，6 信号=132 bits，20-cycle 处理窗口）：
```
// 每 demand miss 到达（LSQ 经地址总线送 miss 虚拟地址与 PC）
classify_miss(pc, dem, at_hit, missed_in_pt, mshr_inflight):
    # dem: demand miss 指示; at_hit: Accumulation Table hit
    # missed_in_pt: PHT 在 region 毕业时是否含模式
    # mshr_inflight: 预取 in-flight 状态
    if not at_hit:                 # AT miss → 冷区，预取从未学到此区域
        category = COLD_REGION
    elif missed_in_pt:             # AT hit 但 PHT 无模式
        category = NO_LEARNED_PATTERN
    elif mshr_inflight:            # 预取已发出但在途（太晚/未覆盖本次 miss）
        category = LATE_PREFETCH
    else:
        category = PREFETCH_FAILURE  # 预取过但被逐出/失败
    bucket = hash(pc)              # 按 PC 累积
    hist[bucket][category] += 1    # 4 类失败模式计数
    every(20 cycles): fifo_send(hist)   # 周期性输出紧凑直方图
```
丢数据建模：20-cycle 处理窗口内到达的新 miss 被丢弃（L1D miss 事件相对稀少：Gaze 激活后残留 miss 率低，间隔 <20 cycles 的连续 miss 罕见），平均每 trace 类别比例误差 <3.5%、中位 2.46%；输出为每 PC 四类计数直方图（384B/报告周期，主机流量可忽略）。诊断含义：x264_s 的 PREFETCH_FAILURE 占比高 → cache 压力逐出预取数据 → 需改 cache 容量/替换/节流；NO_LEARNED_PATTERN 主导 → PHT 容量不足或训练触发保守；LATE_PREFETCH 主导 → 访存调度延迟或 lookahead 距离问题；COLD_REGION/NO_LEARNED_PATTERN 的 PC 是显式预取 hint 的候选。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：IPU_lite 上 50 行 introspection 代码（scratchpad 读 + RISC-V 指令，行为与 PICS 演示相似，面积 0.019 mm²、功耗 15mW）；模拟验证用 ChampSim + Gaze 作者实现（189 条 SPEC traces），离线分析脚本按 IPU 20-cycle 处理延迟建模丢数据并分类每个 demand miss（Gaze 是 HPCA 2025 空间预取器，SJTU，Zenodo 数据 artifact https://zenodo.org/records/14252372 提供 traces）。使用方式：硬件设计者据失败模式分布决定微架构投资方向（PHT/训练触发/访存调度/lookahead/cache 替换/节流）；软件开发者把标记的 PC 作为显式预取 hint 候选；fleet 部署同一二进制对比生产 vs 设计期根因分布。局限：需要预取器暴露内部决策点信号（论文证明 Gaze 的 AT/PHT/MSHR 信号在合理实现中现成可得），分类粒度受信号可见性与 20-cycle 处理窗口约束。

涉及论文标题：
- Enabling Continuous, In-Field Introspection: The Programmable IPU Architecture

## NTT（Number Theoretic Transform，数论变换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- NTT 是定义在有限域（素数模 q 上）的离散傅里叶变换（DFT）变体，用单位原根与模算术代替复指数，把多项式乘法（卷积）从 O(N²) 降到 O(N log N)，是 RLWE/Ring-LWE 类 FHE 方案（BGV/BFV/CKKS）中多项式算术的性能基石。iNTT（逆 NTT）把乘积域结果变回系数域。在 HE-CNN 推理中，NTT/iNTT 是乘法（CMult/PMult）、旋转（keyswitch 的 automorphism）、bootstrapping 等几乎所有原语的共同子操作。
- 本论文角色：NTT 是旋转（含 keyswitch）的 dominant 消耗者——Ring-LWE 方案的 keyswitch/automorphism 需多次正向与逆变换在分解基间切换（Fig.1(a)：旋转 4.8ms vs PMult 0.15ms）；每次 (i)NTT 调用伴随大量 twiddle factor 与密文系数的内存搬运，是 FHE 加速器内存带宽瓶颈的来源之一。FEnc² 通过减少旋转/keyswitch/NTT 的"数量"（相对 HELayers 最多降 94%）来降低 NTT 单元需求，属应用层优化与 NTT 硬件加速（ARK/Neo/TensorFHE 等）正交互补。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一次长度为 N 的 NTT kernel（Cooley-Tukey 基 2，GPU 上按 stage 调度）：
```
input:  a[0..N-1] (系数域), w = primitive N-th root mod q
for stage s in 1..log2(N):
    len = 2^s; half = len/2
    for i in 0..N-1 step len:            # 每 block 一个 warp/thread block
        for j in 0..half-1:
            u = a[i+j]; v = a[i+j+half] * w^{j*N/len} (mod q)
            a[i+j] = u+v; a[i+j+half] = u-v
bit_reverse(a)
```
- Annotations：kernel 输出喂给 keyswitch（与旋转求值密钥做点积再逆 NTT）或 CMult；FEnc² 的 GPU 评测（Table VI）显示 kernel 调用从 48,015 降到 5,775（-88%）、GPU 内存传输从 12,021MB 降到 1,461MB（-87.8%），即 NTT kernel 的调用次数与随之而来的内存流量同时被压缩。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件：SEAL/OpenFHE/Liberate-FHE/TenSEAL 内嵌的 NTT 实现（GPU 上做多多项式批量 NTT、radix-2/4/8 混合、双缓冲掩盖传输）；专用 GPU 库 TensorFHE/HE-Booster/Cheddar 用 tensor core 或 CUDA core 加速 NTT。硬件：ARK、Neo、FAST、EFFACT、CraterLake、F1 等加速器内置专用 NTT/iNTT 单元（多 lane、多基分解）。使用场景：任何 RLWE 多项式乘/keyswitch/自举，是 HE 加速器面积与功耗的主要构成；本论文的架构启示是"算法层面削减 NTT 需求后可缩减 NTT 簇规模、降低面积功耗并缓解内存带宽"。

GenZA 补充视角（ISCA'26，面向 ZKP 的 NTT 映射）：ZKP 的 NTT 大小可达 2^23（Groth16 29.32% 时间、Plonky2 多项式乘），stride 不规则重排（stage i 的 stride 2^{N-1-i}）是大 N 下 off-chip 访问的主要挑战。GenZA 的 NTT 映射：(1) 2D-NTT 分解（four-step NTT）——把数据当 2D、每维做 √N 大小（如 2^13）子-NTT 片内完成，维度间转置用全局 transpose buffer；(2) MDC（Multipath Delay Commutator）流水线——每逻辑级 = radix-2 蝴蝶 + FIFO 延迟缓冲（容量∝stride），沿 PE 行实例化，PE scratchpad 分区作 FIFO；(3) 折叠流水线 + scratchpad 借贷——L=13 长 MDC 管线映射到 2×8 PEs，SRAM 饥饿的首段向邻近空余 PE 借 FIFO 空间（借/贷 PE 距离 ≤2 hops、FIFO 访问下同时至多一对活跃、NoC 流量 ≤2× 前向数据），免去 LegoZK 的 3D 分解（3 次 off-chip 往返）与 UniZK 昂贵的专用 transpose buffer；2^23 NTT 流量 7.4→3.0 GB、PE 利用率 16%→38%、时间 27.1→11.4 ms；(4) 小 bitwidth（64-bit Goldilocks）整条 MDC 管线合并进单 PE（32 lanes 够两条 L=13 管线，只用内部 crossbar/forward chain，完全避免 NoC 压力）；(5) NoC 带宽分析：每蝴蝶 II≈6.75（256/384-bit）或 10.125（768-bit）PE 周期（KO 乘+Montgomery 归约），worst-case 256-bit 约 152 GB/s = per-hop 容量 30%（32×64-bit links @ 2 GHz）。NTT 算术强度低（每元素 1 模乘），本质访存受限，故融合/流水线（见"核间融合与流水线"条目）进一步消除中间数据传输。
- HyperDrive 补充视角（ISCA'26，GPU FP64-TCU 上的分层 NTT kernel）：把 NTT kernel 拆为 Inner-NTT（radix-64 基例，全片上）与 Outer-NTT（EWMult、Residual NTT、转置、GMEM 搬运）两级两 kernel（Kernel 1/2）。Inner-NTT 用 FP64 TCU 的 8×4×8 MMA（PTX m8n8k4）以 warp 粒度执行——每个 warp 用 4 次 MMA（MMA1/2 乘 twiddle factor matrix 的高/低 16-bit 分量、MMA3/4 完成第二级 radix-8）+ Bit-Merge + ModRed + EWMult 完成一个 64 点 NTT，寄存器内完成（TLMOP），零中间 SMEM；转置用 MMA 的 Fragment A/B/D 跨线程分布隐式完成（TransOP）。Outer-NTT 用 RowMaj（预转置布局，3/4 GMEM 访问变行主序全 coalesced）+ TFOP（negacyclic 顺序读、TF-XY 预排序、TF256/TFM 驻 SMEM）。效果（N=2^16、36 limbs）：SMEM stall -33.2%、occupancy 55.0%→76.8%（NTT+ 92.5%）、GMEM stall -59.5%、总延迟 -61.1%、吞吐 932.6 KOPS（2.0× WarpDrive、5.5× Neo，图 15-18）。

涉及论文标题：
- FEnc2: Unifying Data Packing for Efficient Private Inference via Convolution and Architecture-Aware Fragment Encoding
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols
- HyperDrive: Hierarchical Exploitation of Memory Efficiency for GPU-Based FHE Acceleration

## MSM（Multi-Scalar Multiplication，多标量乘法）与 Pippenger 算法

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MSM 是椭圆曲线密码学的核心原语：计算 R = Σ_{i=0}^{N-1} s_i ⊗ P_i（N 个标量 s_i 与 N 个椭圆曲线点 P_i 的带权求和），是 Groth16/HyperPlonk 的 KZG 承诺/打开的 dominant kernel（占 prover 时间 59–70%）。Pippenger 算法（FOCS 1976）用"加窗口 + 桶累加"把昂贵的 PMUL 换成便宜的 PADD：标量 s 按 c-bit 窗口切分，桶累加（bucket accumulation）阶段按窗口值把点 P 分发进对应桶并 PADD 累加，桶归约（bucket reduction）阶段把桶值按桶号加权求和 Σ i⊗B_i，最后窗口聚合按 2^c 幂加权合并。算术强度高（170 modmul/元素），是计算密集型 kernel。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- Pippenger 三段调度（GenZA 映射）：
```
# ① 桶累加（可并行）：每点每窗口独立
for each point P_i, for each c-bit window w of s_i:
    bucket[window][w] += P_i          # 分发到对应桶 PADD
# ② 桶归约（先 PE 内并行、后跨 PE 串行）：
for each window: B_window = Σ i ⊗ bucket[i]   # 两个顺序 PADD 求和优化 [59]
# ③ 窗口聚合（忽略级）：R = Σ B_window ⊗ 2^(c·win)
```
- Annotations：GenZA 的调度要点：(1) 动态 window size 选择——离线成本模型按曲线 bitwidth、MSM 大小 N、片上 SRAM、带宽选 c（MNT4-753 从 2^14 的 c=11 到 2^23 的 c=16），对照 zkSpeed 固定 c=9 最高 2.90× 加速；(2) window-major 映射——桶数超片上 SRAM 时每轮从所有 window 各取子集桶（而非整 window 的桶），提高桶归约阶段并行度（同 window PE 串行归约，并行度正比于片上 window 数），免去 LegoZK 的树式归约；(3) 单 PADD 用 complete addition formula（齐次射影坐标，免模逆）需 12 模乘+2 常数乘：MNT4-753 全 32 lane 组单 PADD 单元（2 个宽乘法器时间复用 14 次模乘），BN128 每 PE 2 个 PADD 单元（4 个宽乘法器，常数乘化简为加法）；(4) 附加优化：signed-digit 把桶数从 2^c−1 减半到 2^{c-1}，sparse MSM 预累加标量 1 的点；(5) 分发由 MSM decoder & dispatcher 在 NoC 前复制点注入对应行（BN128 c=16 平均每点到 ~4 PEs），桶命中均匀无热点。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：CPU/GPU 用 cuZK/GZKP/Bellman 的 MSM（GPU 用多 GPU 或 Pippenger 并行化）；FPGA 用 CycloneMSM/PipeMSM/BSTMSM；ASIC 用 PipeZK/SZKP/zkSpeed/GenZA 的专用或统一单元。使用：任何配对型 PCS（KZG）的承诺与打开；GenZA 中由 PE 阵列 + decoder&dispatcher 执行，MSM 大小与 bitwidth 决定最优 window c，调度器离线选定后硬件配置 window/桶数。NoC 评估（packet-level 模拟）：BN128 c=16 下 dispatch stall 仅 3.97%、平均 link 利用 5.9%、最热 link 峰值 44.51%。

涉及论文标题：
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols

## 核间融合与流水线（Inter-kernel Fusion & Pipelining，空间分区流水线）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 这是面向加速器多 kernel 串行执行的调度概念：把协议计算图按 cut 点切分成子图，调度器用简单 roofline 模型估计各 kernel 行为后贪心合并可并行子图，把 PE 阵列空间分区，让多个 kernel 同时片上执行成流水线，并按吞吐匹配调整各 kernel 的 PE 比例。目标：消除 kernel 间中间数据落片外（fusion）与 kernel 间空闲等待（pipelining），缓解访存瓶颈，提高 PE 利用率。GenZA 中属于静态调度器的输出（Section VI-F），一次性离线调度，成本在多次证明实例间摊销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 调度流程（Groth16 例子）：
```
1) 把 PIOP 计算图按 cut 点切分（三类 cut：数据转置 / 全局归约 / Fiat-Shamir 串行点）
2) 对每个子图，贪心合并 kernel 为空间流水线（若 roofline 估计提升性能）
3) 调各 kernel 的 PE 比例匹配吞吐；子图间数据直接前递（fusion 消除中间写回）
例子：子-iNTT 尾部与后续子-coset-NTT 头部直接融合，中间数据留片上
```
- Annotations：效果（Table XI，对比简单 LRU 缓存）：Groth16 流量 247.9→237.5 GB（MSM 计算密集、收益小）；HyperPlonk 196.7→117.0 GB（1.7×）；Plonky2 1220.0→128.3 GB（fusion 单独 8.1×、pipelining 再 1.2×，共 9.5×），但 Plonky2 离线调度需 517.8 s（复杂计算图，一次性成本）。cut 点类型：数据转置（多项式阶段间/2D-NTT）、全局归约（MSM 标量点积、sumcheck 向量和）、Fiat-Shamir 哈希挑战（强制串行点）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：GenZA 静态调度器（分析成本模型+贪心合并+roofline 估计）产出 PE 分配与流水线方案，运行时每 PE 收模式配置（kernel 类型、bitwidth、场模、数据流模式）；tiling 相同的 producer/consumer 融合，树式 workload 分片、串行归约用 segmented-parallel。使用：访存密集、多串行 kernel 的协议（Plonky2）收益最大；与 kernel 级映射优化（折叠 NTT、动态 MSM window）正交叠加。类比：GPU 库的 kernel fusion（cuDNN/cuBLAS 手调 kernel 模板），GenZA 把融合决策自动化到调度器。

涉及论文标题：
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols

## 分层静态-动态协同调度（Hierarchical Scheduling / Static-Dynamic Co-design）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
分层静态-动态协同调度是把"离线全局规划"与"在线局部自适应"组合成闭环的调度范式，核心前提是：稀疏工作负载的 intra-tile（tile 内数据流）与 inter-tile（tile 形状/占用/遍历顺序）参数互相耦合，任何一层单独优化都会次优。Harmonia 将其实现为三层：(a) 静态分析层（Static Analytical Layer）——只用粗粒度描述符（矩阵形状 M,K,N、全局密度 ρA/ρB、数据类型）与硬件参数（PE 阵列大小 P、SRAM 容量 S_SRAM）离线枚举候选块形状 (T_M,T_K,T_N)，以操作强度 OI=OPs/Bytes 为目标、受 SRAM 可行性约束 s_val(E[nnz_A]+E[nnz_B])+s_psum·T_M·T_N ≤ β·S_SRAM（β=0.8）过滤，输出基线块形状、块间遍历顺序与 SRAM 分区 S_A/S_B/S_C，提供一个安全（1.0×）性能下限并缩小在线搜索空间；(b) 动态 Profiling 层（Dynamic Profiling Layer）——块进入 SRAM 后按 tile 精确 nnz 采样（tile 密度 ρtile=nnz/(T_M·T_K)、行/列方差、非零聚类），细化 tile 形状并选择 InP/Row/OutP 数据流；(c) 动态 Tuning 层（Dynamic Tuning Layer）——消费硬件反馈计数器（SRAM pressure、psum spill、MRN merge-depth、PE stall），用成本模型 Gain>α·Cost 决定是否在 tile 边界切换数据流/微重切块，滞回（T=2~4 周期）防振荡、最坏情况回退 1 次重构延迟（50 cycles）。三层共同把"同构硬件"变成"逻辑异构"执行引擎。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
调度流水（伪代码，Harmonia 处理一次 SpMSpM 的全过程）：
```
# 静态层（离线，每矩阵一次）
T_set = gen_candidates(M,K,N)            # 按可整除性与 PE 对齐剪枝
for (TM,TK,TN) in T_set:
    if feasible(TM,TK,TN, rhoA,rhoB, S_SRAM):   # 约束(2)
        OI[T] = est_OPs(T) / est_Bytes(T)
(TM*,TK*,TN*) = argmax OI; order = pick_traversal(M,K,N)
# 动态 Profiling 层（每块一次）
for block in blocks:
    rho = block.nnz / (TM*TK);  var = row_variance(A_block)
    if rho < thr_low:  expand(block)           # 低密度扩张提复用
    if rho > thr_high: shrink(block)           # 高密度收缩防 psum spill
    df = select_dataflow(rho, var, cluster)    # InP/Row/OutP
# 动态 Tuning 层（每 tile 执行中，反馈驱动）
while executing(tile):
    if anomaly(merge_depth, spill, stall):
        if gain(candidate_df) > alpha*cost(reconfig): switch_dataflow(tile)
        elif gain(micro_tile) > alpha*cost(micro_tile): micro_retile(tile)
```
关键设计：静态层决策独立于逐 tile nnz（保证 buffer 可行性）；切换严格在 tile 边界发生（全局矩阵是一串独立 tile），pipeline flush + DN/MRN 重编程 + buffer reset 共 20–50 cycles，总 stall <1%。结果：16 个 SpMSpM workload 平均 1.75×（orani678 3.46×，接近 2.03× oracle 上界），端到端 DNN 1.87×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
对比既有单层调度：Misam 用决策树在 tile 内选数据流但忽略 tile 间依赖与共享 SRAM 压力；HYTE 用静态分析+运行时细化调整全局 tile 边界但假定固定 tile 内数据流；Vesper 用统一解析模型离线搜全局最优但假设均匀稀疏、无运行时反馈。Harmonia 的实现要点：(1) 静态层用"保守可行性"而非逐 tile 精确建模，避免离线采样过拟合；(2) 在线层每块只做一次轻量采样（A 矩阵驻留部分），开销可忽略；(3) 反馈路径与执行数据通路完全解耦（每 PE 行 128 个计数器占 <0.5% PE 阵列面积），保证 profiling 不扰动时序；(4) 成本模型用 α 调节激进程度（不规则负载 α 大、规则负载 α 小）。论文未提供开源实现。

涉及论文标题：
- Harmonia: A Unified Hierarchical Scheduling Framework for Sparse Matrix Multiplication

## 运行时数据流切换与动态调谐（Runtime Dataflow Switching / Dynamic Tuning / Micro-retiling）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
运行时数据流切换是 Harmonia 动态 Tuning 层的核心机制：在执行一个 tile 的过程中，根据硬件反馈计数器观测到的异常（psum 溢出、merge 深度超预期、PE stall、SRAM pressure），在 tile 边界把 intra-tile 数据流在 InP/Row/OutP 之间切换，或在块内做微重切块（micro-retiling），使同构 PE 阵列呈现"逻辑异构"（每个 tile 用最适合当前稀疏模式的数据流）。切换规则（论文给出的显式映射）：InP→Row 当 A 行极不平衡、InP→OutP 当稀疏近均匀；Row→OutP 当 merge 深度低但 B 行重载压 SRAM、Row→InP 当 tile 稠密；OutP→Row 当 spill 源于局部性、OutP→InP 当 spill 源于高密度。每次切换的硬件开销 = pipeline flush + DN 路由表/MRN 模式（merge-before-store vs column-accumulate）重编程 + AGU/buffer 控制器策略重置 = 20–50 cycles。只有 Gain=T_before−T_after > α·Cost 才触发，滞回机制（异常计数器连续 T=2~4 周期超阈值）吸收瞬时抖动，切换失败时最多损失 1 次重构延迟（50 cycles）后回退到静态基线。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 动态 Tuning 反馈环（每 tile）
loop over cycles:
    spill  = read_counter(psum_spill)
    depth  = read_counter(mrn_merge_depth)
    stall  = read_counter(pe_stall)
    if (spill > TH_spill or depth > TH_depth) and anomaly_hold >= T:
        cand = candidate_switches[dataflow]        # 表驱动切换候选
        if best_gain(cand) > alpha * reconfig_cost:   # 式(3)
            flush_pipeline(); reprogram_DN_MRN(cand); reset_buffers()
            # 20-50 cycles
        elif micro_tile_gain > alpha * micro_cost:
            shrink_K(tile)   # 降 merge 深度 / 收缩 M,N 限 DN/buffer 负载
```
具体例子（email.mtx）：静态层选 OutP，执行中发现部分 tile 密度高导致 psum 溢出本地 buffer、merge 深度超预期 → 反馈计数器连续触发 → 评估切换成本（~40 cycles）后在该 tile 边界切到 Row（选择性路由 B 行片段、merge 深度更浅更可预测），后续 tile 恢复 OutP；整体 16 个 workload 平均 1.75× 加速、总 stall <1%。bcsstk10 场景：静态模型预测 OutP 最优但实际 OutP 延迟为 Row 的 1.6×，动态 Tuning 通过反馈纠正了这一离线误判。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现层面依赖三项轻量硬件支撑（合计 3.3% 面积、<0.5% PE 阵列面积）：Feedback Counters（每 PE 行一组，经轻量 metadata crossbar 汇聚到 Tiling Controller，反馈路径与主数据通路解耦）、Reconfiguration Engine（重编程 DN 路由表、MRN 模式、AGU 与 buffer 策略）、Tiling Controller（执行切换决策与成本模型）。使用上：作为调度器最末级，只纠正 tile 级偏差、不改变全局计划；适用于稀疏模式在运行时剧烈变化的负载（LLM attention/MLP 投影的 token 级稀疏抖动、CNN 剪枝后的 channel 级波动、高度不规则矩阵如 orani678 的聚类长尾）。论文未提供开源实现。

涉及论文标题：
- Harmonia: A Unified Hierarchical Scheduling Framework for Sparse Matrix Multiplication

## Tiling（分块调度：tile 形状与 occupancy 对数据流的影响）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Tiling（分块）是把无法整块驻留片上 SRAM 的稀疏矩阵按块（tile）切分执行，块形状 (T_M,T_K,T_N)、块占用（tile occupancy，SRAM 中同时驻留的 tile 规模）与块间遍历顺序共同决定数据复用、buffer 压力与 psum 行为。Harmonia 的核心洞察（Insight 1）是：inter-tile 参数会从根本上重塑 intra-tile 数据流的相对性能，因此 tile 形状与数据流必须联合优化而非分层孤立。论文用 16×16 PE 阵列、16KB 本地 buffer、1MB SRAM 的 cycle-accurate 模拟验证：(a) tile 形状改变最优数据流——ResNet-0.1 与 Llama-0.2 负载下，保持操作量不变改变 (64×K×N)，K 小 N 大时 OutP 最优，K 大时 OutP 因 buffer 溢出性能骤降而 InP/Row 受益于更高 K 的复用与 PE 并行度，(64,128,64) 使 Row 优于 OutP；(b) tile occupancy 重塑 SRAM 访问——从 16×16（1×PE 阵列）到 256×256（256×）缩放，OutP 最早到最小流量但对超大 tile 因 psum 溢出变差，InP 受益于大 K 但超大 tile 增加重载成本，Row 最容忍大 tile（逐行处理）但极端大 tile 放大 B 冗余访问。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Harmonia 静态层选块过程（Algorithm 1）：候选块数 n_M=⌈M/T_M⌉、n_K=⌈K/T_K⌉、n_N=⌈N/T_N⌉；独立近似下 E[nnz_A]=ρA·T_M·T_K、E[nnz_B]=ρB·T_K·T_N；可行性约束 (2)：s_val(E[nnz_A]+E[nnz_B])+s_psum·T_M·T_N ≤ 0.8·S_SRAM（留出索引/元数据/运行时变化余量）；选取最大 OI=OPs/Bytes 的 (T_M,T_K,T_N)。动态层按 tile 实际密度细化：低密度扩张 tile 提升 PE 利用率与复用、高密度收缩防 psum spill 降 merge 深度、聚类稀疏时把 tile 边界对齐非零簇防局部热点。微重切块（micro-retiling）只在块内进行：spill/SRAM pressure/深 merge 背压触发收缩（降 K 降 merge 深度、限 M/N 降 buffer/DN 负载），持续低密度无异常时扩张提复用。遍历顺序启发式：M,N 小（C 可驻留）时用 k-outer 保持 C 驻留、流式 A/B 块；K 小则强调 M/N 复用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
既有方案对照：Tailors 用离线平均稀疏度静态 overbooking 分块；DRT/HARP 运行时重构 tile 但控制开销高；HYTE 用静态分析+运行时细化调整 tile 边界但数据流固定。Harmonia 的差异：静态层保证所有候选块在"合理稀疏模式"下 buffer 可行（不依赖逐 tile nnz），在线层只改静态层显著偏差的维度（轻量、只改 tile 切分与分发逻辑、无硬件重构），动态层以反馈为据做块内微调——三层 tiling 决策逐级细化。论文未提供开源实现；评估用 bcsstk10.mtx、email.mtx、orani678、rajat19（SuiteSparse）与剪枝 DNN 权重。

涉及论文标题：
- Harmonia: A Unified Hierarchical Scheduling Framework for Sparse Matrix Multiplication

## CSR/CSC 与 bitmask 稀疏存储格式（Sparse Storage Formats）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CSR（Compressed Sparse Row）与 CSC（Compressed Sparse Column）是稀疏矩阵的标准压缩存储格式：CSR 用三个数组（values 存非零值、col_indices 存每非零的列号、row_ptr 存每行非零的起始偏移）把 O(nnz) 存储降至 O(nnz+rows)；CSC 是转置版本（列压缩）。bitmask（位掩码）格式对轻度稀疏区域用一位表示一个元素是否有值，索引解码开销低、带宽省。Harmonia 的硬件数据通路（Semi-independent PE rows + DN + MRN）对高稀疏 tile 用 CSR/CSC + 显式坐标列表（explicit coordinate lists），对轻度稀疏区域用 bitmask；进入 PE 数据通路前，非零元素必须做动态对齐（dynamic alignment / intersection）——只让坐标匹配的非零对进入乘法器，用可重构 DN + 轻量 on-row index-matching 逻辑完成，避免把零塞进 MAC（这正是固定数据流加速器在高稀疏下利用率崩塌的原因，如 SIGMA Flex-DPE <10%）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CSR 的 SpMSpM 内核骨架（C=A×B，A 按 CSR 存）：
```
for r in 0..M-1:                      # A 的每个非零行
    for (k, a) in A_row[r]:           # A 行内每个非零 (列号 k, 值 a)
        for (n, b) in B_row[k]:       # B 第 k 行的非零 (列号 n, 值 b) —— 索引匹配
            C[r][n] += a * b          # 只有匹配坐标的非零对被乘
```
Harmonia 的 Row 数据流硬件化该骨架：每个 PE 行驻留一个 A 行，DN 按该 A 行的 nnz 列号选择性路由所需的 B 行片段（B 片段在 on-row BUF 缓冲一次供本行所有 PE 共享），MRN 沿单一 A 行轨迹做行顺序归约——选择性路由直接消除了冗余 B 传输与深度 psum 归并。CSC 对应 InP 的按列取数，bitmask 用于低开销索引对齐。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
标准库实现：cuSPARSE、MKL（mkl_sparse_?_mm）、SuiteSparse 矩阵以 CSR/CSC 分发；GPU 稀疏内核常结合 CSR + 行级平衡或转 COO/bitmap 变体。Harmonia 中的使用方式：格式选择是 tile 粒度决策（与数据流选择联动），全局 SRAM 存 tile 与元数据（含逐 tile nnz，供 pre-execution profiling）；DN 的可编程路由与 on-row 索引匹配在 tile 边界重配置即可适配不同格式，无需改 PE 数据通路。论文未明确说明对格式切换的开销建模细节。评估矩阵来自 SuiteSparse Matrix Collection（bcsstk10.mtx、email.mtx、orani678、rajat19 等，均为标准稀疏矩阵格式分发）。

Ultra-CSR 补充视角（ParetoES，ISCA'26）：Ultra-CSR 是 AccelES（HPCA 2025）提出的 FPGA 友好 CSR 变体——用位掩码（bit-mask）压缩指针/行索引开销，最小化元数据、最大化每 512-bit HBM packet 的非零承载数；ParetoES 沿用它做检索矩阵编码：配合 INT6 量化后每 512-bit 传输 30 个非零（FP32 下约 11 个），带宽效率提升 6×；并新增 Random-CSR（AccelES 提出，动态逐向量访问变体）的簇探测用法。ParetoES 的流水用法：质心与簇子矩阵都预编码为 Ultra-CSR，质心放 HBM 通道头部（每 packet 30 非零、1 packet/cycle 流入 x-decoder）；x-decoder 用位宽 popcount 单周期解析行索引，产出 (x,y,val) 元组直接进乘法器；选中簇按簇局部有序索引组织（簇感知数据布局），随机访问被限制在活跃簇块内、呈流式 burst 访问——这是"全局不规则 → 有界流式"访存重塑的关键一环。

涉及论文标题：
- Harmonia: A Unified Hierarchical Scheduling Framework for Sparse Matrix Multiplication
- ParetoES Hardware-Accelerated Sparse Embedding Similarity via Pareto-Optimal Pruning

## MPA（Multi-Precision Arithmetic，多精度算术流水线）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MPA 是 FHE NTT/多项式乘法在无原生宽整数乘法的硬件（如 INT8/FP16/FP64 Tensor Core）上模拟 32×32-bit（或更宽）模乘的技术：把宽乘法拆成 bit-splitting（按位拆分成多段）、多次窄乘法、bit-merging（分段结果合并回宽整数）三步。INT8 TCU 方案（TensorFHE/WarpDrive）需要把 32-bit 乘法拆成 16 次 8×8-bit 子乘法，MPA 流水线开销占 INT8-TCU-NTT 总计算时间的 26%~28%（图 3）。
- HyperDrive 用 FP64 TCU（53-bit 尾数精度）实现轻量 MPA：单次 32-bit 乘法把一个乘数拆成两个 16-bit 部分（高/低），分别与另一个 32-bit 乘数做 FP64 乘法，乘积 ≤48-bit，MMA 累加 8 个乘积后仍不超 FP64 精度，最后用 INT64 位合并——单次 32-bit 乘法仅需 2 次 FP64 乘法（对比 INT8 方案的 16 次）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- FP64-TCU 上的 32-bit 模乘（配合 Inner-NTT 的 MMA 执行）：
```
# 输入：a, b ∈ Z_q (32-bit)，q < 2^32；FP64 MMA 累加 8 个乘积
a_hi = a >> 16;  a_lo = a & 0xFFFF        # bit-splitting：高/低 16 位
# MMA1/2：数据矩阵 A 分别乘 TFM 的高/低 16-bit 分量（同一数据两次 MMA）
P_hi = a_hi * b;  P_lo = a_lo * b          # 各 ≤48-bit，FP64 精确表示
acc = Σ (P_hi<<16 + P_lo)                  # 8 个乘积累加 ≤53-bit
r   = BitMerge(acc) mod q                  # INT64 位合并 + 模约减（ModRed）
```
- Annotations：radix-64 Inner-NTT 中每个 warp 的 4 次 MMA（MMA1/2 乘 TFM 高/低 16-bit 分量、MMA3/4 第二级 radix-8）+ 中间的 Bit-Merge/ModRed/EWMult 即一条完整 MPA 流水；相对 INT8 方案，子乘法数从 16 降到 2，MPA 占 NTT 总延迟从 26%~28% 降到 5.0%~7.3%（相对 -84%~-91%，图 16）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：把 NTT 的 32-bit 乘法改写为"数据矩阵 × twiddle factor matrix 高低位分解"的两次 MMA + 寄存器内位合并/约减；INT8 方案（WarpDrive/TensorFHE）则用 8×8-bit 子乘法链 + 进位处理。使用场景：任何用 TCU 加速高精度整数/模运算的 kernel（FHE NTT、多项式乘）；HyperDrive 用它支撑 radix-64 Inner-NTT 的 FP64 TCU 映射，是"FP64 低 MPA 成本 vs INT8 高 MPA 成本"设计取舍的核心（Table III）。

涉及论文标题：
- HyperDrive: Hierarchical Exploitation of Memory Efficiency for GPU-Based FHE Acceleration

## TLMOP / TransOP（Thread-Level Memory Optimization 与隐式转置）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- TLMOP（Thread-Level Memory OPtimization）是 HyperDrive 的 Inner-NTT 优化：用细粒度寄存器数据访问替换 TCU MMA 的自动 SMEM fragment 填充。自动填充下，每步 MMA 前后 fragment 都要经 SMEM 读写（单次 N=2^16 Inner-NTT 产生 6.13MB SMEM 流量、单次 KeySwitch 达数 GB）。TLMOP 先分析 FP64 MMA（8×4×8，PTX m8n8k4）fragment 在 32 线程间的数据分布（每线程 Fragment A/B 各 1 元素、Fragment C 2 元素、结果 D 均分），据此把 64 点 Inner-NTT 的计算流与 lane 映射设计成纯寄存器流水：每个 warp 执行 4 次 MMA + Bit-Merge + ModRed + EWMult，SMEM 只在读全局输入与写最终结果时访问一次。
- TransOP（Transpose OPtimization）解决 4-step NTT 内嵌的转置：常规转置是 SMEM 读-写全周期的内存操作，HyperDrive 利用 MMA 的 Fragment A/B/D 跨线程分布实现隐式转置——MMA1/2 把数据矩阵当 Fragment B、MMA3/4 当 Fragment A 并交替选取奇/偶列，使数据无需跨线程交换即可从 MMA1/2 结果直接进入 MMA3/4，只有 Bit-Merge 等线程本地操作。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 单个 warp 完成一个 64 点 Inner-NTT 的寄存器内流水（图 8A/9）：
```
# 数据布局：warp 的 32 线程持有 8×8 数据矩阵的 fragment 分布
# ① 两级 radix-8 的第一级：两次 MMA（乘 TFM 高/低 16-bit 分量）
D1 = MMA1(A_data, B_tfm_hi);  D2 = MMA2(A_data, B_tfm_lo)   # 8 个 radix-8 NTT
# ② 位合并与约减
r = BitMerge(D1, D2);  r = ModRed(r);  r = EWMult(r, twiddle)
# ③ 第二级：数据变 Fragment A、twiddle 变 Fragment B，隐式转置
D3 = MMA3(A=r[奇列], B=tfm);  D4 = MMA4(A=r[偶列], B=tfm)
# ④ 最终位合并与约减
out = ModRed(BitMerge(D3, D4))
```
- Annotations：关键点——(1) MMA1/2 的 fragment 分布与 MMA3/4 相同，转置由"数据在 Fragment A/B 之间换角色 + 奇偶列选取"完成，无 SMEM 往返；(2) 多 Inner-NTT 在一个 block 内并行执行，采用多 warp/多 block 并行 + warp 内串行循环的混合策略平衡并行度与片上资源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：手写 CUDA kernel，用内联 PTX mma.sync.aligned.m8n8k4.f64（或 WMMA API）并手动管理寄存器 fragment 的线程映射，替代 auto-filled fragment 的 SMEM 中转。使用：任何在 FP64/其他 TCU 上执行 NTT 的 GPU FHE 库；效果——SMEM 相关 stall -33.2%（加 TransOP 累计 -38.9%）、scheduler stall cycle -44.2%（累计 -50%）、occupancy 从 55.0%/62.3% 提到 76.8%/77.0%（NTT+ 达 92.5%）、单次 Inner-NTT SMEM 流量大幅下降；TLMOP 是消融中贡献最大的单项优化（图 15）。

涉及论文标题：
- HyperDrive: Hierarchical Exploitation of Memory Efficiency for GPU-Based FHE Acceleration

## Row-Major NTT（RowMaj，行主序 NTT 与预转置布局）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- RowMaj 是 HyperDrive 的 Outer-NTT 内存布局优化：4-step NTT 把多项式看成 2D 数组、先按列做内层 NTT，导致 3/4 的 GMEM 访问是列主序（strided、非连续），无法 coalescing，带宽利用率低并引发 pipeline stall（baseline stage1/2 中 GMEM 相关 stall 占 45.9%/22.6%）。RowMaj 引入预转置（pre-transposed）数据格式：密文、密钥、明文在 GPU 生命周期内以转置布局存储，使原本的"列"变为"行"访问，3/4 的 GMEM 访问变为行主序全 coalesced。
- 预转置只改内存布局不改数值；转置成本近零——通过多项式操作时直接按转置布局输出（而非单独转置 kernel），且在编码/解码阶段对单 limb 数据完成（RNS 分解前/重构后），避免对多 limb RNS 表示做昂贵转置。配合 TFOP（Twiddle Factor access OPtimization）：negacyclic 卷积 twiddle 顺序读（INTT 预乘 N^{-1} 省一次模乘）、外层 Hadamard 用预排序的 TF-XY、内层 Hadamard/Residual NTT 用 SMEM 中的 TF256（≤256 元素）、Inner-NTT 的 8×8 TFM 驻 SMEM 跨 warp 复用。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- Row-Major NTT 的两 kernel 数据流（Alg. 1 + 图 10）：
```
# 编码阶段（单 limb）：按转置布局写入（零成本）
encode(): coeff → 预转置布局（RNS 分解前）
# Kernel 1：并行块行主序 coalesced 读 GMEM → SMEM
a = GMEM-Load(行主序)            # 全 coalesced 128B 事务
SMEM ← a ⊙ ζ_2N[1:N]             # negacyclic 卷积（顺序读）
for each 64 点 Inner-NTT: 寄存器内 MMA 流水（TLMOP/TransOP）
外层 Hadamard EWMult（TF-XY，与 Kernel 2 起点对齐）
行主序写回 GMEM（转置并入写回阶段，无单独 kernel）
# Kernel 2：读回 → 内层 Hadamard（TF256 复用）→ Residual NTT → 全局转置 → 输出
```
- Annotations：优化前基本 NTT kernel 有 7 次 sparse + 4 次 limited-locality GMEM 访问 vs 1 次 coalesced；优化后只有 1 次 limited-locality、其余 9 次全 coalesced。twiddle 存储：每模数 negacyclic 2N 个、外层 Hadamard N 个、TF256/TFM 各 256 个，36 层共 27MB GMEM、每 block 2KB SMEM。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：在 GPU FHE 库的编码/解码路径维护预转置布局，NTT/INTT kernel 按行主序读写，Auto（索引置换）kernel 保持全 coalesced 输入读，元素级 kernel（EWAdd/EWMult）不受影响；关键收益是它消除 NTT 的"多 pad"约束，使跨多项式 kernel（BConv/IP）能与 NTT 融合（见 COOP 条目）。效果：TFOP+RowMaj 使 GMEM stall -59.5%、scheduler stall -27.4%（累计 -39.3%）；H100 上 NTT 吞吐 1669.5 KOPS 验证跨代可扩展。

涉及论文标题：
- HyperDrive: Hierarchical Exploitation of Memory Efficiency for GPU-Based FHE Acceleration

## COOP（Memory-Aware Kernel Co-optimization，内存感知 kernel 协同优化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- COOP 是 HyperDrive 的操作级 kernel 融合优化：把跨多项式（cross-poly）kernel（BConv、IP）与 NTT 融合成协同 kernel，让高维中间数据驻留片上、隐藏内存延迟、避免 off-chip 往返。此前融合不可行的根因是维度冲突：NTT kernel 以"多项式内"多 pad 方式访问（2D 足迹 N1×#pad），而 cross-poly kernel 需"多项式间"数据交换（3D 足迹 N1×#pad×#poly），融合会耗尽寄存器/SMEM 并降低并行度。RowMaj 消除多 pad 约束后，每 block 数据维度降低，融合成为可能。
- 具体融合：(1) (BConv2-NTT1)——BConv 的矩阵乘法阶段与 NTT Stage-1 融合，每 thread block 处理单个 limb i、BConv 约减沿 α' 维（GMEM→Reg→SMEM），SMEM 中间系数直接喂 NTT Stage-1（SMEM→Reg→GMEM），避免物化 L+α-α' 维（Alg. 2）；(2) (NTT2-IP)——NTT Stage-2 与 IP 融合，NTT 中间结果保持寄存器内直接做 IP 累加（Alg. 3）；(3) INTT2-BConv1——BConv1（EWMult 阶段）与前置 INTT2 融合；BConv1 与 BConv2 不融合（BConv2 的矩阵乘结构会导致重复计算）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- (NTT2-IP) 融合 kernel 的伪代码结构（Alg. 3 简化）：
```
for i in 0..(ℓ+α)-1:                 # 逐 limb
  for k1 in 0..N1-1:
    # NTT Stage-2 内层：寄存器内完成内层 NTT
    for k2 in 0..N2-1:  t[k2] = InnerNTT(c_partial[i][k1][k2])   # 寄存器驻留
    # 外层循环做 IP：与 evk 做 MAC（放弃 lazy reduction，每乘即约减）
    for d in 0..β-1:
      evk_d ← GMEM 预取（在 NTT 计算期间预取到片上，隐藏延迟）
      acc += t[k2] * evk[i][k1][k2][d] mod q_i      # 立即约减，寄存器减半
    out[i][k1][k2] = acc
```
- Annotations：数据 prefetching 的关键是 IP 的 evk（尺寸为输入 2 倍、访存密度高、算术强度低）——融合后 evk 在 NTT 执行阶段预取，与 NTT 计算重叠；寄存器压力缓解：放弃 lazy reduction（每乘即约减）把中间结果位宽减半、寄存器需求减半，NTT-IP kernel 寄存器 72/线程（35.3% occupancy）仍优于分开执行。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：改写 KeySwitch/bootstrapping 的 kernel 流水，把 (BConv2-NTT1)、(NTT2-IP)、(INTT2-BConv1) 写成融合 kernel，并用 evk 预取与即时约减控制寄存器；配合操作重排（ModDown 后紧跟 ModUp 时提前 EWSub，使 ModDown/ModUp 的 INTT 可批处理合并，图 13）。效果：KeySwitch 相对 NTT+ 提速 1.24×，(BConv2-NTT1) 1.36×、(NTT2-IP) 1.32×；BConv 的 stall long scoreboard 从 60.6%、IP 的 74.6% 显著下降；bootstrap 中 COOP 再贡献 1.21×（KeySwitch 1.24×、Rescale 1.21×）。

涉及论文标题：
- HyperDrive: Hierarchical Exploitation of Memory Efficiency for GPU-Based FHE Acceleration


## Buffer Table（BT，缓冲表 / 软件页表）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Buffer Table 是 MANATEE 运行时 page manager 的软件页表：条目数 = SPM page frame（颜色）数，每个条目记录对应 frame 当前驻留的 NVM 页号。其作用是在每次 load/store 前判定"编译器预测驻留的页是否真的在对应 frame 中"，从而处理投机着色的 misspeculation。
- 关键设计：编译器为每条 load/store 插桩 (页号, 颜色) hint，运行时按颜色直接索引 BT 对应条目（无需查表搜索），命中则按页内偏移访问 SPM，未命中则加密驱逐旧页、解密载入新页并更新 BT。这让每次访问的驻留判定降到"一次直接索引 + 一次比较"的开销。
从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 伪代码（page manager 处理一条 store）：
```
store(addr, val, pageNum, color):
  frame = color                       # 编译器 hint 直接定位 frame
  if BT[frame] == pageNum:            # 命中：页已驻留
      SPM[frame][offset(addr)] = val
      WTQ.enqueue(frame, pageNum)     # 记脏
  else:                               # 未命中（misspeculation）
      if BT[frame] != EMPTY:
          encrypt_and_persist(SPM[frame], BT[frame])   # 加密驱逐旧页
      page = decrypt(fetch_NVM(pageNum))               # 解密载入新页
      SPM[frame] = page; BT[frame] = pageNum
      if is_store: WTQ.enqueue(frame, pageNum)
```
- 例子（论文 Figure 8）：SPM 有 5 个 frame、BT 条目 5 个；`store m1 100` 携带 (page=3, color=1) → 检查 BT[1] 是否为页 3，是则命中直接写 SPM[1] 并记 WTQ。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：运行时数据结构（软件），与 WTQ 配合；page manager 代码 491B。BT 依赖编译器 metadata（每条指令的页号/颜色）实现无查找访问。论文未给出公开代码，无法确认是否开源。
涉及论文标题：
- Intermittence-aware Speculative Page Coloring for Secure NVM

## Write Tracking Queue（WTQ，写跟踪队列）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Write Tracking Queue 是 MANATEE 运行时的轻量结构，跟踪哪些 SPM 页已被写脏（dirty）：每个条目记录 (脏页号, SPM frame 号)。它的作用是让断电前只持久化被修改过的页（按页粒度加密写回 NVM），避免把整个 SPM 都加密——这是相对 NVSRAM/Mapi-Pro"整 SPM checkpoint"的核心节能点。
- 正常执行中 WTQ 满时，最老条目被逐出，对应脏页先加密写回 NVM 再复用条目；断电（V_backup 触发）时，JIT checkpoint 按 WTQ 中所有条目把对应脏页加密持久化，再保存寄存器等程序状态。WTQ 代码 1,028B，远小于整 SPM。
从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 伪代码：
```
on_store(frame, pageNum):
  SPM[frame][offset] = val
  if WTQ 满:                         # 逐出最老脏页
      (oldFrame, oldPage) = WTQ.dequeue()
      encrypt_and_persist(SPM[oldFrame], oldPage)
  WTQ.enqueue(frame, pageNum)        # 记当前脏页
on_power_failure():                  # V_backup 触发
  for (frame, pageNum) in WTQ:       # 只持久化脏页
      encrypt_and_persist(SPM[frame], pageNum)   # 4x16B 块凑 64B 原子 flush
  checkpoint(registers, heap, stack); flip(flag)
```
- 例子：程序连续写页 3、页 5，WTQ 记录 (frame1,page3)、(frame2,page5)；断电时这两页被加密写回，未修改页无需持久化。
术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：运行时软件队列（1,028B），与 BT、page manager 协同；脏页以 AES-XTS 页粒度（64B = 4×16B 块）原子持久化。论文未给出公开代码，无法确认是否开源。
涉及论文标题：
- Intermittence-aware Speculative Page Coloring for Secure NVM

## Shape-aware Dispatch（跨平台 kernel 部署：离线生成+验证每平台专属变体，在线按 shape 选最优并回退 vendor library）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Shape-aware Dispatch 是 KernelEvolve 的异构部署策略：因为针对一个平台优化的 kernel 不必然迁移到其他平台（异构平台内存层次、执行模型、编程抽象不同），对每个硬件平台离线（offline optimization phase）生成并验证平台专属 kernel 变体，部署时对每个输入 shape 选择该 shape 下性能最高的变体，生成 kernel 表现不佳时回退到 vendor library（conv1d/conv2d 等）或 PyTorch 基线，确保"自动化合成带来性能收益而不引入回归"（safe production deployment）。论文中的 fallback 触发条件：conv1d 在 out-of-distribution shape（64×768×768×1024）上生成 kernel 仅 0.49-0.63×（相对 PyTorch），此时部署走回退；Optimized FM 对 N≤64 生产 shape 用 fused kernel（2-4×），更大 N（tiling overhead 占优）回退 PyTorch 非融合 baseline；MapId 在 MTIA v2i/v3 的 edge case（batch 2000 时 0.78×）按输入维度 runtime dispatch 回退 PyTorch。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 调度逻辑（伪代码）：
```
# 离线（每平台 p，每算子 op）：生成+验证专属变体
for p in {NVIDIA_H100, AMD_MI350, MTIA_v3, ...}:
    K_p[op] = KernelEvolve(op, hardware=p)   # 图搜索 + 知识库检索生成
    validate(K_p[op])                        # TritonBench 正确性 + speedup
# 在线：shape-aware 选择
def dispatch(op, shape, platform):
    if shape in PROFILED_SHAPES[op] and speedup(K[platform][op], shape) > 1:
        return K[platform][op]               # 生成 kernel 最快
    else:
        return vendor_library[platform][op]  # cuDNN/cuBLAS 或 PyTorch 回退
```
- 具体例子（conv1d 跨 5 平台生产 shape (2048,96,96,200) FP16）：离线为 A100/H100/MI300/MI350/MTIA v3 各生成专属变体（NVIDIA 走 Tensor Core tile 化+3D grid+double-buffer，AMD 走 Infinity Cache 感知 tiling，MTIA 走 SFU/跨 PE/dual-core）→ 在线对生产 shape 选生成 kernel（1.77×/2.30×/1.75×/2.54×/6.54× vs conv1d），对 out-of-distribution shape（64×768×768×1024）自动回退 PyTorch（生成 kernel 在此 shape 仅 0.49×）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：离线阶段由 KernelEvolve 的图搜索流水线完成（每个平台用对应解释器 meta_kernel_gpu/amd/mtia_interpreter 编译执行 + TritonBench 测速），部署时 wrapper 内含 shape-keyed 变体表 + fallback 路径；评估 harness 由 evaluation code generator 确定性生成保证一致性。收益：跨平台一致加速 + 无回归风险（论文强调所有 PFFN 配置 speedup ≥1.0、MapId/MBDT edge case 走回退），使自动生成能安全进入生产。局限：离线 profile 只覆盖目标 shape 分布，out-of-distribution shape 表现下降（论文明示"optimization targets production distributions rather than arbitrary inputs"）。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

## C3（Concurrent Computation and Communication，并发计算与通信重叠）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- C3 指把通信 kernel 与计算 kernel 在同一设备上并发执行以隐藏通信延迟的技术，源自 HPC（LogP 模型时代用计算掩盖数据搬移），GPU 系统上表现为同一 GPU 同时调度两个并发 kernel（一个计算、一个通信）。分布式 LLM 训练中广泛用于把 AllGather（AG）、ReduceScatter（RS）、AllReduce（AR）等通信集合与 GEMM 重叠（FSDP 前向 AG 与输入投影 GEMM、反向 RS 与 MLP down/up 投影 GEMM 重叠），端到端平均 1.1×–1.6× 加速。C3 不是免费的：有限 GPU 资源被并发 kernel 瓜分，计算与通信互相干扰（共享计算单元与内存带宽），计算 kernel 运行时平均慢 18.9%、最高 40%（ConCCL 等报告）。
- Lit Silicon 论文（ISCA'26）的核心发现：C3 重叠不是均匀的——同一节点 8×MI300X 上同一 kernel 的重叠率跨 GPU 显著不同：straggler GPU（更热更慢）重叠率恒定最低（29.6%），leader GPU 重叠率动态增长（最高 52.7%，约为 straggler 的 1.8×），且重叠率与 kernel 时长强相关（Pearson 相关与余弦相似度均高）——C3 是节点级性能波动的主要贡献者。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# FSDP 一层前向的 C3 调度（每个 GPU 独立执行，虚拟并发）
compute_stream  : GEMM_qkv_in  →  GEMM_attn_op  →  GEMM_mlp_* ...
comm_stream     :      AG(next_layer_params)  ── 与 GEMM_qkv_in 起重叠
# 反向
comm_stream     : RS(prev_layer_grads) ── 与 MLP down/up GEMM 重叠
```
Annotations：两条硬件队列（compute/comm stream）并发运行，GPU 抢占式/流式调度下共享 SM 与内存带宽。重叠率（overlap ratio）= 通信 kernel 与计算 kernel 并发执行的时间占比；lit silicon 用 Chopper 解析 trace 计算每 kernel 每 GPU 的重叠率与起始时间。leader 提前发 AG 但必须等 straggler 的 AG 完成（集合是同步点），等待期间计算流继续推进使 leader 的"重叠"变长、资源竞争加剧反而更慢。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：(1) 软件——PyTorch/FSDP 用独立 CUDA stream（或 AMD 等价）把通信 kernel 放到非计算流，训练框架在层间自动插入（torch.distributed 的 async 集合）；RCCL/NCCL 集合本身支持异步；(2) 硬件——DMA 引擎（如 GPU 上的 copy engines）卸载通信减少计算干扰（ConCCL），或专用通信加速器。Lit Silicon 论文用 Chopper（作者自研 GPU 特性分析工具，arXiv:2512.08242）对 PyTorch trace 做 kernel 级分析，量化重叠率与 kernel 时长，用于检测 straggler（重叠率与时长作为 lead value 的输入）。使用场景：任何用集合通信同步的分布式训练/推理；注意 C3 重叠率差异本身就是节点级性能波动的放大器（与热致掉队耦合形成 Lit Silicon 负反馈）。

涉及论文标题：
- Lit Silicon: A Case Where Thermal Imbalance Couples Concurrent Execution in Multiple GPUs

## Lead Value 与 Straggler Wave（领先值与掉队波，kernel 起始时间差的检测指标）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Lead value 是 Lit Silicon 论文（ISCA'26）提出的检测指标：对每个 kernel k，比较其在所有 GPU 上的起始时间戳，leader GPU 比最晚（straggler）GPU 早开始的时间差即该 kernel 上该 GPU 的 lead value；把同一 GPU 上所有 kernel 的 lead value 聚合（sum=曲线下面积，默认；也可 max/last）得到每 GPU 的聚合 lead 值。straggler 的 lead≈0（总是最晚），leader 的 lead>0。Straggler wave 指 trace 中连接各 GPU 同一 kernel 起始时间形成的波前图：straggler 的波前最晚、leader 波前领先。
- 与 C3 的关联：straggler 通信起始晚 → leader 等待延长（C3 重叠变长、资源竞争）→ leader 变慢，lead 值在迭代内动态积累到 equilibrium 后重置，跨迭代重复。检测使用窗口平均（默认窗口 3 个采样）平滑。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Algorithm 1: LEADVALUEDETECT
Input: Timestamp vector T[g,k] for g in GPUs, k in kernels
Output: Lead value vector L[g]
for each Kernel k:
    T_max <- max(T[all GPUs, k])
    for each GPU g:
        lead_value[g,k] <- T_max - T[g,k]
for each GPU g:
    L[g] <- sum_k lead_value[g,k]     # 或 max / last
return L
```
Annotations：T_max 是该 kernel 在节点内最晚的起始时间（straggler 的时间）。示例：GPU0 比 GPU1 早 10ms 开始某 kernel，则 GPU0 该 kernel lead=10ms；若某 GPU 的 lead 在 100 个 kernel 上从 0 线性涨到 10ms，其 sum 聚合 lead≈500ms。聚合方式选择：sum 在 equilibrium 期间也惩罚 GPU（利于在乘法性 C3 干扰下识别 leader），max/last 收敛更快但信息少。输出 L[g] 作为 Algorithm 2 功率上限增量计算的输入（norm_lead 归一化后乘 global 衰减与 max_inc 默认 15W）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：运行时解析 PyTorch profile trace（Chopper 工具）获得每 GPU 每 kernel 起始时间戳，按上述算法计算聚合 lead 值；实际部署每 10 迭代采样一次（每样本 dump+处理约 4 秒），窗口平均后用于调整各 GPU 功率上限（amd-smi），收敛（约 20 样本/80 秒）后功率分布可停用或长周期复用。使用场景：检测多 GPU 节点内的 straggler/leader 归属（热致掉队的量化）、驱动节点级功率重分配（GPU-Red/GPU-Realloc/CPU-Slosh）、评估 C3 引起的性能波动；也用于 MoE 训练（all-to-all 不重叠时 lead 值小但有大 spike，仍可收敛）。

涉及论文标题：
- Lit Silicon: A Case Where Thermal Imbalance Couples Concurrent Execution in Multiple GPUs

## FP8 kernel 量化缩放（tensorwise / rowwise / blockwise scaling）与快速累加（fast accumulation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- FP8 GEMM 执行前需把高精度输入/权重量化到 8-bit 浮点，量化精度取决于缩放（scale）的粒度——这是低精度 kernel 的"recipe"核心选择维度：tensorwise（整张量共享一个 scale，开销最小但精度最差）、rowwise（每行一个 scale，精度与开销折中）、blockwise（如 DeepGEMM 按 128×128 块每块独立 scale，精度最好、需额外缩放计算）。fast accumulation（快速累加）指 GEMM 累加器用 FP32 而非低精度，避免累加误差累积，是精度-性能权衡的另一个开关。论文对比的三库 recipe：TorchAO（tensorwise TW / rowwise RW / 混合 RW GW HP——前向 rowwise、输入梯度高精度、权重 backward tensorwise）、DeepGEMM（blockwise BW）、FBGEMM（rowwise RW）。
- LoKA 的量化开销实证（H100，27 个生产 LRM shape，从 (2048,256)@(256,768) 到 (2048,123200)@(123200,1024)）：端到端 FP8 相对 BF16 平均仅 1.6×；最大有效 TFLOPS < 硬件峰值 20%；量化开销占端到端 GEMM 延迟 >30%；计入 layout 操纵等内存分配开销后 FP8 可能差于 BF16——量化/反量化与布局转换是低精度收益的主要吞噬者。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一次 FP8 GEMM kernel 执行流程（以 rowwise 为例）：输入 x∈R^{M×K}、W∈R^{K×N} → 量化 kernel：对 x 每行求 max→scale_x[i]=max|x[i,:]|/127，x8[i,:]=round(x[i,:]/scale_x[i])（W 按列或块同法）→ FP8 张量核 GEMM：C_fp32[m,n]=Σ_k x8[m,k]·W8[k,n]（FP32 快速累加）→ 反量化：C[m,n]=C_fp32[m,n]·scale_x[m]·scale_w[n]（tensorwise 为单标量相乘；blockwise 需按块拼装）。
- 纯计算 vs 端到端拆解：纯 GEMM 吞吐（只计时张量核部分）掩盖量化/反量化/布局成本；端到端含 quantize→layout→GEMM→dequantize→写回。论文 Fig.4 显示二者差距即量化开销占比（>30%），且 shape 越小开销占比越高（小 GEMM 是 LRM 主导形态）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现载体：DeepGEMM（https://github.com/deepseek-ai/DeepGEMM，块级 scaling 的 FP8 GEMM，CUDA 手写 kernel + 数值测试）、FBGEMM（https://github.com/pytorch/FBGEMM，INT8/FP8 高性能力矩阵 kernel）、TorchAO（https://github.com/pytorch/ao，PyTorch 原生低精度训练到服务优化库，drop-in 替换层）。使用：库提供不同 recipe 选项，用户/调度器按 shape 与精度需求选择；LoKA Dispatch 即自动做此逐算子选择（见 LoKA Dispatch 条目）。局限：scaling 粒度越细精度越好但量化/缩放开销越大，这正是"库优化解决不了、需系统层编排"的原因。

涉及论文标题：
- LoKA: Low-precision Kernel Applications for Recommendation Models At Scale

## LoKA Dispatch（逐算子低精度 kernel 编排与库选择）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- LoKA Dispatch 是 LoKA 三大组件之一，实现"逐算子 kernel 编排"原则：把每个 GEMM 当作独立优化问题，从多个低精度库（TorchAO、DeepGEMM、FBGEMM）及其 recipe（tensorwise/rowwise/blockwise、快速累加、前向/反向不同 datatype 等）的候选实现中，选"满足精度约束下吞吐最高"的 kernel。依据：没有任何单一库/recipe 在所有 shape 与硬件上都最优（实测单库统一策略最好仅 1.08×，混合策略 1.12×）。
- 选择算法：候选先按 LoKA Probe 的 MERE 分析过滤——期望误差低于保守阈值（典型 MERE<0.2）且 Probe 测得加速比 >1.05× 才入选，再从过滤集选实测吞吐最高者。实现为自定义 autograd function（通用适配器）：模型初始化变换 pass 把目标线性层替换为 LoKA-aware wrapper（语义与标准 PyTorch Linear 一致），前向/反向分别路由、可各用不同最优实现（前向/反向 shape/layout/datatype 常不同）。
- 动态性取舍：推理时 kernel 选择完全静态确定（分布漂移由在线连续训练自然处理）；训练时动态切换仅作为大幅分布漂移（如节假日用户行为突变）的安全阀——频繁切换有重编译税，不值得。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 调度流程（一次前向）：输入 (x, W) 到达 LoKA-aware linear wrapper → 查算子映射表（由 Probe 离线生成的 (shape, 库, recipe) 最优映射）→ 路由到该 GEMM 的指定实现（例：该 (2048,123200)@(123200,1024) GEMM 走 DeepGEMM blockwise，另一 (2048,256)@(256,768) 走 TorchAO rowwise）→ 前向 kernel 执行；反向 input-grad GEMM 查另一映射（如混合 recipe RW GW HP 的 backward 分支）→ 反向 kernel 执行。
- 约束优化形式：对每个 GEMM g：候选集 C_g={(lib,recipe)}；过滤 F_g={c∈C_g | MERE_c<thresh 且 speedup_c>1.05×}；选择 c*=argmax_{c∈F_g} throughput(c)。表 VI 结果：TorchAO TW 1.05×、RW 1.01×、混合 1.08×、DeepGEMM BW 0.85×、FBGEMM RW 0.98×、LoKA Dispatch 1.12×（compute-only，Wukong）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：PyTorch 内自定义 autograd.Function + 模型初始化时线性层替换 pass；依赖三个低精度库的公开 API；与 torch.compile 协作时引入新 kernel 需手动干预集成（论文 limitation）。使用：训练/推理框架透明接入（wrapper 保持标准 Linear 语义），配合 torch.compile 达最佳性能。作用：把跨库跨 recipe 的逐算子 kernel 选择自动化，避免对数百模块手工调优，比任何统一低精度策略更快；局限：引入新低精度 kernel 时需人工集成。

涉及论文标题：
- LoKA: Low-precision Kernel Applications for Recommendation Models At Scale

## GEMM epilogue 融合（归一化 / 激活 / 量化融合进 GEMM 尾处理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- GEMM epilogue 是 kernel 在张量核算完矩阵乘后、把累加结果写回 HBM 前的尾处理阶段：通常含反量化/缩放、加 bias、激活、写回。epilogue 融合（epilogue fusion）指把归一化、激活、量化/反量化等后续算子合并进该阶段，在输出 tile 尚在片上（L1/L2/寄存器）时完成，避免中间张量往返 HBM——这是降低低精度开销的关键手段（Triton 等框架原生支持 epilogue 融合）。
- LoKA 的用法：BlockNorm（块级 RMS）设计目标就是把归一化融合进 GEMM epilogue——GEMM 输出 tile 在片上时立即按 256 元素块算 RMS 并归一化，再融合 Hard Swish 与反量化后写回；相比标准归一化（先写 HBM、读回、算全局统计、再写），省掉两轮全局内存流量。约束：融合要求归一化统计在单 thread block 内可算（Case 1 大 batch 小 N 成立；Case 2 小 batch 大 N 需跨 block 同步，收益被抵消，BlockNorm 用固定块规避）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 融合 kernel 尾处理流程（一次 Wukong 线性层）：张量核算完 C_tile∈R^{BM×BN}（BM×BN 为 tile 尺寸，驻留 SMEM/寄存器）→ 反量化 C=(C_fp32·scale_x·scale_w) → 块归一化：C.view(BM, BN/256, 256)，每块 rms=sqrt(mean(block²)+ε)，C_b/=rms → Hard Swish：C_b·clamp(C_b+3,0,6)/6 → 写 HBM。全过程零全局内存往返。
- 对比未融合路径：GEMM→写 HBM→（下一 kernel）读回→全局统计 LayerNorm→激活→写 HBM→（下一 kernel）读回量化……每多一次往返就多一次 HBM 流量与 kernel 启动开销，小 GEMM 下开销占比更高。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Triton（kernel 内 tl.dot 后直接算归一化/激活）、CUDA 手写 kernel 的 epilogue 段、torch.compile 的算子融合 pass（fuse attention/norm）。LoKA 通过把归一化改成 BlockNorm 使统计本地化，从而让融合可行（标准 LayerNorm 沿特征维全局统计无法在 tile 内完成）。使用：低精度/小 GEMM 场景最大化片上复用；论文 Fig.13 消融显示 BlockNorm 因使融合可行而带来显著延迟降低。参考：epilogue fusion 通用概念见 Triton/FlashAttention 相关工作。

涉及论文标题：
- LoKA: Low-precision Kernel Applications for Recommendation Models At Scale

## SM 波量化效应（SM Wave Quantization）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SM wave quantization（SM 波量化效应，NVIDIA 官方文档 [1] 的 wave quantization 概念）指 GPU 上 SM（Streaming Multiprocessor）数量有限导致的调度粒度量化：一个 GEMM 的 thread block（CTA）数不是 SM 数整数倍时，最后一"波"（wave）的 SM 利用率不足，出现部分 SM 空闲的尾波效应。tile 越小、CTA 数越少，波量化损失占比越高——小 GEMM 场景尤其明显。
- 在 LoKA 中的角色：BlockNorm 设计时考虑"小 batch 大输出维"情形——为让单 thread block 装下整行做归一化统计，可把 batch 维 tile 缩小，但这会加剧 SM wave quantization（CTA 数下降、尾波空闲）并降低 W 矩阵的 L2 缓存命中率（W tile 复用减少），端到端加速比趋近于零——这正是论文选择"固定块 BlockNorm 放松数学等价性"而非"调 tile 适配归一化"的原因：后者需对每个新 shape 手工调 tile，牺牲通用性。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 量化损失计算：设 GPU 有 S 个 SM，GEMM 按 tile 得 C=ceil(N_blocks/S) 波；利用率 = N_blocks/(C·S)。例：S=132（H100），若 tile 设计使 CTA 数=130 → 2 波但第二波只占 2/132，利用率≈50%；若 CTA=264 → 2 满波，利用率 100%。LoKA 的 Case 2 情景：为容纳整行归一化缩小 batch tile → CTA 数从 264 降到 ~130 → 波量化损失 50%，且每 CTA 的 W 复用变小、L2 命中率下降。
- 与归一化融合的权衡：论文 Fig.7(c) 展示该路径（缩 batch tile）端到端加速比趋零；Fig.7(d) 最终选择 BlockNorm（固定 256 块，不依赖 tile 适配），在任意 shape 下鲁棒。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：kernel 设计时选择 tile 尺寸使 CTA 数尽量为 SM 数整数倍（或满足最小化波量化）；NVIDIA 性能指南明确该效应（docs.nvidia.com 的 dl-performance-matrix-multiplication wave-quant 一节）；实际框架（cuBLAS/Triton heuristics）自动在 tile 选择中权衡。在 LoKA 场景中它是设计约束而非优化目标：促使归一化设计向"不依赖全局统计/不依赖 tile 适配"的 BlockNorm 收敛。关联概念：tile 尺寸、CTA/SM 调度、L2 缓存命中率。

涉及论文标题：
- LoKA: Low-precision Kernel Applications for Recommendation Models At Scale

## 预加密调度（Pre-encryption Scheduling / 机会性预加密）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
预加密调度是 LÆGIS 提出的运行时调度机制：在 GPU CC 的 UVM 故障批处理（fault batch handling）期间，把 AES-GCM 页加密这一运行时计算从关键路径卸载到 UVM driver 线程的空闲时段执行。前提是加密与访问顺序解耦（IV Bank 显式 IV，见硬件架构条目），使预加密可乱序执行。调度器识别两类空闲：**true idle**（两个 fault batch 之间的 driver 线程睡眠期，平均占 driver 执行时间 87%）与 **false idle**（batch 内 fault preparation 阶段——取 fault、预处理的时段，AES 指令未执行）。候选页选择策略决定变体：F-LÆGIS（false idle + fault buffer 下一批条目）、IR-LÆGIS（true idle + 随机 CPU 驻留页）、IN-LÆGIS（true idle + fault buffer 条目）、IFN-LÆGIS（false+true idle 全用：先 fault buffer 候选，剩余空闲顺序预加密 CPU 驻留页）。预加密完成的页到达时直接标记 ready、跳过关键路径加密，且批量提交提升 PCIe 突发利用率。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
预加密调度器运行在 CPU driver 侧（伴随 nvidia-uvm fault 服务链，加密走 Linux Kernel Crypto API，实测 1.3 GB/s），伪代码：
```
// 每处理完一个 fault batch i 后，调度器进入 idle 窗口
on_batch_dispatched(batch_i):
  # false idle: fault preparation 期间（CE 尚在处理 batch_i 的数据）
  for pg in next_fault_buffer_entries:          # F/IFN：取 fault buffer 下一批候选
      if pg.preencrypted == false:
          iv = ivbank_lookup(pg.id)             # CPU IV Bank 按 ID 重建 IV
          preencrypt(pg, AES_GCM(K_h2d, iv))    # 提前加密，标记 ready
  # true idle: 批次之间（driver 睡眠窗口，平均占 87%）
  while driver_idle:
      pg = IR: random_cpu_page() | IN: next_fault_buffer() | IFN: next_seq_cpu_page()
      preencrypt(pg, AES_GCM(K_h2d, ivbank_lookup(pg.id)))  # 结果直接提交
on_fault_served(pg):
  if pg.preencrypted: mark_ready(pg)            # 免关键路径加密
  else: encrypt_on_critical_path(pg)            # 未预加密页照旧
```
（Annotations：候选集 S_a 为全部 UVM 管理页；预加密页移出 S_a；DMA 时附带 MAC||ID；IV 从 IV Bank 随机访问故预加密顺序无需匹配实际访问顺序，这是与 PipeLLM 预测式加密的关键差异——LÆGIS 密文总是可提交、无误预测 NOP/丢弃。）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：调度器在 GPGPU-Sim+UVMSmart 中建模——显式建模 fault preparation 时间、driver 线程 idle 窗口（按真实硬件 profile 注入），预加密线程利用 idle 窗口执行 AES；评估四种策略 × 默认（Pt=51%）/aggressive（Pt=1%）预取。结果：F-LÆGIS 1.51×、IR-LÆGIS 1.38×、IN-LÆGIS 2.17×、IFN-LÆGIS 2.22×（最大 3.13×）；aggressive 预取下 pIFN-LÆGIS 2.74×（最大 5.05×）、driver active 占比 88.3%；与硬件加速对比（MT 多线程 -35%、X-Baseline 1.19×）证明"利用空闲窗口的机会性预加密"是核心杠杆，无需更快加密硬件即可逼近 Ideal（差距 5.8%）。使用场景：任何 fault-driven 且存在 driver 空闲的 UVM 机密迁移路径；与 UVM 预取/预测研究（TBNp、预测器）正交兼容（IR-LÆGIS 随机选页仍得 1.38× 证明不依赖预测）。

涉及论文标题：
- LÆGIS: Pinpointing and Addressing Performance Overheads of GPU-based Confidential Computing

## Skinny GEMM（瘦矩阵乘）与 PIM buffer 级复制实现

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Skinny GEMM 指一维（token 或 batch）维度远小于另一维的"细长"矩阵乘法，算术强度低、权重复用有限。RAG 推理中它是结构性的：文档 KV 预计算后 prefill 只编码短 query（平均 ~16 token），QKV 投影/FFN 退化为 16×d_model 的 skinny GEMM；个性化/隐私敏感部署 batch 很小，decode 的 FFN 也退化为 skinny GEMM 或 GEMV。roofline 分析（MERIDIAN 图 3）显示这类算子在 H100 上算术强度远低于计算饱和点，attention 与 FFN 全程 memory-bound——这是集中式 RAG 的"计算低效"瓶颈（Bottleneck 2）的根源。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# skinny GEMM（batch=16, d_model=3584）：M=16 ≪ K=N=3584
Y[16, 3584] = X[16, 3584] @ W[3584, 3584]   # 输出行数少 → 每次只复用同一 W 16 次
# 朴素 PIM 顺序执行（单 GEMV 单元）：
for i in 1..16: Y[i,:] = X[i,:] @ W          # 每次重新读 W（近存带宽/能量浪费）
# MERIDIAN buffer 级复制（只复制 buffer 不复制算术单元）：
# 4 个双缓冲 4KB buffer 存多份输入/中间结果，共享 16 乘法器+16 加法器
# 同一 W 从 DRAM 读一次、喂给多个 buffer 里的输入向量 → 权重复用 4 倍
```
对比：PAPI 用"完整复制 GEMV 数据通路"实现真并行，但 HBM 功率预算下可行、LPDDR 下不可行（通道窄、功率低）；MERIDIAN 只复制约占 GEMV 单元面积 14% 的 buffer 结构而共享算术单元，同时 DRAM 访问占能耗 >96%、更大 buffer 提升 row locality、减少 row-switch 开销——在带宽/功率受限的 LPDDR5X 上实现权重复用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：GPU 上 skinny GEMM 靠 batch 聚合/continuous batching 提高复用（或 FlashInfer 等 kernel 库的 ragged 处理）；PIM 上 MERIDIAN 用 buffer 复制 + All-Bank-Mode（命令广播到所有 bank 同地址）让各 bank 的 16-lane PU（16 FP16 乘法/加法器）并行执行多路 GEMV。使用场景：KV-precomputed RAG 的 prefill（Qwen-TB/Tulu3-Block-FT 的 QKV/FFN 投影）、小 batch decode 的 FFN；效果上 MERIDIAN 相对 CENT（GEMM 拆多次 GEMV）3.98×、相对 PAPI（GEMV 单元复制）3.32× 吞吐优势。极端长响应场景（response 远长于文档）MERIDIAN 可退回集中式执行并保留内存侧高效处理。

涉及论文标题：
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition

## GEMV（General Matrix-Vector Multiplication，通用矩阵向量乘）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GEMV 是矩阵×向量的线性代数原语：Y = A·x（A∈R^{M×K}，x∈R^K，Y∈R^M）。自回归 decode 每步只处理 1 个 token，attention（QK^T、SV）与 FFN 均为 GEMV 形态——算术强度极低、输出计算量 O(MK) 而每个输出只需读整行权重，典型 memory-bound，是"decode 内存墙"的直接来源。RAG 中 decode 更糟：query KV 每步新增但文档 KV（数千到上万 token）完全复用，GEMV 的 K 维被文档 KV 拉长、且 batch 小无法用权重批复用补救。MERIDIAN 把文档侧 GEMV 移到 PIM 就地执行（KV 静止），decode 每步只传 query 向量并回收紧凑统计量。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 一次 decode 步的文档注意力 GEMV（单 head，d_model=3584，文档 14630 token）：
s_d = q[1,3584] @ K_d[14630,3584]^T      # GEMV：1×3584 @ 3584×14630，输出 14630
# MERIDIAN PU 执行（bank 内）：
#   K_d 按 head shard 静止在各 bank；q 经 CXL 广播到设备
#   每 bank 的 PU：16-lane FP16 乘加，256-bit/周期消费数据，All-Bank-Mode 并行
#   结果与 o_d/m_d/l_d 紧凑摘要经 NMU（channel 内归约）与 BOOMv2 核（跨设备）聚合
# 对比集中式：K_d（~96MB FP16）从 host DRAM 经 PCIe 搬上 GPU 再算 GEMV
```
decode 的 QK^T/SV 都随 query 变化、无 batch 复用，因此即便 H100 高算力也无法加速——roofline 上始终 memory-bound（MERIDIAN 图 3）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：GPU 上为高带宽受限 kernel（FlashInfer/vLLM 的 decode attention、cuBLAS 的 GEMV 路径；低比特变体如 BitBLAS INT2 GEMV kernel 用 Tensor Core）；PIM 上为 bank 级 MAC 阵列（HBM-PIM/AiM 的标准 GEMV 形态），MERIDIAN 进一步支持内存侧非线性与专用 softmax。使用场景：任何自回归 decode、RAG 的文档注意力（MERIDIAN 的 DocumentAttention 分支）、小 batch FFN。调度要点：GEMV 适合 PIM（数据就地、免搬运）；若批变大 FFN 可转为 GEMM（权重复用）、attention 仍 GEMV（KV query 相关）。

PuD 视角（ISCA'26，PuDGhost 论文）：GEMV 是 PuD 的主要目标负载（LLM 推理的 memory-bound 层）。PuDGhost 论文在真实 DDR4 上以 MAJ3 位串行实现 GEMV：8-bit 精度、4096×N（N∈{4,8,16,32,64}）随机矩阵，用 32768 列（4096×8 bits）通过筛选的列执行；MAJ3 用 8 行 SiMRA（3 操作数×2 冗余 + 常量 0/1 行）。PuDGhost 使长 MAJ3 链累积错误——Base-worst（screening 全 1、执行全 0 相邻行）NMSE 达 2.2×10⁻²（N=16）与 5.8×10⁻²（N=64）；配合 CS-1/CS-2 + 隔离行布局缓解后 NMSE 全维度 <10⁻³（相对 Base-worst 413× 降 @N=32、114× 降 @N=64，BER 1.3×10³× 降）。
  - SHyLA 补充：decode 阶段以 GEMV 为主（memory-bound，逐 token 用 KVCache + Weight 计算），是混合内存系统的带宽瓶颈。SHyLA 的 ATTN 层在 decode 用"GEMV 配对"：每个 tile 处理一对 fused ATTN 的 GEMV（Q·K^T 与 S·V 连续），中间 QK^T 结果留在片上不写 DRAM；KVCache 单 memory plane 放置以支持并行 GEMV（溢出跨两 plane，因 decode 占端到端运行时间主导、prefill 影响小）。tile 数 = 微批 b × 每 die attention head 数。

XtraMAC 补充视角（ISCA'26，FPGA 混合精度 GEMV kernel）：GEMV 是 FPGA 带宽受限负载——kernel 吞吐上限 ≈ HBM 带宽/权重字节数。XtraMAC 论文实现 tile 并行混合精度 GEMV kernel：M 个 tile 各映射到一个连独立 HBM channel 的 PE，权重存 HBM、激活片上缓冲；HBM 每 channel 512-bit 接口字按 per-lane 拆成权重段分发到 PE 内级联的 XtraMAC 链，per-tile datatype 控制信号与操作数同步传播，逐 lane 部分和经级联 MAC 链累加写回。每 channel 级联数 N_MAC = BitWidth_channel/(BitWidth_weight×P)：INT4 权重 + P=2 lanes 时单 channel 512/(4×2)=64 个 MAC 输入/cycle，32 HBM channel 理论 2048 个、实际 1920 个（30 活跃 channel，留 1 读激活 + 1 写回保证布线收敛）。结果：≤1024 实例维持 300 MHz、1920 实例 250–270 MHz（HBM 接口路由拥塞）；512-XtraMAC 占 LUT 98.5%/FF 95.6%/DSP 100%；U55c（460 GB/s）相对 CUTLASS H100（2 TB/s）GEMV 1.2× 低时延（0.0246 vs 0.0294 ms @4096²）、1.9× 能量效率（85 vs 135 W），FPGA 维持 ~74% 有效 HBM 利用率、接近带宽 roofline——靠 2× lane 打包（每 DSP 每 cycle 2 个 INT4 权重）与无格式转换开销抵消带宽差距。
涉及论文标题：
- PuDGhost: Experimental Analysis of Computation Result Corruption in Processing-using-DRAM Operations on Real DRAM Chips and Implications for Future Systems
- XtraMAC An Efficient MAC Architecture for Mixed-Precision LLM Inference on FPGA
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity

## Interleaved Cluster Execution（ICE，交错集群执行）与动态负载迁移

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MERIDIAN（ISCA'26）为去中心化 RAG 推理设计的调度机制，解决文档注意力分解引入的集群间时序失衡：DAC（文档注意力集群）处理大量文档 KV、CEC（上下文执行集群）只处理少量 query/生成 token KV，attention 阶段 CEC 先完成闲置；而 FFN 等上下文重阶段 DAC 闲置。ICE 动态在本应空闲的集群上启动后续 batch：tensor 并行下交替把 batch 分给 DAC/CEC 使两集群并发推进；pipeline 并行下允许 DAC/CEC 在同一 stage 内处理不同 micro-batch 实现 intra-stage overlap。残余失衡（DAC 提前完成文档注意力）由**动态负载迁移**补足：初始化时把部分 CEC 参数静态复制到 DAC，DAC 空闲时协助上下文计算——因 DAC/CEC 微架构同构、迁移开销可摊销到全部推理请求。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# ICE（tensor 并行，B 个 batch）：
for batch in 1..B:
    if batch 属文档注意力阶段: 派发到 DAC（doc KV 就地算）
    else:                      派发到 CEC（QKV/FFN/融合）
    # 关键：DAC 算完本 batch 文档注意力后不等待 CEC，而是预取并启动下一 batch 的文档注意力
    #       CEC 同理在空档启动其可独立部分 → 两集群交错推进、减少 idle bubble
# pipeline 并行（2 stage、micro-batch）：
#   stage 内 DAC 处理 micro-batch i 的文档注意力，同时 CEC 处理 micro-batch i-1 的上下文
# 动态负载迁移：DAC 空闲时执行复制来的 CEC 参数（FC/FFN 部分），DAC 满载时 CEC 自己算
```
对比集中式注意力的调度（NeuPIMs/HeterRAG 的直接复用）：它们无 DAC/CEC 时序失衡（注意力集中在一处），直接套用会导致集群利用率低下——ICE 是针对去中心化数据流的调度新机制。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：host 侧统一调度器实现——初始化静态分配（tensor/pipeline/hybrid 并行策略，CEC 的 FC 跨设备分片、DAC 按 head 分配文档 KV 避免广播），运行时按设备负载动态下发推理任务；ICE 与"单 batch 计算通信重叠"（如 MoE 的 SBO 交错调度）同类思想，但作用对象是 PIM 集群而非通信 kernel。效果（组件消融，图 16）：M-pim 2.19× → M-ad（分解）2.12× → M-ad+ire（+ICE）再 +1.27×；扩展性：32 设备 pipeline 并行 4.19× vs tensor 并行 3.68×（pipeline 只传轻量激活、tensor 需同步部分和）。

涉及论文标题：
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition

## 滑动窗口注意力 kernel 的 CDC 分层映射（Sliding-Window Attention as CDC Layers，MLX 视角）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Sliding-Window Attention（SWA）是注意力矩阵只在局部窗口 W 内非零（token 只 attend 前 W 个 token）的线性复杂度注意力变体，用于长序列/流式推理降低 O(N²) 复杂度。MLX 论文把它作为"蝴蝶 kernel 之外的第二类结构化 workload"证明 MLX 不限于 FFT/蝴蝶类同构 kernel：SWA 的 tile 计算虽混合不同原语（矩阵累加、归约、指数、归一化），其数据流仍可表达为少量 CDC 层、每层阶段严格对齐、依赖链相邻——直接映射到 MLX 在同一 2D 阵列上的折叠执行（Fig.12）。
- 本地知识库旁证：已有 Sliding-Window Attention 条目（算法知识笔记）与 Hybrid Windows Attention（多方向滑动窗口注意力，EasyAnimate 语境）覆盖算法/模型视角；本条目补充 MLX 特有的 kernel/硬件映射视角——SWA 作为 CDC 分层的 kernel 调度对象。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SWA tile 的 CDC 分层 kernel 调度（MLX 折叠执行，窗口 W、块 Q）：
```
# CDC 层序列（相邻依赖链，每层只消费前层 CDC 边界输出 + tile 本地状态）：
# 层0: 窗口化 score 累加（QK^T，FMA 主导）
for i in 0..N-1:
    for j in max(0,i-W)..i:
        S[i,j] = sum_k Q[i,k] * K[j,k]        # FMA-dominant
# 层1: 行向 max 归约
m[i] = max_j S[i,j]                           # FMAX
# 层2: 指数与归一化统计（FEXP + sum/broadcast）
P[i,j] = exp(S[i,j] - m[i]) ;  l[i] = sum_j P[i,j]
# 层3: 加权累加与归一化（SV，FDIV/FMA）
O[i,:] = (sum_j P[i,j] * V[j,:]) / l[i]
```
调度要点：不同层压不同 FU 原语（FMA/FMAX/FEXP/FDIV）→ tagged-block 执行利用异构性；折叠使 CDC batch 部分 in-flight，层间通信只经显式 CDC-boundary xfer 操作（可检查、有界）；层粒度延迟窗口覆盖下达到稳态重叠，并发活跃层数有界（避免全矩阵中间驻留）。实验：SWA 上 MLX 平均归一化加速 3.6×/2.3×（vs AGX Orin/RTX-3090，batch 32，W/Q 两参数扫），FMA 利用率 43%-75%（vs GPU 10.8%-31%/8.9%-28%）；剩余缺口主要来自窗口 KV 流量的带宽损失。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现与使用：(1) GPU 通用实现——SWA 通常作为融合 attention kernel（FlashAttention 变体支持窗口 mask，或流式 chunk 组织），O(N·W·D) 计算；(2) MLX 实现——把 SWA tile 编译为 4 个 CDC 层（上述伪代码）的 tagged blocks，折叠到 4×4 网格，FMA/FMAX/FEXP 异构单元在活跃窗口内重叠，全部层间通信经 CDC-boundary xfer；(3) 使用场景——长序列/流式 transformer 的注意力加速，验证 MLX 从"蝴蝶同构 kernel"扩展到"混合原语结构化 kernel"的通用性。局限：窗口 KV 流量仍占带宽（利用率天花板），更细动态模式需 predicated transfer（mask/segment 编码）与额外控制状态（论文 E 讨论的灵活性-效率权衡）。
- 涉及论文标题：MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures

## 四步 FFT（Four-Step FFT）与 Tensor Core 递归分解

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 四步 FFT 是 Cooley–Tukey 的 cache 友好变体（Bailey 4-step）：把 N=n1×n2 的一维 DFT 重排为 n1×n2 的二维数据布局，分四步执行——(1) n2 个 n1 点列向 FFT；(2) 逐元素乘 twiddle factor $W_N^{j_2k_1}$；(3) 矩阵转置；(4) n1 个 n2 点行向 FFT。由索引映射 $j=j_1n_2+j_2$、$k=k_1+k_2n_1$ 可得 $X[k_1+k_2n_1]=\sum_{j_2}\big[\big(\sum_{j_1}x[j_1n_2+j_2]W_{n_1}^{j_1k_1}\big)W_N^{j_2k_1}\big]W_{n_2}^{j_2k_2}$。关键性质：列向与行向子变换都是批处理 DFT，可表达为矩阵-矩阵乘——天然映射到 GPU Tensor Core；且分解递归（每个子 DFT 可再用四步法细分），形成分层 radix 分解。在 MNEMOS 中，"radix"指四步分解的基数（如 radix-8 即基为 8 的分解）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- MNEMOS 的分层策略：N>64 优先 radix-64 一步；8<N≤64 回退 radix-8；<8 点用 CUDA Core + warp shuffle。基例为 8 点 FFT（匹配 FP64 WMMA 8×8×4：n 点 DFT 是 n×n 矩阵-向量积，批量即矩阵乘）。以 N=512（TFHE 中 512 点？实际 Tangent FFT 用 N/2=256 点）说明 256 点四步 FFT：
```
# 256 = 32×8：列向 32 个 8 点 FFT + 转置 + 行向 8 个 32 点 FFT
# 32 点再拆 = 4×8：列向 4 个 8 点 FFT + 转置 + 行向 8 个 4 点 FFT（CUDA Core）
# 每个 8 点 FFT = 8×8 矩阵 × 8×1 向量 → WMMA 8×8×4（FP64，2 次 MMA/复数分解后 4 个实乘×2）
A = reshape(x, (32, 8))                  # 列优先
Y = FFT8_batched(A, axis=0)              # 32 个 8 点列向 FFT（WMMA）
Y *= twiddle[32, 8]                       # 逐元素 twiddle
Y = transpose(Y)                          # 共享内存转置（swizzle 防 bank 冲突）
Z = FFT32_batched(Y, axis=0)              # 行向 32 点 FFT（再递归）
out = reshape(Z, (256,))
```
- Annotations：64 点特化算法（图 8）把两级 8×8 分解融合为一级，利用 WMMA fragment 在 warp 内的布局隐式完成转置，省去一次共享内存往返与同步；Fourier 矩阵不预存共享内存而在寄存器内生成（8 点 DFT 元素来自 {0,±1,±sin(π/4)}），消除共享内存载入延迟、bank 冲突与布局管理。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：cuFFT 等库内部即用类四步分解；MNEMOS 自研 CUDA kernel 以 WMMA（FP64 mma.sync.m8n8k4）实现列/行向批量子变换，转置在共享内存完成并用宽数据类型 swizzle 消除 bank 冲突。使用场景：TFHE 的 Tangent FFT（N/2 点复数 FFT）与 NTT 类变换（HyperDrive 的 radix-64 Inner-NTT 同思路）；凡"小变换批量、数据规整"的变换都能用四步法 + Tensor Core 加速。注意点：转置步骤是共享内存带宽与延迟的关键（bank 冲突 + 中间往返），MNEMOS 用"fragment 布局隐式转置 + swizzle"两招消除。

涉及论文标题：
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization

## WMMA（Warp Matrix Multiply-Accumulate）与 FP64 Tensor Core 上的 FFT 映射

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- WMMA（Warp-level Matrix Multiply-Accumulate）是 CUDA 中暴露 Tensor Core 的 warp 级 API：整个 warp（32 线程）协作执行 $D_{M\times N}=A_{M\times K}\times B_{K\times N}+C_{M\times N}$ 的小块矩阵乘加，tile 形状由精度决定——半精度 16×16×16、双精度仅 8×8×4（A100 原生 mma.sync.aligned.m8n8k4.f64）。Tensor Core 不原生支持复数算术，需把复数矩阵乘分解为实数运算：$\mathbf{AB}=(\mathbf{A}_r\mathbf{B}_r-\mathbf{A}_i\mathbf{B}_i)+i(\mathbf{A}_r\mathbf{B}_i+\mathbf{A}_i\mathbf{B}_r)$，四个实数乘各映射到 2 次 WMMA（数学恒等、无额外开销）。这是"把 FFT/NTT 变成矩阵乘"类工作（TensorFHE/WarpDrive/Neo 用 INT8 TCU 做 NTT、tcFFT 做 FP16 FFT）在 TFHE 场景的关键差异点：TFHE FFT 需要 FP64 精度（≥30 小数位），故 MNEMOS 直接使用 A100/H100 的 FP64 Tensor Core。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- MNEMOS 中一次 8 点 FFT 的 WMMA 计算（图 8）：8 点 DFT 矩阵 F 与 8×batch 数据矩阵 X 相乘（批量向量），复数场景下分解为 4 个实数矩阵乘：
```
# A = F (8×8 复数 DFT 矩阵，元素 ∈ {0, ±1, ±sin(pi/4)}，运行时生成)
# B = X (8×batch 复数数据矩阵，按实/虚部分开)
for each 实数矩阵乘 (如 Ar@Br):
    mma.sync.aligned.m8n8k4.f64(D_frag, A_frag, B_frag, C_frag)  # 8×8×4 FP64 WMMA
# 4 个实乘 × 每乘 2 次 WMMA（K=4 需 2 步覆盖 8 列）→ 共 8 次 m8n8k4
# 结果按 (ArBr−AiBi) 与 (ArBi+AiBr) 合成复数输出
```
- Annotations：WMMA fragment 在 warp 内按 lane 分布（每线程 Fragment A/B 各 1 元素、Fragment C 2 元素、结果 D 均分）——MNEMOS 利用该寄存器布局省去 64 点 FFT 的一次显式转置；FP64 Tensor Core 吞吐：A100 19.5 TFLOPS、H100 67 TFLOPS，数据中心 GPU FP64 为 FP32 的 1/2（消费级仅 1/64），故该映射对数据中心 GPU 有效。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：`nvcuda::wmma` 命名空间 API（load_matrix_sync/store_matrix_sync/mma_sync）或内联 PTX mma.sync；MNEMOS 以 CUDA C++ 编写、针对 A100 并对 H100 兼容（sm_90 起支持更大 m16n8k4 等）。使用要点：(1) 复数乘必须先分解为实数 MMA（无复数指令）；(2) fragment 布局是转置/数据复用优化的杠杆（寄存器内数据再排布免共享内存）；(3) FP64 WMMA 仅数据中心旗舰（A100/H100 及后续）可用，消费级 GPU FP64 吞吐被阉割。对照：CKKS 的 INT8/FP16 TCU NTT 映射（TensorFHE/WarpDrive）因精度不足不能直接用于 TFHE FFT。

涉及论文标题：
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization

## 共享内存 Swizzling（宽数据类型 bank conflict 消除）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 共享内存 Swizzling 是对共享内存地址做位操作重排（典型为 XOR 高位/低位）以消除 bank conflict 的布局技术：NVIDIA 共享内存组织为 32 个 bank（每 bank 4 字节），同一 warp 内多个线程同周期访问同一 bank 的不同地址会触发冲突、访问被串行化（最多 32-way）。MNEMOS 的场景特殊在"宽数据类型"：每个元素是复数 FP64（16 字节），横跨 4 个连续 4 字节 bank——普通每元素单 bank 的 swizzle 模式失效，需设计"任意 8 个连续线程映射到 32 个互异 bank"的模式（图 9 用 4×8 矩阵示例，不同颜色为 8 lane 子组）。这与 Hopper 上 TMA 的 32B/64B/128B swizzle（ThunderKittens/TMA 文档）同族，但为 16B 元素定制。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- MNEMOS FFT kernel 的转置在共享内存完成，朴素行列访问引发严重 bank 冲突。伪代码（16B 复数元素的 XOR swizzle）：
```
# 线程 t 访问逻辑地址 addr(t) = row*W + col（16B 元素，索引为元素单位）
# swizzle：把元素地址的若干位 XOR 到高地址位，使同周期活跃线程分散到 32 bank
swizzled = addr ^ ((addr & mask_lo) << shift)      # 位交换
# 设计目标：任意 8 个连续线程 → 32 个互异 bank
# 校验：每线程 16B 占 4 bank，8 线程 × 4 bank = 32 bank 全覆盖、零重叠
```
- Annotations：`mask_lo`/`shift` 按元素宽度与矩阵形状选取（16B 元素需跨 4 bank 粒度设计）；消除转置阶段的 stall_MIO_throttle（共享内存/L1 争用）——MNEMOS 实测 FFT 的该 stall 延迟降 3.2×（N̄=256）；64 点特化进一步用 WMMA fragment 布局省掉一次显式转置（详见"四步 FFT"条目）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：CUDA kernel 内手工地址计算（如 `__shfl_xor` 无关的位运算），或由框架自动布局——Triton/TileLang 的 MakeSwizzleLayout、ThunderKittens 的 32/64/128B swizzle、TMA 的 swizzle 描述符（cuTensorMapEncodeTiled）均自动处理标准宽度元素。MNEMOS 因 16B 复数 FP64 超出框架默认宽度而手工设计。使用要点：(1) 先确认元素宽度与 bank 宽度的关系（16B=4 bank）；(2) 转置/对角访问是 bank 冲突高发点；(3) swizzle 与 padding 可组合（padding 破坏对齐、swizzle 保持对齐，TMA/HGMMA 兼容性上 swizzle 更优）。

涉及论文标题：
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization

## BSK 分块与跨密文密钥复用（BSK Tiling / Cross-Ciphertext Key Reuse）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- BSK（Bootstrapping Key）分块复用是 MNEMOS 针对 GPU 上 TFHE 外部乘积访存瓶颈的核心 kernel 级设计：一次 MAC 需取 (k+1) 倍于 GLWE 体积的 BSK（形状 (k+1)ℓ×(k+1)），且一批内多个 PBS 密文（同一卷积层共享参数）访问同一份 BSK。朴素做法（整 BSK 缓存进共享内存）不可行——部分参数集 BSK 超过 A100 每 SM 192KB 合并 L1/SPM 上限，且过度分配共享内存会蚕食 L1 缓存容量。由于 BSK 与傅里叶系数之间是逐元素 Hadamard 积（非一般矩阵乘），线程块无需持有整 BSK，只需处理对应的一块 TBSK 对一块 TGLWE；同一 BSK 分块被多个线程块并发复用（跨密文复用），把复用层级从 L2 提升到 SM 级，缓解"BSK 热数据打爆 L2 带宽/延迟"的瓶颈（baseline 中 stall_long_scoreboard >50%）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 分块与合并访问的协同（图 7，Para-B：N=512、k=4、ℓ=2）：分块几何取 8 个连续复数 FP64 元素（16B/个 = 128B），对齐 GPU 128 字节内存事务粒度，保证每块 TBSK/TGLWE 读天然合并（coalesced），且不改变 FFT 输出的数据布局（避免显式重排的额外开销）：
```
# 单个线程块处理：BSK 的一个 tile（128B）× 一批 GLWE 的对应 tile
for tile_idx in 0..t-1:                    # t = 总 tile 数
    tbsk = load_tile(BSK, tile_idx)         # 128B 合并读，被整批 GLWE 复用
    for glwe in batch:                      # 同参数的一批 PBS 密文
        tglwe = load_tile(FFT(glwe), tile_idx)
        acc[tile_idx] += tbsk ⊙ tglwe       # Hadamard 积（逐元素乘累加）
# 多个线程块并行覆盖不同 tile → 一次 kernel 启动处理大量 GLWE MAC
```
- Annotations：tile 大小是带宽与复用度的权衡——32/64/128B 都满足合并访问（现代 NVIDIA 内存事务最优 128B），更小 tile 提高 BSK 复用因子但增加循环/指令开销，更大 tile 提升 ILP；经优化后 BSK 访问带宽不再是主要瓶颈（相对傅里叶系数访问），故 MNEMOS 取 128B（8 元素）最大化 ILP。k 越大（安全级别依赖 kN，Concrete 常用大 k）BSK 足迹占比越高、复用收益越大（消融 +MAC 1.10×~1.77×）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：CUDA kernel 内按 tile 索引加载 BSK 与 GLWE 傅里叶系数并做分块 Hadamard 乘累加；配合 Tensor Core FFT 的连续输出布局（无需重排）。使用场景：TFHE 批量 PBS（加密 CNN 逐层、AES 等 192 独立输入）；batch 越大收益越稳（图 15：baseline 在 batch>1024 因工作集超 40MB L2 而性能骤降，MNEMOS 全程稳定）。效果：GMEM→L2 流量降 15.7%、L2→SM 降 69.4%，stall_long_scoreboard 从 >50% 降至 ~20%。

涉及论文标题：
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization

## 跨迭代 kernel 融合（Cross-Iteration Kernel Fusion）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 跨迭代 kernel 融合是 MNEMOS 针对 FFT/IFFT 的 kernel 级融合技术：盲旋转主循环每迭代含一次 FFT（正变换）与一次 IFFT（逆变换），而 FFT 与 IFFT 使用同一组系数的共轭版本——twiddle factors（旋转因子）与 precomputation factors（预处理因子，如 Tangent FFT 的 ω^j）。标准实现中每迭代都从全局内存重载这两套系数；MNEMOS 构造跨迭代边界的融合 kernel，把迭代 i 的尾部（IFFT）与迭代 i+1 的头部（FFT）作为单个 workload 执行，使两套系数直接从片上（寄存器/共享内存）跨迭代复用，完全消除主循环内对这些系数的冗余全局载入。收益随分解层数 ℓ 线性增长（ℓ 越大，融合窗口覆盖更多迭代）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 融合 kernel 的结构（图 6，盲旋转循环展开 2 次示意）：
```
# 原（未融合）：每迭代 IFFT 输出写回全局，下次 FFT 再读；系数每迭代重载
for i in 0..n-1:
    c_f = FFT_i(decomp(rot))              # 用 twiddle/precomp 系数（全局读）
    acc = IFFT_i(c_f ⊙ BSK)               # 用共轭系数（全局读）
    acc += prev

# MNEMOS（融合）：迭代边界内一次 kernel 同时处理 IFFT(i) 尾部 + FFT(i+1) 头部
fused_kernel(i):
    c_f = FFT_i(decomp(rot_i))            # 系数已在片上（上一轮融合 kernel 复用）
    acc = IFFT_i(c_f ⊙ BSK)               # 尾部：IFFT 输出留在片上
    # ——迭代边界（无 kernel 启动、无全局往返）——
    c_f2 = FFT_{i+1}(decomp(rot_{i+1}))   # 头部：复用同源共轭系数（片上）
    acc2 = IFFT_{i+1}(c_f2 ⊙ BSK)
    ...
```
- Annotations：`⊙` 为 BSK Hadamard 乘；跨迭代融合消除的是"系数集（twiddle/precomputation）"的重复全局载入，而非 BSK（BSK 复用由分块机制解决）；ℓ 增大 → 每迭代 FFT/IFFT 次数增多 → 系数复用收益线性放大（图 16(b)）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：把相邻迭代的 FFT/IFFT 段合并进单个 CUDA kernel（循环体内跨迭代连续执行），系数表预载入共享内存/寄存器并跨迭代保留。与一般 kernel fusion（消除中间张量 HBM 往返）的区别：此处消除的是"每迭代重载的系数表"访存。使用场景：任何逐迭代重复使用同源系数的变换循环（FFT/IFFT、NTT、负循环卷积）；在 TFHE 盲旋转中与 BSK 分块、Tensor Core FFT 组合，共同把 PBS 从 memory-bound 转为 compute-bound（Para-D 最高 3.01× A100）。

涉及论文标题：
- MNEMOS A GPU-based TFHE Acceleration Framework with Memory Access Optimization

## WQE（Work Queue Entry）与流控字段（wqe_sync/fence/rx_sync/sync）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
WQE（Work Queue Entry，工作队列条目）是 MTIA 300（ISCA'26）HCCL 集体通信在设备端的执行单元：collective 被翻译为 subgraph，subgraph 表示为 WQE 数组，每个 WQE 描述一类操作并直接映射到 RDMA work request（SEND/RECV/WRITE）或本地动作。WQE 类型：**SEND/RECV/WRITE**——与 RDMA WR 同语义（queue pair ID、本地/目标地址、长度、lkey/rkey 等）；**SET**——向本地内存（HBM 或 cache）写一个值；**WAIT**——阻塞直到某内存位置的比较器满足（如 wait 地址 0xabcdef > 10，可视为内存级条件变量）；**REDUCE**——执行和操作 S=A+B（S 可与 A 或 B 重叠，或可选做内存拷贝、充当 DMA 引擎）。流控字段定义 WQE 间顺序，支撑 ring/recursive doubling/ordered tree 等通信模式：**wqe_sync**（本 WQE 等到指定前序 WQE 完成才发出）、**fence**（本 WQE 完成后才发任何后续 WQE）、**rx_sync**（等所有 outstanding receive WQE 完成）、**sync**（等所有前序 WQE 完成）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
一次 4 节点 AllReduce ring 的 WQE 映射（ReduceScatter + AllGather 两阶段，见 AllReduce Ring 条目）：
```python
# ReduceScatter 阶段（每节点处理 1/N 分片，N=4）
wqes_rs = []
for step in range(N-1):                       # 3 步
    wqes_rs.append(SEND(buf=chunk[k], to=next))          # 并行可发
    wqes_rs.append(RECV(buf=recv_chunk, from=prev))      # 无依赖可并行
    wqes_rs.append(REDUCE(S=A+B, dst=local_chunk))       # 依赖前 RECV（wqe_sync）
    # 归约完成再解阻塞下一轮 RECV/SEND（wqe_sync 回指）
# AllGather 阶段（每步依赖前一步数据搬移）
for step in range(N-1):
    wqes_ag.append(SEND(gathered_chunk, to=next))        # 依赖上一 SEND 完成
    wqes_ag.append(RECV(buf=gathered_chunk, from=prev))  # sync 保证顺序
```
执行：WQE 顺序发出但仅按流控字段阻塞；subgraph 间逻辑并行（硬件可用性排队）；16 ME 并发多 subgraph。作用：把 ring/树等算法编码成数据依赖图，让 NMC/ME 在无主机参与下自动流水。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：HCCL 先验（a priori）按 outstanding work/拓扑/通信类型选算法与通道 → 生成 work packets/subgraphs/WQEs → 经 MTIA streaming interface 提交、ME 的 CPU-M 执行（共享 CQ 收 WQE）；WQE 的内部字段（流控字段 + RDMA 字段）在 ME 硬件执行。使用场景：AllReduce/ReduceScatter/AllGather/AllToAll 的 ring/recursive doubling/ordered tree 模式；SET/WAIT 还用于 subgraph 间依赖（内存比较器同步）。与 GPU 对照：NCCL 的通信由主机驱动 kernel 与 ring buffer，WQE 模型把整个 collective 变成设备端数据依赖图。信息缺口：论文未给出 WQE 的内存布局（位宽/字段编码）。

涉及论文标题：
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
- Optimizing 3D Gaussian Splatting with Axis-Shared Rasterization and Order-independent Transmittance

3DGS 补充视角（ISCA'26，GPU radix sort 作为 3DGS 渲染排序 baseline）：gsplat[51] 库的 3DGS 渲染管线用 GPU radix sort（Merrill & Grimshaw PACT 2010 [31]）做 tile 内 Gaussian 深度排序——把 (tile_id, depth) 打包为 64-bit 键后按键排序，得到按 tile 分组、组内深度升序的 Gaussian 列表再 α-blending。本论文把深度排序整体替换为 MLP-OIT（cuBLAS GEMM 直接输出 F(d_i)，跳过排序）：GPU 上因 MLP 算术强度低（1 深度参数仅 6 MAC vs 光栅化每 GS 256×6 MAC，约 30 倍差）而 memory-bound，几何均值延迟为 radix sort baseline 的 1.59×（更慢），论证 GPU 上排序仍更优；专用加速器上 MLP-OIT 相对 32 并行 bitonic 排序网络 21.1~32.4× 加速。与 MTIA 300 的硬件 radix sort（SFU 内桶化+直方图，用于 embedding 反向索引重排）用途不同：此处是通用 GPU 库排序 kernel 作为渲染管线 baseline。

## AllReduce Ring 算法（ReduceScatter + AllGather 两阶段）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AllReduce Ring 是带宽最优（bandwidth-optimal）的 AllReduce 实现（Patarasuk & Yuan, JPDC 2009）：N 个节点组成逻辑环，每节点把数据分成 N 个分片，分两阶段——**ReduceScatter**（N-1 步，每步节点把本地分片发给下一个节点、从上一个节点收分片并归约，最终每节点持有全局和的 1/N 分片）与 **AllGather**（N-1 步，每步转发已归约分片，最终每节点持有完整全局和）。总通信量 2(N-1)/N × 数据量，是渐近最优；代价是依赖链长（N-1 步串行），小消息/大集群下延迟高。MTIA 300（ISCA'26）把 ring 算法编码为 WQE 数组在 Message Engine 上执行（见 WQE 条目）：HCCL 按消息大小/拓扑在 ring/recursive doubling/ordered tree 间选择。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
4 节点 AllReduce ring 的具体过程（每节点数据 B 分 4 片）：
```python
# 阶段1: ReduceScatter（3 步，每步: RECV(prev) → REDUCE(A+recv) → SEND(next)）
#   步1: 节点0 把 c0 发节点1, 节点1 归约 c0 到本地...
#   步2: 继续转递已归约分片
#   步3: 完成后每节点持有一个全局归约分片 g_i
# 阶段2: AllGather（3 步, 每步: SEND(已归约分片) → RECV(上一节点分片) 拼接）
#   步1: 节点0 把 g0 发节点1, 节点1 拼接 g0
#   步2-3: 继续, 最终每节点持有 [g0,g1,g2,g3] = 完整 AllReduce 结果
```
MTIA 300 中阶段间/步间依赖用 WQE 流控字段表达（图 10：自下而上第一对 RECV/SEND 无依赖并行发布，ADD 依赖前 RECV、再解阻塞下一 RECV/SEND；AllGather 每步依赖前一步）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：NCCL/HCCL 库自动选 ring；MTIA 300 上 ring 由 HCCL 翻译成 WQE（REDUCE 在 NMC 执行归约、SEND/RECV 走 RDMA NIC）并在 16 ME 上并行 subgraph。使用场景：DLRM 训练梯度 AllReduce（40 卡、1.6 GB 入站，MTIA 300 通信整体超 H100 3.9×）；大消息/多卡时 MTIA 300 靠 16 节点 scale-up 域与 2.2× 带宽占优，小消息（依赖链延迟主导）NCCL 更优。信息缺口：论文未披露每消息的 ring vs tree 选择阈值。

RoCC 补充视角（ISCA'26，ring AllReduce 的 ROP 硬件执行）：RoCC 论文采用 4-GPU/8-GPU 的 NCCL 式 ring 算法，把 ring AllReduce 分解为 7 阶段（4 GPU）primitive 序列：send → recvReduceSend×2 → recvReduceCopySend → recvCopySend×3 → recv，每 primitive 再译成 ROP μOp（如 recvReduceCopySend = ReadDoorbell→DepBarrier→ReadDoorbell→Add→Write→RingDoorbell），由 ROP 的 collective/primitive 双译码器查表执行、doorbell 门铃跨 GPU 接力（8 GPU 时最多 15 阶段）。与软件 ring 的区别：ReduceScatter/AllGather 的归约（Add）与转发（RingDoorbell）由近内存 ROP 的 4 路 ALU 完成，SM 全程只算 GEMM；结果平均 51% 加速 vs SM 顺序 baseline、23% vs oracle 软件重叠（20% SM 专做 CC）。
涉及论文标题：
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
- PipeComm Maximizing Link Utilization through Pipeline-Aware Collective Communication Synthesis
- RoCC Harnessing Raster Operations Pipeline for Efficient Tensor Collective Communication

PipeComm 补充视角（ISCA'26，ring 作为 baseline 的对照）：PipeComm 把 Ring AllReduce 视为"通用但不拓扑感知"的 baseline——NCCL 用高带宽 ring 算法（低延迟则用 tree），但在物理拓扑与逻辑环不对齐时产生显著低效。仿真对照（8×8 2D Torus，α=150ns、1/β=16GB/s）：同质 Torus 上 ring 近似最优（常数轮通信即可饱和对分带宽），Pipe-Sol 需放宽 II 约束回到最优非流水最短路径才勉强追平 Themis；但在 2D Mesh/异构拓扑上 Pipe-Sol 相对 ring 系 baseline 大幅领先（vs MultiTree 2.23×、vs BlueConnect 1.98×）。这量化了 ring 的优势边界：高对称拓扑（环/Torus）上 ring 已最优、流水线化收益递减（细粒度 stage 的延迟开销反而抵消收益），非对称/异构拓扑上 ring 依赖链与链路错位使其带宽利用率不足 65%——正是拓扑感知合成（PipeComm）的用武之地。

## Table-Batched Embedding（TBE，表批式嵌入）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Table-Batched Embedding（TBE，表批式嵌入）是 DLRM 等推荐模型中把"批量的多用户/多特征 embedding 表查表"合并成单个算子批处理的嵌入计算范式（源于 AutoShard [Zha et al., KDD'22] 与 Meta 生产栈）：一次 kernel 调用处理一个 batch 中所有用户的所有（稀疏）特征对多张 embedding 表的查找、求和/拼接，输出稠密特征供 interaction/MLP 使用。稀疏部分特征不规则、embedding 表超单卡容量（MTIA 300 的 150B 参数 DLRM 99% 在稀疏侧），故 TBE 是 memory-bound/instruction-bound，需要 embedding cache、索引 DMA（gather/scatter）与排序加速。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TBE 前向/反向的 kernel 计算过程（MTIA 300）：
```python
# 前向: 每个 (batch, feature) 的 sparse index → 查 embedding 表 → 聚合
for u in range(B):                          # 批量用户
    for f in features(u):                   # 稀疏特征
        idx = indices[u, f]                 # 连续子集映射到单输出索引（打包）
        emb = embed_table_gather(idx, table[f])   # 索引 DMA gather（专用 cache）
        out[u, f] = sum_or_concat(emb)
# 反向: 稀疏索引需排序使"连续子集 → 单一 embedding 表索引"
sorted_idx = radix_sort(indices)            # 硬件 radix sort（桶化+直方图）
grad = scatter_grad(sorted_idx, table_grad) # 索引 DMA scatter
```
MTIA 300 加速手段：dedicated embedding caches、硬件索引 DMA（scatter/gather）、硬件 radix sort（反向排序）；性能 TBE 前向 2.0×/1.6×、反向 2.1×/1.6×（几何均值 vs H100/H200）。注意 skewed 输入（多数索引指向同一特征）使算子变 cache-bound/instruction-bound 而非带宽-bound，性能不随 HBM 带宽线性扩展。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：TorchRec 提供 TBE 高层实现（MTIA 300 上经 TorchInductor 编译）；GPU 上 AutoShard 做 embedding 表分片。MTIA 300 侧专用硬件（embedding cache/索引 DMA/radix sort）由编译器生成对应 kernel。使用场景：DLRM 训练/推理的稀疏特征处理（Facebook/Instagram 广告、短视频、好友流推荐）。信息缺口：论文未给出 TBE kernel 的具体缓存容量与索引 DMA 的并发度。

涉及论文标题：
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines

## 硬件 Radix Sort 与索引 DMA（embedding 反向加速）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
硬件 Radix Sort 是 MTIA 300 SFU（SIMD 引擎）内的专用排序单元，用于加速 embedding 反向（backward）的稀疏索引重排；索引 DMA（indexed DMA）是 Fabric Interface/Command Processor 支持的按索引列表做 gather/scatter 的 DMA（含字节对齐 tensor 切片 DMA）。动机：DLRM 前向把稀疏 offsets 与 indices 打包成"单个输出索引映射到连续输入子集"，而反向需把稀疏 indices 排序成"连续子集映射到单一 embedding 表索引"——排序是 embedding 反向的关键前置；MTIA 300 用硬件 radix sort 替代软件排序。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
radix sort 的硬件执行（MTIA 300 描述）：从 LS 取元素 → 按位桶化（bucketization）→ 生成直方图（histograms）→ 把桶化元素写回内存。与索引 DMA 组合的 embedding 反向流程：
```python
# 1. 硬件 radix sort: LS 元素 → 桶化 → 直方图 → 写回（SFU）
sorted_idx = radix_sort(indices)            # 连续子集 → 单一 embedding 索引
# 2. 索引 DMA: Command Processor 用 LS 中的索引列表生成读/写序列
grad = indexed_dma_gather(sorted_idx)       # 按索引 gather 梯度
embed_grads = indexed_dma_scatter(grad)     # 按索引 scatter 到表
```
索引 DMA 也用于前向 embedding 查表（scatter/gather），字节对齐切片 DMA 消除 tensor 切片布局变换的软件开销。TBE 前向 2.0×/1.6×、反向 2.1×/1.6×（vs H100/H200）部分归因于此。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：radix sort 为 SFU 内固定功能单元（按位桶化 + 直方图 + 写回）；索引 DMA 在 FI/CP 中由硬件生成读/写序列（索引列表存 LS）。使用场景：DLRM 稀疏侧前向（embedding 查表 gather/scatter）与反向（梯度 scatter + 索引排序）；与 TBE/embedding cache 配套。信息缺口：论文未给出 radix sort 的位宽/基数（radix）与桶数量细节。

涉及论文标题：
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines

## BALD 通信调度算法（Balanced Allocation with Load and Distance awareness）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BALD（Balanced Allocation with Load and Distance awareness）是 BusyBarn 提出的通信调度算法，为 wafer-scale 层次化 2D mesh 上的 point-to-point 与多组 multicast（区域受限 collective 与广播）做链路分配（link allocation），同时优化吞吐与延迟并内建故障容错。算法三步：(1) Path Profiling——对拓扑 T=(N,E) 用 Dijkstra 全源最短路径（Algorithm 1），输出最短距离映射 S 与唯一路径映射 U（记录等长多路径并把 pair 标记非唯一），天然处理非对称节点度与异构边权重；(2) Path Scheduling——对每个通信任务 C=P(s,D)（源 s、目标集 D），按优先级 score = α×branch_cost + β×link_load + γ×neighbor_distance 迭代选择分支与邻居分配链路（Algorithm 2）：branch_cost 为当前分支最早可调度时间、link_load 为链路当前占用（mesh 上 collective 的主要瓶颈）、neighbor_distance 为到最近目的地的距离（预计算）；(3) Heuristic Backtracking——维护 tabu forbidden/candidate 列表，找出最大负载链路 l*、对其上任务按概率 ρ 重分配（成功回退进 tabu candidates、失败进 tabu forbiddens），最多 I 次迭代（Algorithm 3），消除随机选择造成的链路争用。算法基于 BFS/DFS 拓扑无关，只要存活图连通即可容忍任意节点/链路故障。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Path Scheduling 的优先级计算（Algorithm 2 核心）：
```
while P not empty:
    for each task p in P:                       # 每个 multicast 任务
        for each branch in path[p]:             # 当前分支（源/已分配节点）
            for each neighbor in available_neighbors:
                priority = alpha * branch_cost + beta * link_load \
                         + gamma * neighbor_distance   # 选最小优先级
        path[p] += (best_branch, best_candidate)   # 分配链路
        update branch_cost, link_load              # 更新网络状态
        if candidate in destinations: remove from p
```
Heuristic Backtracking（Algorithm 3）伪代码：
```
for iter in 1..I:
    l* = link with maximum load; overloadedTasks = tasks using l*
    u ~ Uniform(0,1)
    if u < rho: pick t from overloadedTasks not in tabuForbiddens
    else:       pick t from tabuCandidates
    re-run Path Scheduling on backtrackedTasks
    if total load of overloaded links decreased:
        tabuCandidates ∪= overloadedTasks      # 成功回退→继续优化
    else:
        tabuForbiddens ∪= overloadedTasks      # 失败→禁止重访
```
执行例子（Fig.7 4×4 mesh 两个并发 multicast：任务1 8→{7,14}、任务2 9→{3,11}）：XY 路由共享 (9,10)(10,11)(11,7) 造成争用热点；BALD 先 profiling 得 8→7 与 8→14 等最短路径与备选等长路径，再按优先级给两任务分配不同链路（如任务2 改走 9→5→6→7→3 之类非冲突路径），β 大时优先均衡链路负载，最后回退重排过载链路任务。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：在事件驱动模拟器中实现三步算法作为通信调度器（替代 XY/XY-YX-FT），输出路径作为路由器 LUT 下发（主机每推理任务加载、执行中更新，类似 TPUv4）。使用与超参：集体通信实验首轮 α=100、β=1、γ=100（偏好最短路径），随后 α=1、β=100、γ=1（均衡链路负载）；ρ 为 tabu 概率、I 为最大迭代。评估结果：All-Gather 峰值有效带宽 533.3 GB/s（与 TACOS 持平，超 MultiTree 1.25×、XY 1.5×、Hierarchical Ring 近 2×）；All-to-All 峰值 213.3 GB/s（XY 的 2.4×，链路故障下 1.84–2.25×）；6×6 mesh 多故障下 All-Gather 1–1.94×、All-to-All 1.56–2.55×（vs XY）。开源：https://github.com/redbird-arch/isca2026-busybarn-artifact.git。

涉及论文标题：
- Mapping and Communication Optimizations with Fault Tolerance for Wafer-Scale LLM Inference

## XY 路由与 XY-YX-FT（mesh 维度顺序路由与故障容错变体）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
XY 路由（dimension-ordered routing 在 2D mesh 上的特例）是 mesh 上通用的最短路径确定性路由：数据包先沿 x 轴走到目标列、再沿 y 轴走到目标行（YX 为反向顺序，XY-YX 表示按维度顺序二选一/交替）。它简单、无死锁、路径最短，但对并发通信不感知：多个任务共享相同链路时产生严重带宽争用（Fig.7a 中任务 1/2 共用 (9,10)(10,11)(11,7) 形成橙色热点）。XY-YX-FT 是论文采用的 baseline 增强版——在 XY-YX 基础上加入回溯（backtracking）规则以覆盖更多故障情形：故障打破 mesh 对称性后，XY 固定顺序路由会失效或产生代价高昂的 detour，回溯规则允许在死路/故障时退回换向，从而容忍更多节点/链路故障，但路径长度增加、延迟劣化。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
XY 路由对单源单目标的路径计算：
```
# 从 (sx,sy) 到 (dx,dy)
if sx < dx: 先向东走 (dx-sx) 步
elif sx > dx: 先向西走 (sx-dx) 步
then: 再沿 y 轴方向走 |sy-dy| 步
```
失败例子（Fig.7b）：XY 下任务 2（9→11）走 9→10→11，与任务 1（8→7）走 8→9→10→11→7 在 (9,10)(10,11) 争用；XY-YX-FT 用回溯让任务改走 detour（如 9→5→6→7→11 类非最短路径），避开冲突但拉长路径、抬高延迟；而 BALD 通过负载感知的链路分配（见"BALD"条目）在最短路径内达成均衡，无需 detour。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：NoC 路由器按确定性维序转发（广泛用于 BookSim2、gem5-Garnet、HD-MoE discrete-event simulator 等）；XY-YX-FT 需路由器支持回溯/换向（论文作为评估 baseline，未给具体规则实现细节）。使用：作为 BusyBarn/BALD 的对照 baseline（合成通信、映射敏感、端到端三组实验均与 XY/XY-YX-FT 对比）；工程上 Tesla Dojo、TSMC SoW 类 wafer-scale GPU 的 D2D 通信亦用 XY routing（见知识库_硬件知识笔记 Wafer-Scale Multi-Chiplet GPU 条目：1.7 TB/s D2D、200 ns/hop、XY routing）。局限：无全局争用感知（MultiTree 亦缺乏全局意识），故障下性能退化显著（BALD 相对其 1–2.55× 加速）。

涉及论文标题：
- Mapping and Communication Optimizations with Fault Tolerance for Wafer-Scale LLM Inference

## 层次化 SA 映射（Hamiltonian Loop / ZigZag / 四元混合损失）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
层次化 SA 映射是 BusyBarn 的映射方法：用两个嵌套的模拟退火（Simulated Annealing）迭代器分别优化 inter-die 与 intra-die 两级映射。Inter-die 映射：把 transformer block 层分配到 die 组。对比 ZigZag 分配（Tangram [19]：按蛇形顺序把块排到 die 组、多组时 folding 折行，在常规 DNN 中低通信距离但自回归 LLM 下最后一个与第一个 die 组之间出现近直径通信路径），BusyBarn 提出 Hamiltonian Loop 策略——把 die 组排成哈密顿环，使相邻 die 组（尤其最后↔第一）距离最小，适配自回归解码的递归数据依赖；SA 把每个 die 组当作环上节点、交换两节点位置以最小化"相邻节点对距离按拓扑约束与每链路带宽加权的总和"，处理拓扑约束 [28]、硬件故障、层-die 数不匹配，迭代到局部最优或预设次数。Intra-die 映射：在给定 die 组内把层算子分配到 core，第二个 SA 采用 Gemini [10] 的移动策略（算子对交换、算子重分配、HBM 数据重分配），但损失函数从"仅通信距离/hop 数"扩为四元混合损失——总通信距离 + 最大链路负载 + 最大 tensor workload + 最大 vector workload，线性时间可算（遍历事件统计，无需 cycle 级仿真），使 tensor core、vector unit、通信链路三类资源同时均衡（消除 Fig.6 的 core 计算不平衡、Softmax 冗余通信、K/K^T 长距离）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Inter-die SA 目标函数与移动（Algorithm 级描述）：
```
# 每个 die 组 = 环上一个节点；层按块分配到组
loop_cost = Σ_{(g_i,g_{i+1}) in loop} distance(g_i,g_{i+1}) * weight
# SA 移动：随机交换两个 die 组在环上的位置
# 接受准则：Metropolis——Δcost<0 必接受；否则以 exp(-Δ/T) 概率接受
# 温度 T 按退火计划递减，直至局部最优或最大迭代
```
Intra-die 四元损失：
```
loss = w1 * Σ_events dist(event)                    # 总通信距离
     + w2 * max_link_load                           # 最大链路负载
     + w3 * max_tensor_workload                     # 最大 tensor core 负载
     + w4 * max_vector_workload                     # 最大 vector unit 负载
```
执行例子（OPT-30B transformer block 映射到 2×2 die 组、4 core/die，D2D 链路故障 Die1-Core2↔Die3-Core0）：inter-die SA 把层排成 Hamiltonian Loop 使 PP 相邻层近距；intra-die SA 在故障拓扑上重排算子与 HBM 数据，热力图（Fig.11）显示 Gemini 在 core/link 上负载失衡、纯通信时间长，BusyBarn 均衡分布显著降低纯通信时间与端到端延迟。评估：die 形状 1×1~3×3 上相对 Gemini 1.25–1.75× 延迟降低；core 形状 5×5~10×10 1.18–1.80×；计算能力 8/16/32 TFLOPs 无故障 1.19–1.31×、1 故障 core 1.24–1.30×；缺陷率 10–20% 1.24–1.53×；收敛：Qwen2.5-7B 2×2 mesh 1000 次迭代达 100 万随机搜索参考值的 12.4% 以内（搜索空间 (4!)^16≈1.21×10^22）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Python + simanneal 库（artifact 依赖），SA 带种子 --seed 123 保证可复现；mapping 阶段占运行时 79%（runtime breakdown，Fig.15）。使用：`bash run_all.sh 16` 复现映射敏感实验（die/core 形状、计算能力、缺陷率）与端到端实验；ZigZag 由 Tangram [19] 提供、Gemini [10] 提供 intra-die baseline（仅距离目标）。相关工程实践：Cerebras CGC 编译器用 SA/力导向做 kernel→PE 放置（EDA 启发式）；Gemini [10] 用 SA 做 layer-pipeline 映射（每 core ≤2 算子）。信息缺口：论文未给出 SA 的温度计划/初始温度/最大迭代次数的具体数值。

涉及论文标题：
- Mapping and Communication Optimizations with Fault Tolerance for Wafer-Scale LLM Inference

## LR（Location Relationship）数据流记号

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LR（Location Relationship）记号是 BusyBarn 的形式化数据流表示，用数据位置与生产者-消费者关系描述 LLM 的算子级并行：对给定数据组，标注所有数据切片在片上的位置、生产者与消费者，覆盖自注意力的张量/序列并行（TP/SP）与相邻层间的流水并行（PP）。它把"数据标注"与"模型架构和互连拓扑"解耦，从而可系统地生成统一与非统一的并行模式、跨模型与跨硬件平台可扩展兼容。作用：把 LLM 推理负载转成可调度的事件序列——用 LR 刻画每层算子的输入/输出数据切片→按目标并行度把数据切成细粒度切片（依赖执行函数的基本单元）→追踪数据依赖得到函数并行计算 DAG→由 DAG 系统生成对应通信事件，作为层次化映射与通信调度的输入。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FFN 层例子（Fig.3）：FFN = LN + 两个线性层（PyTorch 中为 Conv1d）+ 下一 block 的 LN。
```
Act0 = LN(x)                      # 两个设备都需要的输入
Comm 0-0, Comm 0-1                # LN 输出广播到两个设备（TP 分片）
Act1-1 = Conv1d_a(Act0_slice1)    # 设备 1 的部分和
Act1-2 = Conv1d_b(Act0_slice2)    # 设备 2 的部分和
Reduce(Act1-1, Act1-2) -> Act1    # TP reduce 得到完整输出
Comm + reduce -> LN1 的输入切片    # 下一 LN 的数据布局决定归约结果切分
```
运转流程：LR 记号明确每个算子输入/输出数据切片与其位置/生产者/消费者 → 追踪依赖生成函数并行计算 DAG → 依 DAG 生成通信事件 → BusyBarn 的 Event Synthesizer 把记号转成已映射、已调度的事件集合（Notation Building → Hierarchical Mapper → Communication Scheduler 迭代优化，见"BusyBarn Overview"）。对比：它类似数据流 IR/调度描述（如 Alpa 的 tensor-splitting 框架），但显式解耦数据标签与硬件拓扑，兼容多样模型与平台。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：BusyBarn 框架内的 Notation Building 阶段（输入模型参数 JSON 与硬件配置 JSON），产出计算事件与数据依赖，供 SA 映射与 BALD 调度消费。使用：指定混合并行度（SP/CP/TP/PP 组合）后自动生成调度事件集；端到端经事件驱动后端评估。信息缺口：论文未给出 LR 记号的语法/文件格式规范（artifact 代码中体现）。

涉及论文标题：
- Mapping and Communication Optimizations with Fault Tolerance for Wafer-Scale LLM Inference

## 计算-通信重叠（Computation-Communication Overlap）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
计算-通信重叠是分布式训练/推理中的核心优化：让通信（如 All-to-All dispatch/combine、AllReduce）与计算（如专家 GEMM、attention）在同一时间窗口内并行执行，用计算隐藏通信延迟，从而把吞吐推近"通信零暴露"的理想水平。对 MoE 专家并行尤其关键——forward pass 平均 47% 的执行时间花在 All-to-All dispatch/combine 的 device-to-device 数据交换上，瓶颈从计算转向通信。MoE-Hub 论文按粒度把现有重叠方法分为两类并指出其共性缺陷：粗粒度（FasterMoE/Tutel 张量切片流水）在计算图级 pipeline，但因 MoE 动态路由导致通信量/专家负载每轮变化，产生 pipeline bubble；细粒度（Comet/CCFuser 等 kernel 融合、tile/指令级调度）重叠效果好，但需大量硬件专属的软件编排（同步、内存屏障、SM 分组），开销大、可移植性差。二者合计的调度开销+暴露通信占 MoE 层时间 >24%，即使最优实现也达不到理想层性能。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# baseline 细粒度重叠（Comet 风格）：专用 SM 做通信 + 软件就绪轮询
# 阶段1：地址解析 kernel（dispatch 前必须完成）
allreduce(index_expert)                  # 跨 GPU 同步确定 per-token 目标地址
shuffle_and_layout(tokens, addr_table)   # 重排 token、CPU/GPU 计算偏移
# 阶段2：dispatch 与 expert GEMM 重叠
for tile in tokens:
    if SM_group_A: send(tile, dst_gpu, addr)     # 专用通信 SM 子集
    if SM_group_B: while !poll(ready[tile]):;    # 消费者轮询等数据
                   gemm1(tile); gemm2(tile)      # 计算 SM 子集
# 阶段3：combine 反向重叠 + 轮询
```
Annotations：baseline 在"发数据前"必须先做地址解析（同步+shuffle，无法与计算重叠）；细粒度重叠虽让 dispatch 与 GEMM 并行，但消费者靠原子轮询 semaphore 检测就绪，busy-wait 的 warp 占用带宽与算力；且一旦某 token 的地址/顺序未就绪，GEMM tile 无法开始，暴露通信。

```
# MoE-Hub：硬件接管控制平面后的重叠（routing→GEMM1 窗口）
# 生产者（routing kernel）
for token, (e, gpu) in route_result:
    st.rowsp(MallocID[e], RowID=token, RowOffset)   # 立即发出，无需地址
# hub 硬件（透明）
RPM:  合并/整形 → 调度传输（round-robin + 最小 RowID 优先）
AAU:  到达即分配地址（RAT/APT，BaseAddr+LocalRowID*RowSize+RowOffset）
DAM:  写应答→Dependency Table→TB 计数器→Ready→派发 TB
# 消费者（专家 GEMM，CUTLASS）
for tb in ready_tbs:  gemm1(tb); act; gemm2(tb)    # 数据一就绪即执行
```
Annotations：生产者拿到路由结果立即发 st.rowsp（无地址解析、无同步）；RPM 保证整行连续到达；AAU 按到达密集打包；DAM 用写应答自动触发 TB 派发，无轮询。routing→GEMM1 窗口内通信被计算完全隐藏。消融：RPM（MH-PKT）平均 1.13×、DAM（MH-DEP）平均 1.14×、全设计达理想 MoE 层 96.8%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
通用实现：粗粒度——算子级张量切片流水（FasterMoE、Tutel 的图级 overlap）；细粒度——kernel 融合（CCFuser 用 inter-GPU shared memory 把 All-to-All 与 GEMM 融进一个 kernel、FlashDMoE 单 kernel）、tile 级调度+专用 SM（Comet）、GPU-resident 地址同步（Primus-Turbo）；工业库——DeepEP 的 NVLink/RDMA token 分组路由重排。使用上，软件方案要求开发者手写数百至数千行调度/通信代码（Table I）并做硬件专属调优。MoE-Hub 的做法：把重叠能力下沉为硬件原语——软件只发 st.rowsp + 用 rowspMalloc 注册区域，RPM/AAU/DAM 在 hub 透明完成"整形、寻址、就绪信号"，实现无缝透明重叠且调度代码 0 行、通信指令 <10 条。MoE-Hub 论文还指出重叠优化与计算侧 MoE 优化（并行策略、专家不均衡缓解、expert 复制/动态放置、TP+EP 混合并行）正交可叠加。

RoCC 补充视角（ISCA'26，ROP 硬件的细粒度计算-通信重叠）：RoCC 把重叠的执行引擎从 SM 换成 GPU 的 ROP 硬件——warp 完成自己 GEMM tile 后立即发 rocc_allreduce 触发 ROP 做 CC、继续算下一 tile，实现 warp 级（tile 级）细粒度重叠，SM 无需为 CC 让出算力（对比 oracle 软件重叠 20% SM 专做 CC 导致 GEMM 指数级减速：80% SM 时 GEMM 慢 20%）。实测：RoCC 平均重叠率 83.4%（未重叠部分为首段 GEMM 与末段 CC），并发 CC 仅使 GEMM 慢 6.25%，相对 oracle 软件重叠（GEMM 与 CC 完美重叠、SM 分区）平均高 23%，相对顺序 baseline 平均 51%；端到端（Astra-Sim+Chakra）44%，32-256 GPU 13%-21%。核心是门铃同步 + ROP 异步执行，避免软件方案的 cache/NoC 污染与 busy-wait 轮询。
涉及论文标题：
- MoE-Hub Taming Software Complexity for Seamless MoE Overlap with Hardware-Accelerated Communication on Multi-GPU Systems
- RoCC Harnessing Raster Operations Pipeline for Efficient Tensor Collective Communication

## 数据流调度（mapping engine + operator-level + system-level 三层调度，DAGS 依赖感知贪心调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 数据流调度指把算子图按依赖与硬件资源编排成可执行计划的过程，覆盖"算子如何映射到硬件单元、每个算子何时在哪个单元执行、数据如何搬移"。NeRArch-Sim 的模块化数据流调度器把神经渲染加速器的调度问题分解为三层：**mapping**（算子-硬件绑定，mapping engine 按统一分类学匹配算子与硬件模块，多候选取最高吞吐、多实例均衡避免瓶颈，不匹配报 mismatch，输出 mapped IR）、**operator-level scheduling**（各硬件模块内的局部优化：查优化库选 domain-specific 优化策略，输出带 start_cycle/duration 的 operator-scheduled IR）、**system-level scheduling**（跨模块全局编排：Dependency-Aware Greedy Scheduler，DAGS，消费 operator-scheduled IR 生成最终执行计划与 PPA）。本库既有"分层静态-动态协同调度（Hierarchical Scheduling）"条目是静态/动态分层概念，与此处的"三层调度器"是不同上下文。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 算子级时长模型（式 1）：duration(op) = max(n_op/Θ_hw · s_comp, v_off/B_hw · r_bytes)，n_op 处理元素数、v_off 通信量、Θ_hw 硬件吞吐、B_hw 有效带宽（来自 mapped IR）；s_comp/r_bytes∈(0,1] 是优化库带来的计算/访存削减因子（如 tile culling 优化用 active Gaussian ratio 得 s_comp）。DAGS 伪代码（Algorithm 1）：
```
Q ← 所有就绪源算子
while Q ≠ ∅:
    for op in Q:
        d = 后继算子数(op)          # 启发式1：优先解锁整阶段
        c = 关键资源影响(op, G, S)   # 启发式2：异构资源需求(n_op, v_off)
        score(op) = α·d + β·c        # α/β 可配置建模不同加速器
    sel = argmax score(Q)
    st = FindEarliestSlot(sel)       # 尊重依赖/带宽/SRAM bank-端口冲突
    SS[sel] = (st, S[sel].duration)
    更新就绪队列
```
- 例子（GSCore 3DGS）：CCU→(排序 QSU/BSU)→VRU 链；mapping 把 culling 算子绑 CCU、排序绑 QSU/BSU、混合绑 VRU；operator-level 给各算子算 duration（表 VI：CCU 128/128、BSU 4/4、QSU 64/64、VRU 192/192 cycle），system-level DAGS 按依赖与资源约束排全局计划。三种代表性 dataflow 策略都能被同一调度算法捕获：ICARUS weight-stationary、NeuRex pipelined encoding-MLP overlap、GSCore tile 化 sorting-rasterization overlap——通过算子图结构+硬件配置+memory binding 表达，不改调度算法。DAGS 目标建模保真而非最优性：依赖与硬件约束相同则关键路径不变，故端到端延迟/能耗在误差界内。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：调度器 C++/Python（NeRArch-Sim 的 Scheduler/ 目录），CLI 分步 `./nerarch_sim map <dag.pkl> <hw.json> -o mapped_ir.json` → `./nerarch_sim schedule mapped_ir.json --hardware <hw.json> -o scheduled_ir.json` → `./nerarch_sim report scheduled_ir.json --format html`；调度延迟（表 XI）：mapping 2.0~5.3s、op-level 7.1~24.1s（ICARUS/NeuRex 算子多、GSCore/GBU/Uni-Render op-level 高因 inter-operator 调度复杂）、sys-level 1.0~21.1s（随算子数）。优化库按三维分类（优化类型 reuse/skip/low-bit × 作用域 element/region/frame × 决策准则 boundary/threshold），如 element-level Gaussian skipping（GSCore/GS Processor）、per-ray early termination（NeuRex/GSCore）、region-level restricted hashing（NeuRex）、tile-based Gaussian processing 与 bitmap culling（GSCore）、frame-level sparse radiance warping（CICERO）、threshold 触发的 sensitivity 精度降低（SRender）。使用价值：秒级端到端调度支持快速 DSE、决策可解释、接口可换更复杂调度算法（扩展新加速器只需新增算子与其调度器，<300 行 C++）。

涉及论文标题：
- NeRArch-Sim: A Unified Simulator for Benchmarking and DSE of Neural Rendering Accelerators

## 松散调度（Loose Scheduling，晶格手术动态调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 松散调度是 O3LS（ISCA 2026）提出的晶格手术执行调度策略："loose"指调度的灵活性——在表面码 patch board 上，不按固定的预定义模式执行，而是根据电路上下文与布局约束，动态重指派 patch 功能（位置/朝向）、消除冗余 patch 移动与不必要操作，最小化总时间步。它针对的痛点是静态调度（如 SPC/LAPBC）：固定策略对所有电路统一施加"多 patch 测量前必做 patch 旋转对齐 X/Z 算子"，忽略了可免旋转的场合；在紧凑/不规则布局中 patch 旋转（3 时间步）可占 >50% 时间步（O3LS Fig.7），是空间受限布局的主要瓶颈。
- 从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- Algorithm 2（Loose Scheduling Algorithm）伪代码：
  ```
  输入：Pauli 算子序列 S={P_1..P_l}，qubit 数 n，board B
  输出：可执行操作序列 S'
  1 初始化 S'={}
  2 从 S 构建 Pauli DAG G
  3 while G 非空 do
  4    for P_i in G.frontier 且 P_i 可执行 do
  5        对 P_i 用 Dijkstra 求最短 bus patch 列表 L_{P_i}（已占路径视为零成本节点）
  6        在 L_{P_i} 上执行 P_i，S' ← S' + P_i
  7    end for
  8    从 G.frontier 弹出一个 Pauli 算子 P_i
  9    while P_i 不可执行 do
  10       枚举 B 上全部候选 patch 操作 O_B
  11       选 r(o_b, P_i) 最大的 o_b ∈ O_B
  12       在 B 上执行 o_b，S' ← S' + o_b
  13   end while
  14 end while
  15 return S'
  ```
  奖励函数 r(o_b,P_i) 三部分：①施加 o_b 后 board 状态 B_o 中使 P_i 可执行（存在有效路由路径）的数据 patch 数；②破坏数据 patch 连通性的操作奖励为 0（连通性为后续 lattice surgery 必需）；③同奖励时优先低时间开销的操作。每步至少增加一个满足执行要求的 patch，总复杂度 O(n²)；每个可执行算子的 Dijkstra bus 路径复杂度 O(|B|²)。例子（Fig.4）：执行 Z_0Z_1Z_2Z_3Z_4 测量，松散调度只把 q_0 下移并旋转暴露不同边缘（而非整 patch 旋转），compact 布局下 6 个 ancilla patch → squeezed 布局 5 个，时间步显著减少。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为编译器调度 pass，运行在 O3LS 编译流水线末端（布局搜索 → Y-synthesis → 边感知初始映射 → 松散调度），输入是 PDAG frontier + board，输出是时间步序列与每时间片 patch 布局。效果：相对 SPC 时间步降 36.07%（compact）/24.76%（standard）；相对 LAPBC 平均降 35.10%（最大 80.6%）；与 SPARO 调度组合再降时间步 78.24%、路由空间 27.17%、LER 77.1%。可与高并行策略（LAPBC 风格）集成再降 9.31% 时间步。论文未声明开源（arXiv:2604.15099，GitHub 仓库未能定位）。
- 涉及论文标题：
- O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and Loose Scheduling

## 边感知初始映射（Edge-aware Initial Mapping，EA 映射）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 边感知初始映射是 O3LS Module 4：在把逻辑 qubit 映射到数据 patch 时，用 Pauli DAG（PDAG）估计每个 qubit 的旋转需求（预期旋转频率），把旋转需求高的 qubit 优先映射到同时邻接 ancilla patch 的 X 与 Z 边缘（双边缘）的 patch。动机：patch rotation 是 3 时间步的昂贵操作，在 squeezed 布局中频繁发生；若高频切换 X/Z 算子的 qubit 落在只暴露单边缘的 patch，会反复触发旋转。映射到双边缘 patch 后多数旋转可省。复杂度 O(n log n)（PDAG 中旋转需求提取 O(n) + 两次 quicksort）。
- 从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算过程：①构造 PDAG 时对每个 qubit 统计其参与的 Pauli 算子中 X/Z 类型的切换次数，得到旋转需求排序；②统计 board 上每个 patch 暴露的边缘类型（X-only / Z-only / 双边缘）；③两次 quicksort：qubit 按旋转需求降序、patch 按"边缘丰富度"降序，一一对应放置。例子：加法器电路某 qubit 频繁参与 Z_0Z_1 与 X_2X_3 测量 → 映射到 ancilla 旁的双边缘 patch；而只参与 Z 测量的 qubit 可放 Z-only 边缘 patch。适用性：高度紧凑布局中位置变化难以暴露新边缘（收益依赖电路结构）；稀疏布局中每 patch 天然双边缘（收益小）；介于两者之间的 squeezed 布局收益最大。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为编译流水线中的初始映射 pass（在松散调度之前、Y-synthesis 之后运行）。效果：相对 SPARO [28] 的 greedy mapping，时间步减少 15.0%、LER 减少 8.4%；在完整 O3LS 栈（O3LS-2+3+4）中贡献平均 38.62% 时间步、35.17% LER 的改善（ablation Fig.21）。论文未声明开源（arXiv:2604.15099，GitHub 仓库未能定位）。
- 涉及论文标题：
- O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and Loose Scheduling

## Pauli 乘积测量（Pauli Product Measurement，PPM）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PPM 是晶格手术中的核心原子操作：在表面码 patch board 上测量多 qubit Pauli 算子（Pauli 串乘积，如 Z_0Z_1Z_2Z_3Z_4）的期望值/本征值。实现机制：当目标 patch 的相关边缘与 ancilla 路径（路由空间）相邻时，初始化 ancilla patch A、合并 A 与目标 patch 群（沿共享边界做稳定子测量）、测量、再分离，时间成本 1 时间步（round of code cycles）。多 patch π/4 与 π/8 测量用标准 gate teleportation 协议（Litinski [34]）。PPM 是 Pauli-Based Computation（PBC）编译模型的执行原语——Clifford+T 线路转译后以 PPR/PPM 序列执行。
- 从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- PPM 执行流程（在 O3LS 松散调度中，Z_0Z_1Z_2Z_3Z_4 为例）：①调度器确认该 PPM 可执行（各目标 patch 边缘可经路由空间连通）；②Dijkstra 求最短 bus patch 列表 L（已占路径节点视为零成本），得到最小 ancilla 路由补丁集；③在 L 上初始化 ancilla（0 时间步）→ 依次合并目标 patch（1 时间步）→ 测量 → 分离。伪代码层面即 Algorithm 2 的 Step 5-6。PPM 的错误率主要由路由空间与码距决定；在 LER 分层模型中 PPM 是 p_layer 的组成部分之一（P_PPM）。squeezed 布局下 Y 型 PPM 需先经 Y-synthesis 分解为 X/Z 组合（因 X/Z 不能同时访问）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为编译产物中的可执行指令（lattice surgery instruction，含目标 patch 集与 ancilla 路径），物理上由表面码稳定子测量电路执行。评估：STIM 仿真（d=9、p=10⁻³ 电路级去极化噪声，Monte Carlo ≥10⁶ 次）表征每个原子操作错误率，PyMatching 2 解码。O3LS 中 PPM 是调度与布局设计的核心目标——布局评分函数保证"所有数据 patch 可测量"（连通性 C(B)），即任何 PPM 都可执行。论文未声明开源（arXiv:2604.15099，GitHub 仓库未能定位）。
- 涉及论文标题：
- O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and Loose Scheduling

## Patch 操作（Patch Rotation / Deformation / Movement，patch 旋转/变形/移动）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Patch 操作是表面码 patch 在 board 上的基本几何操作，是晶格手术执行的最小动作集（O3LS Fig.3 规则表）：①初始化——单 qubit patch 可初始化为 |+⟩ 或 |0⟩，双 qubit patch 可初始化为 |+⟩⊗|+⟩ 或 |0⟩⊗|0⟩，成本 0；②Patch Deformation——patch 可扩展到覆盖更多 tile（1 时间步）或收缩到更少 tile（0 时间步），扩展+收缩组合实现 patch 移动到相邻 tile（movement，1 时间步）；③Patch Rotation——patch 可经角移动+平移组合旋转（3 时间步，拆成变形/角移动/移动三片）；④Measurement——相关边缘邻接 ancilla 路径时测量 Pauli 算子（1 时间步）。这些规则源自 Litinski [34] 的 tile-based 协议。
- 从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 调度视角：每个 patch 操作是调度器可选动作。在 O3LS 松散调度 Algorithm 2 中，当目标 PPM 不可执行时，从候选操作集 O_B（board 上全部可能的 rotation/deformation/movement）中选奖励最高者（r(o_b,P_i) 最大化可执行 patch 数、保持连通、同奖励取低时间开销）。Rotation 是最昂贵操作（3 时间步），在紧凑布局中可占 >50% 时间步（Fig.7），因此 O3LS 用布局搜索（多边缘 patch 减少旋转）+ Y-synthesis（减少需要旋转的操作数）+ 边感知映射（把高频旋转 qubit 放双边缘 patch）三个模块联合削减旋转。例子：Fig.4 中只移动 q_0 并旋转暴露不同边缘（1+3 时间步）替代整 patch 旋转（3 时间步×多 patch），省掉多轮旋转。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为编译器生成的指令（lattice surgery instruction set，Triage 论文称 LLI），物理上由表面码稳定子测量电路执行；LER 评估中 rotation 错误率分解为三片分别用 STIM 仿真（d=9、p=10⁻³）。O3LS 在固定布局（compact/sparse/standard）与自动生成的 squeezed 布局上都以 patch 操作序列为执行目标，输出时间步数与 LER。论文未声明开源（arXiv:2604.15099，GitHub 仓库未能定位）。
- 涉及论文标题：
- O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and Loose Scheduling

## Ancilla 路由与 Bus Patch（Ancilla Routing / Bus Patch）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Ancilla 路由是晶格手术中为多 patch 测量提供"总线"的空间机制：ancilla patch（路由空间，即 bus）是一组被保留用于临时初始化和测量的空 tile，数据 patch 的 X/Z 边缘须邻接路由空间才能被测量。Bus patch 列表 L_{P_i} 是执行 Pauli 算子 P_i 所需的最短路由补丁集合。路由长度直接决定：①PPM 错误率（路由空间越大错误越多，P_PPM 由路由空间与码距决定）；②idle 记忆错误（ancilla 路径越长，patch 闲置时间越长）；③空间开销（tile 数）。因此最小化路由路径是布局设计（squeezed 布局）与调度（Dijkstra 最小 bus）的共同目标。
- 从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 在 O3LS 松散调度 Algorithm 2 Step 5：对可执行 P_i 用 Dijkstra 依次求所需 patch 间最短路径，之前已识别的路径节点视为零成本（复用），得到最小 bus patch 列表 L_{P_i}——最小化总路由路径长度既增加其他操作的并行机会又降低 LER，每算子复杂度 O(|B|²)。布局设计侧：评分函数 S(B)=C(B)×(N_x(B)+N_z(B)−α_e·N_e(B)) 中，C(B)=路由连通存在性（ancilla 的 X/Z 边缘都须连到路由空间以支持 Y 测量）、N_e=数据 patch 到路由空间的边缘数（α_e∈[0.1,0.3] 密度因子）。例子：Z_0Z_1Z_2Z_3Z_4 测量在 standard 布局需 6 个 ancilla 路由 patch，O3LS squeezed 布局只需 5 个；相对 SPARO 布局平均 ancilla 路由空间降 17.35%，完整 O3LS 栈再降 27.17%。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为编译时布局与调度决策：布局搜索决定每个 patch 的位置与边缘暴露（squeezed 布局缩短路由），调度时 Dijkstra 决定每步 PPM 的 ancilla 路径；物理上 ancilla patch 就是表面上被临时激活做稳定子测量的空 tile。评估指标：ancilla patch 长度/路由空间（每时间片记录）、时间步数、LER。论文未声明开源（arXiv:2604.15099，GitHub 仓库未能定位）。
- 涉及论文标题：
- O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and Loose Scheduling

## 逻辑错误率分层累加模型（Layer-wise Logical Error Rate Model，LER 分层模型）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- LER 分层模型是评估晶格手术执行质量的度量模型（O3LS 采用 SPARO [28] 的方法）：把整个执行按时间片（time slice / layer）划分，每层的逻辑错误率由三类独立错误源复合，再逐层线性累加得到总 LER：
  $$p_{\text{total}} \approx \sum_{t=1}^{T} p_{\text{layer}}^{(t)} \approx \sum_{t=1}^{T} \big(1 - (1 - P_{\text{PPM}}^{(t)})(1 - P_{\text{PR}}^{(t)})(1 - P_{\text{idle}}^{(t)})\big)$$
  假设稀有失败与独立错误事件。P_PPM=Pauli 乘积测量错误率（主要由路由空间与码距决定）、P_PR=patch rotation 错误率（分解为变形/角移动/移动三片分别仿真）、P_idle=idle 记忆错误率。该模型使"时间步 vs ancilla 路径长度"的 trade-off 可量化——大布局时间步少但路径长（idle 错误多），小布局路径短但旋转频繁。
- 从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算过程（评估 pipeline）：①把调度输出的 lattice surgery 指令序列解析为时间片级操作；②用 STIM 对每个原子操作（PPM/PR/measurement）在 d=9、p=10⁻³ 电路级去极化噪声下 Monte Carlo 采样（≥10⁶ 次）得到各错误率，PyMatching 2 解码；③对每个时间片按公式复合 p_layer；④累加所有时间片得总 LER。例子：O3LS 在 7×7 squeezed 布局 vs 10×10 standard 布局跑 adder_28——前者时间步相近但 ancilla 路径短 → P_idle 低 → LER 低（相对大布局降最高 16.9%，相对 SPC 降 43.11%（compact）/44.98%（standard），相对 LAPBC 最高 93.95%）。码距敏感性（d∈[3,5,7,9]）：距离相关的指数级解码抑制对 O3LS 与 baseline 同样适用，故 O3LS 只优化架构因素、收益不随码距衰减。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为编译器的 LER 评估模块（Python 3.10，Intel Core i9-14900K 32 核 + 188GB RAM），输入 = 调度指令序列 + 布局 + 噪声参数，输出 = 总 LER 与每层错误率。该模型也是 O3LS 优化目标——布局搜索的 sweet spot、松散调度、Y-synthesis、EA 映射都以压低 p_layer 为目标。论文未声明开源（arXiv:2604.15099，GitHub 仓库未能定位）。
- 涉及论文标题：
- O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and Loose Scheduling

## Bitonic Sort（双调排序 / 双调排序网络）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bitonic sort 是一种并行排序算法/排序网络（Batcher 1968；优化见 Ionescu & Schauser, IPPS 1997）：把输入序列视为双调序列（bitonic sequence，先单调增后单调减），通过反复做 compare-exchange 归并为有序序列，总复杂度 O(N log²N)。它的核心优势是无数据依赖的比较结构固定（比较器网络），非常适合硬件实现为固定拓扑的 sorting network（比较器级联，流水/并行执行），因此被 3DGS 硬件加速器（GSCore[25] 及其层级排序）用作 tile 内 Gaussian 按深度排序的排序引擎；32 并行 bitonic 网络是硬件消融中的 baseline 排序配置。本论文用其作为"排序硬件"的代表来论证排序的痛点：硬件面积随输入并行度 k 按 k·log²k 增长、复杂度 O(N log²N) 与光栅化 O(N) 异构导致 pipeline 失衡。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
对 8 个深度值 d0..d7 的双调归并（bitonic merge，升序）：
```
# 输入为双调序列（先增后减），目标升序；距离从 N/2 递减到 1
for (k = N/2; k >= 1; k /= 2):          # N=8 → k=4,2,1
    for (j = k; j >= 1; j /= 2):
        for i in [0, N):
            l = i XOR j                  # 比较对索引（XOR 位翻转）
            if (l > i):
                # 按方向决定升降：取 (k&i)==0 时升序、否则降序
                if ((i & k) == 0 and d[i] > d[l]): swap(d[i], d[l])
                if ((i & k) != 0 and d[i] < d[l]): swap(d[i], d[l])
```
每轮 j 下所有比较对可并行执行（网络的一级）；硬件实现为 k·log²k 个比较器节点，k=32 并行时固定 32 输入并行度——tile 内 Gaussian 数（80~10000+）与固定并行度失配：小负载利用不足、大负载需多轮（层级排序）或成为瓶颈。本论文硬件消融：BS 变体用 32 并行 bitonic 网络[20]（按 GSCore 风格）+ 层级排序；与 MLP-OIT（替代排序）对比：21.1~32.4× 加速，且消除 pipeline 失衡。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
硬件实现：比较器（compare-exchange）单元级联成网络，支持完全流水/并行；GPU 实现：CUB/thrust 等库有 device bitonic sort kernel（warp 内共享内存 + 全局归并）。3DGS 场景（本论文 II-B2 章）：每 tile 对相交 Gaussian 按深度升序排序后再 α-blending；tile 负载 80~10000+ 波动两个数量级（MipNeRF-360 profiling），固定并行度模块要么小负载闲置、要么大负载成为光栅化瓶颈（Fig.4 调度图）——这是本论文用 MLP-OIT 完全替换排序的动机。

DMSU/Bitonic-16 补充视角（ParetoES，ISCA'26）：Bitonic 网络在检索加速器中被用作核内局部排序器——ParetoES 用 32 个核内 Bitonic-16 替换单体 Bitonic-512 全局排序器（Distributed Micro-Sorting Unit，DMSU），比较器从 11,520 降到 2,560、流水 45 级降到确定性 10 级，32 排序器完全并行、每 ACPE 各持一个；每核做两阶段：质心分数筛选（簇探测选 Top-nprobe）与簇内 Top-16 排序（索引与分数按 score 联合排序），32 核 Top-16 在 host 聚合为 Top-512 候选超集。32×Top-16 分解的精度代价：K≤200 时 Recall 恒 100%，仅 K>200 尾部略有偏差（Table III）。对比：3DGS 场景用 32 并行 bitonic 排 tile 内 Gaussian（负载 80~10000+ 波动导致失衡），ParetoES 用 16 输入微排序 + 分解规避单体网络资源爆炸（N=512 需 11,520 比较器/45 级，耗尽 FPGA）。

涉及论文标题：
- Optimizing 3D Gaussian Splatting with Axis-Shared Rasterization and Order-independent Transmittance
- ParetoES Hardware-Accelerated Sparse Embedding Similarity via Pareto-Optimal Pruning

## Kernel 分解（Kernel Decomposer）与 Task/CTA/Tile 定义（PIPEWEAVE）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Kernel Decomposer 是 PIPEWEAVE（ISCA'26 GPU 性能预测框架）的第一个模块，把一次 kernel 启动的完整工作量拆成一组基本调度单元 task：`{τ1,...,τt} = F(X, S)`，其中 X 是 kernel 输入参数、S 是硬件规格。task 的精确定义随执行范式变化——传统 GPU 执行模型（如 FlashAttention-2）中 task=CTA（Cooperative Thread Array，即 thread block），一次 launch 生成 CTA grid，硬件调度器把每个 CTA 分给一个 SM；persistent kernel 模型（如 Ping-Pong GEMM、FlashAttention-3）中 CTA 长期驻留 SM 作为 worker，真正的基本调度单元是从全局 work queue 取的更小计算包 tile。每个 task 用维度参数向量 d_i（如 GEMM 的 tile 几何 {tile_M, tile_N, tile_K}）刻画规模；causal attention 下即使是"名义相同"的 task，实际工作量也因掩码而不同。与 Neusight/Habitat 从 profiling 数据反推简化 tiling 不同，PIPEWEAVE 对开源 kernel（FlashInfer、SGLang、vLLM）直接读源码提取并行化与 thread block 映射逻辑得到确定性的 F。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 BF16 GEMM (M×N×K) 为例，若 kernel 用 tile (128,128,64) 且是常规（非 persistent）实现：
```
# 伪代码：Kernel Decomposer 输出 task 集合
grid_m = ceil(M / 128); grid_n = ceil(N / 128)
tasks = []
for i in range(grid_m):
    for j in range(grid_n):
        tasks.append(Task(d = {tile_M: 128, tile_N: 128, tile_K: 64,
                                origin: (i*128, j*128)}))   # 一个 CTA = 一个 task
# 后续 Scheduling Simulator 把 tasks 按 RR/软件 tile 调度分到各 SM
```
对 cuBLAS 这类闭源 kernel，F 无法从源码提取，改用经验推断：用 PyTorch Profiler 在大量 (M,N,K) 组合上 profiling，分析 kernel 名、CTA 数与输入尺寸的相关性，反推出代理（surrogate）映射函数近似其隐式 task 切分；unseen GPU 上闭源 kernel 借用架构最相近 GPU 的切分逻辑。各 kernel 的 Decomposer 实现仅 10–50 行代码。论文验证：分解出的 CTA 数与数据集 ground-truth 完全一致。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现上开源 artifact 位于 https://github.com/zksainx/pipeweave，analytical_model/ 目录为每类算子（GEMM/FA2/FA3/RMSNorm/SiLU×Mul/MoE）提供计算器。使用流程：给定 kernel 类型与输入维度，Decomposer 产出 task 集合→Scheduling Simulator 按调度范式（硬件 RR 或软件 tile scheduler）把 task 映射到 SM 得到 task 分布→Feature Analyzer 按每 pipeline 汇总 demand/理论周期→MLP 预测执行效率。该分解同时保证 kernel 泛化性：任何 kernel 都被转成统一 task 分布，与来源无关，因此新 kernel 只需写一个 Decomposer。

涉及论文标题：
- PIPEWEAVE: Synergizing Analytical and Learning Models for Unified GPU Performance Prediction

## Persistent Kernel（持久化 Kernel）与软件 Tile 调度器（PIPEWEAVE 视角）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Persistent Kernel 是让少数长生命周期 CTA 在整个 kernel 执行期间驻留 SM、反复从全局 work queue 取小粒度工作单元的 GPU kernel 编程范式（如 CUTLASS Ping-Pong GEMM、FlashAttention-3、Stream-K）。区别于传统"一 CTA 一任务、硬件调度器动态派发"的模型，persistent kernel 中 CTA 只 launch 一次，硬件调度器（GigaThread Engine）的角色退居其次，任务分配（哪个 tile 给哪个 SM 的哪个 worker）由软件 tile scheduler 决定。PIPEWEAVE 明确指出：此时"基本调度单元 task"不再是 CTA 而是 resident CTA 每次取回的 tile，且调度语义必须显式建模软件调度器（如 FA3 的 MinHeap 调度逻辑）才能准确预测 per-SM 负载分布。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 FlashAttention-3（Hopper，persistent）为例的软件 tile 调度模拟：
```
# PIPEWEAVE Scheduling Simulator 复刻 FA3 的 MinHeap 调度（约 40 行）
# N_SM 个持久 CTA 各占一个 SM；每 CTA 从共享 work queue 取 tile
queue = MinHeap(tiles, key=按序列/头优先顺序)   # 或 work-stealing
while queue 非空:
    t = queue.pop_min()
    sm = 负载最小的空闲 CTA      # 软件决定去向，而非硬件 RR
    sm.task_list.append(t)
# 输出 task 分布 {T_1..T_N_SM}，供 Feature Analyzer 求每 SM 的 pipeline demand
```
对硬件调度范式（FA2、RMSNorm 等），则模拟 RR 策略：先给每个 SM 至少一个 CTA，资源够再第二轮，直到 SM 饱和，之后新 CTA 在旧 CTA 完成时补位。论文验证（Nsight Compute）：FA3（persistent、确定性调度可显式模拟）的 per-SM op 计数误差仅 0.45%，而 FA2（动态硬件调度）为 6.34%——persistent kernel 的可确定性正是其可建模性来源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现上：CUTLASS/Triton 提供 persistent 编程支持（Triton `@triton.jit` 内 while 循环 + 原子指针取 tile、CUTLASS 的 tile scheduler 模板），开源库 FlashInfer 的 FA3、cuBLAS 的 Hopper persistent GEMM 均属此类。PIPEWEAVE 中调度范式由表驱动（Table V：GEMM/Attention 等 HW/SW 双范式、RMSNorm/SiLU&Mul/Fused MoE 为 HW）。使用意义：persistent kernel 消除反复 launch 开销、减少 tail 效应，但其性能强依赖软件调度质量——这正是 PIPEWEAVE"beyond simulation"用 P80 性能上限模型诊断 Fused MoE（SGLang Triton、persistent）在 A40 上 921 个 underperforming points 并 autotune（BLOCK_SIZE/num_stages/num_warps）提速 1.61× 的前提。

涉及论文标题：
- PIPEWEAVE: Synergizing Analytical and Learning Models for Unified GPU Performance Prediction

## Fused MoE Kernel（融合 MoE Kernel，SGLang Triton 后端）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fused MoE Kernel 是把 MoE 层的路由后 token-专家映射、多个专家各自的 GEMM 计算（甚至 SwiGLU 的 gate/up/down 三次 GEMM）融合进单个 GPU kernel 的算子实现（代表性：SGLang 的默认 MoE 后端、FlashMoE、MegaBlocks）。相比逐专家独立 launch GEMM，融合后消除了 N-1 次 kernel launch overhead 与中间张量的 HBM 往返，还能利用 grouped-GEMM/block-sparse 布局让不同专家共享一次调度。在 PIPEWEAVE 中它是 6 类被建模 kernel 之一（Triton 语言、BF16/FP16、Tensor pipeline、硬件调度），其 33,264 个样本数据集同时用作"beyond simulation"优化指导的 case study。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Fused MoE kernel 的关键调度结构（以 SGLang Fused MoE Triton 为例）：
```
# 1. 路由（上一阶段）：每个 token 经 router softmax 选 top-k 专家
# 2. 单 kernel 内按专家分组计算：
for expert e in activated_experts:
    tokens_e = tokens 按路由结果分到 e 的索引集合
    h_gate = tokens_e @ W_e^gate      # GEMM1
    h_up   = tokens_e @ W_e^up        # GEMM2
    h = silu(h_gate) * h_up           # SwiGLU 激活（element-wise）
    out_e  = h @ W_e^down             # GEMM3
# 3. 按原 token 顺序 scatter 回输出
# Triton 中由 @triton.jit kernel + BLOCK_SIZE/num_stages/num_warps 三个参数控制 tile 划分、流水线深度与并行度
```
PIPEWEAVE 对它的解析建模：M ∈ [2,8192]、E ∈ [8,128]、topk ∈ [2,8]、H ∈ [1024,4096]、N ∈ [512,3072]，每 task 的 Tensor ops = α·tile_M·tile_N·tile_K。优化指导流程：P80 分位模型预测执行效率上限 ŷ_p80，perf_gap = ŷ_p80 − y_actual > 0.1 记为 underperforming point；对 A40/L20/A100/H800 各选约 70 个配置 brute-force autotune 三参数，A40 从 921 个 underperforming points（占 30.4%）经调参平均 gap 从 0.187 降到 0.083、几何平均提速 1.61×，且 underperforming points 数与提速成 Pearson 0.86 正相关；残余 gap 归因于 Triton 编程模型/结构设计限制而非参数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SGLang 默认 MoE 后端为 Triton Fused MoE kernel；vLLM 侧有集成路径；FlashMoE 用 CUTLASS device-side GEMM 在 persistent kernel loop 内直接调用（单 kernel 完成分布式 MoE）；MegaBlocks 用 block-sparse 布局。使用上，PIPEWEAVE 证明融合 kernel 的性能预测比 element-wise 拆分更难也更重要——融合破坏了"算子边界=launch 边界"假设，只能按 kernel 级 task 分解建模。开源 artifact（github.com/zksainx/pipeweave）提供 Triton MoE 的 roofline 计算器，可对给定 (M,E,topk,H,N) 预测 latency。

涉及论文标题：
- PIPEWEAVE: Synergizing Analytical and Learning Models for Unified GPU Performance Prediction

## GPU 性能预测（Grey-box 解析-学习混合建模）与执行效率（Execution Efficiency）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GPU 性能预测是用模型在运行前估算 kernel/推理的耗时，服务于硬件选型与系统探索。三范式各有短板：cycle-accurate 模拟器（Accel-Sim、MGPUSim、AMALI、LLMCompass）保真但慢且不可移植；解析模型（Roofline、GPUMeCH、GCOM）快但精度受限、依赖硬件专属 microbenchmark；数据驱动（Habitat、Neusight）学 tile 级延迟但把 tile 当原子、假设 SM 均匀、静态 wave 假设、无法刻画 fused kernel 与跨 SM 负载不均。PIPEWEAVE 提出 grey-box 混合：解析模型产出 pipeline 级 demand/理论周期特征（知识驱动），轻量 MLP 学习跨 pipeline 交互与资源争用（数据驱动）。其训练目标是**执行效率**（execution efficiency）η = 理论执行时间 / 实测延迟，MLP 输出层用 Sigmoid 限到 [0,1]，最终延迟 = 理论时间 / η——这让模型学的是"离理论多近"而非绝对延迟，天然具备跨硬件泛化性。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# PIPEWEAVE 预测单 kernel 延迟的完整流程（kernel 输入 → 性能输出）
features = []
for each task τi in KernelDecomposer(kernel, X, S):     # 拆 task
    for p in [Tensor, FMA, XU]:                          # math pipeline
        N_ops,p = α·tile_M·tile_N·tile_K (Tensor) 或源码解析的 EW ops
        C_p = N_ops,p / Th_p                              # 理论周期
    B_i = sum(data loaded by τi from memory hierarchy)    # MIO 需求
    features.append({N_ops,p, C_p, B_i, C_mem})
# task → SM → GPU 三级聚合，得到 Table IV 特征向量
y_hat = MLP(features)                       # 预测执行效率 η ∈ [0,1]
latency = T_theoretical / y_hat             # 最终延迟（µs）
```
MLP 为 3 隐层（256/128/64，ReLU+BN+Dropout 0.1），per-kernel 训练，MAPE loss + AdamW + early stopping。效果：seen GPU kernel MAPE 6.1%、unseen 11.4%（Neusight 42.6%/45.1%），E2E 8.5%/10.7%，比 AMALI/LLMCompass 快 3–7 个数量级且 MAPE 6.4% vs 28.3%/29.7%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源 artifact 在 https://github.com/zksainx/pipeweave（Apache-2.0），依赖仅 torch/numpy/pandas/sklearn/joblib。使用：`pip install torch numpy pandas scikit-learn joblib` → `python3 train_mlp.py` 训练各 kernel 的 MLP（输入 dataset/ 的 profiled CSV + hardware/ 的 GPU 规格）→ `python3 aggregator.py --workload workload/<model>_<bench>_<kernels>_tp<k>_pp<k>.json --hardware <GPU> --model_dir mlp_models --dataset_dir dataset --hardware_dir hardware --output e2e/...json` 预测 E2E 延迟 → `compare_pred_real.py`/`compare_vllm_pred_real.py` 与 Roofline/Habitat/Neusight/vLLM 实测对比 MAPE。ground-truth 用 PyTorch Profiler 在物理 GPU 上实测（5 warmup + 10 次取平均）。局限：per-kernel 建模（新 kernel 需新 Decomposer+训练）、假设 E2E 串行无重叠、通信 kernel 用 Random Forest 简化建模。

涉及论文标题：
- PIPEWEAVE: Synergizing Analytical and Learning Models for Unified GPU Performance Prediction

## SASS（Streaming Assembler，NVIDIA GPU 机器汇编）与 PTX

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SASS 是 NVIDIA GPU 的实际机器码（每 GPU 微架构专属的 ISA 编码，如 sm_86 Ampere、sm_89 Ada、sm_120 Blackwell），PTX 是设备无关的虚拟 ISA（Parallel Thread Execution）。编译链：nvcc 把 CUDA C++ 编译为 PTX → ptxas 把 PTX 汇编为 SASS → SASS 打包进 cubin/ELF 并嵌入 nv_fatbin；运行时驱动选择与 GPU 匹配的 SASS（无匹配才 JIT 编译 PTX）。Web 证据：cuobjdump（CUDA Binary Utilities）可 --dump-sass/--dump-ptx 反汇编 fatbin 内的 SASS/PTX。PRowhammer（ISCA'26）观察 O2：NVIDIA GPU 共享库（cuBLASLt、GGML）主要含 SASS；单 bit-flip 可把 SASS 指令变成"不同但合法"的指令（表 I 四类：寄存器变化、opcode 变化、offset 变化、指令变化，如 MOV→MOV 换寄存器、FFMA→FSET.F.FTZ.AND、LDS [R17+0x140]→[R17+0x148]、SHL→LOP3.LUT），避免崩溃并改变 kernel 语义——这是精度降级攻击可行的关键。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SASS 在 kernel 执行中的角色：GPU 驱动按 GPU 架构从 .nv_fatbin 选 SASS → 动态链接 → 传至 GPU → SM 前端取指、按 warp 调度执行。PRowhammer 的指令翻转例子（表 I，RTX 4090，64-bit 机器码）：FFMA R11,R22,R11,R8（0x5980040000b7160b）单 bit-flip → FSET.F.FTZ.AND R11,R22,R11,!P0（0x5880040000b7160b）——opcode 从浮点乘加变比较集位，同一 kernel 后续计算语义全变；SHL R15,R3,0x6 → LOP3.LUT R15,R3,0x6,R0,0x48；LDS.U.32 R23,[R17+0x140] → [R17+0x148]（访存偏移 +8）。压缩码中单 bit-flip 经解压常产生 2–5 个（最多 25 个）改义但合法的指令（Fig. 5），部分 kernel 崩溃、部分（cuBLASLt 3–83、GGML 41–99 个可利用位）不崩溃只改输出。验证工具：cuobjdump 反汇编 + diff 比较翻转前后 SASS 是否仍合法（artifact 用 cuobjdump 检查合法性）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SASS 由 ptxas 生成、存于 fatbin；用户/攻击者用 cuobjdump --dump-sass 或 nvdisasm 观察。使用：正常场景下 SASS 对用户透明；PRowhammer 场景下攻击者把 SASS 当作攻击面——因共享库闭源且压缩，攻击者不做反编译，而是"翻转 bit → 执行 kernel → 观察输出"黑盒验证（500–700ms/次，自定义 CustomLib 100ms/次），配合剪枝定位可利用 bit；再对解压后 SASS 用 cuobjdump 确认指令合法性与改义类别。定位 kernel：profiling 模型（单线性层）确定目标模型调用的 cuBLASLt kernel（sm_86 共 3508 个，实际只调 1–2 个）。注意点：SASS 因架构而异，profiling 需对每个 (库版本, GPU 架构) 组合重复；RTX 4090（Ada sm_89）与 RTX A6000（Ampere sm_86）用同一 cuBLASLt 库代码，翻转位可跨模型/数据集转移。

涉及论文标题：
- PRowhammer Propagating Bit-flips from CPU to GPU

## cuBLASLt（NVIDIA 闭源张量核线性代数库）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
cuBLASLt 是 NVIDIA 的轻量级（Lt）BLAS 库，内含针对张量核（Tensor Core）优化的 GEMM 等 kernel 集合，被 PyTorch、TensorFlow、TensorRT-LLM、Triton Inference Server 等广泛调用（cuBLASLt 由 cuBLAS 内部调用）。Web 证据：cuBLASLt 用 heuristics（cublasLtMatmulAlgoGetHeuristic，按问题规模/GPU 配置/数据类型选最合适 matmul kernel）做运行时 kernel 选择；CUDA 13.x 新增 CUBLAS_GEMM_AUTOTUNE 实测选优。PRowhammer（ISCA'26）把它作为攻击目标：(1) 闭源、GPU kernel 压缩存储（nv_fatbin 压缩后 255MB，整库 335MB）；(2) 含 sm_86 的 3508 个 kernel，但给定矩阵形状/输出维度只调 1–2 个；(3) 分类模型末层线性层调用 cuBLASLt kernel——单 bit-flip 在 cuBLASLt 中即把 ResNet-18/34/50、VGG-16 在 MNIST/FMNIST/CIFAR-10/ImageNet 上的准确率打到随机猜测（ImageNet 最坏 0%，RPL 84.95–100%）；218（MNIST/FMNIST/CIFAR-10）+93（ImageNet）个可利用翻转位，且同一翻转位跨模型/数据集转移有效。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
cuBLASLt 在推理中的调度链：ML 框架（PyTorch）的线性层 → cuBLAS/cuBLASLt API → heuristics 按 (M,N,K,数据类型,GPU) 选 kernel（如 ImageNet 1000 类输出 vs CIFAR-10 10 类输出各对应不同 kernel）→ 运行时从 .nv_fatbin 动态链接 SASS → GPU Tensor Core 执行。PRowhammer 的利用链（black-box）：(1) profiling 模型（单线性层、输出维度=目标类别数）定位实际调用 kernel——输出 10 类的 MNIST/FMNIST/CIFAR-10 共用一个 kernel，输出 1000 类的 ImageNet 用另一个；输入维度未知时扫 2–10000 验证 kernel 集合稳定；(2) 剪枝定位：把 nv_fatbin 二分（n=2）分段、每段全 bit 翻转后执行 kernel 与 golden 比对，保留崩溃/改输出的有用段直到 1KB 阈值，再随机抽 10000 bit 逐个模拟，得可利用 bit（mnist/fmnist/cifar10 用偏移 0x95c787a 的 bit 4；imagenet 用偏移 0xc56745c 的 bit 8）；(3) 攻击者 Rowhammer 翻转该 bit → 受害者末层 GEMM 输出被破坏 → 分类概率乱序。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：NVIDIA 闭源共享库 libcublasLt.so（Linux 位于 /usr/local/lib/libcublasLt.so.12），内含压缩 nv_fatbin（NVCC --compress-mode 闭源压缩算法）；artifact 用 get_golden_lib.sh 拷贝库、run_profile_cublas.sh 跑五阶段 profiling 管线（kernel_locater → choose_target_region → run_flipper_watchdog → segregate → extract_useful_flips）输出 bitflip_data.csv。使用：正常推理时框架透明调用；攻击侧把它作为"单一共享点"——一个库被破坏影响所有依赖它的模型（含 Triton Inference Server、TensorRT-LLM 的生产 LLM serving 后端）。限制：kernel 选择随库版本/GPU 架构/autotune 变化，profiling 需对每个 (库版本, 架构) 对重复；PRowhammer 对防御的启示：库代码路径需完整性校验（压缩/解压 ECC/CRC、dispatch 前哈希）。

涉及论文标题：
- PRowhammer Propagating Bit-flips from CPU to GPU

## GGML（开源 GPU 加速张量库）与 ggml_mul_mat kernel

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
GGML 是开源的机器学习张量库（ggml-org/ggml，C/C++ 实现），llama.cpp 的底层算子库，面向本地 LLM 推理，提供量化张量与 GPU kernel（CUDA/Metal/Vulkan/SYCL/WebGPU/CPU 多后端）。其核心算子 ggml_mul_mat（含 ggml_mul_mat_vec 变体）执行量化矩阵乘：对 4-bit 类型（Q4_0/Q4_1/Q4_K）按块解量化后做 GEMM/GEMV；Web 证据显示现代版本按 batch 在 MMVQ（逐行 GEMV，低延迟）与 MMQ（MFMA-tiled GEMM，高通量）间分派。PRowhammer（ISCA'26）把它作为 LLM 攻击目标：llama.cpp 高层函数大量调用 ggml_mul_mat；压缩 nv_fatbin 14MB；单 bit-flip 使 Llama-2-7B/Mistral-7B/Falcon-7B（4-bit 量化）在 Google Natural Questions 100 问上的平均 BERTScore 从 0.58–0.62 跌到 0.25–0.30（输出 # 串或跨语言乱码；部分位产生语法连贯但语义错误的文本）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
运转链：llama.cpp 前向（QA 任务）→ 高层函数调 ggml_mul_mat（每 token 的矩阵乘，如 M=1 解码或大 M 预填充）→ CUDA 后端按量化类型选 mul_mat 实现（4-bit Q4 在 GPU 上 dequant+GEMM）→ SASS 从 GGML 的 .nv_fatbin 动态链接执行。PRowhammer 的利用：以 ggml_mul_mat 构造 wrapper 做 bit-flip profiling（RTX A6000/5060/4090 分别得 33/55/64 个可利用翻转）→ 因该 kernel 被模型反复复用，翻转位跨模型转移显著 → 攻击阶段对真实模型应用 → 解码输出被破坏。flip 模拟开销 500–700ms/次，50000 次 trial 中 cuBLASLt 无可利用位、需剪枝策略（GGML 41–99 个可利用位/10000 trial）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：开源（https://github.com/ggml-org/ggml），llama.cpp（https://github.com/ggml-org/llama.cpp）调用；GGML 共享库含多架构 SASS 的压缩 nv_fatbin。使用：正常使用 `llama-cli -p "..."` 跑量化模型推理；攻击侧用 run_profile_ggml.sh 跑五阶段 profiling（bitflip_data_ggml.csv 第一列索引对应 outs_ggml/stdout/out_err_<index>.log 的 corrupted 输出），再对真实模型施加单 bit-flip 验证 BERTScore 退化。攻击例子（Listing 3/4）：正确输出 "Google is a multinational technology company" → 翻转后 "Unterscheidung sehialog Dhorn Jurivers H"（乱码）；或 "The dog's name on Tom and Jerry is Spike." → "In the Tom and Jerry cartoon series, the dog's name is Momo."（连贯但错误）。

涉及论文标题：
- PRowhammer Propagating Bit-flips from CPU to GPU

## 收益感知迁移调度（benefit-aware thread migration）与 Load State

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PhaseWeave 的线程迁移决策算法（Algorithm 1）：不无条件迁到预测的"最优" chiplet，而是显式权衡预测收益与迁移/排队代价。对线程 t、预测 phase p、当前 chiplet c，到候选 chiplet c' 的迁移效用为 U(t,c')=S(p,c')-S(p,c)-λ·Q(c')-C_switch：S(p,c) 是 phase p 在 chiplet c 的离线预期稳态性能（如 IPC），Q(c') 是 c' 当前 runqueue 长度（由硬件 Load State 模块报告），λ 把排队线程折算成预期延迟惩罚，C_switch 是单次上下文切换代价；若 c'=c 则 U=0。调度器选 c*=argmax U(c')，仅当 U(c*)>θ 且线程在现 chiplet 驻留 ≥T_min 才迁移（T_min 防预测噪声导致的振荡）。这是把"负载感知"并入迁移决策的运行时调度策略，属于异构后端上的运行时任务调度。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
调度循环（每 100µs epoch 边界触发，不在线程关键路径上）：

```
for each thread t at epoch boundary:
    p <- predicted phase for t            # 硬件 RF 预测器输出
    c <- current chiplet of t
    for each chiplet c' in C:
        S_c' <- expected perf of phase p on c'   # 离线 phase 特征表
        S_c  <- expected perf of phase p on c
        Q_c' <- runqueue length reported by Load State of c'
        U(t,c') <- S_c' - S_c - lambda*Q_c' - C_switch
    c* <- argmax_{c'} U(t,c')
    if U(t,c*) > theta and residency_time(t) > T_min:
        Migrate t from c to c*             # 标准 context switch 入队目标 chiplet runqueue
```

例子：某线程在 compute chiplet 跑完 GEMM phase，预测下一 phase 为 DeepCopy；若 fast-memory chiplet 空闲（Q≈0），U≈(fast-mem 相对 compute 的 DeepCopy 加速) - C_switch > θ → 迁移；若 fast-memory chiplet runqueue 很长（λ·Q 大），U 变负 → 留在 compute chiplet 或迁往次优但空闲的 chiplet（论文统计 18.5% 的迁移因最优核满载而落到次优核）。硬件支持：每 chiplet 有软件可写 task-count 寄存器（OS 在 runqueue 变化时更新），Load State 模块读取后导出给调度器，全程不占线程关键路径。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：调度器在 OS 侧运行（PhaseWeave 不改 OS 调度器数据结构——chiplet 暴露为 CPU 簇，硬件经 MSR/MMIO 给 placement recommendation，OS 周期读并设 affinity），迁移=常规跨核任务迁移（页表根/TLB 走既有机制）；Load State/task-count 为硬件支持。使用与收益：对照实验显示无条件迁移（PhaseWeave-NoMigrationAlg）只达 1.26× 吞吐（专精+多核收益），加收益感知算法后 1.56×（算法额外贡献 1.24×）；53.2% 的 epoch 迁移≥1 线程；真实系统迁移开销平均 23.8µs、中位 9.5µs，远小于 phase 时长。通用启示：负载感知迁移是异构资源池（chiplet/GPU/加速器）调度的通用模式——把"排队惩罚"与"架构优势"统一进效用函数，避免异构系统被"最优目标过载"反噬。

涉及论文标题：
- PhaseWeave Phase-Aware Execution on Heterogeneous Chiplet Architectures for Datacenters


## Online Kernel Profiler（在线 kernel 画像器）与 wave-based Latency Predictor（wave 比例延迟预测）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Online Kernel Profiler 是 PowerWeave（ISCA'26）的组件：在 serving 运行过程中透明地逐 kernel 记录执行延迟，构建"kernel 身份 → 频率-延迟行为"的在线画像，为后续 DVFS 频率选择提供依据。它运行在 PowerWeave Interposer（透明拦截 CUDA driver API 的 Rust 层）的专用后台线程中。kernel 身份 key = 函数句柄 + grid/block 维度 + 共享内存大小 + CUDA stream——把"同函数不同序列长度/batch 的实例"区分为独立条目，避免输入相关行为（如 attention 内核随序列长度变化的访存模式）被平均进一条曲线。
- Latency Predictor：对新出现的 kernel 配置（同 kernel 家族不同 grid/thread-block 配置），用 wave 比例从已画像实例泛化延迟：l = waves × (l_old / waves_old)，其中 waves = 总 launched blocks /（每 SM 可驻留 block 数 × 分配给该 kernel 的 SM 数）。直觉：每个 SM 并行驻留若干 block，一个 kernel 的 block 需要多"波"串行执行完，waves 与延迟成正比。论文实测平均误预测 3.9%（prefill 4.55µs vs 平均 118.75µs；decode 0.84µs vs 16µs），对 SLO 安全足够。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 运转流程（伪代码级）：
```
// Interposer 在每次 kernel launch 时：
key = (funcHandle, gridDim, blockDim, sharedMemSize, stream)
startEvt.record(stream); kernel.launch(); endEvt.record(stream)
// 后台线程异步查询事件耗时，避免阻塞关键路径
if 首次见到 key:
    baseline 阶段: 以最大频率执行，记录 latency[key][f_max]
    profiling 阶段: 把 key 分配到多个频率点(1965MHz..915MHz, 12步)执行
                    latency[key][f] = measured  # 每频率点在不同请求中完成
else if key 未见过的变体(同家族不同配置):
    l_pred = waves(key) * latency[donor][f_max] / waves(donor)
    # waves = launchedBlocks / (blocksPerSM * numSMs)
# 进入 operating 阶段后每 kernel 完成后对照预测延迟，
# 偏离 > profiling-threshold(5%) 则重启 profiling
```
- 例子（Llama-3.1-8B decode，148 SM 中 decode 域占 37 TPC=74 SM，blocksPerSM=8）：某 attention kernel 启动 1024 个 block → waves=1024/(8×74)≈1.73；若已画像的 donor 配置 waves_old=1、l_old=16µs，则预测 l≈27.7µs。跨序列长度/batch 的新配置无需重新全量扫描即可得到延迟估计，profiling 开销被摊到 ≈150 个请求（12 频率点×2 轮），单个请求不感知完整成本。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：kernel launch 拦截（LD_PRELOAD 式/CUDA driver API interpose）+ CUDA event 对异步计时（不阻塞关键路径）+ 后台线程执行 profiler/predictor/controller。kernel 执行通常数百 µs 到数 ms，profiling 对单个 kernel 选择性监控、影响最小。新 kernel 无 donor 时保守用最大频率直到其运行时间占比被评估（论文：这类 kernel 平均占 1.9% 运行时间，低于 5% 重画像阈值，实验从未触发重画像）。在线画像使系统无需离线 profiling 即可适应：不同 batch/序列长度/模型的频率-延迟行为被持续学习，权重（kernel 占应用运行时间比例）在线更新以跟踪 prefill/decode 比例变化。

涉及论文标题：
- PowerWeave: Unlocking Energy-Efficient ML on GPUs with OS-Level Spatial Power Management

## Frequency-Latency Scaling（频率-延迟缩放模型：kernel 级曲线合成 per-application 缩放函数）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Frequency-Latency Scaling 模块把 Online Kernel Profiler 收集的 per-kernel 频率-延迟曲线，合并成 per-application（每租户/每域）的频率-性能模型，回答"降到某个频率会让应用端到端延迟慢多少"。核心是一阶 Taylor 近似启发式：对目标性能退化 k，调整后频率 f(k) = f_max / S，其中 S = 1 + k / Σ(s·w)：w 是每 kernel 占应用总运行时间的权重（在线更新，跟踪 prefill/decode 比例漂移），s 是每 kernel 的频率敏感度因子（曲线斜率，刻画"降频 1% 延迟涨多少"，由指令混合决定、profiling 后固定）。
- 直觉：权重 w 平衡应用内 compute-bound（高敏感）与 memory-bound（低敏感）工作的比例；敏感度 s 防止低敏感 kernel 把目标频率拖到频率敏感 kernel 无法接受的低点。per-application 独立建模使每租户/每阶段可独立缩放频率，且不绑定固定 slowdown 假设，能适应负载波动与 SLO 变化。论文消融（Fig. 12）：完整系统平均偏差 1.7%（最坏 5.2%，且平均低于目标——不会高估性能损失）；去掉 live weight updates 平均偏差升到 4%、最坏 75%（高负载下预测严重失准）；去掉 sensitivity 平均 4% 但 10%/20% slack 处过度估计损失导致 SLO 违反（最坏 10.6%）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算过程（伪代码级）：
```
# 对每个应用(域)：
# 从 profiler 得到每 kernel 的 {latency(f_max), latency(f_i)} 曲线
s_k = 平均每降频 1% 的延迟增幅(曲线斜率)        # 固定
w_k = kernel_k 运行时间 / 应用总运行时间          # 在线更新
S(k) = 1 + k / sum(s_k * w_k)                    # k = governor 允许的性能退化
f_target = f_max / S(k)                          # DVFS Controller 下发的频率
```
- 例子（disaggregated prefill，Llama-3.1-8B）：prefill 应用里 GEMM 类 kernel 权重 w 大、敏感度 s 高 → 允许 20% 退化时 S≈1.15，频率仅从 1965MHz 降到 ~1710MHz（TTFT 仍满足 SLO）；decode 应用里 memory-bound kernel 权重 w 大但 s≈0（降频不加速）→ 同样 20% slack 可把频率降到很低（~915MHz 级），TPOT 几乎不变。Governor 每监控窗口更新 slack：s₂ = ((1−s₁)×l₁)/SLO（由当前延迟反推最大频率下理论延迟再除以 SLO）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：模块位于 PowerWeave Interposer 地址空间（Rust），输入 per-kernel 画像与 governor 的 slack 指令，输出每域目标频率经 NVML 下发。使用：profiling 阶段结束后进入 operating 阶段，持续用 live weight 更新跟踪 workload 构成；每 kernel 完成后对照预测延迟，偏离超 5%（profiling-threshold）重启 profiling。该模型与 serving 框架、应用解耦：应用只通过 governor 声明 SLO 与上报指标，频率策略在控制平面内实现——同一套机制可支撑 latency-driven / per-tenant / throughput-balancing 等多种策略。

涉及论文标题：
- PowerWeave: Unlocking Energy-Efficient ML on GPUs with OS-Level Spatial Power Management



## AllToAll（全对全通信，Expert Parallelism 的 token 路由）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AllToAll 是集体通信（CC）操作之一：每个 rank 向所有其它 rank 发送/接收一个 tensor 的不同分片，实现数据的完全重新分布（full redistribution），是三类典型 CC（AllGather、AllReduce、AllToAll）中唯一"每个 rank 都既是发送方又是接收方且分片各不相同"的操作。在 MoE（专家并行）中，AllToAll 承担 token 路由：gate/router 决定每个 token 去哪个（可能位于其它 rank 的）专家，AllToAll 把各 rank 的 token 子集分发给对应专家，专家算完后反向 AllToAll 把结果送回。RoCC 论文把 AllToAll 分解为最简单的 primitive 序列（send → recv，仅 2 阶段）并支持在 ROP 上执行；AllToAll 无归约计算，本质是"数据搬运"，与 AllGather/AllReduce 一样是网络/内存 bound。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RoCC 论文中 AllToAll 在 ROP 上的执行（Table I，4-GPU ring 分解为 2 primitive）：send 展开为 ReadDoorbell→RingDoorbell（读本地分片、门铃转发给目标 rank），recv 展开为 ReadDoorbell→Write（收上一 rank 的分片、写本地）。伪代码（k/V 层 AllToAll dispatch 概念）：
```
for e in experts:                         # 每个专家一个目标分片
    tokens_to_e = gate(tokens, e)        # 路由决策：哪些 token 去专家 e
    if e on remote rank r:
        send(tokens_to_e, rank_r)        # 发给远端专家（RoCC: RingDoorbell）
for r in ranks:
    recv(tokens_from_r)                  # 收各 rank 发来的 token（RoCC: ReadDoorbell→Write）
    expert_gemm(tokens_from_r)           # 本地/远端专家计算
```
在本文评估中，AllToAll 场景模拟专家并行压力测试：每个专家与所有其它专家交换 token。CC-only 延迟比较中 RoCC 对 AllToAll 达 25% 加速（大消息，因 ROP 近内存避免 NoC 与 L1 miss 开销）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：NCCL/RCCL 库提供 AllToAll 原语（如 ncclSend/ncclRecv 或 ncclAllToAll），底层按拓扑组织多阶段点到点传输；RoCC 将其编译为 ROP primitive（send/recv）执行。使用场景：MoE 训练/推理的 token dispatch 与 combine、序列并行（Sequence Parallelism）的激活交换、专家并行梯度同步。与相关概念区分：AllGather 是"每 rank 广播自己的分片、拼接全员数据"（RoCC 分解 4 阶段 recvCopySend），AllReduce 是"归约后全员可见"（RoCC 分解 7 阶段），AllToAll 只做分片交换无归约无拼接（RoCC 分解仅 2 阶段）。

- STAGE 补充视角（ISCA'26）：STAGE 的 Collective Communication Matcher 把 AllToAll 作为"Pull+Push"组合的特例系统化匹配：producer 分布 [B/dp,S,H@1/tp] 与 consumer 分布 [B,S/dp,H@1/tp] 之间（dp 作用于 batch、sp 作用于 seq），匹配结果即 AllToAll；还识别出此前被忽略的组合模式，如 [B/dp,S,H@1/tp]→[B/tp,S,H/dp] 匹配 ReduceScatter+AllToAll、[B/dp,S,H@1/tp]→[B,S,H] 匹配 AllReduce+AllGather（Table IV）。MoE 的 EP 层 AllToAll 通信量在 STAGE 中按张量尺寸精确计算，用于 DeepSeek-R1 推理与 MoE 训练的通信验证（Table VII 中 Mixtral 8x7 TP4-EP8-PP4 的 Send/Recv 误差 2.755%）。

涉及论文标题：
- RoCC Harnessing Raster Operations Pipeline for Efficient Tensor Collective Communication
- Scalable Synthesis of Distributed LLM Workloads Through Symbolic Tensor Graphs

## 带宽利用中心数据流（Bandwidth-Utilization-Centric Dataflow：tile 分组 / plane-aware 布局 / 输出输入驻留 / GEMV 配对）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 带宽利用中心数据流是 SHyLA 的运行时算子映射与数据布局策略，目标是在 3D 堆叠混合内存（NVM 存 Weight/KVCache、DRAM 存 IA）的放置约束下最大化 areal bandwidth 利用，避免 2D/2.5D 风格访问模式在 3D 堆叠下带宽利用不足。组成：(1) memory-plane-aware 数据布局——tile group 内 NVM plane 存其 NTile 与同组 DTile 的 Weight 均匀切块、DRAM plane 存 IA 行，防 plane starvation 与 compute-tile-memory-plane 失配；(2) tile 分组——#DTile<#NTile 时每组 1 个 DTile + 若干 NTile（同相对位置配对），组内专用高速链路、跨组 AXI fabric；(3) intra-chiplet 输出驻留+输入驻留（output-stationary with input-stationary）强调 Weight 读复用、最小化 IA 流量，tiling 因子 B_I/B_K 由片上 buffer 容量决定；(4) decode GEMV 配对 + KVCache 单 plane 放置；(5) GQA attention-group/request 级并行。仿真标定达到 DRAM 读/写 90%、NVM 读 70%、NVM 写 10% 的带宽利用。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 以 FFN GEMM（IA∈DRAM、Weight∈NVM、B_I×B_K tiling）为例的调度伪代码：
```
for i_batch in range(0, I, B_I):               # IA 行块（DRAM 全局输入 buffer）
    load IA_rows[i_batch] -> global_in_buf     # DRAM 读，双缓冲与计算重叠
    for k_block in range(0, K, B_K):           # Weight 列块（NVM）
        load W[:, k_block] -> tile_local_buf   # NVM 读（每 NVM plane 存 NTile+组内 DTile 切块）
        compute C[i_batch, k_block] = IA_rows[i_batch] @ W[:, k_block]   # tile MAC 阵列，output/input-stationary
        write C -> DRAM（或片间 all-reduce 于 Attention Output/FFN2 输出）
    # decode 阶段 ATTN：每 tile 一对 GEMV（QK^T→SV 连续、QK^T 中间结果不写 DRAM），KVCache 单 plane 本地访问
```
时间模型即解析模型 Eq.(2)：IA 批读占 DRAM 读带宽、每批内 Weight 块占 NVM 读带宽、IA 写回占 DRAM 写带宽。消融（SHyLA-D 无 tile 分组/细粒度切分、one-to-one NTile-DTile 配对）显示该数据流贡献 1.35× geomean 系统吞吐提升（更高带宽利用、更低 NVM 访问开销）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 在 GPGPU-Sim（https://github.com/gpgpu-sim/gpgpu-sim_distribution）中以 CUDA 实现：CUDA 管理片上双缓冲与 plane-aware tile 映射，GPGPU-Sim 内存控制器地址映射被修改为把连续地址跨 channel 分布（提高带宽利用）、channel 数按 CACTI-3DD 推导带宽配置、DRAM/PCM 时序按 workload 配置。SHyLA 本体（CUDA 映射/解析模型）未开源（联网未找到仓库）。使用流程：输入 LLM 层算子形状与混合内存配置 → GPGPU-Sim 逐周期模拟 tile 计算与 Weight 加载重叠、按 plane 分布的内存请求 → 输出每层执行周期与带宽利用 → 汇总系统 token 吞吐。

涉及论文标题：
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity

## GPU Patch 聚批与 TopK 选择（unfold 聚批 / AvgPool2D PSM / GPU TopK kernel）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SLICE 把 patch 级上采样调度全部放在 GPU 上执行的运行时方案：① 用 AvgPool2D 一次池化把码流元数据网格聚合为每 patch 统计量（PSM）；② 用 GPU TopK kernel 按推理面积预算（35%）选出得分最高的 patch；③ 用 unfold 把 patch 网格转为紧凑 GPU 张量、按 SR mask gather 出需推理的 patch 组成 batch，一次/少数几次 EDSR forward；④ 按行分带（row-wise banded）GPU 拷贝合并写 framebuffer。设计目标：避免 CPU 往返与 kernel 启动开销，让 patch 提取、调度、SR、合并全链路驻留 GPU。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# PSM 聚合（各一次 AvgPool2D，P=16）
mv_mean      = AvgPool2D(G^mv,  kernel=4,  stride=4)   # 4×4 块粒度 MV 网格
res_pixel_mean = AvgPool2D(|G^pix|, kernel=16, stride=16)  # 像素粒度
hf_ratio     = AvgPool2D(G^hf, kernel=4, stride=4) / AvgPool2D(G^t, kernel=4, stride=4)
# 调度决策
M^reuse = (mv_mean==0) & (res_pixel_mean==0)
score   = 0.9·hf_ratio + 0.1·(1 − clip(mv_mean/10,0,1))
M^SR    = TopK(score, k=35%)                            # GPU TopK kernel
# 聚批推理
patch_batch = unfold(frame, 16, 16)                     # (N_patch, 3, 16, 16)
selected    = patch_batch[M^SR]                         # gather → (≈35%·N_patch, 3, 16, 16)
HR_patches  = EDSR_fp16(selected)                       # 单次/少数几次 forward，输出 4× (…,64,64)
# 合并：reuse 行按水平相邻合成连续带整段拷贝；插值 patch 各自拷贝；SR patch 按目标位置写回
```
例子：270p 帧 30×17=510 个 patch，TopK 选约 178 个组成 batch 单次 EDSR forward；静态 patch 走 HR cache 带拷贝；其余 bicubic。论文强调不循环逐 patch 构建 batch，避免 CPU 往返与 kernel 启动开销，这是吞吐优化的关键实现细节。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
基于 PyTorch 标准张量算子实现：torch.nn.functional.avg_pool2d、torch.topk、tensor.unfold / F.unfold、布尔 mask 索引（gather）、GPU 上的 banded 拷贝。论文未说明是否自定义了 CUDA kernel（记为论文未明确说明——实现描述为 PyTorch 算子级）。合并阶段把水平相邻的复用 patch 合成连续 band 整段拷贝，以提升带宽效率、降低每 patch 更新开销。

涉及论文标题：
- SLICE A Selective Local Inference Framework with Codec Exploitation for Accelerating Video Super-Resolution

## Expert Prefetching（专家预取，临时共享专家 + CUDA 异步流重叠）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Prefetching 是在 CPU-GPU 异构 MoE 推理中，把将被激活的专家权重从 CPU 主存提前经 PCIe 加载到 GPU 并与计算重叠、以隐藏专家取数延迟的运行时机制。STEP 中 profiling（Qwen3-30B-A3B、A100、INT8）显示专家取数占 MoE 推理执行时间约 88%，远高于 gating/计算/聚合合计的 12%，因此隐藏 expert-fetch 延迟是核心优化。STEP 的预取策略：①把窗口内票选的 top-c 临时共享专家在计算开始前整体预取并常驻 GPU（结构 j+c shared + k−c routed 后每步动态加载从 k 降到 k−c）；②预取实现为独立 CUDA stream 上的异步数据传输 kernel（cudaMemcpyAsync H2D），在专家计算 kernel 之前发起，借助 CUDA 非抢占式 kernel 执行让传输与计算并发；③每个解码步序列的最后一个预取 kernel 后插入 CUDA event 记录完成，CPU 用非阻塞查询（cudaEventQuery）保证数据可用同时避免同步阻塞。预取质量由命中率（Prefetch Hit Rate）衡量：CNN/DM 85.5–98.8%、LongBench 72.1–95.6%（Table II-IV），窗口自适应在准确率 >75% 时保持积极预取、<40% 时减窗甚至停用。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 每层 decode step（Mixtral top-2 → STEP 后 1 shared + 1 routed 为例）
# stream 0（计算流）                    # stream 1（预取流）
launch(gating_kernel)               
                                        launch(cudaMemcpyAsync, H2D,  # 临时 shared 权重
                                               elected_expert_w, 0)   # 已常驻则跳过（hit）
topk_idx = topk(gate_out, k-c)
                                        launch(cudaMemcpyAsync, H2D,   # 非驻留 routed 权重
                                               routed_expert_w)
launch(expert_gemm, stream=0)        # 与 stream1 传输并发（非抢占 kernel 执行）
...（每步最后一个预取 kernel 后）
                                        cudaEventRecord(ev, stream=1)
cudaEventQuery(ev)  # CPU 非阻塞轮询，数据就绪后继续，避免 blocking 同步
```
Annotations：stream 0/1=两条 CUDA 流（默认流 + 独立预取流），H2D=host-to-device 异步拷贝，cudaEventRecord/Query=记录/查询事件（非阻塞同步原语），hit=权重已在 GPU 显存（命中时不发起传输）。重叠条件：预取 kernel 先于计算 kernel 入队、流间无依赖（非抢占 kernel 执行允许并发），事件提供"就绪查询"而非"等待"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：pinned memory（cudaHostAlloc/cudaMallocHost）保证 H2D 异步传输可用；cudaMemcpyAsync + 独立 stream + cudaEvent 组合；预取目标来自窗口投票选举（临时共享）或 gating 预测。STEP 部署在 HuggingFace Transformers 推理路径上（batch=1 实时推理），与 EP 正交（每 EP group 独立预取，peer GPU HBM 可作二级缓存）。评估方法：以 Cached Expert Ratio（CER 25%/50%/75%）控制显存约束，测 TTFT（prefill）/TPOT（decode）与命中率；与 MoE-Infinity（activation-aware 预取）、HybriMoE（CPU-GPU 调度+缓存）、AdapMoE、DAOP、APTMoE、MoE-Lightning、llama.cpp 对比，decode 平均几何加速 1.54×–2.22×。Fig.14c 定量：命中率 >75% 时增大预取数显著降延迟，<40% 时过度预取浪费带宽——这是自适应窗口阈值（th_s/th_f）的 kernel 层依据。

涉及论文标题：
- STEP: Adaptive Spatio-Temporal Expert Prefetching for Low-Latency and Memory-Efficient MoE Inference

## CUDA Stream 与 CUDA Event（异步数据传输与事件同步）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CUDA Stream 是 GPU 上串行执行 kernel/传输操作的独立命令队列，不同 stream 之间可并发执行（无依赖时），是 CUDA 实现"传输与计算重叠"的基本手段；CUDA Event 是流内/跨流的同步原语，标记某流中已入队操作完成的时刻。STEP 用这两个机制实现专家预取与计算的流水线重叠：预取 kernel（异步 H2D 拷贝）在独立 stream 上先于专家计算 kernel 入队，二者并发执行；在每个解码步序列的最后一个预取 kernel 后记录 cudaEvent，CPU 侧用 cudaEventQuery 非阻塞查询其完成状态——查询返回"未完成"时 CPU 可继续做其他工作，避免 cudaStreamSynchronize 式阻塞同步，从而在不阻塞主线程的前提下保证"计算开始前所需专家已就绪"。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// STEP 式预取-计算重叠（伪代码）
cudaStream_t s_prefetch, s_compute;   // 两条流
cudaStreamCreate(&s_prefetch); cudaStreamCreate(&s_compute);
cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, s_prefetch); // 预取入队
expert_gemm<<<grid, block, 0, s_compute>>>(...);                       // 计算入队（并发）
cudaEventRecord(ev_done, s_prefetch);  // 记录预取完成点（非阻塞）
while (cudaEventQuery(ev_done) == cudaErrorNotReady) { /* CPU 可做其他事 */ }
// 数据可用后继续下一层
```
Annotations：s_prefetch=预取流、s_compute=计算流、cudaMemcpyAsync=异步 H2D 拷贝、cudaEventRecord=在流中插入事件标记、cudaEventQuery=非阻塞查询（返回 cudaErrorNotReady 表示未完成）、cudaStreamSynchronize 是应避免的阻塞版本。CUDA 的非抢占式 kernel 执行保证同 SM 上 kernel 完整运行，多流并发主要依赖资源余量（拷贝引擎与 SM 计算并行、或不同 kernel 占用不同资源）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：CUDA Runtime/Driver API 的标准组件（cudaStreamCreate/cudaMemcpyAsync/cudaEventRecord/cudaEventQuery）；STEP 在 Transformers 推理路径上按"预取 kernel 先入队、计算 kernel 后入队、事件标记同步点"的模式组织每层执行。使用场景：一切"数据搬运在关键路径上"的 GPU 系统——MoE 专家权重 H2D 预取、KV cache 搬运、pipeline 阶段间传输等；与 CUDA Graph 结合可减少 launch 开销。注意事项：异步传输要求 pinned memory；同一流内保持顺序、跨流靠 event/依赖保证顺序；过度并发会争抢带宽（STEP 用命中率反馈控制预取激进性，命中率 <40% 时减窗停用）。


- Symbiotic MLLM Serving: Dynamically Balancing Parallelism Across GPUs and Resources Within GPUs
RESONATOR 补充视角（ISCA'26，stream 到 SM 子集的绑定）：CUDA Stream 除"异步并发/事件同步"外还可作为 SM 配额载体——RESONATOR 用 green-ctx/libsmctrl 把 CUDA 流绑定到 SM 子集（GPC 配置掩码），预创建每任务两条流：wide 流（SMCTRL.SetQuota 1.0，全部 SM）与 narrow 流（SetQuota q_narrow，窄 SM 子集），contending 模式下每 kernel 查 profile 表路由到 wide/narrow 流（compute-bound→wide、memory-bound/低占用→narrow），实现 kernel 级 time-space sharing；complementary 模式下 decode 流固定绑 SM_dec 切片（SM 分区，兼容 CUDA Graph 重放）。与 STEP 的"预取/计算两流并发"不同，RESONATOR 用流既做并发也做 SM 隔离/配额。
涉及论文标题：
- STEP: Adaptive Spatio-Temporal Expert Prefetching for Low-Latency and Memory-Efficient MoE Inference

## Per-Kernel Stream Binding（wide/narrow stream 按 kernel 类型路由，RESONATOR contending 模式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Per-Kernel Stream Binding 是 RESONATOR Intra-GPU Sharing Engine 在 contending 场景（encoder 与 compute-bound prefill chunk 共跑）下的 kernel 级调度机制：不做静态 SM 分区，而是让两个任务都看到完整 SM 池，在 kernel 粒度控制资源使用。每个任务 T∈{enc,llm} 预创建两条 CUDA 流——wide stream s_T^wide（SMCTRL.SetQuota 1.0，可用全部 SM）与 narrow stream s_T^narrow（SMCTRL.SetQuota q_narrow，0<q_narrow<1，只跑窄 SM 子集）。每个到达 kernel k 查 profile 表 P 的 TYPE(k)∈{comp,mem}：compute-bound kernel 路由到 wide stream（占满 SM 填计算空洞），memory-bound/低占用 kernel 路由到 narrow stream（限制在小区间、不阻塞全局计算）。这样 encoder 与 prefill 的重 kernel 以 time-space sharing 填满 GPU，轻/带宽型 kernel 不干扰——对应论文"单一全局流或单一静态 SM 分区无法利用 kernel 级空洞"的洞察。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Algorithm 1（RESONATOR）的 per-kernel 路由伪代码：
```
Input: 到达 kernel k，任务 T∈{enc,llm}
State: kernel profile 表 P，SM manager SMCTRL
Streams: 每任务 wide 流 s_T^wide、narrow 流 s_T^narrow（预创建）
InitContendingMode:
  for T in {enc,llm}:
    SMCTRL.SetQuota(s_T^wide, 1.0)        # wide 流可占全部 SM
    SMCTRL.SetQuota(s_T^narrow, q_narrow) # narrow 流只占 q_narrow 比例 SM
DispatchKernel(k, T):
  τ ← P.TYPE(k)                            # τ∈{comp,mem}
  if τ = comp: s ← s_T^wide                # compute-bound 用全 SM
  else:        s ← s_T^narrow              # memory-bound/低占用 用窄 SM
  LaunchOnStream(k, s)
```
Annotations：kernel 类型由离线 profile 表给出（每 kernel 典型 SM 用量与 HBM 带宽），运行期只做查表+选流；wide/narrow 流与 SM 配额预创建，dispatch 只加一次元数据查找与流选择；因 launch 流运行期才定，contending 路径用 eager 执行而非 CUDA Graph 重放（对 compute-heavy encoder+prefill 场景 CPU launch 开销相对 kernel 时间可忽略）；narrow 流绑 SM 子集依赖 green-ctx/libsmctrl 机制。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现依赖"把 CUDA 流绑定到指定 SM 子集"的运行时支持：RESONATOR 用 green-ctx 或 libsmctrl（[26]，Bakita & Anderson RTAS'23 的硬件计算分区）——libsmctrl 通过修改 CUDA stream 内部 metadata（GPC 配置掩码）让 GigaThread Engine 只把 grid 分到 mask 内 SM，mask 更新开销 ~4us（微秒级 SM 重分区），比 GreenContext 的 context 切换便宜。使用场景：一切"两个 compute-heavy 负载共享一卡、靠 kernel 特性互补"的 GPU 服务；RESONATOR 用它实现 encoder+prefill 共存（Figure 11 消融：Stream-based Sharing 相对静态 SM Partitioning 在 TTFT 上最高 6.5× 提升、平均 1.6×，因为 compute-bound 阶段 co-scheduling 更有效）。局限：仅在 contending（compute-bound chunk）场景启用，complementary decode 路径仍用 SM 分区以保护 decode 尾延迟。

涉及论文标题：
- Symbiotic MLLM Serving: Dynamically Balancing Parallelism Across GPUs and Resources Within GPUs

## Logical Sharding（strided GEMM 逻辑分片，零开销 TP/DP 切换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Logical Sharding 是 RESONATOR 实现"零开销、Just-in-Time 并行切换"的机制，解决 canonical TP 的根本局限：canonical TP 把 transformer 层权重物理切分到多 GPU、每 GPU 只存自己的分片，改 TP 度需在网络里重分布权重数据（成本高、不可能逐 batch 适应）。logical sharding 利用 cuBLAS/CUTLASS 等现代 GPU 计算库的 strided memory access 能力——GEMM 可通过 leading dimension（ld）参数在非连续内存切片上计算：启动时把完整未分片的 encoder 模型预载到每张 GPU，运行时只改 kernel launch 参数（ld），把计算逻辑约束到想要的 1/k* 分片上，把"数据搬移问题"变成"元数据更新"。DP 执行时 kernel 作用在完整本地 tensor；TP=k* 时每 worker 用 strided GEMM 只算自己的 1/k* 逻辑分片。代价：encoder 权重全量复制到每 GPU（ViT-675M 1.3GB、MoonViT 0.8GB，A100 80GB 上 HBM 开销 1.6%/1.0%），且 strided 布局的 GEMM 内存合并/缓存局部性略差。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
logical sharding 的 GEMM 执行（Figure 16 微基准，Qwen2-VL-7B 三类 encoder 线性层）：
```
# canonical TP（materialized）：权重物理连续分片 W_shard [h, 4h/k]
Y = GEMM(X, W_shard)                       # 每 GPU 算自己的列分片，再 all-reduce
# logical sharding：完整权重 W_full [h, 4h] 预载每 GPU，只改 ld
Y = GEMM_strided(X, W_full, ld=4h, n_cols=4h/k, col_offset=worker*4h/k)
# DP 执行：kernel 用完整本地 tensor
Y = GEMM(X, W_full)
```
Annotations：ld（leading dimension）=行间内存步长；strided GEMM 通过指定 ld 与起始偏移让每 worker 只计算连续权重中的 1/k 列；因逻辑分片仍含宽矩阵 tile（分片列宽 896–7168 元素，远大于 64 元素 L2 cache line），kernel 在 tile 内保持规整向量化访存，多出的 stride 只影响行间地址计算而非内层循环——微基准显示 strided vs contiguous 中位差仅 0.7%、91% 配置 <2%（延迟与 MFU 归一化到 A100 FP16 峰值 312 TFLOPS）；少量低序列长配置有残余差距（固定 launch/寻址开销未被摊薄）。Performance Atlas 直接 profile strided 路径，布局代价已计入调度延迟估计。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：基于 cuBLAS/CUTLASS 的 strided GEMM API（ld/batch stride 参数），权重全量预载 + 控制面改 launch 参数。使用：Inter-GPU Parallelism Engine 每 batch 跑 PRISM 选好 DP/TP 计划后，统一运行时按计划给各 worker 下 ld/offset 参数即可，不触碰数据面（无 reshuffle/reload）——这正是"动态 per-batch 并行"可行的使能器。论文 §IV-H 用三类 encoder GEMM（QKV projection、FFN up、FFN down）× L_seq∈{1k,4k,8k,16k} × TP∈{1,2,4} 验证布局代价可忽略。

涉及论文标题：
- Symbiotic MLLM Serving: Dynamically Balancing Parallelism Across GPUs and Resources Within GPUs

## libsmctrl / Green Context（CUDA 流到 SM 子集的硬件绑定）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
libsmctrl 是底层 CUDA 库，运行时修改 CUDA Stream 的 SM（Streaming Multiprocessor）执行掩码：调用 libsmctrl_set_stream_mask() 把某 stream 上后续 kernel launch 限制在 GPU 特定 SM 子集，实现进程内 SM 空间分区。Green Context（green-ctx）是另一套 SM 配额机制（上下文级方案），两者都是"把 CUDA 流/上下文绑定到指定 SM 集合"的运行时支持。libsmctrl 通过直接修改 CUDA stream 内部 metadata（GPC 配置掩码）工作：stream 上发射 grid 时 GigaThread Engine 按 mask 决定把 thread block 分发到哪些 SM；改 mask 后已排队后续 kernel 自动遵从新 mask，无需重建 stream/context，更新开销 ~4us（相对 GreenContext 的 context 切换需重初始化 CUDA Graph 等资源更便宜）。RESONATOR 用它实现 Intra-GPU Sharing 的 wide/narrow stream SM 配额与 SM 分区。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
libsmctrl 绑定流的 kernel 执行流程（RESONATOR 场景）：
```
gpc_info = libsmctrl_get_gpc_info()          # 获取 GPU GPC/TPC 拓扑
# 构建 SM mask（A100: 108 SM，以 16 SM 为粒度）
mask_wide  = build_sm_mask(all_sms)           # wide 流：全部 SM
mask_narrow = build_sm_mask(sm_ids[0..q_narrow*SM_total])  # narrow 流：窄子集
libsmctrl_set_stream_mask(s_enc_wide,   mask_wide)
libsmctrl_set_stream_mask(s_enc_narrow, mask_narrow)
libsmctrl_set_stream_mask(s_llm_wide,   mask_wide)
libsmctrl_set_stream_mask(s_llm_narrow, mask_narrow)
# 运行期（contending 模式）：每 kernel 查 profile 表选流后 launch
LaunchOnStream(k, s)   # s 已是绑定了 SM mask 的流，kernel 自动只在 mask 内 SM 执行
```
Annotations：RESONATOR 的 SMCTRL.SetQuota(stream, quota) 抽象在 libsmctrl 层实现为 set_stream_mask；complementary 模式（decode 保护）也是同机制：decode 流绑 SM_dec 切片、encoder 用其余 SM；依赖特定 NVIDIA driver 版本兼容性（libsmctrl 用未文档化内部 API）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：用户空间库（Bullet 的修改版 https://github.com/zejia-lin/BulletServe；论文引 [26] Bakita & Anderson RTAS'23 的硬件计算分区）。使用：创建多条 CUDA 流、各自 set_stream_mask 后按需发射 kernel，适合"一卡内分时/分空承载多个负载"的 serving 场景——RESONATOR 的 encoder+LLM 共存（wide/narrow 双流每任务）、Bullet 的 prefill/decode SM 动态重分区（平均 4.1us repartition）。相对 MIG（多实例 GPU 硬件分区）粒度更细、切换更快；相对默认 SM 抢占式调度提供显式隔离/配额控制。

涉及论文标题：
- Symbiotic MLLM Serving: Dynamically Balancing Parallelism Across GPUs and Resources Within GPUs

## cuStateVec（NVIDIA GPU 量子态矢量模拟 kernel 库）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- cuStateVec 是 NVIDIA cuQuantum SDK（Bayraktar et al., QCE 2023）的核心组件之一：GPU 上高性能态矢量（statevector）模拟的 C/C++ 库，提供稠密/稀疏矩阵作用、张量积、测量、采样等 kernel（如 cuStateVecApplyMatrix、cuStateVecMeasure、cuStateVecBatchApplyMatrix），是 Qiskit-Aer GPU 后端与 CUDA-Q statevector 后端的底层模拟引擎。本论文用 cuStateVec v1.12.0 作为 TUSQ 的 SVS kernel 后端（backend-agnostic，用户可换任意模拟 kernel）。
- 在 TUSQ 中的角色：ECM+DFTT 决定"何时对哪个态矢量乘哪个门"，实际矩阵向量乘由 cuStateVec 在 GPU 上执行；DFTT 的 compute（乘 U）与 uncompute（乘 U†）都映射到 cuStateVec 的 apply-matrix kernel；多 GPU 时按子向量切分态、跨 GPU 门搬移振幅到单 GPU 运算。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- TUSQ 调用 cuStateVec 的 kernel 调度过程（一次 DFTT 边遍历）：
  ```
  // DFTT 树遍历逻辑（CPU）驱动 GPU kernel（伪代码）
  for edge in dfs_order(tree):
      if 正向: cuStateVecApplyMatrix(handle, gate_matrix, state, ...)   // |ψ'> = U|ψ>
      else:    cuStateVecApplyMatrix(handle, gate_matrix_dagger, state, ...) // uncompute：U†
  // 叶子处：cuStateVecSample / cuStateVecMeasureBatch 按 |ψ[i]|^2 采样 s_i 次
  ```
- kernel 语义：单比特门作用 2^n 元素（每个元素独立更新，GPU 并行度 2^n）、双比特门作用 4·2^n 元素（本论文操作计数 1/4）；TUSQ 通过与 Qiskit/CUDA-Q 相同的 cuStateVec v1.12.0 保证对比公平——加速全部来自算法层（冗余消除与树遍历复用），而非 kernel 本身。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：cuQuantum SDK 经 `pip install cuquantum-python` 或 C API 集成（https://github.com/NVIDIA/cuQuantum）；TUSQ 在 NERSC Perlmutter（NVIDIA A100 40GB）上以 CUDA_VISIBLE_DEVICES=0 单 GPU 运行，后端 kernel 为 cuStateVec v1.12.0，30-qubit Adder ×10^6 shots 约 820s（baseline >10 小时）。对比：Qiskit 2.1.0 与 CUDA-Q 0.11.0 的 baseline 后端同样是 cuStateVec v1.12.0，TUSQ 平均 59.06×/13.38× 加速归因于算法。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation

## cuTensorNet（NVIDIA GPU 张量网络模拟 kernel 库）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- cuTensorNet 是 NVIDIA cuQuantum SDK 的另一核心组件：GPU 上高性能张量网络（tensor network）模拟与收缩（contraction）库，支持 MPS/MPO 张量网络态模拟（含 bond dimension 截断）、张量网络收缩路径规划（contraction path finding）与量子线路到张量网络的转换，是 CUDA-Q tensornet-mps 后端的底层引擎。本论文用 cuTensorNet v2.9.1 作为 TUSQ 的 TNS kernel 后端。
- 在 TUSQ 中的角色：TUSQ 的 ECM+DFTT 只依赖"矩阵向量乘 + 从向量采样"，因此可直接叠加在 TNS 上——DFTT 树遍历的每条边是张量网络上的门收缩，uncompute 是对应逆收缩；输出向量供频率加权采样。TNS+TUSQ 对 40-qubit 电路（bond dimension=16）平均 248.39× 加速于未优化 TNS。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- TNS+TUSQ 的 kernel 调度过程：
  ```
  // 每个待模拟电路实例（ECM 输出）在 cuTensorNet 上计算（伪代码）
  tn = 把电路转化为张量网络（MPS 表示，bond dimension ≤ D）
  for edge in dfs_order(tree):
      if 正向: state = cuTensorNetContract(tn, gate_tensor)   // 门收缩，truncate 到 D
      else:    state = cuTensorNetContract(tn, gate_dagger)   // uncompute 逆收缩
  输出向量 → 按频率加权采样
  ```
- 对比：未优化 TNS（CUDA-Q tensornet-mps）每个电路实例独立从头收缩（时间随 shots 线性增长，100/1k/10k shots 外推），QFT40/Adder40/QAOA40(p=2) 分别 1119642/628889/158407 秒（40 小时超时未完成）；TNS+TUSQ 3444/2625/805 秒完成。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：cuTensorNet 集成于 cuQuantum SDK（https://github.com/NVIDIA/cuQuantum，pip install cuquantum-python）；CUDA-Q 通过 `--target tensornet-mps` flag 调用做未优化 TNS baseline。TUSQ 场景：ECM 在 CPU 预采样 ER/剪枝（与 SVS 相同），DFTT 在 GPU 上调度 cuTensorNet 收缩；bond dimension=16、100k shots、α=0.01、β=100。适用性：TNS 针对 SVS 内存 O(2^n) 不可行的大/深电路，与 TUSQ 冗余消除正交。

涉及论文标题：
- TUSQ Tracking, Uncomputation, and Sampling for Noisy Quantum Simulation

## BLIMP 数据库算子内核（select/semijoin/join/aggregate kernel 与异构编排）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BLIMP 数据库算子内核是运行在 DDR bank 内 200MHz RISC-V（-V）核上的 OLAP 算子内核，与 host CPU 形成异构编排：select（FILTERTOBITVECTOR，位图输出）、hash semijoin/join（SEMIJOINPROBE，探测侧）、aggregate（非分组/分组）。内核以 1KB row buffer 粒度访存，执行 fetch-read-apply-store 流程：读入一行元素 → 逐元素应用谓词/哈希/聚合（BLIMP-S 串行、BLIMP-V 用 32×64b vALU 向量化）→ 结果 coalesce 打包进位图或输出区。host 负责建哈希表、切分 PIMDT chunk（适配 32MB bank）、relayout 载入与结果编排；跨 bank 无直连，数据交换走 host 读-写中转。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
select 内核（Algorithm 1 FILTERTOBITVECTOR）伪代码：
```
procedure FILTERTOBITVECTOR(in *d, out *b, func p)   # d=PIMDT 分区, b=位图, p=谓词, w=元素宽
    v2 ← [0,...]; eproc ← 0
    for each data row r in d:                        # 逐 1KB row buffer 行
        v1 ← FetchMem(r)                             # 读入整行（如 1024 个 8-bit 或 256 个 32-bit）
        v1 ← apply(p, v1)                            # BLIMP-S 串行 / BLIMP-V 向量化逐元素谓词
        v1 ← coalesce(v1, w, mod(eproc, 8192))       # 布尔结果打包进位图行位置
        v2 ← v1 ∨ v2                                 # 累积位图
        eproc ← eproc + ElementsPerRow(w)
        if mod(eproc, 8192) = 0 then                 # 每 8192 元素对应一个位图行 HitmapRow
            StoreMem(v2, b[HitmapRow(eproc)]); v2 ← [0,...]
    if mod(eproc, 8192) ≠ 0 then StoreMem(v2, b[HitmapRow(eproc)]); ZeroMaskRemainder(...)
```
semijoin 内核（SEMIJOINPROBE）：对每行元素 v1[i]，先向量化 hash 得桶索引 BucketRow(hash(v1[i]))，再串行探测——FetchMem 该 row buffer 桶、检查 key 是否在桶 slot 列表、命中则置位、否则沿 BucketNext 取链上下一桶（新 row buffer 读）直到 IsNull 或 hit。join 大哈希表时按分区多轮 build-probe（每轮 offload 一个分区、结果 OR 到下一轮）。调度约束：BLIMP 一次只能执行一个算子（不像 host 可并行多 kernel），compute mode 下 bank 数据对 host 锁定，下游/并行相邻操作必须 stall（host 可趁机做无关工作）。性能数据（10 亿值列）：select 位图 2.0×/12.9×（BLIMP-S/-V），值输出 2.0×/4.2×；semijoin 1.4×/2.1×、join 2.1×/3.0×；SUM 非分组 2.1×/33.7×、分组 1.9×/2.1×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现与评估：论文用 validated cycle-level simulators（prior work [25] + riscvovpsim/Imperas + DRAMSim2 DDR4 时序）模拟单 bank 内核执行（假设对称计算、取最慢 bank 周期），host 侧等价 kernel 在真实 2× Xeon Silver 4114（40 线程 AVX512 手调 C++）上计时；算子级评估只测 PIM 域计算（不含 relayout/build/加载/post-processing），端到端 SSB SF100 评估叠加全部开销。使用：查询执行器对 PIMDT 列上的算子识别是否支持，支持则走"预处理（切分/建表）→ relayout 载入 → 执行 → 部分物化/原位保留 → host 取回"工作流，不支持则回退 host；host 哈希表 build 时间与 CPU 侧 Swiss Table 相当（论文引 abseil）。内核与仿真脚本开源情况论文未明确说明，可依托同组 dovedevic/blimp（https://github.com/dovedevic/blimp）框架。

涉及论文标题：
- Taking Analytic Databases to the Bank

## 图原生 Push/Pull 数据流（Graph-Native Push/Pull Dataflow，外积/内积/Gustavson 张量收缩推广）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
数据流决定"谁驻留、谁流动、谁广播"——即循环嵌套中稀疏/稠密操作数的数据移动策略。三类经典 SpMM 数据流：Pull/Inner-product（输出驻留点积，输出复用高但稠密输入重复取）；Push/Outer-product（稠密行 B[k,:] 广播与相关非零外积累加 $Partial\_C_{M,L}=\sum_k A_{:,k}B_{k,:}$，输入复用高但部分和散布需大量缓冲/同步）；Gustavson/row-wise（每次取一条稀疏行并流式取对应稠密行累加，平衡输入输出局部性）。TensorPrism 的图原生数据流（§V）不是固定单一模式，而是按共现图遍历顺序动态切换：收缩模式顶点 PUSH 稠密行到目标顶点集（等效外积）、自由模式顶点 PULL 从源顶点拉特征累加（等效内积/row-wise），且一个收缩顶点向多个不同模式的自由顶点集广播实现 inter-mode（跨模式）复用——超出 2D 空间的表达能力。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# TensorPrism PE 内收缩引擎执行一个分区（图遍历序）
for clique (I,J,K) in partition P_i:    # 非零元素
    # 收缩顶点 K PUSH: feed unit 广播稀疏输入 A[I,J,K] 给 8 个 MAC
    # 寄存器堆供稠密行 B[K,:] (32 FP32), 多累加器存不同部分和
    partial_C[I,J,:] += B[K,:] * A[I,J,K]   # 标量-向量乘+向量累加
# 列向遍历完成后转 PULL: 自由顶点从剩余源顶点拉特征累加输出行
```
例（Fig.7）：contraction 顶点 K0 向目标集 {I2,J0}/{I0,J0} PUSH（B[K0,:] 与稀疏切片 A[J0,:,K0] 逐非零乘、部分积存 C[J0,:,:]）；K2 向 J1/J2（不同模式）目标集广播实现 inter-mode 复用。硬件支撑：feed unit 8 路广播+连续周期重发（空间×时间复用，最高 128× 复用/取数）；PUSH 写不同输出地址→无写冲突免同步。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：PE 微架构（fetch unit 分片缓存稠密行、ring 跨单元转发；feed unit 广播；寄存器堆单端口 SRAM+cache；多累加器；commit unit+MAG 映射输出地址）+ CoG Scheduler 按式 6 划分后按分区分派。相比 baseline：inner-product 牺牲输入局部性、outer-product 需昂贵部分和同步（GSpTC 用 outer 在 chcr 上归约竞争占 73% 执行时间）、Gustavson 平衡但限 2D；TCP 编译期固定数据流+电路交换网络无法适配不规则（power-of-2 padding 浪费 2.89× 带宽）。图数据流动态切换使吞吐量平均 2.07×/1.71×/1.55×（vs GSpTC/TCP/SPADE&HotTiles），nel1 上 2KB feature 时 2.95×。

涉及论文标题：
- TensorPrism: Rethinking Sparse High-order Tensor Acceleration via Co-occurrence Graph

## Ring Attention（环注意力，含 zigzag 交错 / striped 分区变体）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Ring Attention 是 Ring-style（环形）序列并行注意力 kernel/通信模式：把一条长序列按 token 均分到多个 GPU，GPU 排成逻辑环，每个 GPU 持有本地 token 的 Q 分片与 K/V 分片；每一步计算本地 Q 与当前 K/V 的局部 attention（含局部 softmax 统计量），然后把 K/V 传给下一个 rank、从上一个 rank 接收新 K/V，经过 H-1 轮（H=GPU 数）后每个 token 与完整序列的 K/V 交互完毕。计算与通信可重叠（计算当前块时异步发送前一块）。关键变体：(1) Striped Attention——把序列按等间隔 stripe 轮转分给各实例（每实例可对每个 KV shard 计算，缓解因果掩码负载不均）；(2) Zigzag Ring Attention（zigzag 交错）——把序列切成 2N 个 shard S0..S2N-1，实例 i 分配 (Si, S2N-i-1)，使各实例计算量相等（负载均衡）。Tetris（ISCA'26）在 prefill 阶段扩展 Flash Attention 支持 zigzag ring attention，并用 NVSHMEM 降低 ring 通信开销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Zigzag Ring Attention prefill（N 个 SP 实例，序列分 2N shard）
rank i 本地持有 shard S_i 与 S_{2N-1-i} 的 Q/K/V
for step = 0 to N-1:
    o_i, lse_i = FlashAttention_zigzag(Q_local, K_cur, V_cur)   # 本地 partial attention（在线 softmax）
    Send(K_cur, V_cur) -> GPU_{(i+1)%N}; Recv(K_cur, V_cur) <- GPU_{(i-1)%N}   # P2P ring
    # 通信与下一轮计算重叠（NVSHMEM async put / NCCL P2P）
# 结束时各 GPU 已 attend 全部 K/V，合并 partial softmax 统计量得到精确注意力输出
```
Annotations: Q_local 为本地 token 的 query；K_cur/V_cur 为当前轮持有的 KV shard；o_i/lse_i 为 partial output 与 log-sum-exp（online softmax 用）；ring 步数=N；因果掩码下 zigzag 交错使各实例工作量相等。
Tetris 中的跨 chunk 扩展：每个 chunk 内用 zigzag 交错保证均衡；计算新 chunk 前把前序 chunk 的 KV cache 在 zigzag 布局下均匀重分布到当前实例组（每个 Pi 只需把后半段 KV 传给 P4+i），复用 ring communicator 与下一层 prefill 跨层重叠（cache balancing 隐藏）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Ring Attention（Li et al. ACL'23）、ring-flash-attention 开源库、Striped Attention（arXiv:2311.09431）、ZigZag Ring Attention；Tetris 在 vLLM/PyTorch 后端扩展 Flash Attention 的 zigzag 变体 + NVSHMEM one-sided put 做 ring 传输（NVLink intra-node / InfiniBand inter-node），decoding 阶段 ring 只传 Q（体积小，减少通信）。使用场景：长上下文 prefill/decode 跨多 GPU 的上下文并行（CP/SP）；局限：ring 步数随 GPU 数线性增长、所有实例须同步开始（单实例延迟会拖累整环），SP 大小变化产生资源碎片（Tetris 的 CDSP 正是为解决此点）。Web 证据：Ring Attention 论文与 ZigZag 说明（上下文并行常见实现）。

涉及论文标题：
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism

## Sequence Parallelism（SP，序列并行 / 上下文并行 CP）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sequence Parallelism（SP）是把 transformer 输入序列沿 token 维度切分到多个 GPU 的分布式并行策略，用于满足长上下文请求的计算与内存需求（KV cache 与 attention 激活不放在单卡）。与 TP（沿 hidden/head 维切权重）、PP（沿层切）、DP（沿 batch 切）正交。两类主流实现：DeepSpeed-Ulysses 式（attention 前后两次 All-to-All 在 sequence layout 与 head layout 间切换）与 ring-attention 式（P2P 环形传递 KV，即 CP/context parallelism）。在 serving 中 SP 的核心优势：调整 SP 大小只重分 token、不需要重分片模型权重（TP 调整需 reshard 权重、挂起设备），且跨节点扩展优于 TP（TP all-reduce 对低带宽网络敏感）。局限：SP 环要求实例同步，大 SP 下短请求计算量不足、无法掩盖 ring 通信，性能反而劣于小 SP。Tetris（ISCA'26）在 prefill 阶段用 SP 池（TP=1 统一 SP 组）最大化资源分配灵活性，decoding 用大 TP 的 DP。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# ring-attention 式 SP（N 个实例，序列 S token）
# 每实例持有 S/N token 的 Q 分片 + 本地 K/V 分片
for step in 0..N-1:
    # 本地 partial attention（Flash Attention）与 KV ring 传输重叠
    o += FlashAttn(Q_local, K_cur, V_cur)
    K_cur,V_cur = ring_sendrecv(K_cur, V_cur)   # P2P
# 非 attention 算子（MLP/LN）各 GPU 只算自己的 token 分片，无通信
```
Annotations: 每实例计算量 S/N × d（attention）+ 自身 token 的 FFN；通信量每步 O(KV chunk)，总 O(N·KV)。
Tetris 的调度粒度：SP 大小（1/2/4/8/16，2 的幂）作为 CDSP 调度的候选，实例组扩展按"先节点内后跨节点"策略（GetGroup），prefill 统一 SP 池的 SP 大小在集群初始化（initialize_model_parallel）时显式配置建立 ring communicator。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Megatron-LM sequence_parallel（Korthikanti et al.，配合 TP 消除冗余 LN/Dropout）、DeepSpeed-Ulysses（All-to-All）、ring-flash-attention / Context Parallelism（Meta，P2P ring）、LoongServe 的 ESP（动态 SP）、Tetris 的 CDSP（chunk 级动态 SP）。使用：长上下文 serving/training 中作为与 TP/DP/PP 组合的 4D 并行之一；serving 里 SP 大小可随请求动态调整（无需权重重分片）。Web 证据：Context Parallelism 文档（Meta）与 DeepSpeed-Ulysses 论文确认两类实现与通信模式。

涉及论文标题：
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism

## Flash Decoding（闪速解码注意力 kernel）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Flash Decoding 是 FlashAttention 团队提出的长上下文 decoding 优化 kernel：把 attention 计算沿 KV 序列长度维度拆分（split）到多个 SM 并行执行，各 SM 算部分 KV 块的 partial attention（含 partial logsumexp），最后通过 log-sum-exp reduction 合并各 partial results 得到精确注意力输出。与 FlashAttention 的区别：FlashAttention 针对 training/prefill 的单 query-多 KV 并行（一个 thread block 处理一个 query block + 逐步滚动 KV），解码时 query 只有 1 个、KV 很长，一个 thread block 顺序滚完整个 KV 会浪费并行度；Flash Decoding 用多个 thread block 并行扫不同 KV 段，吞吐接近同时处理所有 KV 段。Tetris（ISCA'26）在 decoding 阶段采用 Flash Decoding 计算 attention，配合 CUDAGraph 消除 kernel launch 开销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Flash Decoding（1 query、KV 长 L、split 到 M 个 thread block）
for m in 0..M-1:                    # 每个 block 并行处理一段 KV
    o_m, lse_m = scan(Q, K[m·L/M:(m+1)·L/M], V[...])   # 段内 FlashAttention 式滚动 + online softmax
# 合并：lse = logsumexp(lse_0..lse_{M-1}); O = Σ_m exp(lse_m - lse) * o_m
```
Annotations: 每 block 独立扫描一段 KV，无跨 block 通信；合并阶段按 online softmax 规则加权，保证与顺序扫描数值等价；M 增大摊平单 block 顺序扫描延迟。
在 Tetris 中与分布式 decoding ring 配合：每个 decoding 实例作为部分请求的 master，Flash Decoding 负责本实例内多请求 batch 的 attention；CUDAGraph 把 decode 的 kernel 序列（含 Flash Decoding）录制为图，逐 token 重放消除 launch 开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：FlashAttention 2.6+ 集成（decode kernel）、vLLM/SGLang/TensorRT-LLM 默认 decode attention、SeerAttention-R 的 Block Sparse Flash Decoding 变体（跳过无效 KV blocks）；Tetris 中作为 decoding 计算组件（A100 集群，配合 Flash Attention prefill）。使用：长上下文 decode 阶段（KV 长、batch 小）的标准 kernel；支持 GQA（多 query head 打包同一 thread block）与 PagedAttention 兼容。Web 证据：Flash Decoding 官方 blog（Dao et al.）与 FlashAttention 2.6 release notes。

涉及论文标题：
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism

## NVSHMEM（GPU 对称内存 / PGAS one-sided 通信库）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NVSHMEM 是 NVIDIA 基于 OpenSHMEM 标准的 GPU 集群通信库，提供 Partitioned Global Address Space（PGAS）编程模型：所有 GPU 显存被抽象为统一全局地址空间（对称内存 symmetric memory——各 GPU 分配相同大小、相同虚拟地址的内存），GPU 通过 one-sided put/get/atomic 直接读写远端 GPU 对称内存，无需远端显式参与，也无需 CPU 中转；配合 GPUDirect（IBGDA）让 GPU 直接访问 RDMA 网络。相比 NCCL collective，NVSHMEM 支持 kernel 内 fine-grained、device-initiated 通信（如 nvshmem_putmem_nbi + 信号量信号）。Tetris（ISCA'26）用 NVSHMEM 实现 ring attention 的 KV 传输，降低 ring 通信开销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Tetris ring attention 的 KV 传输（rank i → rank i+1）
nvshmem_malloc(&sym_kv, size)          # 初始化对称内存（每 rank 同址）
nvshmem_putmem_nbi(peer_sym_kv, local_kv, size, peer=i+1)   # one-sided put（非阻塞）
nvshmem_fence(); nvshmem_quiet()       # 完成语义
# 接收侧 nvshmem_signal_wait_until 轮询信号，无需 rank i+1 显式 recv
```
Annotations: putmem_nbi 为非阻塞 bulk put；对称地址使收发双方无需地址交换；信号/等待原语提供 completion 通知；one-sided 消除 two-sided MPI/NCCL 的 CPU 参与与同步配对。
在 Tetris 中：ring attention 每步把 K/V chunk 直接 put 到下一实例的对称 buffer 与下一轮 attention 计算重叠；相比 NCCL P2P，kernel 内 fine-grained 传输更易与 Flash Attention zigzag 计算融合/重叠。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：NVIDIA NVSHMEM 库（https://developer.nvidia.com/nvshmem，OpenSHMEM 规范）；常见用法——初始化（nvshmem_init/nvshmem_malloc）→ kernel 内 put/get + 信号同步 → 完成（nvshmem_quiet/fence）。生态：DeepEP（DeepSeek MoE 通信库）构建于 NVSHMEM 之上、FlashMoE 用其做单 kernel 分布式 MoE、Comet/Mirage 等 fused kernel 用其做 kernel 内通信。使用场景：需要 device-initiated / fine-grained / 与计算融合的跨 GPU 通信（MoE dispatch-combine、ring attention、replicated expert 预取）。Web 证据：NVSHMEM 官方文档（docs.nvidia.com/nvshmem）与各 MoE 论文实现说明。

涉及论文标题：
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism

## CUDAGraph（CUDA 图捕获 / 图回放）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CUDAGraph（CUDA 10 引入）允许把一系列 GPU kernel launch、memory copy、memory allocation 预录制为一个有向无环图（DAG），后续通过单次 cudaGraphLaunch 回放整图，消除逐个 kernel launch 的 CPU-GPU 同步开销与 CUDA driver 调度开销。生命周期三阶段：Graph Construction（Stream Capture 或显式 API 构建节点+依赖）→ Instantiation（编译为可执行图 cudaGraphExec_t）→ Launch（每输入重复回放，仅替换输入 buffer 指针）。核心约束：capture 时 kernel shape/launch 配置必须固定（静态输入 shape），动态 batch 需按 bucket 预编译多张图。Tetris（ISCA'26）在 decoding 阶段用 CUDAGraph 消除 kernel launch 开销（decode 每 token 都跑同一组 kernel）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Tetris decode 的 CUDAGraph 使用（vLLM 风格）
for bucket_b in [1,2,4,8,16,32]:
    graph[b] = cudaStreamBeginCapture(stream)      # 录制 decode 一步的全部 kernel
    forward(decoding_batch(b))                     # Flash Decoding + 各算子
    cudaStreamEndCapture(stream, graph[b])
    exec[b] = cudaGraphInstantiate(graph[b])
# 运行时每 decode iteration：
cudaMemcpyAsync(input_buffer, tokens, ..., stream)  # 替换输入
cudaGraphLaunch(exec[bucket(batch_size)], stream)   # 单次提交整图
```
Annotations: 每 batch bucket 一张图（动态 batch 的近似）；输入/输出用固定 buffer 地址（kernel 读新数据只需更新 buffer 内容）；回放省去每 kernel 的 host launch 与同步。
在 Tetris 中：decode 每 token 的 kernel 序列（Flash Decoding attention、FFN、采样）录制为图逐 token 重放，与 Flash Decoding 共同压低 decode 单步延迟（对 TBT 指标关键）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：CUDA Runtime（cudaStreamBeginCapture/EndCapture + cudaGraphInstantiate + cudaGraphLaunch）、PyTorch torch.cuda.CUDAGraph、vLLM 的 CUDA Graph compilation framework（按 batch size 预编译 decode 图）。使用：decode 阶段 kernel launch 开销占比高的场景（小模型、短 kernel、高吞吐 serving）；约束——需静态 shape/固定 buffer，动态控制流需 host 侧分支或 persistent kernel（FlashInfer 用 persistent kernel 兼容 CUDAGraph）。Web 证据：CUDA Graph 官方文档与 vLLM 设计文档确认 capture/instantiate/launch 生命周期。

涉及论文标题：
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism

## 中断亲和（Interrupt Affinity / effective_affinity）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 中断亲和是 Linux 中把每个 IRQ 绑定到特定核子集的机制。虽然名义上的 affinity mask 可包含多个核（默认所有核），但内核最终通过 effective_affinity 把每个中断绑定到单一核；effective_affinity 的更新是事件驱动的，可由 irqbalance 等守护进程根据运行条件和用户配置的 affinity mask 触发。结果：即使 mask 允许多个候选核，同一时刻中断投递也集中在单一核上。
- 在本文中：中断亲和是 Linux/x86 与 Arm 上 SPI 投递的默认行为（SPI 恒定路由到 affinity 指定的核，无论该核是否 idle）；Apple silicon + macOS 经 TIDE 反推证明**不采用**中断亲和（Observation 1），SPI 被均匀投递到所有 active core——这一差异使基于"监听固定核"的既有攻击假设失效，也让 Mwait/IdleLeak 类依赖 idle 唤醒的攻击不可行。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- Linux 中断投递流程（对比）：
```
# Linux（x86/Arm）：
#   设备产生 IRQ → 中断控制器 → 按 effective_affinity 选单一核 → 该核处理
#   /proc/irq/<irq>/smp_affinity 可配置 mask；irqbalance 按负载/拓扑动态改 effective_affinity
#   ⇒ 攻击者必须先确定目标中断被路由到哪个核，再在该核上检测
# Apple silicon + macOS（AIC）：
#   设备产生 SPI → AIC → 均匀投递给所有 active core（idle core 忽略）
#   ⇒ 攻击者任意 active core 上运行 TIDE 即可捕获 victim 中断，无需 core 绑定
```
- 论文验证（efficiency 指标 = 生成的 SPI 请求数 / 成功检测到的中断数）：默认设置下各机均可检测到三类 SPI；Mac mini 2023（8 核）上把 sender/receiver 用 CoreBinder 钉到不同核遍历全部 56 种组合，效率稳定在 10.73±0.20，与 sender 所在核无关——证明"均匀投递"而非"固定核亲和"。把 receiver 换成 idle 进程后，检测到的中断数随 active receiver 数成比例下降（idle 核被忽略）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Linux 内核 IRQ 子系统（smp_affinity / /proc/irq/）、irqbalance 守护进程；Apple 侧为 AIC 硬件均匀广播（无此配置接口）。使用：性能调优（把网卡中断绑定到专用核，如 Mellanox ConnectX 127 个 IRQ 绑定独立核避免与 memcached 争抢）、防御（把 SPI 绑到非攻击者核）。限制：macOS 用户态无 affinity 接口（需 CoreBinder kext 才能钉核）；Apple 的均匀广播使"绑定防御"失效。

涉及论文标题：
- Towards Practical Interrupt Side-Channel Attacks on macOS for Apple Silicon

## Counting-thread 定时器（自建线程计数定时器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Counting-thread 定时器是在架构定时器不可用（timer-constrained）场景下攻击者自建的计时器：一个独立线程在紧循环中自增共享内存中的计数器，计时通过读写该计数器实现——计数器增量近似流逝时间。要求并发、不中断执行与共享内存资源。它比架构定时器（如 cntvct_el0、rdtsc）更不精确、更嘈杂，但能在定时器被限制/防御的环境下"复活"计时类攻击（SysBumps、S2C、ARMageddon 等使用）。
- 在本文中：counting-thread 定时器作为 TIDE 的对照基线：TIDE 检测到 100,000 个中断时，counting-thread（阈值 2,000）只检出 45.0%（M3 Pro）/62.9%（M1 Pro）/80.5%（M3 Air），不可靠；且它本身是一个 active 线程，会改变 Apple 中断投递（增加 active core 数）。论文用 TIDE 增强它（见下）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 定时线程与测量流程：
```
# 定时线程（独立核）：
while (1): counter++
# 测量线程：
t0 = counter;  <被测代码>; t1 = counter    # 用时 ≈ (t1 - t0) / 每秒增量
# 中断发生：定时线程所在核被抢占 → counter 停止增长 → t1-t0 偏小 → 误判/噪声
```
- TIDE 增强版（去噪）：
```
# 定时线程每个循环迭代：
if x18 == 0:            # x18 被清零 → 本核被中断过
    tide_counter += 1   # 记录中断次数
    x18 = 0x1           # 重置 x18
counter++
# 测量时同时读 counter 与 tide_counter，把被中断的测量丢弃
```
- Annotations：中断会让定时线程暂停，因此任何"被中断的测量"都偏短；TIDE 增强只在检测到中断时额外执行寄存器写+内存写，开销仅 0.06%（100 runs，M3 Air 空闲环境）；在视频噪声下把 SysBumps 成功率从 54% 提升到 81%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：用户态 C 线程 + 共享内存计数器（无需特权）；SysBumps 攻击（macOS KASLR 破解）与缓存侧信道（Prime+Probe）使用；SSBench 论文中亦作为软件计时基线（与 kperf 检测的对抗场景）。使用：测量被测代码执行时间、探测缓存/TLB 状态。限制：要求定时线程所在核不被中断（在 Apple silicon 上 SPI 均匀投递使该假设更难成立）；增加 active core 数会改变中断分布；比架构定时器精度低一个量级。

涉及论文标题：
- Towards Practical Interrupt Side-Channel Attacks on macOS for Apple Silicon

## SysBumps（基于共享 dTLB 投机执行的 macOS KASLR 破解）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SysBumps（Jang et al., CCS 2024）是破解 macOS 13.1–15.1 KASLR 的攻击：虽然 Apple 的 Double Map 隔离了用户页表与内核页表，但 data TLB（dTLB）跨特权级共享。当对已映射物理地址触发投机执行时，架构效应会在共享 dTLB 中驱逐 priming 的条目（投机结果本身被隔离，但 TLB 效应可见）。攻击用 Prime+Probe 技术 + counting-thread 定时器测量 probe 时间，遍历 32,768 个候选槽位，判定每个地址是否有有效物理映射，从而去随机化内核基址。
- 在本文中：SysBumps 作为 TIDE 去噪原语的验证载体——原版 SysBumps 在 idle 下成功率 92%（与论文 95.7%–98.8% 相当），在 Chrome v130 播放 YouTube 短视频造成的中断噪声下降到 54%；用 TIDE 增强定时器丢弃/重测被中断污染的 Prime+Probe 测量后提升到 81%。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- Prime+Probe 于共享 dTLB 的测量流程：
```
# 对每个候选内核地址 slot（共 32768 个）：
for slot in candidates:
    Prime:    访问 priming 地址集，把 dTLB 条目填满
    Spectre:  在系统调用内触发对 slot 的投机访问（对有效映射会产生 TLB 驱逐）
    Probe:    重访 priming 地址集并计时（counting-thread），被驱逐则重访慢
    判定:     probe 慢 ⇒ slot 有有效物理映射 ⇒ 内核基址候选
```
- 中断噪声破坏点：中断可能发生在 Prime 阶段（priming 条目被打乱）或投机执行阶段（内核上下文切换本身做大量 dTLB 访问），导致 probe 时间失真。TIDE 去噪：在测量前后各一次 x18 写/读，若 x18 被清零（本核被中断）则丢弃该 slot 测量并重做。剩余差距来自无中断的 dTLB 级噪声（视频播放本身的内存活动），无法由中断过滤消除。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：SysBumps 开源代码（论文原样部署、不做优化）；本论文在其上叠加 TIDE 增强 counting-thread 定时器。使用：KASLR 去随机化 → 为进一步内核攻击铺路。指标：成功率（100 runs 下 92% idle / 54% 噪声 / 81% TIDE 增强）。限制：噪声源不止中断（dTLB 干扰无法完全消除）；依赖共享 dTLB 的架构事实。

涉及论文标题：
- Towards Practical Interrupt Side-Channel Attacks on macOS for Apple Silicon


## 在线激活/KV 量化 kernel（crest factor 计算 + 与 GEMM 执行重叠）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 在线激活/KV 量化 kernel 是量化 LLM 推理中负责把运行期产生的激活（以及 KV cache 的 K/V）按 group 量化到低比特浮点格式的运行时计算 kernel。UNICORE 论文（ISCA'26）在硬件模拟器层面评估其开销：激活量化在 Llama-2-7B 上占 prefill 时延 7.1%–20.7%、decode 仅 0.3%–1.6%（序列 512–8192），且可大部分与 GEMM 执行重叠。UNICORE 的量化 kernel 额外融合 crest factor（CF）计算（max-abs 归约 + RMS 归约 → κ=峰值/RMS → 阈值查表选 DynFP E/M 布局），使算术强度（arithmetic intensity）从 0.63 提到 0.87（额外 reduction、sqrt、division），但仍 memory-bound、无可见开销；对 L≥2K 序列 CF 计算占 QKᵀ FLOPs 不足 0.2%。
- 从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 在线量化 kernel 伪代码（每 group 流式处理）：
    ```
    # 对每个 group 的激活张量 g（32 元素 group）
    # 单趟 streaming 归约：
    max_abs = max(|g|)                 # reduction 1：峰值
    sq_sum  = sum(g*g); rms = sqrt(sq_sum/n)   # reduction 2 + sqrt：RMS
    kappa   = max_abs / rms            # crest factor
    layout  = cf_threshold_map(kappa)  # 阈值查表 → DynFP E/M 布局
    scale   = max_abs / max_rep(layout)  # 8-bit scale
    q       = dynfp_quantize(g, layout, scale)  # 量化输出 + 格式索引/scale 元数据
    ```
  - 调度/重叠：量化 kernel 与后续 GEMM 在硬件流水上重叠执行（prefill 中激活量化占时延 7.1%–20.7% 但大部分重叠、decode 中占 0.3%–1.6%），量化后的低比特激活/缓存以原始格式存储传输（E3M2 仅存在于 UNICORE 计算数据通路内）；CF 计算使算术强度 0.63→0.87 但仍 memory-bound（QKᵀ FLOPs <0.2%）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：UNICORE 中该 kernel 运行在加速器 Vector Unit/在线量化数据路径（artifact Software/Accuracy/ 的 unicore_kernel 提供 PyTorch 实现），与 GEMM 在硬件流水上重叠；K/V 与 softmax 输出 P 用与激活相同 group size/位宽量化，仅 K、V 做 crest factor 在线格式选择。使用：在 prefill/decode 全流程对激活（及 KV cache）逐 group 量化，配合离线权重量化（贪心 palette 搜索）形成完整 W/A/KV 低比特推理管线；其 memory-bound 特性使量化开销在带宽受限的 decode 阶段几乎可忽略。开源：https://github.com/CLab-HKUST-GZ/isca53-unicore。

涉及论文标题：
- UniCore: A Bit-Width Scalable GEMM Unit for Unified LLM Inference

## Interleaved-Fusion（IF）交错融合映射策略（HMUX 融合分组与 chiplet 间交错映射）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- IF 是 CASCADE 的 HMUX 任务映射策略（论文 IV 节）：把 BSP 的 n 个 HMUX 划分为若干"连续 HMUX 的融合组"，再把这些组按循环时空顺序交错映射到 C 个 HMUX Chiplet（HC）。两步：(1) 划分——把 n 个 HMUX 分成 k 个连续组 (G_0…G_k)；(2) 交错——组按 t mod C 循环交错到 chiplet（G_0→C_0、G_1→C_1……），用二维时空矩阵 f(t,c) 表示。动机：naive 把每个 HMUX 单独跨 chiplet 交错最大化并行时，ICT（中间密文传输）使 D2D 通信成为瓶颈（D2D 时延 > HMUX 计算时间 → HC 严重欠利用），且 inter-HC batching 按比例增加跨 chiplet ICT 总量、无济于事；IF 把连续 HMUX 融合在本地执行，使组内 ICT 留在单个 chiplet（降低 D2D 通信频率），同时组间交错保持流水并行。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 融合组在 HC 内的执行（Fig.8，组内 2 个 HMUX）：RLWE1 遍历 Rotation→Decomp→FFT→VMA→IFFT 完成 HMUX_1，输出回馈 HC 输入再遍历同一功能单元完成 HMUX_2——因功能单元是 PCG 系数粒度流水，两次遍历在时间上重叠（一个 HMUX 时延≈最长流水级）；为避免功能单元气泡，注入多个 RLWE（intra-HC batching）让不同密文的计算重叠。
- 映射调度伪代码（f(t,c) 的构建，n=17、C=4 示例）：
```
# 目标：Σ|f(t,c)| = 17，最小化 T_task = T_run + T_bubble
f = 2D 矩阵 (t=0..T-1, c=0..3)
t0: f(0,0)=H0,H1   f(0,1)=H2,H3   f(0,2)=H4,H5   f(0,3)=H6,H7   # 组尺寸 2
t1: f(1,0)=H8,H9   f(1,1)=H10,H11 f(1,2)=H12,H13 f(1,3)=H14,H15
t2: f(2,0)=H16,H17 f(2,1)=NA     f(2,2)=NA      f(2,3)=NA       # 3 个空槽（empty-slot）
# OIFS 改为可变尺寸：t1 用 H8,H9 | H10,H11 | H12-H14 | H15-H17 消除空槽
```
- Annotations：|f(t,c)| 为组内 HMUX 数，T_exe(t,c)=max(T_comp×|f(t,c)|, T_comm)；融合变大隐藏 T_comm（D2D 时延），但过粗融合增大 bubble（启动/排空）；空槽（|f|=0）浪费时间槽。权衡由 OIFS 的 DP 求解（见 编译框架 库 OIFS 条目）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：作为调度策略由 OIFS 离线生成 f(t,c) 并写入硬件配置；HC 端支持"组内回馈"（输出重入本地输入执行下一 HMUX）与 intra-HC batching（多 RLWE 注入）。使用场景：TFHE 自举流水线的跨 chiplet 部署——解决"流水并行 vs D2D 通信"的折中；评估中相对两种基线策略（SHM 均匀分段、FFM 固定融合尺寸）在 DeepCNN-50/XG-Classifier 上总执行时间最低、HC 利用率与 D2D 带宽利用率最高（DeepCNN-50 参数集 I：95.9%/76.8%）。注意：IF 是映射/调度策略（决定 HMUX 到 chiplet/时隙的分配），与 OIFS（离线求最优配置的调度器+成本模型+DP）是"策略-求解器"关系。

涉及论文标题：
- Unlocking Pipeline Parallelism for Bootstrapping: A Pipelined Multi-Chiplet TFHE Accelerator
