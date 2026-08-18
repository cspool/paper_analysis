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
