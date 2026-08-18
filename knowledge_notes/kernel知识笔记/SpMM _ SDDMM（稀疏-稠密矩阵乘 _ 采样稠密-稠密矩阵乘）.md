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
