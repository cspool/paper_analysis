## HEXA-MoE: Efficient and Heterogeneous-aware MoE Acceleration with ZERO Computation Redundancy

- baseline方法是什么？
  - **Tutel (Static MoE Library)**：使用 expert parallelism + 所有 token 的 all-to-all dispatch/combine 通信，配合 expert capacity 超参数校准各 expert workload。通过 GeMM 接口执行 expert FFN 计算。由于各 expert 的 workload 动态变化，需要 token padding（填充到 capacity）或 discarding（丢弃超出 capacity 的 token），产生冗余 FLOPs 和冗余内存分配/访问。全栈执行例子（以 Swin-MoE-Base, top-2 routing, 4×RTX 4090, Tutel baseline 训练单步为例）：
    - **模型训练算法层**：Swin-MoE-Base, MoE 层替换标准 FFN，每层 E 个 expert (E=4/8)，top-k routing。Forward: Gate softmax(W_g·x) → top-k selection → dispatch tokens 到各 expert → 各 expert 用 GeMM 计算 FFN (需 token padding 对齐 capacity) → combine 聚合输出。Backward: 按 auto-diff 计算各 expert 权重梯度和输入梯度。Auxiliary load balancing loss 鼓励 expert 均匀使用。
    - **系统框架层**：Tutel 基于 PyTorch，expert parallelism 将 expert 分布到多 GPU。执行流程：① attention 计算（数据并行）→ ② MoE gate routing → ③ all-to-all dispatch（跨设备同步发送 token）→ ④ 各设备本地 GeMM（grouped GeMM 或 padding 后标准 GeMM）→ ⑤ all-to-all combine（跨设备同步聚合输出）。All-to-all 通信可占 40%+ runtime。
    - **编译框架层**：论文未明确说明。PyTorch eager mode + NCCL 通信后端。
    - **kernel 调度层**：NCCL all-to-all collective kernel + cuBLAS GEMM kernel。执行顺序：all-to-all（通信）→ GeMM（计算）→ all-to-all（通信），通信和计算严格串行。token padding 引入冗余内存分配和冗余 FLOPs（padding token 的计算结果被丢弃）。
    - **硬件架构层**：4× NVIDIA RTX 4090 (24 GB)，同构 GPU 集群，节点内 PCIe/NVLink 互联。
  - **MegaBlocks (Dynamic MoE Library)**：使用 block-sparse 操作和对应 GPU kernel 替代 GeMM，消除 token padding。但运行时行为动态不确定，可能触发 OOM；且 grouped GeMM 的动态性导致 memory 管理复杂。
  - Baseline 痛点：
    1. **冗余 FLOPs 和内存（核心痛点 1）**：Expert parallelism + GeMM 接口强制 token padding/discarding。Tutel 的 expert capacity 超参数需要 hand-tune——capacity 太小则丢弃 token 损失精度，capacity 太大则大量 padding token 产生冗余计算和冗余内存。MegaBlocks 虽消除 padding 但引入 runtime OOM 风险。
    2. **All-to-All 通信瓶颈（核心痛点 2）**：Expert parallelism 依赖同步 all-to-all dispatch/combine 通信，可占 40%+ runtime。随模型规模和设备数增加，通信开销线性增长。
    3. **同构设备限制（核心痛点 3）**：Expert parallelism 主要部署在同构设备上。异构设备（新旧 GPU 混用）更便宜易获取，但 expert parallelism 的负载均衡依赖同构硬件假设，无法直接利用异构设备的差异化计算能力。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **HEXA-MoE 方法**：通过三个系统性设计解决 baseline 的全部痛点：
    1. **Expert-Specific Operators (ESMM, ESS, ESTMM) 替代 GeMM**（解决痛点 1）：将 MoE 前向和反向传播重新公式化为三个 expert-specific 算子——ESMM（expert-wise 矩阵乘法）、ESS（expert-wise 求和累加）、ESTMM（expert-wise 转置矩阵乘法）。每个 token 仅与其路由 expert 的权重做矩阵乘法，无需 token padding/discarding，实现 **in-place 计算**，几乎零冗余 FLOPs。公式化对比：传统方法对每 expert 构造 N_e×D 的 padded batch（含 padding token），HEXA-MoE 直接对原始 N×D batch 做 expert-specific 计算——ESMM 通过 re-index vector 指导 I/O，thread-block 加载同 expert 的 tokens 和对应权重，跳过 padding 位置（填 -1 跳过），结果原位写回。对比 Tutel 的 "capacity padding + GeMM" pipeline，HEXA-MoE 的 "re-index + ESMM" pipeline 消除了 dispatch/combine 中的 token 重排和 padding 计算。
    2. **Tensor Parallelism 替代 Expert Parallelism**（解决痛点 2）：用 tensor parallelism（沿 FFN intermediate size 切分）替代 expert parallelism。Data-centric 下各设备 all gather 完整 MoE 参数后本地计算（无 all-to-all）；Model-centric 下各设备 all gather 数据批次后本地计算参数 chunk（all reduce 替代 all-to-all）。Tensor parallelism 的 all gather / all reduce 通信 pattern 比 expert parallelism 的 all-to-all 更规整，且配合 pipeline-shared cache 可实现通信-计算重叠。
    3. **Heterogeneous-Aware Expert Allocation**（解决痛点 3）：基于各设备 benchmark 延迟按反比分配 workload——Data-centric 下调整 local batch size B_i，Model-centric 下调整 FFN intermediate sub-dimension h_i。Tensor parallelism 的 workload 可精确预测（由 batch size 或 sub-dimension 决定），使异构调度成为确定性问题，无需处理 expert parallelism 的动态 workload 不确定性。
  - 全栈执行例子（HEXA-MoE, Swin-MoE-Base, data-centric, 4×RTX 4090，与 baseline 同配置对比）：
    - **模型训练算法层**：与 baseline 相同的 Swin-MoE-Base 模型结构，差异在 MoE 层的计算方式：
      - Forward: y1 = ESMM(x, W1, b1, R(x)) → y2 = F(y1) → y = ESMM(y2, W2, b2, R(x))
      - Backward: ∂ℓ/∂b2 = ESS(∂ℓ/∂y, R(x)) → ∂ℓ/∂W2 = ESTMM(y2, ∂ℓ/∂y, R(x)) → ∂ℓ/∂y2 = ESMM(∂ℓ/∂y, W2^T, null, R(x)) → ∂ℓ/∂y1 = ∂ℓ/∂y2 ⊙ F'(y1) → ∂ℓ/∂b1 = ESS(∂ℓ/∂y1, R(x)) → ∂ℓ/∂W1 = ESTMM(x, ∂ℓ/∂y1, R(x)) → ∂ℓ/∂x = ESMM(∂ℓ/∂y1, W1^T, null, R(x))
      - Backward 中 3 个算子（ESS, ESMM, ESTMM）融合为 ESFK（单 kernel），单 MoE 层 backward 仅需 2 fused kernels + 1 element-wise dot product。
    - **系统框架层**：基于 PyTorch + CUDA C++ 自定义 kernel。Tensor parallelism 替代 expert parallelism——各设备沿 FFN intermediate size 切分所有 expert 的权重。Data-centric 执行流程：① attention + router 计算 → ② (并行) all gather MoE 参数到 pipeline-shared cache → ③ ESMM 用 cache 中的完整 MoE 参数本地计算 → ④ 下一层。All gather 与 attention 在分离 CUDA stream 上重叠。对比 baseline Tutel 的 "all-to-all dispatch → GeMM → all-to-all combine"，HEXA-MoE 的 "all gather → ESMM" 流程无 token 重排、无 padding、通信量更可控。
    - **编译框架层**：论文未明确说明。PyTorch eager mode + NCCL 通信后端 + 自定义 CUDA kernel。
    - **kernel 调度层**：ESMM kernel 使用 re-index vector 指导 I/O——thread-block 加载 sub-vector 定位同 expert tokens → 加载对应 expert 权重（仅一次）→ Tensor Core MMA (nvcuda::wmma) → 按 sub-vector 写回。ESS kernel 按 expert+channel 分配 thread-block → 累加同 expert tokens。ESTMM kernel 对 re-indexed 输入做 expert-wise 外积。ESFK 融合 kernel 统一 thread-block shape 为 (WARP, TIMES)，grid 扩展为 3 维聚合 ESS+ESMM+ESTMM。Pipeline-shared cache 在 HBM 额外区域动态缓存 gathered shards。对比 baseline Tutel 的 "NCCL all-to-all + cuBLAS GEMM"，HEXA-MoE 的 "NCCL all gather + ESMM/ESFK kernel" pipeline 消除了 token padding 的冗余 kernel launch 和冗余 HBM 访问。
    - **硬件架构层**：与 baseline 相同（4×RTX 4090）。结果：10%-48% 内存节省，0.5-4.3× 加速 vs Tutel/MegaBlocks。Heterogeneous 实验（TITAN RTX + RTX 2080 Ti）：data-centric 下 optimal allocation 可降低延迟 13.2%-25.3%，model-centric 下降低 6.3%-11.9%。
  - **核心设计洞察**：HEXA-MoE 的本质洞察是 MoE 计算中存在"冗余的源头"——token padding/discarding 不是 MoE 的本质需求，而是 GeMM 接口强加的人为约束。通过将 MoE 从 "GeMM 视角"（先重排为规则 batch 再调 GeMM）重新定义为 "Expert-Specific 视角"（不重排 token，直接做 expert-wise 计算），HEXA-MoE 从根本上消除了冗余 FLOPs 的源头。这一视角转换带来两个连锁效应：(1) Tensor parallelism 变得自然可行——expert-specific 算子天然适配沿 intermediate size 的切分（ESMM 中每个 expert 的权重可独立切分），使 expert parallelism 的 all-to-all 被替换为 tensor parallelism 的 all gather/reduce；(2) 异构调度变得确定——tensor parallelism 下各设备的 workload 由 batch size 或 sub-dimension 精确决定，消除了 expert parallelism 中动态 workload 的不确定性。EXA-MoE 的 elegant 之处在于它不需要在现有系统上"修补"冗余，而是通过重新定义计算范式使冗余在数学层面消失。
