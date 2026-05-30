## Pipeline MoE A Flexible MoE Implementation with Pipeline Parallelism

- baseline方法是什么？
  - Baseline 是 DPMoE（Data Parallel MoE），即传统的 bound data parallel + expert parallel 的 MoE 并行架构。每个 data parallel replica 持有完整 backbone + E/D 个 expert，experts 分布在所有 DP rank 上。每个 MoE layer 需要两次 all-to-all 通信（dispatch 和 gather），每次传输 b×s×h 大小的 hidden embeddings。
  - 全栈执行例子（Baseline: DPMoE, GPT-3 Medium → 6.7B MoE, 64 experts, top-1 gating, DP=4 + TP=8 + EP=64, 32 V100, Megatron-LM v2.5 + DeepSpeed v0.5.10）：
    - **算法层**：MoE layer 执行 gating(W_gate @ h) → softmax → TopK=1 routing → 1st all-to-all（dispatch tokens 到对应 expert 所在 rank）→ per-expert FFN(GeLU(XA)B) → 2nd all-to-all（gather processed tokens 回原 rank）→ Dropout + LayerNorm。两次 all-to-all 占 forward 总时间的 65.5%（实测），MoE 层总时间占 forward 的 82.6%。token→expert → all-to-all 需跨节点 InfiniBand 传输（BW=12.5 GB/s），expert FFN 计算被通信严重拖慢（t'_a2a/t'_FFN > (E-1)E/16 ≈ 252 for E=64）。
    - **系统框架层**：Megatron-LM v2.5 管理 TP + DP + EP 三路并行。DeepSpeed ZeRO optimizer 用于 DP ranks 间 optimizer state 分片以节省显存。bound DP+EP 要求 E 能被 D 整除（如 E=64, D=4 则每 rank 16 experts）。DP 和 EP 维度强耦合——改变 DP 规模会改变 expert 分布，无法灵活配置。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：PyTorch/Megatron 标准的 all-to-all 通信（NCCL backend）、GEMM FFN（cuBLAS）。ring-style all-to-all 的时间随 EP world size 线性增长，跨节点 InfiniBand 带宽远低于节点内 NVLink。
    - **硬件架构层**：华为云 V100 SXM2 服务器（8 GPU/节点），节点内 NVLink 互联，节点间 InfiniBand 100 Gb/s（BW≈12.5 GB/s）。all-to-all 跨节点通信是性能瓶颈——每个 expert 的 EP world size = 64 = 8 节点，ring all-to-all 延时 O(N·m/B)。
  - Baseline 双缺陷：(1) **all-to-all 通信瓶颈**：两次 all-to-all 占 forward 时间 65.5%，严重限制 training throughput（6.7B DPMoE 仅达到 backbone 的 66.2% 吞吐）；(2) **backbone 扩展受限**：DPMoE 的 DP+EP 绑定使得每个 DP rank 只能容纳 ≈single-expert 大小的 backbone，无法通过 tensor parallel 或 pipeline parallel 有效扩展 backbone，而最新研究表明 thick backbone + moderate experts 更有优势。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文提出 PPMoE（Pipeline MoE），通过两大核心设计解决 DPMoE 缺陷：(1) 将 expert parallel 与 tensor parallel 绑定（代替与 data parallel 绑定），所有 experts 位于同一 TP group（同一节点），用 tensor index slicing 替代 all-to-all；(2) 无缝集成 pipeline parallel 扩展 backbone。
  - 全栈执行例子（PPMoE, GPT-3 Medium → 6.7B, 64 experts, top-1 gating, TP=8 + PP=4 + DP=1 + EP=64, 32 V100, Megatron-LM v2.6）：
    - **算法层（解决 all-to-all 通信瓶颈）**：
      - 传统 DPMoE 流程：hidden → gating → all-to-all dispatch → expert FFN → all-to-all gather → output。关键瓶颈是 all-to-all。
      - PPMoE 流程：hidden → TP all-reduce sync（与 TP 的 attention/FFN 相同）→ 各 TP rank 独立执行 gating（因输入相同，产出相同 dispatching order）→ tensor index slicing（index_select, 纯本地操作，零通信）→ 串行执行 N=E/T 个 local expert FFN → index assignment 重建 → inner-node all-reduce（与 TP FFN 的 all-reduce 完全相同，走 NVLink）。
      - 核心替换：**all-to-all (跨节点 InfiniBand) → index_select (本地) + all-reduce (节点内 NVLink)**。MoE all-reduce 时间 = FFN all-reduce 时间（差异仅 1.9% of total forward），通信从 forward 的 65.5% 降至 20.7%。
    - **系统框架层（解决 backbone 扩展受限）**：
      - 传统 DPMoE：DP+EP 绑定，每 DP rank backbone = single-expert 容量，扩展 backbone 需要重新分配 expert 布局。
      - PPMoE：expert parallel 与 tensor parallel 绑定在同一节点内。MoE 层的输入/输出格式和通信模式与非 MoE FFN 完全一致（均为 TP all-reduce）。因此 dense 模型的 TP+PP 框架可"即插即用"地替换部分 FFN 为 MoE 层 → 无需修改 pipeline stage 划分和通信拓扑。
      - 功能等价性：DPMoE 是"空间并行"（不同 DP rank 同时处理 micro-batches），PPMoE 是"时间并行"（同一 pipeline 串行处理 micro-batches，通过 gradient accumulation 等效全局 batch）。两者在数学上等价，但并行架构不同。
    - **编译框架层**：论文未明确说明。基于 Megatron-LM v2.6 实现，核心修改在 model definition 层（expert parallel 与 TP 绑定）和 forward function（index slicing 替代 all-to-all）。
    - **kernel调度层**：PyTorch 标准的 index_select、GEMM（cuBLAS）、all-reduce（NCCL over NVLink）。串行 N 个 expert FFN 的计算速度与处理单个大 batch 几乎相同（因为低层算子优化），无额外性能损失。all-reduce 走 NVLink（300 GB/s）而非 InfiniBand（12.5 GB/s），通信带宽比 DPMoE 高约 24×。
    - **硬件架构层**：同一 V100 集群。关键差异：DPMoE 的 all-to-all 跨 8 节点 InfiniBand（每 rank 传 bsh bytes，ring 延时 O(E)×），PPMoE 的 all-reduce 仅在单节点 8 GPU 间 NVLink（每 rank 传 bsh/T bytes，延时 O(log T)）。因此 PPMoE 的通信效率远高于 DPMoE。
  - 解决 Baseline 缺陷的方式：
    1. **针对 all-to-all 通信瓶颈**：PPMoE 通过将 experts 全部置于同一节点（TP group）内，从根本上消除了跨节点 all-to-all。dispatch 用本地 index slicing 替代（零通信），gather 用节点内 NVLink all-reduce 替代（与标准 TP 通信一致）。结果：MoE 通信从 forward 65.5% → 20.7%，total MoE forward 从 82.6% → 38.2%。
    2. **针对 backbone 扩展受限**：PPMoE 通过使 MoE 层与 TP+PP 框架兼容（因输入/输出格式和通信模式一致），实现了 backbone 在 depth 维度（PP）和 width 维度（TP）的自由扩展。6.7B backbone 可扩展为 143B PPMoE，且 even 达到 backbone（20× smaller）的 90.7% 吞吐。
    3. **功能等价性保证**：PPMoE 与 DPMoE 在数学上等价——相同的 global batch、相同的 gradient accumulation、相同的更新规则。Convergence 验证确认 training/validation loss 与 backbone 一致收敛。PPMoE 仅改变并行架构，不改变模型语义。
