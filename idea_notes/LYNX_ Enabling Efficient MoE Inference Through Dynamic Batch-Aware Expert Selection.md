## LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection

- baseline方法是什么？
  - **vLLM 默认 MoE 推理**：标准的 top-k expert routing，每个 token 独立选择 k 个 expert。在 decode 阶段，随着 batch size 增大，batch 中所有 token 选择的 expert 并集快速增长，最终激活几乎全部 expert。此时 MoE 丧失了稀疏性优势：激活的参数数量接近甚至超过同等容量的 Dense 模型，同时还要额外承担动态 expert dispatch 的开销。现有优化技术分为两类：(1) 静态方法——pruning、quantization、expert clustering 等，依赖离线校准数据集识别 workload 级 expert 冗余，永久修改模型。(2) 动态方法——per-token level 减少激活 expert 数（如 dynamic k），但未解决 batch 级 expert utilization 问题。
  - **全栈执行例子（Baseline vLLM on H200, Qwen2-57B, decode iteration, B=16）**：
    - **算法 Pipeline 层**：Router 计算 B×N=16×64 logits → softmax → top-8 per token → 每个 token 独立选 8 个 expert。由于 load-balancing loss 在训练中强制 uniform expert 利用，16 个 token 的 expert 选择并集覆盖几乎所有 64 个 expert（~55-60 个）。
    - **系统框架层**：vLLM v0.10.1, v1 scheduler, continuous batching。每层 MoE 的 expert 权重需从 HBM 全部加载（因几乎所有 expert 都被激活），decode iteration 中 42% 时间花在 expert weight 的 HBM 数据搬运上，成为 memory-bandwidth-bound。
    - **编译框架层**：论文未明确说明。
    - **Kernel 调度层**：vLLM 默认 fused expert kernel（grouped GEMM），以全部 active expert（~55-60 个）为 dispatch 参数启动。每个 decode iteration 的专家计算 kernel 需加载 ~55-60 个 expert 的完整权重矩阵。
    - **硬件架构层**：H200 GPU (141 GB HBM, SXM NVLink)。Memory bandwidth 是瓶颈——decode 的 arithmetic intensity = (B × k / N) = 16 × 8 / 64 = 2，远低于 GPU compute 能力，decode latency 与 active expert 数量线性相关。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **LYNX 方法**：通过 **AffinityBinning** 技术在 batch 级别动态重映射低置信度 token-to-expert assignment，将 batch 的 expert 激活从"并集"压缩为"最小关键集"。核心创新：
    1. **Insight 1 — Phase Sensitivity**：prefill 和 decode 对 expert fidelity 的敏感性截然不同。Prefill 需要严格 expert fidelity（建立 context），decode 因 attention/残差连接/累积 context 的补偿机制而对 expert 选择高度容错。因此 LYNX 仅在 memory-bound decode 阶段启用 remapping。
    2. **Insight 2 — Router Confidence as Reliability Signal**：Router 输出的 logits 差异（log-ratio to top-1）可靠地标识 token 对 expert assignment 的"在意程度"。高置信度 token（router 强烈偏好某个 expert）的 assignment 必须保留；低置信度 token（router 的各 expert 分数接近）可以安全地重映射到其他 expert 而不影响输出质量。这种区分无需校准数据，是 MoE router 的固有特性。
    3. **Insight 3 — Expert Rank Hierarchy**：top-1 expert 主导输出质量（deny top-1 会 catastrophic accuracy drop），lower-ranked experts (rank 2-8) 高度冗余。LYNX 利用此等级制：保留所有 token 的 top-1 expert 作为"锚点"，仅重映射 lower-ranked experts。
    4. **AffinityBinning**：将 per-token router confidence 按 sparsity ratio (k/N) 决定的 bin width 和 count 离散化。batch-size 为底数的指数加权确保：被多个高置信度 token 偏好的 expert 获得指数级高分，仅被低置信度 token 偏好的 expert 被大幅降权。这实现了 batch 级别专家重要性的自动校准——bin 参数仅由模型架构决定，无需任何 workload-specific tuning。
    5. **Expert Remapping**：在 batch 级别决策最小关键专家集后，将低置信度 token 的 lower-ranked expert assignment 重映射到该集合内。每个 token 仍激活 k 个 expert（保持 top-k 语义），只是 expert 选择在 batch 内被 consolidated 到更小的并集上。
  - **对应解决 Baseline 缺陷**：
    - **Baseline: batch 级 expert 并集随 batch size 线性增长 → memory bandwidth bound** → LYNX 在 batch 级减少 active expert 总数（~25→15），直接降低 HBM 数据搬运量，使 decode latency 与 reduced expert count 相关而非 full expert count。
    - **静态方法：依赖离线校准，永久修改模型，不灵活** → LYNX 完全 runtime，不从模型中永久移除任何 expert，不修改权重，适应 workload 变化。
    - **动态 per-token 方法：减少 per-token k，但 batch 级 expert 利用仍高** → LYNX 保持 per-token k 不变，通过 batch 内 remapping 减少并集大小。
    - **需要 calibration data** → LYNX 仅依赖 router 输出 logits 作为信号，self-calibrating（参数仅由 sparsity ratio 决定）。
  - **全栈执行例子（LYNX on vLLM, Qwen2-57B, decode iteration, B=16）**：
    - **算法 Pipeline 层**：Router logits → AffinityBinning（每 token 的 log-ratio 离散化到 6 个 bin）→ Adaptive Scoring（16^bin 指数加权，高置信度 token 的偏好主导分数）→ 动态确定 active expert set（如从 ~55 降至 ~15）→ Low-confidence token assignment 重映射到 active set → 每 token 仍激活 8 个 expert。
    - **系统框架层**：vLLM batch scheduler 中 Phase-aware Optimizer 识别 decode-only iteration → 启用 LYNX pipeline。每层 router 后插入 4 个 fused kernel → 最终 dispatch 到更小的 expert set。
    - **编译框架层**：论文未明确说明。Triton kernel 编译为 CUDA，CUDA Graph 捕获静态执行图。
    - **Kernel 调度层**：4 个 fused Triton kernel（替代 700+ PyTorch ops, <4% overhead）→ Expert GEMM kernel 仅加载 ~15 个 expert 的权重（vs baseline 的 ~55 个），HBM 数据搬运量减少 ~73%。
    - **硬件架构层**：H200 GPU。LYNX kernel overhead (<4%) 远小于内存带宽节省（expert 加载量减少 ~73%），net latency 降低 1.09-1.30x。准确率偏差 <1%，平均情况甚至提升（因移除了 training load-balancing 强制的低质量 expert assignment）。
