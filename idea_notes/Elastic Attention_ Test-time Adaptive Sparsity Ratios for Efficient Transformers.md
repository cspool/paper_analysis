## Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers

- baseline方法是什么？
  Baseline 是现有的 hybrid attention 方法（DuoAttention、PruLong、InfLLM-V2）以及静态稀疏注意力方法（MoBA、NSA），它们的核心问题是使用**固定计算比例**（static computation ratios），无法根据输入任务动态调整稀疏度。全栈执行例子如下：

  - **算法pipeline层**：DuoAttention 在训练时将 attention heads 分为 retrieval heads（FA）和 streaming heads（SA），head 分配在训练后固定，推理时所有任务使用相同的 Ω_MSR（如始终 0.70）。PruLong 使用 hard concrete reparameterization + Lagrangian penalty 学习 head 二分类，head 分配同样训练后固定。MoBA/NSA 将 attention 计算限制在预定义的 block/chunk 内，sparsity pattern 由固定超参数（block_size, top-k）决定。这些方法在 sparsity-sensitive tasks（如 QA）和 sparsity-robust tasks（如 summarization）上使用相同的计算模式，导致要么 sparsity-sensitive 任务性能下降（sparsity 过高时），要么 sparsity-robust 任务计算冗余（sparsity 过低时）。实验表明：随着 Ω_MSR 从 0 增至 1.0，sparsity-sensitive 任务（Single-Doc QA）性能从 100% 降至 56%，而 sparsity-robust 任务（Code）性能始终 >93%。

  - **系统框架层**：基于 HuggingFace Transformers 推理 pipeline。所有 baseline 的 head 分配模式在训练后固定，推理时不随输入任务变化。DuoAttention/PruLong 的 retrieval/streaming head mask 是预先计算好的常数，MoBA/NSA 的 block selection 由固定规则决定。

  - **编译框架层**：论文未明确说明。

  - **kernel调度层**：DuoAttention 使用 FlashAttention-2 分别对 retrieval 和 streaming heads 执行 attention，需要两次 kernel launch。MoBA/InfLLM-V2 需 reserve 部分计算预算用于 pre-compute sequence-level features。部分 baseline（如 NSA、InfLLM-V2）对 KV attention heads 数量有严格可除性要求（如 16 的倍数），与 Llama-3.1-8B 的 GQA 配置不兼容。

  - **硬件架构层**：单 NVIDIA A800/A100 GPU。MoBA 和 InfLLM-V2 在 256K context 下因 sequence-level feature pre-computation 导致 GPU OOM。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文 Elastic Attention 通过**test-time 自适应稀疏度分配**解决 baseline 的静态计算比例问题。核心观察：下游任务自然分为两类——sparsity-robust（粗粒度上下文即可完成，如 summarization）和 sparsity-sensitive（需细粒度证据检索，如 QA）。基于此，引入轻量级 Attention Router 根据输入动态调整每个 head 的计算模式。

  全栈执行例子（Elastic Attention on Llama-3.1-8B-Instruct, FA-SSA setting）：

  - **算法pipeline层（核心创新）**：
    (a) **Attention Router**（每层 0.27M 参数）：接受 Key hidden states x_K，通过 Boundary Pooling（仅取序列首部和尾部各 100 tokens）得到 task representation → Task MLP 提取 task-specific 特征 → Router MLP 输出 head-wise 二值路由决策 r_hard ∈ {0,1}。r=0 的 head 使用 FA (retrieval)，r=1 的 head 使用 SA (sparse)。训练使用 Gumbel-Softmax 连续松弛 + STE 解决 argmax 不可微问题，温度 τ 从 τ_init 指数衰减至 τ_min。
    (b) **Lagrangian 约束训练**：min-max 优化 L = L_language + λ1·(Ω_MSR - t) + λ2·(Ω_MSR - t)²。t 为 task-dependent 非紧约束（sparsity-sensitive t=0.7, sparsity-robust t=1.0），λ 为可训练 Lagrange 乘子。backbone 参数完全冻结，仅训练 Router（12h on 8×A800）。
    (c) **任务自适应性**：训练时 Router 从随机初始化自动学会区分 sparsity-sensitive 和 sparsity-robust 任务——训练曲线显示 Code 和 ICL 任务的 Ω_MSR 收敛到 ~0.80-0.85（较低 sparsity = 保留更多 FA），QA 任务的 Ω_MSR 收敛到接近 t (~0.65-0.7)。推理时 Router 根据输入的 task representation 动态分配每个 head 的计算模式，无需 task label。

  - **系统框架层**：基于 PyTorch + HuggingFace Transformers + LOOM-Eval 推理框架。Router 作为可插拔模块集成到已有预训练模型中，无需修改 backbone 参数。Router 的 pooling 仅处理 boundary tokens（100+100），复杂度独立于序列长度。

  - **编译框架层**：论文未明确说明。

  - **kernel调度层**：实现基于 Block Sparse Attention (BSA) Kernel 的 fused hybrid attention kernel。将 routing decisions m 直接传给 kernel 作为 metadata，kernel 内部通过 thread-block level branching 判断每个 head 的类型并执行对应的 attention 逻辑。Single kernel launch 处理所有 heads，消除 tensor splitting 的内存开销和多次 kernel launch 的调度开销。保持 grid 完整性（Batch×Heads×Sequence Blocks），GPU SM 可最优调度。

  - **硬件架构层**：8× A800 训练，单 GPU 推理。Router 延迟仅 ~0.196ms/router call，且不随序列长度增长。

  **对比 baseline 的关键差异**：
  - Baseline 静态 sparsity → Elastic Attention test-time 自适应 Ω_MSR：同一模型在 Code 任务上 Ω_MSR ~0.82（高 sparsity），在 QA 任务上 Ω_MSR ~0.68（低 sparsity），动态匹配任务需求。LongBench-E 上 Elastic Attention (FA-SSA) 在 Llama-3.1-8B 上平均分 53.35 > backbone 53.28（因 sparsity 减少 attention dispersion 反而可能提升性能）。
  - Baseline 需 per-task 调参 → Elastic Attention 自动分配：训练时仅需设置两类 target sparsity（t_robust=1.0, t_sensitive=0.7），无需 per-task 超参搜索。
  - Baseline kernel inefficiency（多次 launch + tensor split）→ Elastic Attention fused BSA kernel：prefill 阶段加速显著，减少 memory copy 和 kernel launch overhead。
  - Router 的 Task MLP 隐式学会任务判别：通过对 hidden states 的 pairwise cosine similarity 分析，经过 Task MLP 后不同任务的 representation 被映射到近似正交的子空间（M_uv ≈ 0），使 Router MLP 能做出准确的 head 分配决策——尽管训练中从未提供 explicit task label。
  - 极低训练成本：12h on 8×A800 vs 同类方法需更新 backbone 参数，且 Elastic Attention 冻结 backbone 保持原有能力。
  - RULER 长度外推：256K context 下 FA-XA 仍保持 68.51（Llama-3.1-8B）vs MoBA/NSA 在 128K+ 严重退化（near-zero accuracy）。
  - 与 XA-SSA（全部 head 用 SA）的 scalability：Qwen3-4B 在三个 benchmark 上平均性能差距 <1 点（48.45 vs 48.14），证明全 SA 配置在极限压缩下仍可控。
