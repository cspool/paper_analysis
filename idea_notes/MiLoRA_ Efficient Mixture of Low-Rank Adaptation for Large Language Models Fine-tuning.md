## MiLoRA: Efficient Mixture of Low-Rank Adaptation for Large Language Models Fine-tuning

- baseline方法是什么？
  MOE-style LoRA 方法（如 MOELoRA、LoRAMoE、MoCLE 等）是 baseline。这些方法将每个 LoRA 模块内部分解为多个 sub-rank experts（例如 MOELoRA 中将 r=32 的 LoRA 分解为 32 个 single-rank LoRA，每 4 个组成一个 expert，共 8 个 experts per LoRA module），通过 token-wise router 为每个生成 token 动态选择激活的 experts。以 MOELoRA + LLaMA-2 7B 在 multi-tenant serving 下的全栈执行：
  - **算法层**：每层 Transformer 包含 7 个 LoRA 模块（Q/K/V/O/G/U/D），每个模块内有 8 个 sub-rank experts + 1 个 router。生成每个新 token 时，7 个 router 各自计算 top-4 路由概率，调用 4 个激活的 experts 计算 LoRA 增量：x' = xW_m + x·Σ(e_i·W_m^{A,i}·W_m^{B,i})。总计：每 token 每层调用 7 个 router + 7×4=28 个 sub-expert forward。
  - **系统框架层**：HuggingFace Transformers + PEFT 库。LoRA 参数不 merge 回 backbone（multi-tenant 设置下每个 tenant 有自己的 LoRA weights）。每个 generation step 的 forward pass 中额外执行所有 LoRA modules + routers。
  - **编译框架层**：论文未明确说明（标准 PyTorch forward，无编译优化）。
  - **kernel 调度层**：论文未明确说明（标准 PyTorch linear kernel + softmax，无 custom kernel）。
  - **硬件架构层**：NVIDIA A40 GPU (48GB)。
  - **核心缺陷**：(1) 每 token 每层计算 7 个 router + 大量 sub-expert forward，产生显著推理延迟（tps 比 base model 降低约 20%）；(2) token-wise routing 在 multi-tenant 下每个 tenant 请求都需要独立计算 router，延迟随 tenant 数量线性增长；(3) 多任务学习中共享 LoRA parameters 导致任务间数据冲突（单任务 ST 到多任务 MT 性能下降 0.5-2.0%）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MiLoRA 通过两个核心设计系统性解决 baseline 的延迟与效率问题，全栈执行如下：
  - **算法层 — Prompt-Aware Routing + Layer-Level Expert Selection**：
    - Expert 粒度提升：每个 LoRA 模块（而非其子结构）被定义为一个 expert，N_mod=7 个 experts per layer。每个 Transformer layer 仅激活 1 个 expert（k=1，通过 Top-k=3 的 softmax 概率分布实现），被选中模块用 LoRA 修正其 output，其余 6 个模块以原始 backbone 权重执行。
    - Router 从 token-wise 降为 prompt-wise：Router 仅在 input prompt 首次经过 backbone 时计算一次路由决策（before the first new token），后续所有 auto-regressive token 生成步骤均复用该决策。Router 计算流程：H^l (prompt hidden states) → SelfAttnPool(·) → Rational Activation Ra(·) → Softmax(W_r^l · h^l) → Top-k。
    - Load Balancing：训练时施加 auxiliary loss L_lb = N_mod · Σ f_i^l · p̂_i^l（λ_lb=1e-2），防止 experts 分配不均衡。
  - **算法层 — Learned Rational Activation Functions**：
    - 替代固定激活函数（ReLU/GeLU 统一用于所有层），使用有理函数 Ra(x) = Σ a_j x^j / (1 + ||Σ b_i x^i||)（m=6, n=5，可学习 a_j, b_i），每层 router 有独立参数。通过 DARTS 风格的 bi-level optimization 训练 activation params（architectural params Θ, lr=1e-6）和 LoRA params（Ω, lr=1e-4）。
    - 效果：不同深度的 Transformer layer 学习到不同的激活函数形态，较统一 ReLU 或 GeLU 有更好的 routing 质量和下游任务表现。
  - **系统框架层**：HuggingFace Transformers + PEFT 库。实现与 MOELoRA 等共用框架，但 adapter 结构更简单（per-layer 1 router + 1 activated LoRA module vs. 7 routers + 28 activated sub-experts）。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层**：论文未明确说明。
  - **硬件架构层**：NVIDIA A40 GPU (48GB)。
  - 对比 baseline 的改进映射：
    - **token-wise routing → prompt-aware routing**：Router 从每 token 调用 7 次降为整个序列仅调用 1 次（per layer）。MoE 路由开销从 O(L × T × N_mod) 降为 O(L × N_mod)，在 L=32 层 × T=256 tokens 场景下，router 调用次数从 32×256×7=57344 降至 32×7=224。实测推理加速：beam=1 时 tps 43.7 vs MOELoRA 35.9（+21.7%），beam=3 时 tps 33.5 vs MOELoRA 28.4（+17.9%）。
    - **per-module multi-expert → per-layer single-expert activation**：Activated LoRA 参数量从 MOELoRA 的 30.1M（每 token）降至 25.2M（每 prompt 固定选择），且在 generation 阶段仅执行被选中的 1 个 LoRA module（而非 7 个 module 内的多个 experts），减少 memory bandwidth 和 compute 开销。
    - **固定激活函数 → 可学习激活函数**：每层 router 学习最适合其深度的激活函数形态，缓解了深层/浅层对路由敏感性不同的矛盾。Ablation 显示 learnable activation 在 BoolQ/PIQA/MMLU 上均优于固定 GeLU 或 ReLU/GeLU 混合方案。
    - **多任务学习性能保持**：MiLoRA 在 ST→MT 切换中性能几乎不下降（Avg. 75.4→75.2, Δ=-0.1%），而 LoRA/DoRA 分别下降 2.0%/2.2%。MoE routing 机制天然为不同任务/数据选择不同 expert，缓解数据冲突。
