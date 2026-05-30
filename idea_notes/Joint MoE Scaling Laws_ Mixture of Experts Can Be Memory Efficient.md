## Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient

- baseline方法是什么？
  - **Chinchilla Scaling Laws for Dense Models (Hoffmann et al. 2022)**：L(N, D) = m·N^μ + n·D^ν + c，仅考虑模型参数量和训练 token 数两个变量，未涉及 MoE expert 数量维度。应用于 MoE 时，通常固定 expert 数量和模型规模，按 dense 方式做 compute-optimal 配置。Clark et al. (2022) 的 MoE scaling law L(N_act, Ê) = a·Ê^δ · N_act^(α+γ·ln(Ê)) 仅建模 N_act 和 E 的关系，忽略 dataset size D 的影响。
  - 全栈执行例子（Baseline: compute-optimal dense model, E=1, F=10^20 FLOPs）：
    - **模型推理/训练算法层**：Decoder-only Transformer，标准 FFN（SwiGLU, hidden=3×d_model），无 MoE routing。训练流程：固定 N_act=1.7B, D=9.7B tokens。Optimizer: 固定 LR（无 E-dependent scaling）。
    - **系统框架层**：标准 PyTorch 训练，数据并行/模型并行视 GPU 配置而定。论文未明确说明具体框架。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：标准 dense GEMM kernel。无 MoE-specific sparse computation。
    - **硬件架构层**：PLGrid HPC / Writer.com 计算资源。Memory budget 示例中引用 H100 (80GB)、RTX 4090 (24GB)。论文未明确说明具体 GPU 配置。
  - Baseline 缺陷：
    1. **缺少 E 维度的联合优化**：传统 scaling law 或仅针对 dense (E=1)，或将 E 视为固定超参而非优化变量。无法在给定 compute+memory budget 下系统性地选择最优 E。
    2. **E 与 D 的关系不明确**：Clark et al. (2022) 的形式未含 D 项，无法回答"固定 memory budget 下，增加 E 后应分配多少 token"的问题。
    3. **无 memory optimality 指导**：传统 wisdom 认为 MoE 因 total params 远大于 active params 而 memory-inefficient，实践中倾向于选择 dense 模型以满足 memory 约束。
    4. **超参数 (LR) 无 E-dependent scaling**：不同 E 的 MoE 训练使用相同 LR 策略，导致 suboptimal tuning。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **Joint MoE Scaling Laws 方法**：提出联合 scaling law L(N_act, D, Ê) = aÊ^δ · N_act^(α+γ·ln(Ê)) + bÊ^ω · D^(β+ζ·ln(Ê)) + c，将 D 项引入 MoE scaling，并通过 exponent 中的 ln(Ê) 交互项捕捉 E 与 N_act 和 D 的交叉效应。
    1. **D 项的引入**：在 Clark et al. (2022) 的基础上增加 bÊ^ω · D^(β+ζ·ln(Ê)) 项，使 scaling law 能同时描述不同 E 下的 token scaling behavior。ν(E)=β+ζ·ln(Ê) 中 ζ<0，表明 expert 越多 dataset exponent 越负（更多 expert → 需要更多 data）。
    2. **Memory optimality 的数学化**：将 memory constraint 形式化为 N_total ≤ M（或加上 KV-cache），在 {N_act, D, E} 三维空间求解约束优化的 argmin L。推导出 E≤8 的 rule of thumb：固定 total params 的 MoE 用 E× tokens 训练可超越 compute-optimal dense。
    3. **Inference optimality**：将 inference cost (2·N_act·D_inf) 纳入 joint FLOPs budget，揭示 MoE 在 inference 阶段的额外优势（每 token 仅激活 N_act，而非 N_total）。
    4. **LR scaling law for MoE**：实证推导 LR(N_act\e, E) = exp(8.39-0.81·ln(N_act\e)-0.25·ln(E))，E 的负系数表明更多 expert 需要更低 LR，确保不同 E 配置的 fair comparison。
  - 全栈执行例子（对比 Baseline，论文方法在 E=4, F=10^20 FLOPs 下）：
    - **模型推理/训练算法层**：与 baseline 同架构的 Switch MoE（E=4 experts/layer）。根据 joint scaling law，compute-optimal 配置为 N_act=1.2B, D=13.9B tokens（vs dense N_act=1.7B, D=9.7B）。训练用 LR 由 LR scaling law 按 (N_act, E) 计算。关键张量流变化：每 token 经 Router 从 4 个 expert 中选择 top-1，仅激活对应 expert FFN，FLOPs/token = baseline 的 1.2/1.7 ≈ 70%。
    - **系统框架层**：论文未明确说明。推测标准 PyTorch + expert parallelism（因 E=4 较小，routing cost negligible）。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：与 baseline 相同的 dense GEMM kernel，但 expert FFN 的矩阵维度更小（因 N_act 更小），计算量降低。因 E=4 采用 expert parallelism，无复杂 all-to-all 通信。
    - **硬件架构层**：同 baseline 硬件平台。Memory 对比：dense N_total=1.7B 占用约 3.4GB (BF16)；E=4 MoE N_total≈3.0B（根据 Eq.9 估算，专家参数约 4× baseline FFN），但 N_act=1.2B 比 dense 小 → inference 时仅需 2.4GB 活跃参数，KV-cache 也更小（d_model 更小）。
  - **Baseline 缺陷 → 方法设计映射**：
    | Baseline 缺陷 | Joint Scaling Law 设计 | 效果 |
    |-------------|----------------------|------|
    | E 非优化变量 | 将 E 作为连续优化维度 | 可在 3D 空间系统性选择最优 (N_act,D,E) |
    | 无 D-E 交互 | 增加 D^(β+ζ·ln(Ê)) 项 | ζ<0 揭示 expert 越多需越多 data |
    | MoE 被认为 memory-inefficient | Memory optimality 分析 | 证明 E=4 MoE 在 1.1B total params 下 loss 低于 dense |
    | Inference cost 被忽略 | Joint train+inference budget | MoE 用 36-61% 更少 FLOPs/token |
    | 无 E-dependent LR | LR scaling law with ln(E) term | 不同 E 的 fair comparison 成为可能 |
