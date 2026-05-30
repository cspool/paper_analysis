## Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models

- baseline方法是什么？
  Baseline 是 **(1) Dense Transformer 模型** 以及与先前 MoE scaling law 研究中的方法对比。Baseline 的核心缺陷：(a) **Dense 模型中参数量直接绑定计算量**——每 token 需要计算所有参数对应的 FLOPs，无法像 MoE 一样通过稀疏激活解耦参数量和计算量；(b) **MoE 缺乏统一的效率预测框架**——虽然 MoE 通过稀疏激活提高效率，但给定一个 MoE 架构配置（activation ratio, granularity, shared experts），无法预先知道其相对于等性能 dense 模型的计算效率；(c) **先前 MoE scaling law 研究的局限性**——Clark et al. (2022) 在固定数据集上评估导致 MoE 被 undertrained 的错误结论；Ludziejewski et al. (2024) 使用统一超参导致不公平比较，且他们的 granularity 定义 (G=4d_model/d_expert) 更粗粒度，观测到单调递增而非最优范围。
  
  全栈执行例子（Baseline: Dense Transformer, 训练大模型场景, C=1e22 FLOPs）：
  - **算法Pipeline层**：Dense decoder-only Transformer with GQA + RoPE。每层 FFN 为 dense，每 token 计算所有参数：C_dense_ffn = 6·B·s·d_model·d_ffn。模型参数量 = 计算量，无法通过 sparsity 提升效率。原问题：给定 C=1e22 FLOPs，应该训练多大的 dense 模型？过去凭经验猜测，无 EL 指导。
  - **Serving/系统框架层**：标准训练框架（AdamW optimizer, WSD LR schedule, data parallelism）。论文未明确说明训练系统细节（基于 Ling 系列内部框架）。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：标准 PyTorch/CUDA GEMM kernel 执行 attention 和 FFN。MoE 的 all-to-all communication 和 expert parallelism 论文未在实验中涉及。
  - **硬件架构层**：GPU（具体型号论文未明确说明，基于 Ant Group Ling Team 先前工作推测 A100/H800 级别 GPU）。Dense 模型训练仅需 data parallelism，无 expert 通信开销。
  Baseline 的核心缺陷：(a) **缺乏 EL 预测能力**——不知道 MoE 能带来多少效率增益，无法在训练前决定最优 MoE 配置；(b) **不公平的 MoE vs Dense 对比**——先前工作使用固定数据量评估 MoE（Clark et al. 2022）或统一超参（Ludziejewski et al. 2024），导致 MoE 被低估；(c) **Granularity 最优范围未发现**——Ludziejewski et al. 使用更粗粒度定义，观测到 monotonic trend，未发现 U 形最优范围 G=8~12。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Efficiency Leverage (EL)** 作为量化 MoE 计算效率的核心指标，通过大规模实证研究（300+ 模型，最大 28B 参数）建立 MoE 的统一 scaling law。具体设计解决 baseline 的三重缺陷：
  
  (1) **EL 指标解决"缺乏预测能力"缺陷**——EL = C_dense / C_moe 直接量化 MoE 的效率增益。基于 compute-optimal allocation 和 optimal hyperparameters 训练，保证公平比较。最终 joint scaling law：EL(A,G,C) = Â^{α + γ(log G)² + β log G}，给定 A, G, C 可直接预测 EL。
  
  (2) **三阶段实验方法解决"不公平对比"缺陷**——Stage 1: 推导 MoE 的最优超参 scaling law（η^opt ∝ C^{-0.1529}, B^opt ∝ C^{0.3644}），MoE 需要更大 batch size 和略低 LR（因 expert 梯度稀释）。Stage 2: 推导 MoE 的最优模型-数据分配（MoE 偏向更小 M、更多 D）。Stage 3: 在最优条件下消融各架构参数。这确保每个配置都在 near-optimal 条件下评估。
  
  (3) **更精细的 Granularity 定义解决"U 形最优未发现"缺陷**——使用 G=2d_model/d_expert（而非 4d_model/d_expert），探索更细粒度（G 最高 16），发现 U 形 loss-G 关系，最优 G≈12。此外发现 routing balance 影响：poor balance 使最优点下移至 coarser G。

  全栈执行例子（论文方法：Ling-mini-beta, C=1e22 FLOPs, A=3.4%, G=12, S=1/13≈7.7%）：
  - **算法Pipeline层**：基于 joint scaling law 预测 EL>7x → 选择 A=3.4%, G=12, E^s=1。Architecture: 20 layers, d_model=2048, d_ffn=5120, d_expert=384, 16 heads/4 kv_heads, E=384, E^a=12, E^s=1 (N=17.5B, N^a=0.85B)。每个 token 仅激活 3.4% 的总参数。Training: AdamW (β1=0.9, β2=0.95, wd=0.1), WSD LR schedule, η_max=3.78e-4, B=1792。Dense baseline: 28 layers, d_model=4096, d_ffn=14336, N=6.11B, η_max=2.93e-4, B=2048。
  - **Serving/系统框架层**：Ling 系列内部训练框架。论文未说明 distributed training 细节（DP/TP/EP 配置、all-to-all communication、expert parallelism 等）。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：标准训练 pipeline。MoE routing: top-k softmax gate + load balancing loss (coeff=0.01) + router z-loss (coeff=0.001)。Training tokens=1T (vs dense 的 1T)，C_moe ≈ 1.43e21 FLOPs vs C_dense ≈ 1e22 FLOPs → EL ≈ 7x。
  - **硬件架构层**：GPU 训练（具体型号未明确说明）。MoE 的高 total params (17.5B) 需要更多 GPU memory 存放 expert 参数，但仅 0.85B 激活计算，FLOPs 远低于 6.1B dense。
  
  效果：Ling-mini-beta (0.85B active) 在 1T tokens 训练后，overall benchmark average 45.5 vs Dense-6.1B 44.0，验证 >7x EL。证明了 scaling law 的预测准确性。
