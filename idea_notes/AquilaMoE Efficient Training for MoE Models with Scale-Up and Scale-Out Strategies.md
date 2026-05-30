## AquilaMoE Efficient Training for MoE Models with Scale-Up and Scale-Out Strategies

- baseline方法是什么？
  **Baseline 1: 从头训练（From Scratch）**。大模型（32B MoE）完全随机初始化，在全部 5345B tokens 上从头预训练。需要最大规模的集群（1024 devices × 240 GFLOPS），训练吞吐仅 25B tokens/day，需 213.8 GPU-days 总训练时间。
  **Baseline 2: bert2BERT 初始化（FPI/Stacking AKI）**。使用 FPI（Function Preserving Initialization）或 stacking-based AKI 扩展小模型权重初始化大模型。FPI 导致对称权重（net2net 固有缺陷），stacking 导致层间输出空间不匹配（last layer output ≠ first layer input）。

  **Baseline 全栈执行例子（以训练 32B MoE 为例）**:
  - **算法层**: 随机初始化所有参数 → MoE router N(0, 0.02) → 从零开始学习所有 token 表征 → 需要全量 5345B tokens → load balancing loss + z-loss 稳定训练
  - **系统框架层**: 分布式训练框架（PyTorch + 自研 AI 框架）→ 1024 GPUs 数据并行 + expert 并行 → all-to-all scatter/gather 通信 → 训练吞吐 25B tokens/day
  - **编译框架层**: 论文未明确说明（BAAI 内部 AI 框架，可能基于 PyTorch + 定制算子）
  - **Kernel/运行时调度层**: 每 GPU 执行持有的 expert 的 FFN GEMM kernel → MoE layer 涉及 all-to-all 通信 token dispatch → router 计算 + top-k 选择 kernel
  - **硬件架构层**: 1024 × Ascend-like 240 GFLOPS accelerators → 25B tokens/day 吞吐 → From Scratch 需 5345B/25=213.8 天等效训练时间

  **Baseline 缺陷**:
  1. **计算和数据浪费**: 从头训练需要 5345B tokens，每个 token 都需要从零学习基础语言知识。
  2. **FPI 权重对称**: 扩展时将权重简单复制/拆分，导致对称权重在训练中梯度相同，有效参数减半。
  3. **Stacking 层间不匹配**: StackBERT 的层堆叠方法使第 L_1-1 层输出空间与第 0 层输入空间不匹配，导致训练初期的 loss spike 和不稳定。
  4. **GQA 不兼容**: 原始 AKI 仅支持 MHA，无法处理 Group Query Attention 模型的 attention head 扩展。

- 论文方法是什么？如何对应解决Baseline的缺陷？

  **论文方法**: EfficientScale = Scale-Up (AKI-Pro) + Scale-Out (Sparse Upcycling)

  1. **AKI-Pro 解决 FPI 对称性**: 使用相邻层权重而非同层复制来扩展宽度（继承 bert2BERT AKI），避免对称初始化，保证有效参数不减少。
  2. **Interpolation 替代 Stacking**: 深度扩展使用 `W'_l = floor(l × L_2 / L_1)` 插值而非直接复制堆叠，保证相邻层输出空间平滑过渡，训练更稳定（验证: FPI-Interpolation loss 3.31 vs FPI-Stacking loss 4.30 at M(32,4096)）。
  3. **GQA 兼容性改造**: 在源和目标模型 group 数一致的前提下，将每个 GQA group 视为独立 MHA block 进行 AKI 扩展，使 AKI 支持 GQA 模型。
  4. **Sparse Upcycling 解决从头训练**: 将 dense 模型 MLP 直接复制为 MoE experts，保留已学知识，仅需 545B tokens 微调（vs 5345B 从头训练）。

  **Defect→Design 映射**:

  | Baseline 缺陷 | EfficientScale 设计选择 | 解决机制 |
  |---|---|---|
  | 从头训练需 5345B tokens | Scale-Up 知识迁移 + Scale-Out 复用 dense 权重 | 仅需 3600(7B) + 1200(16B) + 545(MoE) = 5345B tokens total，但 7B 训练可复用已有小模型，实际额外训练仅 1745B |
  | FPI 对称权重导致有效参数减半 | AKI-Pro 用相邻层权重打破对称 | Validation loss 降低: M(32,4096) AKI-Pro 7.81 vs FPI 4.30 |
  | Stacking 层间不匹配导致训练不稳定 | Interpolation 深度扩展 | FPI-Interpolation loss 3.31 << FPI-Stacking loss 4.30 |
  | 原始 AKI 不支持 GQA | 将每个 group 视为独立 MHA block 扩展 | 支持 GQA 架构（16B 模型 8 KV groups），扩展后训练收敛正常 |
  | 从头训练 32B MoE 需 213.8 天 | Scale-Up + Scale-Out pipeline | 时间节省 4.12×，算力节省 3.35× |

  **EfficientScale 全栈执行例子（以 1.3B → 7B → 16B → 8×16B MoE 为例）**:

  - **算法层**:
    Phase 1: 加载 Aquila2-1.3B M(24,2048) 预训练权重 → Phase 2: AKI-Pro 宽度扩展 (768→4096 hidden dim, 2048→14336 intermediate) + Interpolation 深度扩展 (24→32 layers) + GQA 保持 32 groups → 初始化 M(32,4096) 7B 模型 → 连续预训练 3.6T tokens → 再次 Scale-Up AKI-Pro: M(32,4096)→M(40,5120), 深度 interpolation 32→40, 宽度 4096→5120, GQA 32→8 groups → 连续预训练 1.2T tokens → Phase 3: Sparse Upcycling ×8 experts → router N(0,0.02) 随机初始化 → top-2 routing → 连续预训练 545B tokens
  - **系统框架层**: BAAI 自研 AI 框架 → 1024 GPU data+expert parallelism → all-to-all token dispatch/gather → load balancing loss (λ=0.001) + z-loss (λ=0.01) → full BF16 训练
  - **编译框架层**: 论文未明确说明（BAAI 内部框架，可能基于 PyTorch+XLA 或自研编译器）
  - **Kernel/运行时调度层**: MoE layer forward: router Softmax + top-2 selection → all-to-all scatter tokens to expert-holding GPUs → 每 GPU 执行 2 experts' FFN GEMM (W_gate·x, W_up·x, W_down·h) → all-to-all gather → residual add + LN。Dense layer: 标准 Transformer block GEMM kernels
  - **硬件架构层**: Phase 1: 480 × 989.5 GFLOPS GPU → 279B tokens/day。Phase 2-3: 1024 × 240 GFLOPS accelerators → Scale-Up 70B/day, Scale-Out 25B/day。时间节省 4.12× (213.8→51.84 GPU-days), 算力节省 3.35×

  **关键数值验证**（Table 1, 4, 6）:
  | 指标 | Baseline (From Scratch) | EfficientScale | 提升 |
  |---|---|---|---|
  | M(32,4096) validation loss | 12.22 (random init) | 7.81 (AKI-Pro) | 初始 loss 降低 36% |
  | GSM8K-gen | - | 54.51 (8×16B MoE) vs 7.81 (7B) | MoE 大幅超越 dense |
  | 总训练时间 | 213.8 等效天 | 51.84 天 | **4.12× 时间节省** |
  | 总算力 | 52,592,640 GFLOPS-days | 15,705,343 GFLOPS-days | **3.35× 算力节省** |
  | MMLU-ppl | - | AquilaMoE 61.00 vs AquilaDense-16B 57.11 | +3.89 点 |
