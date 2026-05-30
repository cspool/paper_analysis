## Brainformers Trading Simplicity for Efficiency

- baseline方法是什么？
  **Baseline 为 GLaM（manually crafted sparse Transformer）**：标准稀疏 MoE Transformer，采用 uniform block 设计——每个 Transformer block 固定为 attention + FFN（dense block）或 attention + MoE（sparse block），两者严格交替排列。所有层使用相同的 Top-2 token-based gating、固定的 model dimension 和 expansion ratio（FFN hidden = 4× model dim）。架构由人工设计，无自动搜索优化。

  **Baseline 全栈执行例子（以 GLaM 8B/64E 模型训练一个 token 为例）**：
  - **算法层**：input token → Layer 1: Attention(4096-dim, 32 heads) + Dense FFN(4096→16384→4096) → Layer 2: Attention + MoE(Top-2 routing, 64 experts, each 4096→16384→4096) → Layer 3: Attention + Dense FFN → ... 交替重复 → 每 token 固定激活 2/64 experts → 计算 FLOPs 由固定 expansion ratio 决定
  - **系统框架层**：GLaM 训练框架（Google 内部，推测基于 TensorFlow/XLA + TPU） → Expert Parallelism 分布 64 experts 到多 TPU device → all-to-all token dispatch → TPU 间通信
  - **编译框架层**：论文未明确说明（Google 内部 XLA 编译）
  - **Kernel/运行时调度层**：TPU matrix unit 执行 dense FFN GEMM（4096×16384）和 MoE expert GEMM → Top-2 routing kernel → 每层严格交替导致不同层间计算量不均匀（MoE 层重、FFN 层轻）
  - **硬件架构层**：512 Cloud TPU-V4 chips → 143B total params 分布 → steps/sec = 0.39 → 训练收敛慢

  **Baseline 的核心缺陷**：
  1. **Uniform 架构限制效率**：固定 attention-FFN/MoE 交替导致架构缺乏灵活性，无法根据计算需求调整不同层的宽度和类型。GLaM 的 uniform 设计使得 MoE 层和 dense 层计算量差距大，层间负载不均。
  2. **固定 Top-2 gating 不是最优**：所有 token 固定激活 2 experts，但 Expert Choice routing 可能更优（允许 perfect load balance 且每 token 激活数量可变）。
  3. **固定 expansion ratio 浪费参数**：GLaM 使用固定 4× expansion（4096→16384），但 MoE 的多 expert 已提供宽度，不需要如此大的 expansion。
  4. **手动设计无法系统化优化**：人工调整架构维度（层数、宽度、expert 数）缺乏系统性，难以找到最优配置。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**：Brainformer = 通过演化搜索（Regularized Evolutionary Search）自动发现非均匀 Transformer block 架构，打破 uniform 交替限制，在固定训练时间预算下联合优化层类型序列、层宽度、gating 机制、routing 策略和激活函数。

  **Defect→Design 映射**：

  | Baseline 缺陷 | Brainformer 设计选择 | 解决机制 |
  |---|---|---|
  | Uniform attention-FFN 交替 → 架构不灵活 | Non-uniform block-wise architecture: 每层独立选择 F_attn/F_moe/F_ffn | 搜索发现最优层序列（如 Brainformer Block 1: 8 sub-layers 含 2 attention + 3 MoE + 3 FFN），减少 attention 频率降低计算 |
  | 固定 Top-2 routing → 负载不均 | 搜索同时优化 gating function（Top-2 vs Expert Choice） | 搜索选择 Expert Choice gating + capacity factor=1，实现 perfect load balance 和极致稀疏 |
  | 固定 4× expansion → 参数浪费 | 搜索可变 model dim + MoE/FFN hidden dim | 搜索选择更大 model dim(1024) + 更小 expansion factor，利用 MoE 多 expert 的宽度替代单层大 expansion |
  | 手动设计 → 非系统化 | Evolutionary search + fixed training time constraint | 在固定 wall clock time 下自动采样、训练、评估、选择最优架构 |
  | 训练预算分配不公 → 稀疏模型吃亏 | Fixed training time search + inference time constraint | 以训练时间和计算成本（而非参数总量）为比较基准，允许模型以更快 step time 换取更多 training steps |
  | Sparse model scaling 效率低 | Block-wise stacking: 搜索到的 block 通过 ScaleModelDim + StackNTimes 扩展到目标规模 | 100M→1B→8B 线性扩展，保持 block 结构不变 |

  **Brainformer 方法全栈执行例子（以 Brainformer-1 8B/64E 训练一个 token 为例）**：

  - **算法层**：input token → Brainformer Block（8 sub-layers）：
    Sub-layer 1: F_attn (model_dim=1024, 20 heads) → Multi-head Self-Attention
    Sub-layer 2: F_moe (model_dim=1024, moe_hidden=2048, ExpertChoice gating, capacity=1) → 每 token 平均路由至 1 expert，64 experts，perfect load balance
    Sub-layer 3: F_ffn (model_dim=1024, ffn_hidden=2048) → Gated GeLU activation
    Sub-layer 4: F_attn → Self-Attention
    Sub-layer 5: F_moe → Expert Choice MoE
    Sub-layer 6: F_ffn → Dense FFN
    Sub-layer 7: F_moe → Expert Choice MoE
    Sub-layer 8: F_ffn → Dense FFN
    → Block 重复 N 次（stacking）→ LM head → token prediction。相比 GLaM：attention 频率降低（2 vs 每层都有），expert 激活数降低（~1 vs 2），model dim 更大（1024 vs 4096 但 expansion 更小），总 activated params 更少（7.4B vs 9.8B）
  
  - **系统框架层**：Google 内部 TPU 训练框架 → Expert Parallelism（64 experts 分布 512 TPU V4）→ Expert Choice routing 天然 load balance → 无 auxiliary loss 即可均衡 → 减少通信等待。Brainformer-1 实现 1.96 steps/sec vs GLaM 0.39（5× faster）
  
  - **编译框架层**：论文未明确说明（Google 内部 XLA 编译，自动融合 TPU 计算图）
  
  - **Kernel/运行时调度层**：TPU V4 matrix unit 执行 MoE expert FFN GEMM（1024×2048, 1 expert/token avg → 远小于 GLaM 的 4096×16384×2）→ Attention kernel（1024-dim, 20 heads → 少于 GLaM 的 4096-dim, 32 heads）→ 总计算量大幅降低 → step time 5× faster
  
  - **硬件架构层**：512 Cloud TPU-V4 chips → 158B total params（高于 GLaM 143B），但仅 7.4B activated（低于 GLaM 9.8B）→ 更少的 per-chip 计算量 + Expert Choice 的 load balance → TPU 利用率更高 → 2× training convergence speedup

  **关键设计对应关系**：
  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | Non-uniform block（8 sub-layers, variable types） | 打破 uniform 交替限制，灵活组合计算 | Brainformer-1 PPLX 1.99 vs GLaM 2.12 at 8B scale |
  | Expert Choice gating (capacity=1) | 替代 Top-2 实现 perfect load balance + 更稀疏 | 每 token avg 1 expert, step time 5× faster |
  | 搜索可变 model dim + hidden dim | 优化 expansion ratio 匹配 MoE 的宽度 | Model dim 1024 + MoE hidden 2048（expansion~2× vs baseline 4×） |
  | Fixed training time search | 公平比较不同架构在相同预算下的质量 | Brainformer 在相同训练时间下达到更低 PPLX |
  | Block-wise stacking (ScaleModelDim + StackNTimes) | 从搜索到的小规模块扩展到生产规模 | 100M block → 1B → 8B 线性扩展 |
  | 减少 attention 频率 | Attention 在长序列上成本高 | Block 仅 2 attention 层（vs baseline 每层 attention） |
  | 2× convergence speedup + 5× step time speedup | 整体训练效率 | 512 TPU V4, same hardware, 更快达到目标 PPLX |
