## Stuffed_Mamba__State_Collapse_and_State_Capacity_of_RNN-Based_Long-Context_Modeling

- baseline方法是什么？
  Baseline 是标准 Mamba-2 模型（Dao & Gu, 2024），训练在 8K 上下文长度上。Mamba-2 使用选择性状态空间模型（SSM），每层 H 个 head 并行计算，状态大小 HPN = 256d（N=128, P=64, H=2d/P），约等于同等 hidden dim Transformer 的 KV cache 大小。Mamba-2 内置指数记忆衰减机制 α_t = exp(-Δ_t·exp(A)) ∈ (0,1)，理论上 α_t → 0 可完全遗忘、α_t → 1 可完全保留。Baseline 在训练长度内表现良好（LM loss 正常，passkey retrieval 近乎完美），但 context > 8K 后：(a) LM loss 急剧爆炸（perplexity 从 ~10 升至 ~100+），(b) passkey retrieval 准确率从 >95% 骤降至 ~0%，(c) 状态分布（mean/variance）出现 outlier channel 驱动的方差爆炸，导致输出 incoherent。

  全栈执行例子（Mamba-2 370M 在 8K 训练长度下处理 32K 上下文）：
  - 算法层：输入 32K tokens → Embedding → 48 层 Mamba-2 block（每层: RMSNorm → 输入投影 expand 2× → CausalConv1d(k=4) → SiLU → SSM selective scan: Δ_t=Softplus(W_Δ u_t+b_Δ), α_t=exp(-Δ_t·exp(A)), B_t=σ(Conv(W_B u_t)), C_t=σ(Conv(W_C u_t)), x_t=SiLU(Conv(W_x u_t))^T → h_t=h_{t-1}·α_t + Δ_t·B_t·x_t → y_t=C_t·h_t + D⊙x_t → SiLU gate + 输出投影 → residual）→ LM head → logits。在 t=8K-32K 时，某些 head 的 α_{1:t} 仍 > 0.997（首 token 几乎不衰减），状态中累积了过多历史 token 的加权信息，导致 memory interference——query C_t 与状态中任意 token 的 B_i 非正交时产生检索误差，token 越多误差越大，最终 loss 爆炸。
  - 系统框架层：PyTorch + HuggingFace Transformers（Mamba-2 官方实现）。FP32 推理，greedy decoding。
  - 编译框架层：论文未明确说明。
  - kernel调度层：Mamba-2 官方 selective scan kernel（Triton/CUDA fused kernel），并行化递归计算。状态为固定大小，O(1) per-token 计算和内存。
  - 硬件架构层：NVIDIA A800 80GB GPU。未涉及硬件修改。

  Baseline 核心缺陷：
  1. **无法遗忘（Inability to Forget）**：虽然 Mamba-2 有 α_t 衰减机制，但训练过程中模型学会了保留几乎所有信息（α_t 始终接近 1）以最小化 LM loss，而非在必要时遗忘。在 8K 训练长度内这可行，但超出后状态被"塞满"，信息干扰导致检索失败。
  2. **状态过参数化（State Overparameterization）**：状态大小 N_S = 256d 相对于 8K 训练长度过大，模型无需学习遗忘即可达到低 loss——本质是一种过拟合。更多训练数据反而加剧此问题：模型学会更激进地保留信息（Figure 8: passkey 精度随数据量增加反而下降）。
  3. **方差爆炸（Variance Explosion）**：超过训练长度后，某些 head 的状态均值和方差急剧变化，由少数 outlier channel（~5% 的 channel）驱动，这些 channel 的值在 t > T_train 时发散。
  4. **大模型更差**：780M 的长度泛化比 370M 更差（更大的状态 → 更严重的过参数化）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文从诊断角度分析了 Mamba-2 的长度泛化失败根因，提出了两条高层次的解决方案方向（遗忘诱导方法）和关键训练指导原则。

  **核心发现与设计原则：**
  (1) **遗忘阈值定律**：T_forget = 5.172 · N_S − 4.469（线性关系，R² > 0.999）。训练长度必须大于此阈值，模型才能学会有效遗忘。对于 370M 模型（N_S=12.9M），T_forget ≈ 66.7K tokens。
  (2) **召回能力定律**：T_recall = 4.756 · (1.365^{N_S} − 1) − 0.742（指数关系，R² > 0.999）。即使超过遗忘阈值，模型仍可检索信息到远超训练长度的上下文——370M 模型持续预训练后在 256K passkey retrieval 上近乎完美。
  (3) **状态过参数化假说**：遗忘只会在训练上下文的信息量超过状态容量时发生。这解释了为什么更大的状态需要更长的训练长度。

  论文方法全栈执行例子（使用 Sliding Window 干预的 Mamba-2 370M 处理 32K 上下文）：
  - 算法层：与 baseline 相同的前向过程，但在每个 Mamba-2 head 的 state update 后添加 Sliding Window 后处理：(a) 正常计算 h_t；(b) 维护 Δ_sum（ Δ 的累积和）和 h_{t-w}（w 步前的状态）；(c) 计算 α_window = exp(-Δ_sum·exp(A))；(d) h_t^{(w)} = h_t − α_window·h_{t-w}（精确窗口状态）；(e) 用 h_t^{(w)} 替代 h_t 进行 query y_t = C_t·h_t^{(w)} + D⊙x_t。这等价于强制遗忘 window 之前的所有 token，将有效上下文缩短为 w tokens。RRI 方法更温和：α_t' = α_t^{0.9999}（轻微加速衰减），B_t' = 0.75·B_t（减弱新信息插入），在 32K 上将 LM loss 从 ~15 降至 ~8。两种方法均无需训练。
  - 系统框架层：修改 Mamba-2 的推理代码，在 state update 后插入干预逻辑。无需重新训练或微调。
  - 编译框架层：论文未明确说明。
  - kernel调度层：Sliding Window 仅需额外存储 h_{t-w} 和 Δ_sum（两种浮点数，忽略不计），额外计算量为一个标量乘法和一个矩阵减法（O(NP)，极小）。RRI 的额外开销为标量乘法和指数运算，同样可忽略。
  - 硬件架构层：NVIDIA A800 80GB GPU。未涉及硬件修改。

  关键设计选择映射到缺陷：
  - 缺陷1（无法遗忘）→ **通过状态过参数化假说揭示了根因**：遗忘机制未有效学习不是因为架构缺陷，而是训练长度不足。RRI 和 Sliding Window 证明了"强制遗忘可以修复长度泛化"，提供了因果证据：问题在遗忘，不在检索。
  - 缺陷2（状态过参数化）→ **建立 T_forget ∝ N_S 的线性缩放律**：对于任意状态大小 N_S，存在最小训练长度使模型学会遗忘。这为训练长上下文 Mamba 模型提供了具体指导——8K 训练长度对 370M 模型（N_S=12.9M）严重不足，需 ~67K。
  - 缺陷3（方差爆炸）→ **通过 per-channel 状态统计揭示了 outlier channel 现象**：发现方差爆炸由少数 channel 驱动，其余 channel 保持稳定。使用 "newlines" prompt（恒定输入）来分离"上下文长度效应"和"输入变化效应"，确认爆炸是 context length 而非输入内容的函数。
  - 缺陷4（大模型更差）→ **状态大小是根本变量**：不是参数数量，而是 N_S = 256d（每层状态元素数）决定长度泛化能力。这解释了为什么 780M 比 370M 更差——本质是 N_S 更大（19.3M vs 12.9M），而非参数多。

  持续预训练的关键实践指导：
  - 训练长度应随状态大小线性增长：T_train > 5.172 · N_S − 4.469
  - 使用 Truncated BPTT（12 序列拼接，stop gradient 在序列边界）使状态初始值分布更多样化
  - WSD LR scheduler（10% decay steps）允许从中间 checkpoint 高效恢复
  - 数据过滤：丢弃短于 4K tokens 的文档，确保训练数据有足够长距离依赖
  - 验证策略：passkey retrieval 对长度泛化敏感度远高于 validation loss，应作为主要验证指标

  370M Mamba-2 持续预训练结果（256K 上下文）：
  - 在完整 256K passkey retrieval 上达到近乎完美的准确率
  - 据论文所知，这是首个 <1B 参数模型在此长度上达到如此性能
  - 超越了同参数规模的 Transformer 模型

  论文的局限与未解决问题：
  - RRI 和 Sliding Window 方法会牺牲短上下文性能（弱记忆插入）
  - Sliding Window 需要选择窗口大小 w，w 的选择对最终性能敏感
  - 论文未提供正式的训练代码仓库（依赖 Mamba-2 官方实现 + 自定义训练脚本）
  - 遗忘阈值的线性关系仅在 Mamba-2 上验证，RWKV 等其他 RNN 架构的关系未充分探索
  - 780M 模型的 T_forget 超出实验资源（>128K），未能直接验证
