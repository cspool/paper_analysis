## AdaMOE Token-Adaptive Routing with Null Experts for Mixture-of-Experts Language Models

- baseline方法是什么？
  **Baseline 为 vanilla MoE with fixed top-k routing**: 每个 MoE layer 中，router 计算 Softmax(W_g · x) 得到所有 n 个 expert 的概率分布，固定选择 top-k 个 expert 激活。所有 token 无论语义重要性或复杂度，都激活相同数量 k 个 expert。典型例子：Mixtral-8x7B 使用 n=8 FFN experts, k=2 top-2 routing；Mo-LoRA 使用 n=4 LoRA experts, k=1 or k=2 routing。

  **Baseline 全栈执行例子（以 Mixtral-8x7B 推理一个 token x 为例）**：
  - **算法层**: token embedding x → Router: G(x) = Softmax(TopK(x · W_g, k=2)) → 2/8 FFN experts 激活 → y = G(x)_1 · E_1(x) + G(x)_2 · E_2(x) → 所有 token 固定使用 2 experts。无论 "apple" 还是 "the" 都消耗相同 FLOPs。
  - **系统框架层**: HuggingFace Transformers / vLLM → Gate 计算 → TopK (k=2) → Expert FFN 计算（各 expert 独立 GEMM）→ Weighted sum combine。无自适应机制。
  - **编译框架层**: 论文未明确说明（标准 PyTorch eager execution）。
  - **Kernel/运行时调度层**: 论文未明确说明（标准 cuBLAS GEMM，2 experts per token）。
  - **硬件架构层**: 论文未明确说明（标准 GPU，所有 expert 参数驻留 HBM）。

  **Baseline 的核心缺陷**：
  1. **固定 expert 激活数不考虑 token 差异**: 语义丰富的 token（名词、动词）和功能 token（标点、连词、<EOS>）消耗相同计算量。论文通过 SocialIQA 上 Mixtral-8x7B 的路由分布分析验证：各层 token 路由概率分布的 sharpness 差异巨大——部分 token 极度倾向单一 expert，而另一部分 token 分散到超过 2 个 expert，证明固定 k 对所有 token 并非最优。
  2. **无法按需分配计算资源**: 简单 token 被过度处理（浪费 FLOPs），复杂 token 可能资源不足。无法根据计算预算灵活调整 expert 负载。
  3. **Expert-Choice Routing 的因果不适配**: Expert-choice routing 可实现不等量选择，但依赖 future tokens 做 top-k token selection，不适合 auto-regressive text generation。

- 论文方法是什么？如何对应解决Baseline的缺陷？

  **论文方法**: AdaMOE = 在 MoE layer 的 expert set 中引入固定数量 m 的 **null experts**（zero mapping, 零 FLOPs），并将 TopK 的 k 值增大。每个 token 仍做固定 top-k 选择，但当 null experts 被选中时无计算开销，因此实际激活的 true expert 数量随 token 自适应变化。通过修改 load balancing loss（null experts 间不做负载均衡）和 annealing α 训练策略控制平均 expert 负载。

  **Defect→Design 映射**:

  | Baseline 缺陷 | AdaMOE 设计选择 | 解决机制 |
  |---|---|---|
  | 固定 k 使所有 token 消耗相同 FLOPs | 引入 m 个 null experts + 增大 k 值 | Token 在 top-k 中可选 0~k 个 true expert（null expert 不消耗 FLOPs），实现 token-adaptive routing。简单 token 倾向选更多 null expert，复杂 token 倾向选更多 true expert |
  | 无法按计算预算调整 expert 负载 | 通过调整 m（null expert 数量）和 α（load balancing loss 系数）控制平均 true expert 使用率 | 增大 m → 更多 null experts → 降低 true expert 负载；annealing α（先紧后松）实现 performance-efficiency tradeoff。ARC-C 上 FLOPs 减少 14.5% 同时 accuracy 提升 1.69% |
  | Expert-Choice Routing 不适合自回归 | Token-choice routing 天然适配 causal LM | 每个 token 独立选择自己的 top-k experts，不依赖 future tokens，与标准 transformer 推理完全兼容 |
  | 传统 load balancing 对 null experts 施加不必要约束 | 修改 load balancing loss: null experts 之间不做负载均衡 | 将所有 null experts 视为同质，用平均 f_j 替代各自 f_j，避免对 router 施加无意义的约束。实验验证 ℓ_null 比 ℓ_bal 在 RTE/COLA/SQA/OQA 上显著提升 accuracy |
  | Top-k 增大后 normalization 方式选择 | 仅对 top-k 中的 true experts 做 Softmax normalization | 保证加权输出与 vanilla MoE 的数值尺度一致（option 2 在 SIQA 上 accuracy 81.27 vs option 1 80.19） |

  **AdaMOE 方法全栈执行例子（以 Mixtral-8x7B + AdaMOE (m=8, k=3) 推理一个 token x 为例）**：
  - **算法层**: token embedding x → Router: G(x) = Softmax(TopK(x · W_g, k=3)) where W_g ∈ R^{d × (8+8)} → 若选出 {E_2, E_5, null_3} → 仅对 E_2, E_5 做 Softmax (option 2) → y = w_2 · E_2(x) + w_5 · E_5(x) → 实际仅 2 true experts FLOPs。若选出 {null_1, null_3, null_6} → y = 0，token 完全绕过此 MoE layer，类似 MoD 的 "bypass" 行为。平均 Load = 1.66（baseline Load = 2.00）。
  - **系统框架层**: HuggingFace Transformers 推理 → 原始 gate module 扩展 gate2 维度（gate2 output=8 for m=8 null experts）→ Router 拼接 gate+gate2 输出 → TopK(k=3) → 仅执行 true experts 的 FFN 计算 → null experts 不触发任何 GEMM kernel → 减少 FLOPs（↓14.5% on ARC-C）。
  - **编译框架层**: 论文未明确说明。
  - **Kernel/运行时调度层**: 论文未明确说明（实际 expert 激活数减少 → 减少 FFN GEMM kernel launch 次数 → 推理延迟降低）。
  - **硬件架构层**: 论文未明确说明（标准 GPU，减少的 FLOPs 直接转化为能耗和延迟节省）。

  **关键设计对应关系**：
  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | Null experts (zero FLOPs) | 实现 token-adaptive routing 而不破坏 top-k 机制 | Load 从 2.00 降至 1.66 (↓17%), FLOPs ↓14.5% on ARC-C |
  | m 和 k 超参数配置 | 按计算预算调整 expert 利用率 | m=32,k=6: Load=1.54, accuracy 仍高于 baseline；m=40,k=8: Load=1.34 |
  | Annealing α (α1=0.02→α2=0.0001) | 先紧后松: epoch 1 建立负载均衡, epoch 2 释放 token 自由度 | WINO accuracy: epoch1=76.24 → epoch2=81.93 (+5.69%) with minimal Load increase |
  | ℓ_null (no balancing among null experts) | 消除对 null experts 间的不必要约束 | ℓ_null vs ℓ_bal: RTE 67.51 vs 56.68, COLA 85.01 vs 83.68 |
  | Plugin-and-play (对 vanilla LLMs 和 MoE-LLMs) | 无缝集成，无需改动模型架构 | Llama2-7B Mo-LoRA: 各配置均超 baseline；Mixtral-8x7B: accuracy +1.69% @ FLOPs-14.5% |
  | 鲁棒性 (不同 epochs/LoRA ranks) | 方法对各种超参数不敏感 | Epoch 1 vs 10: AdaMOE 48.88→88.54 (baseline 45.95→87.19); Rank 8 vs 32: AdaMOE 48.88→49.01 (baseline 45.95→46.72) |

  **创新总结**: AdaMOE 通过在最简单的位置（expert set）插入最简单的操作（null expert = 0 FLOPs），以最小代价实现了 token-adaptive routing。核心洞察：将 "选择多少 expert" 的离散决策问题转化为 "选哪些 expert" 的连续路由问题（增加 m 个 null expert 并增大 k），配合 load balancing loss 自动调整平均 null/true expert 使用率。方法实现简单（仅需扩展 router 输出维度 + 修改 loss），兼容现有 (MoE-)LLM，可直接 fine-tune 使用，无需 pretrain from scratch。
