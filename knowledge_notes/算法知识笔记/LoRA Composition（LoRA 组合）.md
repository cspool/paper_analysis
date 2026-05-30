## LoRA Composition（LoRA 组合）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LoRA Composition（LoRA 组合）是指将多个分别独立训练好的 LoRA adapter（每个适配特定任务/概念，如服装风格、面部特征、语言翻译能力等）组合为一个统一模型，使组合后模型同时具备各 LoRA 的特化能力。核心挑战：(1) 直接叠加多个 LoRA（$\hat{W} = W + \sum \Delta W_i$）在 N≥3 时会破坏预训练模型的生成能力（参数偏移过大）；(2) 归一化加权叠加（$\hat{W} = W + \sum w_i \Delta W_i, \sum w_i=1$）虽保护了生成能力，但会稀释每个 LoRA 的独有特征（各 w_i ≈ 1/N）。LoRA 组合的两大流派：(a) Linear Arithmetic Composition（线性算术组合）——直接对 LoRA 权重矩阵做加权求和；(b) Reference Tuning-based Composition（参考调优组合）——用小规模参考数据重训练整个模型以融合 LoRA 输出。

从算法pipeline角度拆解术语：
Linear Arithmetic Composition 的前向计算：
```
# 对于每个 transformer block 的每个线性层:
W ∈ R^{d×k}              # 预训练权重（冻结）
ΔW_i = B_i @ A_i         # 第 i 个 LoRA 的增量权重, B_i∈R^{d×r}, A_i∈R^{r×k}
w_i                       # 第 i 个 LoRA 的组合权重 (Σ w_i = 1)

# NLA (Normalized Linear Arithmetic) 组合权重:
W_hat = W + Σ_{i=1}^{N} w_i · ΔW_i

# 前向: y = W_hat @ x = W @ x + Σ_{i} w_i · (B_i @ A_i @ x)
```

关键特性：w_i 是全局标量，所有 transformer 层共享同一组 {w_i}。组合权重可在推理前一次性 merge（W_hat 预计算），推理时无额外开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Linear Arithmetic：最简单的实现是加载所有 LoRA weights → 按 w_i 缩放 → merge 到 base model（peft.merge_and_unload）。PEMs (Zhang et al. 2023) 定义 LoRA 算术算子（加法/减法），LoRAHub (Huang et al. 2023) 用 gradient-free 优化（CMA-ES）在 few-shot 样例上自动搜索 {w_i}。
- Reference Tuning-based：Mix-of-Show (Gu et al. 2023) 使用梯度融合 + 可控采样 + 位置 mask，但需要全模型重训练，灵活性差。
- 适用场景：V&L 域的多概念图像生成（同时生成多个视觉主体）、NLP 域的多任务能力组合（翻译+NLI+QA 一次推理）。实际部署中常见于 Stable Diffusion 生态（Civitai 上的 LoRA 组合）和 LLM 多任务 serving。

涉及论文标题：
- Mixture of LoRA Experts
- PEMs: Composing Parameter-Efficient Modules with Arithmetic Operations
- LoRAHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition
- Mix-of-Show: Decentralized Low-Rank Adaptation for Multi-Concept Customization of Diffusion Models
- SVDiff: Compact Parameter Space for Diffusion Fine-Tuning
