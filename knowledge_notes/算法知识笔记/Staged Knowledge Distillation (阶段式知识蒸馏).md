## Staged Knowledge Distillation (阶段式知识蒸馏)

术语解释
Staged Knowledge Distillation 是 DeepSpeed-MoE 在 MoS 中发现的 KD 策略：蒸馏仅在预训练初期（如前 400K steps）进行，后期停用 KD 仅优化标准 LM loss，解决学生模型容量不足导致的 underfitting 问题。

术语是什么？
标准 KD 在整个训练过程中同时优化任务损失和蒸馏损失（L = L_CE + α·L_KD）。DeepSpeed-MoE 发现对于 MoE 预训练的蒸馏，全程 KD 在后期反而伤害精度——因为 PR-MoE 学生已经通过减少层数降低了容量，进一步最小化 KD loss 迫使学生放弃对标准 LM loss 的优化（underfitting）。Staged KD 通过在训练后期停用 KD loss，允许学生在预训练后期专注于标准 LM loss，避免 underfitting。

从算法pipeline角度拆解术语：
公式：
$$\mathcal{L}_{\text{staged}} = \begin{cases} \mathcal{L}_{CE} + \alpha \cdot \mathcal{L}_{KD}, & \text{if step } < K \\ \mathcal{L}_{CE}, & \text{otherwise} \end{cases}$$
其中 K=400K 为 KD 停止步数，α 为加权系数。

Staged KD 有效性验证（Table 5, 350M+PR-MoE）：
- No KD (PR-MoE+L21): LAMBADA 62.33, BoolQ 52.35（baseline 无蒸馏直接减层）
- Full KD (全程): LAMBADA 61.56, BoolQ 57.89（全程蒸馏，LAMBADA 更差）
- Staged KD (MoS): LAMBADA 63.46, BoolQ 58.07（Staged KD 最佳，接近教师 63.65/59.88）

术语一般如何实现？如何使用？
- 实现于 DeepSpeed-MoE 训练流水线中
- K（停止步数）需要根据模型规模和训练总步数调参
- 适用于预训练阶段的 MoE 知识蒸馏场景
- 核心洞察：当学生容量不足时，"少即是多"——减少 KD 的干扰

涉及论文标题：
- DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale

---
