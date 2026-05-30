## NormHead

术语解释
对 LLM 输出投影层权重进行 L2 归一化后再用于 token 预测的训练稳定性技术。Ling 团队在 MoE 训练中使用，抑制 loss spike 期间的输出 norm 波动。

术语是什么？
Ling 团队发现 LM-Head 的输出 norm 在 loss spike 期间不稳定——权重范数波动放大梯度异常。NormHead 强制 W_lm_head 每行范数为 1 以消除此效应。

公式：h_o = W_lm_head / ||W_lm_head||₂ · h

术语一般如何实现？如何使用？
- 在 LM-Head forward 中插入 F.normalize(weight, p=2, dim=1)
- 与 router z-loss 协同（分别稳定输出层和路由层）
- MoE 训练受益更显著

涉及论文标题：
- Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs

---
