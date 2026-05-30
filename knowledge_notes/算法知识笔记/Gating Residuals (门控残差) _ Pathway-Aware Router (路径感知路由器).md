## Gating Residuals (门控残差) / Pathway-Aware Router (路径感知路由器)

术语是什么？
Gating Residuals 是 MoE++ 提出的路由增强机制：将前一层 MoE++ 的路由分数（softmax 概率分布）通过可训练变换矩阵 W_g∈R^{N×N} 融入当前层的路由计算中。具体公式：G(x^j) = W^j·x^j + W_g^j·G(x^{j-1})（j>1 时），首层仅使用 W^1·x^1。这种设计使每个 token 在选择当前层专家时能"记住"前一层走过的路径，保证异构专家架构下的路由稳定性。带有 Gating Residuals 的 Router 称为 Pathway-Aware Router。

从算法pipeline角度拆解术语：
```
# 第 j 层 MoE++ routing
logits = W @ x  # [B, S, N], 当前层的基础路由分数
if j > 1:
    logits += W_g @ prev_gating_scores  # [N×N]@[B,S,N]→[B,S,N]
gating_scores = softmax(logits, dim=-1)  # 存为下一层的 prev_gating_scores
selected_indices, selected_probs = topk_with_capacity(gating_scores, k=2, capacities)
```

W_g 是可学习的 N×N 矩阵，显式建模层间专家选择的相关性。实验证明（Fig.6）：Gating Residuals 降低了路由分数的方差，但不改变均值和值域范围，因此稳定了异构专家架构的路由。

术语一般如何实现？如何使用？
- 实现为 Megatron 中 MoE Router 的扩展：在 forward 时传入 prev_gating_scores，用额外的线性层 W_g 做变换后加到 logits 上
- W_g 初始化为小值或零，在训练中学习层间路由关联
- 消融实验（Tab.6）显示 Gating Residuals 在 1B 模型上提升 average benchmark 约 0.2 个百分点（47.5→47.7）
- 适用场景：任何多层 MoE 架构，特别是异构专家结构（不同层可能有不同专家类型分布时更有价值）

涉及论文标题：
- MoE++: Accelerating Mixture-of-Experts Methods with Zero-Computation Experts
