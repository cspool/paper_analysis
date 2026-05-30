## Lagrangian Constraint Training for Sparsity (拉格朗日约束稀疏训练)

术语是什么？

Lagrangian Constraint Training for Sparsity 是一种通过可训练 Lagrange 乘子（λ1, λ2）在 min-max 框架下自动平衡语言建模损失与稀疏度约束的训练方法。与固定惩罚系数不同，λ 通过 gradient ascent 动态调整，使不同任务能容忍不同的 sparsity-vs-performance gap。Elastic Attention（Tang et al., 2025）和 PruLong（Bhaskar et al., 2025）均采用此方法。

从算法pipeline角度拆解术语。

```
# 训练目标（Elastic Attention）
# 外层: max_{λ1, λ2}, 内层: min_{router params}
Ω_MSR = (1/(L·H)) · Σ_l Σ_h I[r_hard[l,h] == SA]
L_diff = Ω_MSR - t        # t = target sparsity
L = L_language + λ1·L_diff + λ2·L_diff²

# 梯度下降更新 router 参数
θ_router -= lr_router · ∂L/∂θ_router

# 梯度上升更新 Lagrange 乘子
λ1 += lr_λ · L_diff
λ2 += lr_λ · (2·λ2·L_diff)
```

术语一般如何实现？如何使用？

Elastic Attention：sparsity-sensitive tasks t=0.7（更多 FA），sparsity-robust tasks t=1.0（全部 SA）。λ 随机初始化，`lr_λ=1e-3`（高于 router lr=5e-4）。训练时不同 task 的 λ 收敛到不同值——敏感任务 λ 更大（更强约束），鲁棒任务 λ 更小（宽松），自动实现 task-dependent sparsity。PruLong 中 t warmup 0→0.7（800 steps），λ 同样可训练。两种方法都使用 non-tight constraint（不强制精确满足 t）。

涉及论文标题：
- Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers
- Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs
