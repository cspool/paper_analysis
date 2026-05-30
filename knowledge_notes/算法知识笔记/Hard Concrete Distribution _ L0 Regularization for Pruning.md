## Hard Concrete Distribution / L0 Regularization for Pruning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Hard Concrete Distribution（Louizos et al., ICLR 2018）是将离散二值 mask 优化连续化的方法。通过 Gumbel-Softmax 重参数化 + hard sigmoid gate，使 Bernoulli 采样可微且支持 {0,1} 端点的非零概率质量。PruLong 用它学习 attention head 的 retrieval/streaming 分类 mask，消除 DuoAttention continuous gating 的 train-test rounding gap。

从算法pipeline角度拆解术语。

```
// Hard Concrete 重参数化（PruLong）
u ~ Uniform(1e-6, 1-1e-6)
s = σ( (2/3) × log(u/(1-u)) + log_α )
g̃ = -0.1 + 1.1 × s
z̃ = clamp(g̃, 0, 1)  // hard gate: support {0,1}

// 期望 L0 稀疏度（闭式）
P(z > 0) = σ(log_α + log(10))
s(π) = 1 - (1/(L×H)) × Σ P(z_{i,j} > 0)

// Lagrangian: L_reg = λ1(s - t) + λ2(s - t)²，λ1,λ2 gradient ascent

// 最终：max_{λ1,λ2} min_{log_α} E[NTP_loss] + L_reg
```

术语一般如何实现？如何使用？

PruLong 训练：1000 steps, 1M token batch, seq_len 131K, LR=1.0 for log_α/λ, sparsity warmup 0→t over 800 steps, 不更新 model weights。训练后取 top k% log_α = +∞ (z=1), 其余 -∞ (z=0)。原始：https://github.com/AMLab-Amsterdam/L0_regularization。PruLong：https://github.com/princeton-pli/PruLong。

涉及论文标题：
- Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs

---
