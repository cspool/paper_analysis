## Training Time Constrained Search (训练时间约束搜索)

术语解释
Training Time Constrained Search 是 Brainformers 提出的公平模型比较和搜索方法：在固定的训练 wall clock time 预算（芯片数 × 训练时间）下进行架构搜索和评估，替代传统的"固定 training steps + 固定 params"比较范式。更快 step time 的模型自动获得更多 training steps，从而在架构搜索中天然偏向训练效率高的设计。

术语是什么？
传统模型比较的局限：
1. **Fixed params + fixed steps**：歧视总参数多但 activated params 少的稀疏模型
2. **Fixed activated params + fixed tokens**（GLaM 方法）：忽略 Chinchilla 定律——小模型可从更多数据受益
3. **Compute-efficient scaling**（Chinchilla）：固定 FLOPs budget，但未考虑架构变化的影响

Brainformers 扩展 Chinchilla 的理念到架构搜索：固定 wall clock budget，允许搜索算法在 model capacity 和 training steps 之间 trade off。

从算法pipeline角度拆解术语。
```
# Training Time Constrained Search 的优化框架

# 给定:
#   budget_chips = N_chips × training_hours  (固定)
# 约束:
#   step_time(architecture) ≤ baseline_step_time
# 目标:
#   minimize validation_perplexity at end of budget

# Step time 影响 training steps:
training_steps = budget_chips / step_time

# 例如 8B scale on 512 TPU V4:
# GLaM: step_time = 2.56s → 33,750 steps in 24h
# Brainformer-1: step_time = 0.51s → 169,412 steps in 24h
# → Brainformer-1 可在相同 wall clock time 内训练 5x 更多 steps

# 搜索 reward:
R = α × (1/perplexity) + β × (1/step_time)
# 或约束优化: R = -log(perplexity) s.t. step_time ≤ T_baseline
```

术语一般如何实现？如何使用？
- 适用于跨模型架构的公平比较（dense vs sparse, uniform vs non-uniform）
- 需要准确的 wall clock time 测量（包括所有通信、数据加载、同步开销）
- Early stopping 在 25% budget 时淘汰表现差的架构
- 局限性：对硬件平台敏感（不同硬件上的 step time ranking 不同）；FLOPs 不能替代 step time 测量（memory/communication 开销不可忽略）

涉及论文标题：
- Brainformers Trading Simplicity for Efficiency
