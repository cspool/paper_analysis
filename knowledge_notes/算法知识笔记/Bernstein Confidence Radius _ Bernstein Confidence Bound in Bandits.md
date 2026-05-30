## Bernstein Confidence Radius / Bernstein Confidence Bound in Bandits

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bernstein Confidence Radius 是 bandit 中使用 Bernstein 不等式构造的方差自适应置信半径，由 Audibert, Munos & Szepesvári (Theoretical Computer Science, 2009) 在 UCV 算法中引入。与标准 UCB 使用 Hoeffding 不等式（置信区间宽度仅依赖样本数，与方差无关）不同，Bernstein 版同时利用经验均值 μ̂_a 和经验方差 σ̂_a²：β_a(n) = sqrt(2·σ̂_a²·ln(n) / N_a(n)) + 3·ln(n) / N_a(n)。当 arm 方差很小时（如静态场景 clip），界宽显著小于 Hoeffding 界——因为 sqrt(σ̂_a²/N) << sqrt(1/N)。这使算法能更快排除低方差低质量 arm，将更多采样预算投向高方差高不确定性 arm。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 每个 arm a 在 n 轮总采样后
N_a = max(1, pulls_of_arm_a)
μ̂_a = mean(observed_rewards_a)
σ̂_a² = variance(observed_rewards_a)

# Bernstein 置信半径 (FOCUS Eq.5)
β_a = sqrt(2 * σ̂_a² * ln(n) / N_a) + 3 * ln(n) / N_a

# 高概率保证 (Theorem B.1): P[|μ̂_a - μ_a| ≤ β_a] ≥ 1 - 6/n

# 与 Hoeffding 版对比
# Hoeffding: β = sqrt(ln(n) / (2*N_a))   — 固定宽度，忽略方差
# Bernstein: β = sqrt(2σ̂²*ln(n)/N) + 3*ln(n)/N — 方差自适应
```

FOCUS 消融（Table 8）：FOCUS-M（仅经验均值）= 63.0%, FOCUS（加 Bernstein）= 63.5% on LLaVA-Video。增益来自 Bernstein 对高不确定性 arm 的额外探索激励。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Bernstein confidence radius 的实现仅需每个 arm 维护 μ̂_a 和 σ̂_a²（Welford 在线更新算法），计算开销 O(M)。在 FOCUS 中，β_a 用于构造 optimistic mean μ̃_a = μ̂_a + β_a（Stage I 后 arm 粗选），也用于判断 arm 的探索价值。效果取决于 reward 的真实方差（clip 内帧的 relevance 波动）和每 arm 采样数。

涉及论文标题：
- FOCUS__Efficient_Keyframe_Selection_for_Long_Video_Understanding
