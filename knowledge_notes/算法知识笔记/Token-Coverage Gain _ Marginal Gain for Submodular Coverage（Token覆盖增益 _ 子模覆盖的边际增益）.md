## Token-Coverage Gain / Marginal Gain for Submodular Coverage（Token覆盖增益 / 子模覆盖的边际增益）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token-Coverage Gain（也称 Marginal Gain）是 SCOPE 的核心选择机制，量化将候选 token v 加入当前已选集 S 后带来的额外覆盖度。其理论基础是**子模函数（submodular function）**的边际增益性质。SCOPE 定义 set-coverage 函数：

$$f(\mathcal{S}) = \sum_{u \in \mathcal{V}} \max_{s \in \mathcal{S}} \text{sim}(u, s)$$

该函数是 monotone submodular 的（满足 diminishing returns 性质）。Marginal gain 定义为：

$$\Delta(v; \mathcal{S}) = f(\mathcal{S} \cup \{v\}) - f(\mathcal{S}) = \sum_{u \in \mathcal{V}} \max(C(u, \mathcal{S}), \sin(u, v)) - C(u, \mathcal{S})$$

其中 C(u, S) = max_{s∈S} sim(u, s) 是 token u 在 S 下的当前最佳 coverage。这个边际增益刻画：v 的加入能对多少还未被良好覆盖的 token 提供更好的相似度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
边际增益在 SCOPE 贪心选择中的计算：

```
# 每轮迭代, ∀ 候选 token v ∉ S:
Δ = 0
for each u in V:  # 遍历所有 N 个 token
    current_best = c_u              # u 在 S 下的当前最佳相似度
    new_best = max(current_best, S_uv)  # 加入 v 后的新最佳相似度
    Δ += (new_best - current_best)  # 累加增益
# Δ 即为 v 的 marginal coverage gain
```

例子：假设 V = {猫头, 猫身, 背景, 香蕉}，S = {猫头}（c_u 已初始化）。候选 v = 背景 patch：猫头已有 sim=1.0，无新增益；香蕉 sim=0.2→0.3，增益 +0.1；背景 sim=0.1→1.0，增益 +0.9。Δ(背景) = 0.9。候选 v = 猫身：猫头 sim=1.0，无新增益；香蕉 sim=0.2，无新增益；背景 sim=0.2（不变），无增益。Δ(猫身) ≈ 0。因此背景会被优先选择，确保 coverage 扩展。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Marginal gain 的实现要点：
- 需要维护 coverage score 数组 c_u (N 维)，每次选出新 token 后更新
- 计算 Δ 需要遍历 N 个 token，每轮需要 O(N²) 次比较（N 候选 × N 全量）
- 总和 K 轮，总复杂度 O(K·N²)
- 贪心选择的 (1-1/e) 近似保证：对于 monotone submodular 函数，贪心选择可以达到最优解的 (1-1/e) ≈ 63% 近似
- SCOPE 将 marginal gain 乘以 attention saliency 得到 SCOPE score，打破纯 coverage 贪心，使选择同时考虑显著性和覆盖度
- 理论参考：Iyer et al. "Submodular combinatorial information measures with applications in machine learning" (ALT 2021)

涉及论文标题：
- SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs
