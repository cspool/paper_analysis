## Submodular Set Coverage Maximization for Token Selection（子模集合覆盖最大化用于Token选择）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
子模集合覆盖最大化（Submodular Set Coverage Maximization）是 SCOPE 方法的理论基础。一个集合函数 f 是子模的（submodular），如果满足 diminishing returns 性质：向小集合添加元素带来的边际增益 ≥ 向大集合添加相同元素带来的边际增益。SCOPE 定义的 set-coverage 函数 f(S) = Σ_{u∈V} max_{s∈S} sim(u, s) 是 monotone submodular 函数，这意味着贪心选择策略可以达到 (1-1/e) ≈ 63% 的最优近似保证。

子模覆盖最大化在 ML 中的应用广泛，包括：数据摘要（document summarization）、主动学习（active learning）、特征选择（feature selection）、以及 SCOPE 中的 visual token selection。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SCOPE 将 token selection 建模为 cardinality-constrained monotone submodular maximization：

$$\max_{\mathcal{S} \subseteq \mathcal{V}, |\mathcal{S}| = K} f(\mathcal{S}) = \sum_{u \in \mathcal{V}} \max_{s \in \mathcal{S}} \operatorname{sim}(u, s)$$

标准贪心算法：每次选择边际增益 Δ(v; S) 最大的元素，重复 K 次。理论保证：
- f(∅) = 0（空集覆盖度为0）
- f 单调：S ⊆ T ⇒ f(S) ≤ f(T)（更多 token 不会降低覆盖度）
- f 子模：Δ(v; S) ≥ Δ(v; T) for S ⊆ T（diminishing returns）
- 贪心解 f(S_greedy) ≥ (1-1/e) · f(S_opt)（近似保证）

SCOPE 的独特之处：将标准 submodular maximization 的边际增益与 attention saliency 相乘，得到 SCOPE score，打破了纯子模优化的近似保证，但在实践中取得了更好的 saliency-coverage trade-off。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 贪心选择是最常用的 submodular maximization 近似算法，时间复杂度 O(K·N²)
- 更高效的实现可使用 lazy greedy（利用 diminishing returns 减少评估次数）
- SCOPE 未使用 lazy greedy（因为引入了 saliency 加权打破了单调子模性）
- 在每个 MLLM query 中在线运行（training-free）
- 子模性保证来自于 cosine similarity 的 max 聚合的数学性质，而非特定于视觉 domain
- 参考论文：Iyer et al., "Submodular combinatorial information measures with applications in machine learning", ALT 2021

涉及论文标题：
- SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs
