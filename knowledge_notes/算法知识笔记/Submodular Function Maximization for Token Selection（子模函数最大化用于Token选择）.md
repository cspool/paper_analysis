## Submodular Function Maximization for Token Selection（子模函数最大化用于Token选择）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
子模函数（Submodular Function）是一种具有"边际收益递减"性质的集合函数：对任意 A ⊆ B ⊆ N 和 s ∈ N\B，有 f(A ∪ {s}) - f(A) ≥ f(B ∪ {s}) - f(B)（加入同一元素，小集合的增量大于或等于大集合的增量）。MMTok 将 vision token selection 的覆盖函数 f(S; M) = (1/m) Σᵢ max_{j∈S} M_{i,j} 证明为子模函数（Leskovec et al., 2007 的设施选址函数变体）。最大化一般子模函数是 NP-hard（Khuller et al., 1999），但贪心算法可以保证解不差于最优解的 (1-1/e) ≈ 63%（Nemhauser et al., 1978）。MMTok 利用两个关键性质：(1) 覆盖函数是子模函数；(2) 两个子模函数之和仍为子模函数（加法保持子模性），因此 T-V coverage + α × V-V coverage 的合并目标仍可被贪心算法近似优化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
子模覆盖贪心算法的核心步骤：
```
Algorithm: Greedy Submodular Maximization for Coverage
输入: 相似度矩阵 M ∈ R^(m×n), 目标选择数 k
输出: 选择的 token 索引集合 S, |S|=k

S = ∅
# 维护当前每个目标 token 的最佳覆盖值
coverage = zeros(m)       # coverage[i] = max_{j∈S} M[i,j]
for iter in 1..k:
    best_s, best_total = -1, -inf
    for s in 1..n, s ∉ S:
        # 计算加入 s 后的新覆盖值（增量计算）
        new_total = 0
        for i in 1..m:
            new_total += max(coverage[i], M[i,s])
        if new_total > best_total:
            best_s, best_total = s, new_total
    S.append(best_s)
    # 更新 coverage
    for i in 1..m:
        coverage[i] = max(coverage[i], M[i, best_s])
return S
```

理论性质：
- 子模性：覆盖函数满足 f(A ∪ {s}) - f(A) ≥ f(B ∪ {s}) - f(B)（A ⊆ B 时边际收益递减）
- 单调性：f(A) ≤ f(B) 当 A ⊆ B（加入更多 token 不会减少覆盖）
- 近似比：(1-1/e) ≈ 0.632（对单调子模函数 + 基数约束）
- 复杂度：O(kmn)，通过增量计算可优化至 O(kn)（利用 max 操作的结合性）

在 MMTok 中的实例化：source tokens = vision tokens (n), target tokens = text tokens (m) + vision tokens (n), 相似度矩阵 M = softmax-calibrated cosine similarity。两个覆盖子问题的和保持子模性（Corollary 1），因此 Alg. 2 的贪心算法对联合目标仍保持 (1-1/e) 近似保证。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
子模最大化在 MMTok 中使用 PyTorch 向量化实现：(1) 使用 `torch.matmul` 构建相似度矩阵；(2) 贪心循环使用 `torch.max` 和索引操作增量计算覆盖增益；(3) 每次迭代选增益最大的 token。关键优化：维护 running max coverage，每次迭代仅需对新候选 token 计算增量 O(m+n)，而非重新计算全部 O(kmn)。MMTok 的 PyTorch 实现使 2880 tokens 选 160 仅需 6.4ms。子模函数最大化也被广泛应用于其他领域：主动学习（传感器放置）、摘要生成（文档覆盖）、推荐系统（多样性最大化）等。在 VLM token selection 场景，其关键优势是理论保证 + 高效贪心实现 + 可组合性（多目标加权和仍保持子模性）。

涉及论文标题：
- MMTok__Multimodal_Coverage_Maximization_for_Efficient_Inference_of_VLMs
