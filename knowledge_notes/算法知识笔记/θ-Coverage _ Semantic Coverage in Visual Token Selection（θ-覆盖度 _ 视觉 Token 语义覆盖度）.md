## θ-Coverage / Semantic Coverage in Visual Token Selection（θ-覆盖度 / 视觉 Token 语义覆盖度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
θ-Coverage 是 SCOPE 论文提出的度量指标，用于量化选定的 visual token 子集对全量 token 集合的语义覆盖程度。定义：对于 full token set V 和 selected subset V'，一个 token v ∈ V 被 V' "覆盖"，当且仅当存在至少一个 v' ∈ V' 使 cosine similarity sim(v, v') ≥ θ（θ 为相似度阈值）。θ-coverage 即为被覆盖 token 占全量 token 的比例：

$$\operatorname{Coverage}_{\theta}(\mathcal{V}',\mathcal{V}) = \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \mathbb{I}\left(\max_{v' \in \mathcal{V}'} \operatorname{sim}(v, v') \ge \theta\right)$$

高 θ 值要求更严格的相似度标准 → 通常导致较低的 coverage 但确保保留的 token 更具语义代表性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
θ-Coverage 在 SCOPE 中作为**分析工具**而非优化目标使用。论文用它诊断 saliency-only 方法的缺陷：通过测量不同 θ 下的 coverage 曲线，发现 saliency-only 方法的 coverage 低于 random baseline，证明其语义信息丢失严重。

计算流程（分析用，非在线推理）：
```
输入: V (N tokens), V' (K selected tokens), θ
covered = 0
for each u in V:
    max_sim = max_{v in V'} cosine_sim(u, v)
    if max_sim >= θ: covered += 1
return covered / N
```

在 MME benchmark 上，当 K=64（从 576 中选）时，saliency-only 在不同 θ 下的 coverage 显著低于 SCOPE。这个分析间接验证了 coverage-aware selection 的重要性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
θ-Coverage 主要作为离线分析工具：
- 用于评估剪枝后 token 子集的语义完整性
- 支持跨方法比较（saliency-only vs coverage-only vs SCOPE vs random）
- 帮助选择合理的 token 保留数量 K
- 论文未将 θ-coverage 直接用作训练或优化目标（SCOPE 使用 soft set-coverage function f(S) = Σ_{u∈V} max_{s∈S} sim(u,s) 作为优化目标，而非硬阈值版本）

涉及论文标题：
- SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs
