## Saliency-Skewed Attention Distribution in Visual Token Pruning（视觉Token剪枝中的显著性偏斜注意力分布）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Saliency-skewed attention distribution 是 SCOPE 论文揭示的 saliency-based token pruning 方法面临的关键问题：在 CLIP vision encoder 中，CLS token 对各 visual token 的 attention 分布高度偏斜——少数 token（如前景物体区域）获得极高的 attention 值，而绝大多数 token（如背景区域）的 attention 值几乎均匀地平坦分布（flat tail）。这种偏斜导致两个后果：(1) Top-K 选择几乎全部集中在前景区域，丢失背景上下文（semantic incompleteness）；(2) flat tail 区域的 token 之间 attention 差异极小，无法有效区分 informative vs redundant tokens（token indiscriminability）。

论文图 1(b) 展示了 MME benchmark 上前 128 个 token 的平均 attention 分布，显示 attention weights 迅速平坦化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
该观察在 SCOPE 中作为**动机分析**驱动算法设计，而非算法组件。其影响体现在：

1. 解释了为什么 saliency-only Top-K 方法在低 token budget 下性能急剧下降：flat tail 中即使存在 informative tokens（如"cat 旁边的地毯"可能对回答 "Where is the cat?" 有用），其 attention 与纯冗余 background token 几乎相同，Top-K 排序无法区分
2. 论证了引入 coverage metric 的必要性：coverage 不关心 attention 绝对值，而是基于 token 嵌入的语义相似度，能有效区分 semantic content
3. 支持 SCOPE score = Δ(v)·A_v^α 中 α 的设计：α=1.0 在保留高 attention token 的同时允许部分低 attention 但具有高 coverage 增益的 token 被选入

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
论文未提出专门"解决" attention skewness 的方法，而是通过 SCOPE 的覆盖度机制绕过了该问题。实践中，attention 偏斜程度的测量方法：
- 计算 attention 分布的 Gini 系数或熵值
- 绘制 attention 排序后的累积分布曲线（论文图 1(b)）
- 对比不同模型层级的 attention 偏斜程度（layer -2 已在 SCOPE 中被选为 saliency 来源）
- 偏斜度随 model scale 增大可能加剧（更大模型倾向于产生更集中的 attention）

涉及论文标题：
- SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs
