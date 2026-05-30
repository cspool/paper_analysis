## Constituent Redundancy in KV Cache (KV Cache 构成性冗余)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Constituent Redundancy (构成性冗余) 是 SpindleKV 首次识别并提出的一种 KV Cache 冗余类型。与传统的"token 间冗余"（不同 token 注意力贡献不同，低贡献的可被 evict）不同，构成性冗余描述的是：在 Transformer 浅层中，不同 token 的 Key/Value 向量之间存在极高的余弦相似度——这些向量的"构成成分"高度重叠，可被分解为一组有限"基础向量"的线性组合。SpindleKV 实验发现 LLaMA2-7B-chat 浅层中大量 token 对的 KV 余弦相似度超过 0.9（Key），而在深层这种相似性急剧下降。

从算法pipeline角度拆解术语：

**构成性冗余 vs. 注意力稀疏性**：

| 维度 | 浅层 (0-10) | 深层 (20-31) |
|------|------------|-------------|
| 注意力稀疏性 | 低（分布均匀） | 高（集中在少数 token） |
| KV 余弦相似度 | 高（>0.9） | 低 |
| 适合的压缩方法 | CodeBook replacement | Token eviction |

产生原因：浅层 token 经历较少的 Transformer 编码迭代，上下文信息整合有限，KV 向量仍是相对"原始"的基础表示。深层 token 经多次 self-attention + FFN 变换后被上下文信息"分化"，相似度下降。

术语一般如何实现？如何使用？

利用构成性冗余进行压缩的实现是 CodeBook-Based KV Cache Compression。实验验证：仅用码本压缩（无 eviction）在 LLaMA2-7B-chat 上 50% KV Cache 保留率下 LongBench 准确率无下降。构成性冗余的发现区分了 SpindleKV 与纯 eviction 方法的关键洞察：eviction 仅对深层有效，码本压缩弥补了浅层不足，两者结合使 SpindleKV 在所有保留率下均优于 PyramidKV/PyramidInfer。

涉及论文标题：
- SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers
