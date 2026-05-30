## Representative Key Vector for KV Cache Similarity Retrieval（KV缓存相似度检索的代表键向量）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Representative Key Vector 是 StreamKV 中用于高效 KV 相似度计算的每帧/每块的聚合特征向量。对于第 i 段的第 m 帧，其 per-patch key vectors 为 $[\mathbf{k}_{m,p}^i]_{p=1}^{P^2}$（$P^2$ 为 ViT patch 数），representative key 定义为所有 patch-wise key 的均值：$\mathbf{r}_m^i = \frac{1}{P^2} \sum_{p=1}^{P^2} \mathbf{k}_{m,p}^i \in \mathbb{R}^{D'}$。其中 $D'$ 为不区分 attention heads 的拼接维度（将所有 head 的 key 维度拼接为单个向量）。Representative key 的用途：(1) 作为 KV 压缩/检索中 cosine similarity 计算的输入（替代完整的 multi-head key tensor）；(2) 存入 KV Bank 的索引结构中，与对应 KV blocks 一一映射，实现快速相似度检索而不需要加载完整 KV blocks。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

Representative key 在 pipeline 中的计算和使用：

```
# Per-frame KV block 定义
# b_m^i = [(k_{m,p}^i, v_{m,p}^i)]_{p=1}^{P²}  # 所有 patch 的 KV

# Representative key 计算 (Eq.3)
r_m^i = (1/P²) Σ_{p=1}^{P²} k_{m,p}^i  ∈ R^{D'}
# 不区分 attention heads: D' = num_heads × head_dim
# 所有 head 的 key 拼接为一个长向量后取 patch 平均

# KV Bank 存储结构（per layer l）
B_l = [b_1, b_2, ..., b_n]     # KV blocks (存储完整 K, V)
R_l = [r_1, r_2, ..., r_n]     # Representative keys (用于检索)

# 检索时: 使用 R_l 做相似度计算，用 B_l 查表获取完整 KV
{sim_j} = cos_sim(R_l, criterion)  # 仅需轻量 D' 维向量比较
I = top_K(sim_j, K)                 # 选出 top-K 索引
{P} = [B_l[j] | j ∈ I]             # 从 Bank 获取完整 KV blocks
```

关键设计权衡：representative key 是 patch-mean pooled 的一维向量，丢弃了 patch 级空间信息但保留了帧级语义特征，使相似度检索的计算量从 O(P² × D') 降至 O(D')。

术语一般如何实现？如何使用？

实现方式：在每段编码完成后，对每帧的 key tensor（形状 [num_heads, P², head_dim]）做 reshape 到 [P², D'] 后在 P² 维度上取平均。Representative key 与对应的完整 KV blocks 成对存储。适用场景：(1) 需要快速遍历大型 KV 库进行相似度检索的场景；(2) 可推广到任何需要轻量索引来组织压缩后 KV cache 的系统；(3) 相似度计算使用 cosine similarity（方向性）而非 L2 距离，因为 attention 机制天然对向量方向敏感。论文未独立评估 representative key 的消融，但其是层自适应选择模块的必要输入。

涉及论文标题：
- StreamKV: Streaming Video Question-Answering with Segment-based KV Cache Retrieval and Compression
