## Personalized PageRank (PPR) Graph Retrieval（个性化PageRank图检索）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Personalized PageRank (PPR) 是PageRank的扩展，指定个性化起始节点集合(seed nodes/teleport set)使随机游走偏向它们，计算图中每个节点相对于query的"个性化重要性"分数。WorldMM用于知识图谱检索：将query提取的实体节点设为seed(teleport概率偏高)，PPR迭代至收敛，节点分数s[i]为与seed的关联强度。Episodic Memory检索：PPR→节点→关联caption候选。Semantic Memory检索：边得分=两端节点PPR之和(edge_score=ppr(u)+ppr(v))，取top-k边对应三元组。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
输入: KG邻接矩阵A[N,N], seed节点S, α(通常0.85)
s = [1/|S| if i in S else 0 for i in 0..N]
while not converged:
    s_new = α * A^T @ s + (1-α) * s_init
s = s_new

# Episodic Memory: 节点检索
caps = [G.get_caption(node) for node in argsort(s)[:k]]

# Semantic Memory: 边检索
edge_scores = {edge: s[u]+s[v] for edge=(u,v) in G_s}
top_triplets = argsort(edge_scores)[:10]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现基于HippoRAG框架(github.com/OSU-NLP-Group/HippoRAG)。PPR相对标准PageRank优势：teleport bias使结果与query相关而非仅全局重要性。Embedding检索替代PPR导致4.4%精度下降，验证了图结构检索优于纯相似度检索。

涉及论文标题：
- WorldMM__Dynamic_Multimodal_Memory_Agent_for_Long_Video_Reasoning
