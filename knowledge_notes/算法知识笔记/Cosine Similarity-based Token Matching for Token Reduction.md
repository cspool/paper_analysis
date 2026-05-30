## Cosine Similarity-based Token Matching for Token Reduction

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cosine Similarity-based Token Matching是UTRC中连接低重要性和高重要性token集合的机制。对每个低重要性token a_i ∈ M_A，计算其与所有高重要性token b_j ∈ M_B的余弦相似度，选择最相似的b_j作为匹配目标f_i，记录最大相似度g_i。按g_i排序后仅保留top-p%的最相似匹配对。设计动机：a_i与其counterpart f_i越相似，a_i的语义信息已在f_i中充分表示，可被安全删除或融合而丢失信息最少。与bipartite matching（强制一对一匹配不考虑质量）的关键差异：相似度阈值作为质量闸门，低相似度的pair被拒绝。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
A_norm = normalize(M_A, dim=-1)        # [N/2, D]
B_norm = normalize(M_B, dim=-1)        # [N/2, D]
sim_matrix = A_norm @ B_norm.T         # [N/2, N/2] pairwise cosine
g_values, f_indices = max(sim_matrix, dim=-1)  # row-wise best match
sorted_pairs = argsort(g_values, descending=True)
keep_pairs = sorted_pairs[:int(p * N/2)]       # top-p% 过滤
# g_i低: a_i无法在任何M_B中找到好匹配
# → 包含独特关键信息, 不能prune/merge
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch实现：F.normalize + matmul一次性计算所有pairwise cosine。p值由目标FLOPS reduction反推。与bipartite matching (ToMe)的核心差异：质量闸门过滤掉低质量匹配，对被拒绝的a_i不执行任何破坏性操作。

涉及论文标题：
- Rethinking_Token_Reduction_for_State_Space_Models

---
