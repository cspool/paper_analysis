## Semantic Group Embedding (语义组嵌入)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Semantic Group Embedding 是语义组内所有 token embedding 的均值向量：z_S = (1/|S|) Σ t_j。通过将 token embedding 分解为 t_i = s_i + u_i（s_i: 高层语义, u_i: token-identity）后，组内平均压制 token-identity 噪声 u_i 同时保留高层语义 s_i。理论保证：Var(z_S) = (Σ_s + Σ_j)/n < Var(t_t)，n = |S|。类似 GNN 的均值聚合和句子元嵌入。

从算法pipeline角度拆解术语：

```
# 输入: token embeddings T = [t_1, ..., t_n] for semantic group S
z_S = (1/n) * Σ_{i=1}^{n} t_i           # d 维均值
z_S_reduced = W_svd @ z_S               # 可选 SVD 降维到 r 维
expert_id = argmin_k ||z_S_reduced - c_k||  # 路由到最近聚类中心
```

方差分析：Var(z_S) = Var((1/n)Σ s_i) + Var((1/n)Σ u_i) = Σ_s/n + Σ_j/n < Var(t_t)，因为 1/n < 1 for n > 1。

术语一般如何实现？如何使用？
- Streaming 增量更新：z_S_new = (|S|*z_S_old + t_new) / (|S|+1)，O(d)。
- SVD 降维加速 K-means 聚类和 routing 时的距离计算（r << d）。
- 与 token embedding 的关键区别：语义组嵌入消除了单个 token 的 identity 噪声，保留语义上下文的整体特征。

涉及论文标题：
- Oracle-MoE: Locality-preserving Routing in the Oracle Space for Memory-constrained Large Language Model Inference
