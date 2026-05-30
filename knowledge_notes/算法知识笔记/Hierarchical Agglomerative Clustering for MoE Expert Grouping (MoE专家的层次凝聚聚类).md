## Hierarchical Agglomerative Clustering for MoE Expert Grouping (MoE专家的层次凝聚聚类)

术语解释
Hierarchical Agglomerative Clustering (HAC) 是一种自底向上的聚类算法，在 MoE expert grouping 场景中用于将功能相似的 expert 分组为 cluster。C-PRUNE、HC-SMoE 和 Mosaic Pruning 等近期工作均采用此方法，因其不需要预设 cluster 数量（可通过 pruning rate 自适应确定）且能保留 expert 间的层次相似结构。

术语是什么？
HAC 的基本流程：(1) 初始化：每个 expert 作为一个 singleton cluster；(2) 迭代合并：在每步选择 affinity 最高的两个 cluster 合并；(3) 终止：cluster 数量达到目标值。合并标准使用 Ward's linkage 或 average linkage。

C-PRUNE 中的 HAC 实现特点：
- 亲和矩阵 A_ij = σ(α · cos(φ(f_i), φ(f_j)))，基于 expert embedding 的 cosine similarity
- Cluster 合并后的新 affinity 通过 weighted average 更新（average linkage）
- 聚类在每层独立执行（layerwise），每层可有不同的 cluster 数量
- 聚类阈值 τ^(l) 自适应层深度：τ^(l) = mean_deviation + δ·σ^(l)

从算法pipeline角度拆解术语。
```
# HAC for MoE Expert Grouping (per-layer)
Input: expert_embeddings phi[N][d] for layer l
       target_clusters K (derived from pruning rate)

# Step 1: Build affinity matrix
A = zeros(N, N)
for i in 1..N, j in i+1..N:
    cos_sim = phi[i] @ phi[j] / (norm(phi[i]) * norm(phi[j]))
    A[i,j] = A[j,i] = sigmoid(alpha * cos_sim)

# Step 2: Initialize N singleton clusters
clusters = [{i} for i in range(N)]

# Step 3: Iteratively merge until reaching K clusters
while len(clusters) > K:
    (u, v) = argmax(A[u][v] for all u < v)
    new_cluster = clusters[u] ∪ clusters[v]
    clusters.remove(u), clusters.remove(v)
    clusters.append(new_cluster)
    # Update affinity (average linkage)
    for each remaining cluster c:
        A[new_idx, c] = (|C_u|*A[u,c] + |C_v|*A[v,c]) / (|C_u| + |C_v|)

# Step 4: Merge experts within each cluster
for cluster C_k in clusters:
    omega = softmax([gamma * A[i, center] for i in C_k])
    merged_params = sum(omega_i * expert_params[i])
```
注解：
- α: 相似度敏感度。α 越大，聚类越激进
- K = ceil(N × (1 - pruning_rate_layer))
- 时间复杂度 O(N² log N)，N 为 expert 数量（通常 ≤ 128）
- 各层的 HAC 可完全并行执行

术语一般如何实现？如何使用？
- **Linkage 选择**：C-PRUNE 使用 average linkage；HC-SMoE 验证 ward/average 均有效
- **与 K-means 的比较**：C-PRUNE Table 3 显示 HAC (avg 0.449) 显著优于 K-means (avg 0.405)，因 HAC 不假设球形 cluster
- **并行化**：各层 HAC 可完全并行执行
- **在 C-PRUNE 中的角色**：Phase 1 的核心，分组结果决定后续 global pruning 和 expert merging 质量
- 局限：(1) O(N² log N) 在超大 N 时可能成为瓶颈；(2) 聚类对 affinity matrix 构建方式敏感

涉及论文标题：
- Cluster-Driven Expert Pruning for Mixture-of-Experts Large Language Models

---
