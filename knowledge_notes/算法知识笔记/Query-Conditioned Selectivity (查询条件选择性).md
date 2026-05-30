## Query-Conditioned Selectivity (查询条件选择性)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Query-Conditioned Selectivity 是 softmax attention 的一个关键特性：每个 query q_i 产生自己专属的注意力分布 {α_ij}_j，使不同 query 可以从同一序列中提取不同的上下文信息。在 standard softmax attention 中，α_ij = exp(q_i^T k_j/√d) / Σ_t exp(q_i^T k_t/√d)，权重同时依赖 query 和 key，实现了完全的 query-conditioned 逐 token 权重分配。

Standard linear attention 丧失了该特性：所有 query 共享同一个全局 summary G，导致 o_i = q̃_i^T G / q̃_i^T z 中唯一的 query 依赖性来自 q̃_i 本身，而 token 级别的贡献（k_j v_j）已在 G 中不可区分地融合，不同 query 获得几乎相同的 context vector。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**选择性对比（三个 attention 机制）**：

```
// Softmax Attention — 完全 query-conditioned，O(N²)
for each query i:
    scores_i = [exp(q_i @ k_j / sqrt(d)) for j in 1..N]  // 每个 query 独立计算
    alpha_i = softmax(scores_i)                            // 不同的分布
    o_i = sum(alpha_ij * v_j for j in 1..N)

// Standard Linear Attention — 丧失 selectivity，O(Nd²)
G = sum(phi(k_j)^T @ v_j for j in 1..N)  // 所有 token 融合
for each query i:
    o_i = (phi(q_i)^T @ G) / (phi(q_i)^T @ z)
    // q_i 不同，但 G 中 token 贡献不可分 → 选择性弱

// MHLA — 恢复 selectivity via 两阶段，O(Nd² + M²d²)
// Stage 1: query-conditioned block 选择
// Stage 2: block 内 token 级 kernel reweighting
S̃_i = Σ_b m_{i,b} S_b     // query block i 专属的混合 summary
o_i = (q̃_i^T @ S̃_i) / (q̃_i^T @ z̃_i)
     = Σ_t m_{i,b(t)} (q̃_i^T @ K̃_t) @ V_t^T
// 两阶段: block 级 m_{i,b(t)} + token 级 q̃_i^T K̃_t
```

术语一般如何实现？如何使用？

Query-conditioned selectivity 是评估注意力机制表达能力的重要维度。MHLA 论文通过注意力矩阵的 rank 和熵来量化该特性：rank 越高表示注意力空间越多样（更多不同的注意力模式），熵越低表示注意力越集中（更强的选择性）。在模型设计中，选择注意力变体时需要权衡该特性和计算效率。

涉及论文标题：
- MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head

---
