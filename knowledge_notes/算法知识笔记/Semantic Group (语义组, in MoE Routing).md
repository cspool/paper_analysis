## Semantic Group (语义组, in MoE Routing)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Semantic Group 是 Oracle-MoE 中基于注意力分数的因果图（DAG）定义的一组最大化互连 token 集合，用于捕捉 token 序列中的高层语义局部性。定义：将 token 序列建模为有向无环图 G = (V, E)，V 包含所有 token，E 由注意力分数矩阵 A = [a_ij] 的 lower-triangular 部分加权（a_ij 存在仅当 i > j，因果注意力约束）。若 token 组 S = {t_k1, ..., t_km}（k1 < ... < km）满足：(1) 所有 i > j 均有 a_ij > ε；(2) 不存在包含 S 的真超集也满足条件(1)，则 S 为一个语义组。这本质上是 DAG 上的 Minimum Clique Cover 问题的重构。由于注意力矩阵具有块结构（block structure），可用多项式时间贪心算法求解（从左到右扫描 token，尝试合并到已有组中）。

从算法pipeline角度拆解术语：

语义组划分贪心算法：
```
def partition_semantic_groups(attention_matrix_A, epsilon, seq_len):
    """
    A: [T, T] causal attention score matrix (lower-triangular)
    epsilon: 注意力分数阈值，决定"语义相关"的最小分数
    """
    groups = []
    for t in range(seq_len):
        merged = False
        for group in reversed(groups):
            if all(A[t][k] > epsilon for k in group):
                group.append(t)
                merged = True
                break
        if not merged:
            groups.append([t])
    return groups
```

关键性质：(1) 同一语义组内的 token 共享相似的高层语义（由 attention Q·K^T 内积保证）；(2) 1024 token 的序列通常仅产生 < 5 个语义组；(3) 同一序列/用户交互的语义组在 Oracle Space 中倾向于属于同一 K-means 聚类；(4) 组内平均操作将 token-identity 方差从 Var(t_t) 降至 (Σ_s + Σ_j)/n。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 依赖因果注意力分数（decode 时利用 KV cache 可自然获取）。
- 阈值 ε 控制语义组粒度，论文未明确给出具体值。
- 复杂度 O(T × G)，G 为组数（通常 < 5），近乎线性。
- 仅适用于 causal attention 的 auto-regressive 模型。

涉及论文标题：
- Oracle-MoE: Locality-preserving Routing in the Oracle Space for Memory-constrained Large Language Model Inference

---
