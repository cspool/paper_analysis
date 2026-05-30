## Sinkhorn Distance for Fairness

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Sinkhorn Distance（Sinkhorn 距离）是熵正则化的最优传输（Optimal Transport, OT）距离。标准 Wasserstein 距离的最优传输问题求解复杂度为 O(n³ log n)，Cuturi (2013) 提出通过添加熵正则化项 H(P) 使问题变为强凸，从而可通过 Sinkhorn 算法（迭代矩阵缩放）高效求解，复杂度降至 O(n²)。在机器学习公平性领域，Sinkhorn distance 被用于衡量和最小化不同受保护属性组（如不同种族、性别）的特征分布之间的差异——将公平性问题建模为最优传输问题：寻找将一组分布传输到另一组的最小代价方案，用 Sinkhorn distance 作为 fairness regularization term。

数学定义：

$$Sinkhorn_{\epsilon}(\mu, \nu) = \min_{P \in \Pi(\mu,\nu)} \langle P, C \rangle + \epsilon \sum_{i,j} P_{ij} \log P_{ij}$$

其中 μ, ν 为两组分布，C 为代价矩阵，ε 控制正则化强度，P 为传输计划矩阵。通过 Sinkhorn 算法迭代更新缩放因子 u, v 使 P 满足行/列边际约束。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

在 FairCLIP 和 Fair-MoE 中，Sinkhorn distance 用于在对比学习 loss 层面实现公平性：

```
# Sinkhorn distance 在 fairness loss 中的使用
# 输入: 两组样本的 embeddings Z_a, Z_b (来自属性 a 和 b)

# Step 1: 计算代价矩阵 C (通常用 cosine distance)
C[i][j] = 1 - cosine_similarity(Z_a[i], Z_b[j])

# Step 2: Sinkhorn 算法迭代
K = exp(-C / epsilon)          # Gibbs kernel
u = ones(n) / n                # 初始化缩放因子
v = ones(m) / m
for t in 1..T:
    u = a / (K @ v)            # a, b 为边际分布 (通常均匀)
    v = b / (K^T @ u)
P = diag(u) @ K @ diag(v)      # 最优传输计划

# Step 3: Sinkhorn distance = ⟨P, C⟩
L_sinkhorn = sum(P * C)
```

在 Fair-MoE 的 FOL 中，L_distance 即为 Sinkhorn distance，用于最小化不同受保护属性组特征分布之间的距离，与方差优化项互补。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Sinkhorn distance 通过 Python OT 库（如 POT: Python Optimal Transport）或 PyTorch 自定义实现。在 fairness 应用中通常作为辅助 loss 项：L_total = L_task + λ · L_sinkhorn。FairCLIP 使用 Sinkhorn distance 作为唯一的 fairness constraint。Fair-MoE 将其保留为 FOL 的一个子项，同时引入方差优化项以增强公平性。适用于需要对齐不同组特征分布的场景（如医疗 VLMs、面部识别、推荐系统）。

涉及论文标题：
- Fair-MoE: Fairness-Oriented Mixture of Experts in Vision-Language Models
