## Mahalanobis Initialization for Codebook EM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mahalanobis Initialization 是 GPTVQ 提出的 EM 算法 seeding 方法，替代传统的 k-Means++。方法：对 N 个 d 维数据点 X，先按马氏距离（Mahalanobis distance）到数据中心排序，然后从排序列表中均匀间隔采样 k 个点作为初始质心，采样间隔为 ⌊N/(k-1)⌋。马氏距离 = (x - μ)^T Σ^{-1} (x - μ)，在 GPTVQ 中使用 Hessian 加权的距离（等价于加权欧氏距离）。该方法比 k-Means++ 快 3-5×，且最终 perplexity 相当（Table 13：Mahalanobis PPL 6.05 vs k-Means++ PPL 6.16）。直觉：马氏距离排序确保采样点在不同"距离层级"上均匀分布，覆盖从数据中心到远尾的整个分布范围。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Mahalanobis Initialization 伪代码
# 输入: 数据点 X ∈ R^{N × d}, 均值 μ ∈ R^d, 逆 Hessian 对角 D (≡ Σ^{-1})
# 输出: k 个初始质心

# Step 1: 计算每个数据点的马氏距离
distances = []
for i in range(N):
    diff = X[i] - μ  # [d]
    # 加权马氏距离: (x-μ)^T diag(1/H^{-1}_{jj}) (x-μ)
    mahal_dist = Σ_{j=0}^{d-1} diff[j]^2 / H_inv_diag[j]
    distances.append((mahal_dist, i))

# Step 2: 按距离排序
distances.sort()  # 升序

# Step 3: 均匀间隔采样
step = N / (k - 1)
centroids = []
for m in range(k):
    idx = int(m * step)
    centroids.append(X[distances[idx].index])

return centroids  # 形状: k × d
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GPTVQ 默认使用 Mahalanobis initialization 因其速度优势（Table 13: 2D 3B 16384→756s vs K++ 3168s，4.2× 加速）且精度不劣于 k-Means++。实现要点：(1) 马氏距离计算可与 Hessian 计算共享 D 矩阵；(2) 排序使用 PyTorch 的 argsort；(3) 对 70B 大模型，Mahalanobis 的高效性尤为关键（k-Means++ 可能耗时数十小时）。

涉及论文标题：
- GPTVQ: The Blessing of Dimensionality for LLM Quantization

---
