## RaBitQ (Randomized Binary Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RaBitQ（Randomized Binary Quantization）是一种面向高维向量 ANN 搜索的量化方法，由 Gao 和 Long 在 SIGMOD 2024 提出。RaBitQ 将 D 维浮点向量量化为 D 位二进制串（对应 32x 压缩率），提供无偏距离估计器和具有渐近最优误差界的理论保证。核心原理：(1) 码本构造——取超立方体顶点集 C = {±1/√D}^D（所有由 +1/√D 和 -1/√D 组成的 D 维单位向量），乘以随机正交矩阵 P 旋转得到码本 C_r = {Px | x∈C}；(2) 量化编码——对每个数据向量 o，找码本中最近向量 ō₀ 作为量化向量，用 D 位二进制码 x̄_b ∈ {0,1}^D 表示；(3) 距离估计——无偏估计器 ⟨ō₀,q⟩/⟨ō₀,o⟩ 估计内积 ⟨o,q⟩，计算简化为 ⟨ō₀,q⟩ = (2/√D)·⟨q',x̄_b⟩ - (1/√D)·Σq'[i]，其中 q'=P^{-1}q。误差界以高概率为 O(1/√D)。GitHub: https://github.com/gaoj0017/RaBitQ

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RaBitQ 的算法 pipeline 分为 index 和 query 两阶段：
```
# Index 阶段
P = random_orthogonal_matrix(D)  # QR 分解生成
c = centroid(data_vectors)       # 全局质心
for each o_r:
    o = (o_r - c) / ||o_r - c||  # 中心化归一化
    o' = P^{-1} o                # 逆旋转变换
    for d = 1..D:
        x_b[d] = (o'[d] > 0) ? 1 : 0  # 按符号量化
    # ō₀ = P·(2/√D·x_b - 1/√D·1_D)
    store: ||o_r-c||, ⟨ō₀,o⟩, x_b

# Query 阶段
q = (q_r - c) / ||q_r - c||, q' = P^{-1} q, s = Σ_i q'[i]
for each candidate:
    ⟨q',x_b⟩ = FastScan_SIMD(q', x_b)  # 批量 SIMD
    ⟨ō₀,q⟩ = (2/√D)·⟨q',x_b⟩ - (1/√D)·s
    ⟨o,q⟩_est = ⟨ō₀,q⟩ / ⟨ō₀,o⟩
    dist_est² = ||o_r-c||² + ||q_r-c||² - 2·||o_r-c||·||q_r-c||·⟨o,q⟩_est
```
关键不变式：(1) 估计器无偏，即 E[⟨ō₀,q⟩/⟨ō₀,o⟩] = ⟨o,q⟩；(2) 误差界始终成立，不依赖数据分布；(3) ⟨ō₀,o⟩ ≈ 0.8 在高维空间中高度集中。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
RaBitQ 提供两种距离计算实现：(1) FastScan-based：基于 SIMD (AVX512) 批量计算；(2) Bitwise-based：位操作逐个向量计算。RaBitQ 通常与 IVF 索引结合使用，先对数据集聚类再对每聚类内向量量化。也可与图索引结合（如 SymphonyQG, SIGMOD 2025）。代码开源在 https://github.com/gaoj0017/RaBitQ（C++）。局限：仅支持 32x 压缩率 (1 bit/dim)，中等压缩率下需配合 re-ranking 或扩展方案。

涉及论文标题：
- RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
- Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

---
