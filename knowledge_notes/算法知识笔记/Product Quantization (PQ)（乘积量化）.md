## Product Quantization (PQ)（乘积量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Product Quantization (PQ) 是一种高维向量压缩方法，由 Jégou et al. (2010) 提出。核心思想：将 D 维向量空间分解为 M 个低维子空间的笛卡尔积，每个子空间独立训练子码本（通常 256 中心 = 8-bit），原始向量每段映射到最近子码本向量。M 个子码本索引串联构成量化码。码本总大小 = 256^M（概念性），压缩率可达 32x+。主要问题：(1) 距离估计有偏且无理论误差界；(2) 中等压缩率 (B≥4 bits/dim) 下精度反而不如 SQ；(3) 距离计算需频繁查表 (RAM 访问)，效率低于 SQ/LVQ/RaBitQ。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Index 阶段
# 1. 子空间分解: D 维 → M=D/d 个 d 维子空间
# 2. 每子空间 k-means 训练 256 中心子码本
# 3. 量化:
for m = 1..M:
    o_m = o[(m-1)*d : m*d]
    idx[m] = argmin_j ||o_m - c_{m,j}||  # 最近子码本向量
# 存储: idx[1..M]（M bytes）

# Query 阶段: 查表累加
for m = 1..M, j = 0..255:
    dist_table[m][j] = ||q_m - c_{m,j}||²
for each candidate:
    dist_est² = Σ_{m=1..M} dist_table[m][idx_candidate[m]]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
广泛部署于 Faiss (`IndexIVFPQ`)、Milvus 等向量数据库，通常配合高压缩率 (32x+) + IVF + re-ranking。FastScan SIMD 加速版需 k=4 子码本但精度进一步损失。PQ 更适合极高压缩率且能容忍 re-ranking 开销的场景。

涉及论文标题：
- RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
- Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

---
