## Inverted File Index (IVF)（倒排索引）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Inverted File Index (IVF) 是 ANN 搜索中最广泛使用的分区索引方法之一。核心思想：索引阶段将 N 个数据向量通过 KMeans 聚类划分为 L 个聚类（如 4,096 个），查询阶段只需扫描与查询向量最近的 nprobe 个聚类中的向量（而非全部 N 个），从而大幅减少搜索空间。IVF 的优势：(1) 索引体积极小（仅需存储聚类质心和聚类分配）；(2) 顺序内存访问模式——聚类内向量连续存储，对 SIMD 批量计算和 cache 友好；(3) 与各种量化方法（SQ/LVQ/PQ/RaBitQ）无缝结合——量化在聚类级别应用，使用本地质心中心化以减小量化误差。在 Extended RaBitQ 论文中，IVF 是 ANN 查询的主索引结构。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
# IVF + Extended RaBitQ 系统架构

# === Index 阶段 ===
# 1. 全局聚类
centroids = KMeans(data, L=4096)  # L 个聚类中心
for i = 1..N:
    cluster_id[i] = argmin_j ||data[i] - centroids[j]||

# 2. 每聚类独立量化
for each cluster c:
    local_c = mean(cluster_vectors[c])  # 本地质心
    for each v in cluster_vectors[c]:
        v' = v - local_c  # 本地中心化
        code = Extended_RaBitQ_encode(v', P, B)  # Algorithm 1
        # 拆分存储: code = 2^{B-1}·ȳ₀ + ȳ_last
        store_in_cluster(c, ȳ₀, ȳ_last, ||v'||, ⟨ō,o⟩)

# === Query 阶段 ===
q → 计算到 L 个质心的距离 → 选 top-nprobe 最近聚类
for each c in top_nprobe:
    q_local = (q - local_c[c]) / ||q - local_c[c]||
    q' = P^{-1} q_local, s = Σ q'[i]
    # 批量扫描聚类 c 中的所有向量
    batch = load_cluster_codes(c, ȳ₀)  # Stage 1: 仅加载 MSB
    ⟨ȳ₀,q'⟩_batch = FastScan_SIMD(batch, q')  # 批量 SIMD 计算
    for each vector in batch:
        if lower_bound(dist_rough) > best_dist: continue
        # Stage 2: 增量加载剩余位
        ȳ_last = load_remaining_bits(c, id)
        ⟨ȳ_u,q'⟩ = 2^{B-1}·⟨ȳ₀,q'⟩ + ⟨ȳ_last,q'⟩
        dist_est = compute_full_distance(...)
        update best if better
return best_match
```

关键参数：L（聚类数，影响精度的上限和扫描粒度）、nprobe（扫描聚类数，越大精度↑速度↓）。Extended RaBitQ 中 L=4,096（百万级）/ L=262,144（亿级 MSMARCO）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
IVF 是 Faiss (`IndexIVFFlat`, `IndexIVFPQ`, `IndexIVFScalarQuantizer`) 和 Milvus 的核心索引结构。实现要点：(1) KMeans 使用 Faiss 提供的 GPU/多线程加速；(2) 聚类内向量按 ID 连续存储以达到顺序内存访问；(3) nprobe 是运行时参数，可动态调节精度-速度权衡；(4) 与 SQ/LVQ/PQ 结合时，量化在每聚类内独立进行以减少 local variance。IVF 的局限：对高维数据聚类效果可能下降；对极大规模数据（>10B）需要分层 IVF 或与图索引结合。

涉及论文标题：
- RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
- Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

---
