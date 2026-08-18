## Spherical K-means++ Refine（球面 K-means++ 精化聚类）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Spherical K-means 是 K-means 的方向敏感变体（Dhillon 2001，Hornik 2012 的 R 实现）：把向量与质心归一化到单位范数、用余弦相似度（内积）替代 L2 距离度量方向对齐，用于高维稀疏文本/词向量聚类。传统 K-means 在稀疏高维空间的缺陷：(1) 距离度量失配——L2 范数被零元素主导、掩盖判别信息，把仅方向不同的语义相关向量错误聚类；(2) 质心漂移——稀疏向量的算术均值引入噪声、降低簇代表性。ParetoES 的 Spherical K-means++ Refine 在其上叠加三处：(a) K-means++ 初始化（Arthur & Vassilvitskii 2006）——按与已选质心最小余弦距离成比例的 p(x_i)=(1−max_cj x_iᵀc_j)/Σ(1−max x_kᵀc_j) 采样，从 m=min(0.01n,10000) 随机子集采质心，把单次迭代复杂度从 O(nd) 降到 O(md)、总复杂度 O(mKd)；(b) 质心更新取簇内最接近均值归一化向量的成员（argmax_{x∈c_i} x·normalize(Σx)），缓解质心漂移；(c) 动态精化（Post Refine）——merge/split 策略：质心间 cos>θ_merge=0.9 的近重复簇对合并为新簇（新质心取并集中最接近均值者），簇内平均 cohesion=mean(xᵀμ_k)<θ_split=0.6 时用 2-means 分裂并迭代重分配直至两子簇 cohesion≥0.6 或达 max_refine_iter。实验：稀疏空间下聚类内距低 75.8%、紧凑性差 82.6%，精化后（+ReSparse）在相同 Recall 下扫描更少向量（Fig.9 优于 K-means/Spectral/Hierarchical）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ParetoES Algorithm 1 的伪代码（精简）：
```
Input: 稀疏向量 {x_i}, 初始簇数 K, θ_merge=0.9, θ_split=0.6, max_refine_iter
# 初始化：K-means++ 采样 K 个质心
centroids = []
while len(centroids) < K:
    p(x_i) = (1 - max_{c in centroids} x_i^T c) / sum_k (1 - max_{c} x_k^T c)
    sample x_i ~ p(x_i)      # 从 m=min(0.01n,10000) 子集
# 迭代：归一化 + 分配 + 质心更新
repeat until convergence:
    x_i = x_i / ||x_i||_2
    c_i = argmax_k (x_i . mu_k)                    # 余弦分配
    mu_k = argmax_{x in c_k} (x . normalize(sum(x))) # 最接近均值向量
# 动态精化
while max_{i≠j} cos(mu_i, mu_j) > 0.9:             # merge
    (i*,j*) = argmax cos; mu_new = argmax_{x in c_i* ∪ c_j*} (x . normalize(sum))
    merge c_i*, c_j* -> c_new
for each c_k with mean(x^T mu_k) < 0.6:            # split
    2-means -> c_k1, c_k2; repeat max_refine_iter: reassign + update
    if cohesion(c_k1)>=0.6 and cohesion(c_k2)>=0.6: replace c_k
```
pipeline 角色：作为离线索引构建第一步，产出 K=nlist=⌊√(m/2)⌋ 个簇与质心，供在线检索做质心相似度簇筛选（Top-nprobe），决定"扫哪些子矩阵"——聚类质量直接决定同 Recall 下的扫描簇数（论文：稀疏 vs 稠密需多扫 1.83× 簇，精化后减少冗余激活）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
软件实现：Faiss 的 IndexIVFFlat 用 K-means（L2）做量化器、scikit-learn KMeans/SphericalKMeans、Hornik 的 R skmeans 包。ParetoES 在 NVIDIA A100 GPU 上实现全部聚类（nlist=√(m/2)、迭代上限 1000、收敛阈值 10⁻⁴ 按质心位移 L2 范数），全精度浮点执行（与检索端 INT6 混合精度）。对比基准用 Faiss v1.7.2 的 IndexIVFFlat（METRIC_INNER_PRODUCT + IndexFlatIP quantizer）、Spectral Clustering、Hierarchical Clustering。效果：精化版（Spherical Refine + ReSparse）在 Recall@100=0.8 约束下扫描向量更少（Fig.9），同 nprobe 下 Recall 更高。论文未开源。

涉及论文标题：
- ParetoES Hardware-Accelerated Sparse Embedding Similarity via Pareto-Optimal Pruning
