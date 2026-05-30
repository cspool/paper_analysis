## Locality-Sensitive Hashing (LSH) in MoE Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Locality-Sensitive Hashing (LSH) 是一种概率性降维方法，主要用高维空间的近似最近邻搜索。其核心思想是：将高维数据通过一组hash函数映射到低维"桶"（buckets）中，使得相似的数据以高概率被映射到同一个桶，而相异的数据以高概率被映射到不同桶。数学上，LSH hash函数h满足：P[h(x)=h(y)] = 1 − d(x,y)/D，其中d(x,y)是x和y之间的距离，D是空间的直径。

在MoE训练中，LSH被LSH-MoE论文创新性地用作一种在线聚类压缩技术：在all-to-all通信前，对每个expert接收的token集合使用LSH快速聚类，将相似的token归入同一cluster，然后仅传输各cluster的中心（centroid）而非全部token，从而大幅减少all-to-all通信量。压缩率由hash函数数量控制——hash函数越多，bucket越多，cluster越细，压缩率越低（即传输的数据越接近原始量）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

LSH-MoE中LSH在MoE层forward pass中的具体流程：

```
# 输入: X_i = {x_1, x_2, ..., x_N} 为分配给第i个expert的token集合
# LSH参数: hash函数数量H, 旋转矩阵R (d×d随机矩阵)

# Step 1: LSH聚类 — 将每个token映射到bucket
for each token x in X_i:
    # Cross-Polytope Hashing:
    # 将x用随机旋转矩阵R变换，然后映射到最近cross-polytope顶点
    hash_code = argmax_{j in {±1,...,±d}} |(R @ x)[j]|
    # hash_code ∈ [0, 2d-1] 标识cross-polytope的顶点

# Step 2: 按hash_code分组，同一bucket的token归为一个cluster
clusters = group_by_hash(X_i, hash_codes)

# Step 3: 计算每个cluster的中心（替代传输数据）
for each cluster_j:
    centroid_j = mean(cluster_j)                 # 聚类中心
    residuals_j = {x - centroid_j | x in cluster_j}  # 残差（本地保存）

# Step 4: 仅传输centroid集合替代完整token
C_i = {centroid_j for j=1..m}                   # m个centroid << N个token
send_via_alltoall(C_i)                            # 通信量: m*h << N*h

# Step 5: Expert对中心计算
E_centroids = Expert(C_i)

# Step 6: 接收结果并用残差恢复每个token的近似输出
receive_via_alltoall(E_centroids)
for each cluster_j, for each token k in cluster_j:
    Y_jk = E_centroids[j] + residuals_j[k]      # 残差补偿
```

压缩率 = m/N（cluster数量/原始token数量），由hash函数数量H控制。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

LSH在MoE训练中的实现：
- 通过PyTorch实现LSH聚类模块，hash计算为GPU上的矩阵运算（旋转矩阵乘法+argmax）
- 必须高效在线执行（因为待压缩数据是动态实时生成的，无法预压缩或重叠处理），因此选择cross-polytope hashing（O(d)复杂度）
- LSH聚类替代传统的K-Means等迭代聚类算法，因为K-Means的迭代特性不适合在线实时场景
- 默认使用6个hash函数（约20%压缩率时精度无损），可通过调整hash函数数量控制压缩率
- 计算开销远小于通信节省——LSH的矩阵运算相比all-to-all通信可忽略

涉及论文标题：
- LSH-MoE Communication-efficient MoE Training via Locality-Sensitive Hashing
