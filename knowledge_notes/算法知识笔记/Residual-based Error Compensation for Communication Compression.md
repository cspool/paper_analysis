## Residual-based Error Compensation for Communication Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Residual-based Error Compensation（基于残差的误差补偿）是LSH-MoE框架中用于缓解通信数据压缩对模型精度影响的补偿方案。其核心思想是：在压缩阶段（all-to-all通信前）记录每个token与其所属cluster center的差异（残差），在解压后（expert计算完成后）将残差加回到expert对center的输出中，从而近似恢复每个token的完整输出。

数学公式：
1. 残差记录：$$\Delta \text{cluster}_j = \{x - \overline{\text{cluster}}_j \mid x \in \text{cluster}_j\}$$
2. 残差恢复：$$Y_{ij} = \{E(\overline{\text{cluster}}_j) + \Delta \text{Cluster}_{jk} \mid k = 1, 2, \dots, N_j\}$$

关键洞察：该方案利用expert FFN计算的近似线性性——对于cluster内相似的token，其expert输出也高度相似，因此用central的输出加token-specific残差可以很好地近似完整输出。

LSH-MoE实验证明，不使用error compensation时，在相同训练时间下perplexity高出0.3个点；使用error compensation后，模型质量与无压缩训练几乎一致。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 在MoE layer forward pass中

# 阶段1: 压缩与残差记录
for each expert i:
    tokens_i = {x_1, x_2, ..., x_N}  # 分配给expert i的token
    clusters = LSH(tokens_i)          # LSH聚类
    centroids_i = []
    residuals_i = []
    for cluster_j in clusters:
        c_j = mean(cluster_j)                          # 聚类中心
        resid_j = {x - c_j for x in cluster_j}          # 残差
        centroids_i.append(c_j)
        residuals_i.extend(resid_j)

# 阶段2: 仅传输centroids（通信量：m·h << N·h）
E_centroids = Expert(all_to_all(centroids_i))  # expert计算
results = all_to_all(E_centroids)               # 传回

# 阶段3: 残差补偿恢复
for each cluster_j, for each k in 1..N_j:
    Y_jk = results[j] + residuals_j[k]           # 加上各自残差
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 残差计算和存储都在本地GPU上进行，不增加额外通信
- 残差的存储开销为N·h（与原始token相同），但仅在本地保存，不通过网络传输
- 误差补偿的有效性源于"token similarity"——cluster内部的token非常相似，它们的expert输出也相似，所以线性残差近似精度足够
- 需要注意：该补偿是对中间激活值进行压缩的误差补偿，而非梯度压缩。激活压缩对误差的容忍度更低（因为误差会在后续层累积放大），因此误差补偿对保持模型质量至关重要
- 本质上是一种有损压缩的误差控制技术，类似压缩感知中的残差编码思想

涉及论文标题：
- LSH-MoE Communication-efficient MoE Training via Locality-Sensitive Hashing
