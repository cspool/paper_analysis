## Query Grouping for Sparse Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Query Grouping是DSV中优化sparse attention计算效率的技术，利用相邻tokens的critical KV pairs高度重叠的特性（Observation 5: 2×2×2 3D cube内重叠率>92.4%），将相邻queries按3D voxel分组共享critical KV indices。核心收益：(1) 减少critical KV estimation的开销（无需为每个query单独预测，只需一个proxy query）；(2) 改善memory access coalescing（同组queries访问相同KV set，增加data reuse）；(3) 提升tensor core利用率（gathered KV可以更大batch处理）。自适应grouping机制根据输入video scene动态调整group size，保证overlap ratio >80%。

从kernel调度角度拆解：
```
# Query Grouping for Sparse Attention (DSV)
# 输入: Q [H, S, d_k], K [H, S, d_k], V [H, S, d_k]
#       crit_indices_all [H, S, K_per_query]  # per-query critical KV indices

# Step 1: Determine optimal group size
for each attention head:
    # 从3D latent space构造voxel groups
    group_size = AdaptiveGroupSize(video_latent_shape, overlap_threshold=0.8)
    # group_size options: 1x1x1 (no grouping), 2x2x2, 2x4x4, etc.

# Step 2: Select proxy query per group
for each voxel_group g of size [g_F, g_H, g_W]:
    proxy_query_idx = center_of(g)  # 或随机采样
    proxy_crit_indices = crit_indices_all[head, proxy_query_idx]
    # 同组所有queries共享此critical KV set

# Step 3: Sparse attention with shared KV
for each voxel_group g:
    K_gathered = gather(K, proxy_crit_indices)  # [K_shared, d_k]
    V_gathered = gather(V, proxy_crit_indices)  # [K_shared, d_k]
    Q_group = Q[g.queries]                      # [group_size, d_k]
    O[g] = softmax(Q_group @ K_gathered^T / sqrt(d_k)) @ V_gathered
```

关键设计决策：(1) group内共享同一critical KV set（而非每个query独立gather），显著减少gather操作次数；(2) proxy query选择center token（因其critical KV最representative）；(3) group size自适应——高overlap场景用大group（如2×4×4=32 queries共享），低overlap用小group；(4) grouping仅在sparse training stage（Stage 2）使用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

DSV中使用Triton kernel实现query-grouped sparse attention。Grouping策略在CPU或lightweight GPU kernel上完成（仅需overlap ratio profiling）。邻接性基于3D latent space（frames × H × W），而非1D token线性顺序。adaptive机制profiling输入video scene的critical KV overlap ratio后选择group size。限制：(1) 仅当overlap ratio >80%时有效，稀疏度极高时grouping收益可能下降；(2) group内的query可能有略微不同的optimal critical KV set，需trade-off accuracy vs efficiency。

涉及论文标题：
- DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training
