## Topology-aware Expert Remapping（拓扑感知专家重映射）

术语是什么？
Topology-aware Expert Remapping 是 ScaleMoE 论文提出的面向异构网络的 expert-to-GPU 映射优化技术。在 Dynamic Expert Clustering 将 token 聚类后，该技术决定每个 cluster 应放置在哪个 GPU（或节点）上，以最小化跨设备的通信延迟。核心机制：(1) **Coverage Matrix (C×C)**：量化 cluster 间的 expert 覆盖度，cell (i,j) 表示 cluster Ci 对 cluster Cj 所需 experts 的覆盖程度（即 Cj 的 token 有多大比例可以在 Ci 所在的 GPU 群本地计算）；(2) **Bandwidth Matrix (GPU×GPU)**：profiling 得到的点对点网络带宽（含 NVLink 节点内 600 GB/s 和 Ultra Ethernet 节点间 100 Gbps 或更低）；(3) **Genetic Algorithm**：以最小化总通信时间为目标，fitness function = Σ_{i,j} ((b·s - CM[SV[i]][SV[j]]·h) / BM[i][j])，其中 b 为 batch size，s 为 sequence length，h 为 hidden dimension，SV 为 cluster-to-GPU 映射向量。每代执行 uniform order-based crossover + swap mutation，选择最低 fitness 的 solution。

从系统架构角度拆解术语：
Topology-aware Expert Remapping 在分布式训练系统中的工作流程：
```
// 输入: K-means 聚类结果 (C clusters), 网络拓扑, 32 GPUs (4 nodes × 8 GPUs)

// Step 1: 构建 Coverage Matrix
for each cluster Ci:
    for each cluster Cj:
        CM[i][j] = fraction_of_Cj_experts_covered_by_Ci
        // e.g., Cj 需要 {E1,E3,E7,E9}, Ci 所在 GPU 有 {E1,E3,E5}
        // → CM[i][j] = 2/4 = 0.5

// Step 2: 构建 Bandwidth Matrix（网络 idle 时 profiling）
for each GPU pair (i, j):
    if same_node(i, j):     BM[i][j] = 600 GB/s   // NVLink 3.0
    elif same_switch(i, j): BM[i][j] = 100 Gbps   // Ultra Ethernet fast path
    else:                   BM[i][j] = 50 Gbps    // multi-switch slow path

// Step 3: Genetic Algorithm 搜索最优映射 SV
population = [random_SV() for _ in range(pop_size)]
for generation in range(max_generations):
    // 计算 fitness
    for each SV in population:
        fitness[SV] = sum_{i,j} ((b*s - CM[SV[i]][SV[j]]*h) / BM[i][j])
    // 选择: 保留 fitness 最低的
    elites = select_lowest_fitness(population)
    // Crossover + Mutation
    population = uniform_order_based_crossover(elites)
    population = swap_mutation(population)  // 随机交换两个位置

// Step 4: 应用映射
best_SV = population[argmin(fitness)]  // SV[i] = cluster mapped to GPU-i
apply_cluster_to_device_mapping(best_SV)
```

术语一般如何实现？如何使用？
实现为 ScaleMoE 的 CPU 侧模块，每个 superbatch (100 iterations) 与 GPU iteration overlapped 执行。遗传算法的 remapping 开销 2443.32ms（per superbatch），被 overlap 完全隐藏。在异构网络中效果尤为显著：ScaleMoE 在 heterogeneous network 中 speedup 高达 3.31×（vs homogeneous 1.84×），因为 topology-aware remapping 主动将高通信需求的 cluster 对放到高带宽链路上。

涉及论文标题：
- ScaleMoE: A Fast and Scalable Distributed Training Framework for Large-Scale Mixture-of-Experts Models
