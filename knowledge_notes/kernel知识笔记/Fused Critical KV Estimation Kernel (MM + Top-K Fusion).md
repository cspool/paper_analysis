## Fused Critical KV Estimation Kernel (MM + Top-K Fusion)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Fused Critical KV Estimation Kernel是DSV中实现低秩矩阵乘法（Q_lr·K_lr^T）与top-K选择融合的CUDA kernel，避免在GPU内存中物化完整的[S, S]大小的attention score矩阵。问题背景：标准的两步流程——(1) 计算完整attention score矩阵 [H, S, S]（H=16, S=300K时需~288GB@BF16），(2) 对每个query执行top-K选择——是memory-bound且内存不可行的。融合策略：将矩阵乘法的部分积直接流入增量top-K更新，每个query仅保留top-K scores在寄存器中，空间复杂度从O(S²)降至O(SK)。在CUDA cores上（非Tensor cores）执行，因为slim形矩阵乘法（低秩维度d_lr ≪ S）是memory-bound特性，CUDA cores更适合。

从kernel调度角度拆解，kernel的两阶段执行流程：
```
// Fused MM + Top-K Estimation Kernel (DSV)
// 输入: Q_lr [H, S, d_lr], K_lr [H, S, d_lr], K_per_query

// Stage 1: Per-SM computation with online top-K
for each SM assigned rows [r_start, r_end]:
    for query in [r_start, r_end]:     // 每个SM处理多个完整query行
        scores = []                     // 寄存器中保留top-K (score, index) pairs
        for each tile of K_lr rows:
            // CUDA core matmul: q_tile [1, d_lr] @ K_lr_tile^T [tile_size, d_lr]
            partial_scores = dot_product(q_tile, K_lr_tile_rows)
            // Bitonic Select: 在线合并当前top-K与新partial
            scores = BitonicMerge(scores, partial_scores, K_per_query)
        // scores = [(score_1, idx_1), ..., (score_K, idx_K)]

    // Stage 2: Threshold-based index selection
    // 当K_per_query很大时（如20K at 90% sparsity with S=200K），
    // 直接保留K个indices会超出shared memory限制
    // 方案：先确定每个query的top-K阈值，再二次扫描选择indices
    if K_per_query > shared_mem_threshold:
        threshold = scores[K_per_query].score
        // 重新遍历，仅记录score > threshold的indices
```

设计要点：(1) 使用CUDA cores而非Tensor cores——slim矩阵（d_lr=16 vs S=200K）无法有效利用Tensor core的MMA指令；(2) Bitonic Select用于在线top-K合并（避免sorting完整partial results）；(3) 两阶段split应对超大K场景（避免shared memory溢出）；(4) kernel设计为memory-coalesced——相邻query共享K_lr tile，最大化L2 cache hit。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

DSV中使用自定义CUDA kernel实现。输入shape：[H, S, d_lr]，d_lr ≪ d_k（如16 vs 128）。必须在线执行（per-step），因为Q_lr, K_lr随每步输入变化。在Stage 2训练中每个forward pass调用。替代方案：直接用Triton来实现类似的fused kernel（减少开发成本但性能可能略低）。性能数据：forward pass overhead相对可控，backward无额外overhead（索引可在backward中复用）。

涉及论文标题：
- DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training
