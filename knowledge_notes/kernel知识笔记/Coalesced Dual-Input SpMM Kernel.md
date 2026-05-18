## Coalesced Dual-Input SpMM Kernel

术语是什么？

Coalesced Dual-Input SpMM Kernel 是 Swift 论文提出的 GPU SpMM kernel 设计策略，核心目标是同时实现稀疏矩阵 A 和稠密矩阵 B 的 coalesced memory access。传统 GPU SpMM 方法（Sputnik/ASpT/RoDe/cuSPARSE）通常只优化稀疏矩阵侧的存储格式或线程负载均衡，未考虑 warp 内线程在访问 A 的稀疏索引后，用这些索引访问 dense B 时产生的地址跳跃（接近 warpSize 次 memory transaction）。Swift 通过 sparsity-based column sorting + dense row rearrangement + CSC format + warp-size blocking 的组合设计，让 warp 内相邻线程处理的稀疏列对应 B 中连续地址，使 A 和 B 的加载都具备高度合并访问。

从kernel调度角度拆解术语：

Swift 实现 coalesced dual-input 的 kernel 执行流程：

```
// Regular kernel: coalesced dual-input SpMM (1 warp per 32-column block)
// A: M×K sparse (CSC), B: K×N dense (column-major, rows rearranged)
// C: M×N dense output

// Thread block = 32×8 threads (8 warps)
warp_id = tid / 32
lane_id = tid % 32
block_col_start = blkColIdx[warp_id]         // 该 warp 处理的起始列号
block_offset = blkPtr[warp_id]               // value/rowIdx 数组起点

for offset = block_offset to next_block_offset:
    // Step 1: Coalesced sparse A read
    // 所有 lane 读取同一列 block 内不同行的非零元
    value = A_value[offset + lane_id]          // 32 threads → 128B aligned
    row_idx = A_rowIdx[offset + lane_id]       // 32 threads → 128B aligned

    // Step 2: Coalesced dense B read
    // col_idx = block_col_start + lane_id → 连续 32 列
    col_idx = block_col_start + lane_id
    // B 为 column-major，连续列索引 = 连续地址
    for j = 0 to N-1:
        b_val = B[col_idx * K + j]             // 32 threads → coalesced
        partial = value * b_val

    // Step 3: Shared memory store (coalesced per warp)
    smem[lane_id] = partial

    // Step 4: Segment sum in shared memory
    // positionIdx/offsetIdx guide intra-warp reduction (see Segment Sum term)
    ...

    // Step 5: Reduced atomicAdd write-back to C
    segment_sum = smem_reduce(row_idx)
    atomicAdd(&C[row_idx * N + j], segment_sum)
```

coalescing 效果量化：
- 论文实验观察：数据加载相关开销平均超过整体性能的 32%
- Regular part coalesced B access 相对 non-coalesced 版本：N=32 时 1.32×, N=128 时 1.38× 几何平均 speedup
- Column-major B layout 是 coalescing 的关键：warp 内 32 线程处理连续 32 列 → B 地址 = colIdx × K + j → 相邻线程 colIdx 差 1 → 地址差 K（连续）

术语一般如何实现？如何使用？

Coalesced dual-input 的实现要求：
1. **稀疏格式选择**：必须使用 CSC (Compressed Sparse Column) 而非 CSR，因为 column-major 方向使同一列的非零元 row_idx 连续，同 warp 内线程按列分配时访问 B 的列地址也连续。
2. **列排序预处理**：按每列 NNZ 升序排序 A 的列，并同步重排 B 的对应行。排序后相邻列 NNZ 相近 → warp 内各 lane 工作量相似 → 负载均衡改善。B 行重排是关键——不能只排序 A 而让 B 保持原位（否则列索引映射错误）。
3. **Warp-size 对齐**：按 warpSize=32 划分 column block。列宽恰好为 32 的进入 regular kernel，不满足的进入 irregular kernel。
4. **适用条件**：非零元分布较均匀、NNZ 不过大（NNZ > 10^6 时排序+blocking 预处理开销可能超过收益）。

涉及论文标题：
- Swift: High-Performance Sparse-Dense Matrix Multiplication on GPUs

