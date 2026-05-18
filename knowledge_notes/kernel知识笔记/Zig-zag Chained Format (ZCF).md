## Zig-zag Chained Format (ZCF)

术语是什么？

Zig-zag Chained Format (ZCF) 是 Drawloom 为 Tensor Core 和 Sparse Tensor Core 加速设计的稀疏矩阵存储格式，替代 CSR 等传统格式。ZCF 按行 strip 的 NNZ 分类构建三种数据布局：(1) Long ZCF：longPtr（TC block 计数 per strip）+ longCid（列索引）+ longVal（非零值），Long row strip 独占完整 TC blocks；(2) Medium ZCF：mediumPtr（TC block 计数 per window）+ mediumCid + mediumVal，多个 medium row strip 聚合到一个 TC block；(3) Short ZCF：shortCid（remapped 列索引）+ shortVal（2:4 压缩值），直接兼容 SpTC 的 structured sparse 格式。ZCF 的核心特征是 zcf_value_stride 参数控制向量化访存粒度——FP16 stride=8, FP32/FP64 stride=4——使每 thread 可执行 128-bit vectorized load 对齐 GPU memory transaction 粒度。

从kernel调度角度拆解术语：

ZCF 的构建和访问伪代码：

```
// ZCF构建 (preprocessing)
Input: CSR matrix A (row_ptr, col_idx, values)
       TC shape: mma_m, mma_n, mma_k
       Thresholds: T1, T2

row_strip_width = mma_m / mma_n  // e.g., 2

// 按NNZ per row重排序矩阵行
sorted_rows = sort_rows_by_nnz(A)

for strip_id, (row_i, row_{i+1}) in enumerate(sorted_rows):
    nnz = count_nnz(strip)
    if nnz > T1:    // Long
        longVal.append(values[strip])
        longCid.append(col_idx[strip])
        longPtr.append(num_TC_blocks)
    elif nnz > T2:  // Medium  
        if window_full:
            mediumVal.append(values[window])
            mediumCid.append(col_idx[window])
            mediumPtr.append(num_TC_blocks)
    else:           // Short (SpTC)
        remap_2_4_sparsity(values, col_idx)  // 50% compress
        shortVal.append(compressed_values)
        shortCid.append(remapped_column_ids)

// ZCF kernel访问 (runtime)
// 向量化访存: zcf_value_stride对齐128-bit transaction
// zcf_value_stride = 8 (FP16) or 4 (FP32/FP64)
for each TC block in warp assignment:
    // 128-bit vectorized load (每个thread加载连续stride个元素)
    val_tile[i] = longVal[block_base + thread_id * stride]
    cid_tile[i] = longCid[block_base + thread_id * stride]
```

相比 DASP 的离散、非 coalesced 访存，ZCF 通过 zig-zag chain 布局保证 TC block 内数据连续→减少 IMAD（memory index 计算）指令 67.8%、branch 指令 50%、memory bandwidth 提升 48.3%。

术语一般如何实现？如何使用？

ZCF 实现为 CUDA C++ 预处理函数（CPU 端运行一次，overhead 可被多次 SpMV 迭代摊销）。预处理：读取 CSR→按 NNZ per row 排序→按 T1/T2 阈值分类→构建三种 ZCF 数组→输出为 Drawloom kernel 的输入。kernel 中 warp 通过 ZCF 的 ptr 数组索引正确的 Cid/Val 数据块执行 vectorized load（使用 `float4`/`double2` 等 128-bit 类型强制 coalesced global memory transaction）。FP16 场合 stride=8 使每 warp（32 thread）一轮访存 load 256 个值（32×8）。

涉及论文标题：
- Exploiting Efficient Mapping and Pipelined Execution for Accelerating SpMV on Tensor Cores

