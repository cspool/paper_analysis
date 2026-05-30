## CSR Sparse Matrix Multiplication for LLM Inference（面向 LLM 推理的 CSR 稀疏矩阵乘法 CUDA Kernel）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CSR (Compressed Sparse Row) 稀疏矩阵乘法用于 LLM 推理中的 hybrid sparse-dense weight format——当约1%的权重以非结构化稀疏格式保留为高精度（FP16），其余权重以低位量化存储时，需要专门的 GPU kernel 高效执行 sparse + dense 两部分矩阵乘法的叠加。SpQR 论文为 outlier weights 设计了定制 CUDA kernel：(1) outlier 以 CSR 格式存储（按 row-first/column-second 排序，每个 outlier: FP16 value + FP16 col_index，每行一个 cumulative row pointer）；(2) kernel 通过 tile-based 划分实现 load balancing——将权重矩阵划分为等大小 blocks，每个 thread block 加载其 tile 覆盖的 outlier slice 到 shared memory (SRAM)；(3) 每个 GPU core 判断其 tile 范围的 rows 中哪些含有 outlier，仅对有效行加载对应 col_index 和 value；(4) 执行 sparse dot product 累加到 dense dequantized 结果上。因 outlier 的 row-wise pattern，column index/value 的内存访问趋于连续。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SpQR CSR Sparse MatMul Kernel 伪代码（在 A100 上 batch_size=1 token-by-token 生成）：

```cuda
// Input:  activation x (FP16, d_in)
//         CSR: row_ptr[N+1], col_idx[num_outliers], values[num_outliers]
//         dense_quant: Q (packed INT3/4), scales_s (FP16), scales_z (FP16)
// Output: y = W_dense × x + W_sparse × x

// === Dense dequant + matmul (先执行) ===
for each thread block (tile of output rows):
    for each block (beta1 x beta2 weights):
        // 1. Load bilevel statistics to SRAM
        // 2. Second-level dequant: scales/zeros (3-bit→FP16)
        // 3. First-level dequant: weights (3-bit→FP16)
        // 4. Dot product with activation segment
        partial_dense[tile] += dot(weights_fp16, x_segment)

// === Sparse CSR matmul (后执行) ===
// Tile partitioning
tiles = divide_output_rows_into_tiles(W_dense.rows, TILE_SIZE)

for each thread_block b (maps to tile t):
    // Step 1: Identify outlier range for this tile
    row_start = tiles[t].start_row
    row_end = tiles[t].end_row
    outlier_start = row_ptr[row_start]
    outlier_end = row_ptr[row_end]
    num_tile_outliers = outlier_end - outlier_start

    // Step 2: Load outlier slice to shared memory
    __shared__ uint32_t smem_col_idx[MAX_TILE_OUTLIERS]
    __shared__ half   smem_values[MAX_TILE_OUTLIERS]

    for i = threadIdx.x; i < num_tile_outliers; i += blockDim.x:
        smem_col_idx[i] = col_idx[outlier_start + i]
        smem_values[i] = values[outlier_start + i]
    __syncthreads()

    // Step 3: Per-row sparse dot product
    for row = row_start + threadIdx.x; row < row_end; row += blockDim.x:
        o_start = row_ptr[row] - outlier_start
        o_end = row_ptr[row+1] - outlier_start
        acc = 0.0f
        for k = o_start; k < o_end; k++:
            acc += smem_values[k] * x[smem_col_idx[k]]
        partial_sparse[row] = acc
    __syncthreads()

// === Merge ===
y[row] = partial_dense[row] + partial_sparse[row]
```

Load balancing 关键：步骤1-3通过 tile 划分确保每个 thread block 处理的outlier数量大致均匀，步骤4因row-wise outlier pattern获得连续内存访问。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SpQR 开源：https://github.com/Vahe1994/SpQR。CUDA kernel 以 C++/PTX 实现。核心实现要点：(1) 利用 PyTorch CUDA Extension 编写自定义 kernel（而非 cuSPARSE 通用接口），因为通用稀疏库的 format conversion 和间接内存访问开销过大；(2) 在 token-by-token 生成（batch_size=1, memory-bound）场景下，因压缩率 >3.4x，DRAM 读取量大幅减少，即使额外 sparse compute 开销存在，wall-clock time 仍比 FP16 推理快 20-30%。相比之下，PyTorch 默认的 cuSPARSE sparse matmul 比 FP16 推理更慢（因稀疏矩阵乘法在低 batch size 下 overhead 较大）。该 kernel 与 dense quantized matmul 串联使用，未融合（论文将其列为 future work）。

涉及论文标题：
- SpQR A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression
