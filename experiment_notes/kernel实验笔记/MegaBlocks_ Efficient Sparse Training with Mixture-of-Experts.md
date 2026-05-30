## MegaBlocks: Efficient Sparse Training with Mixture-of-Experts

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现自定义 **block-sparse GPU kernels**（SDD, DSD, DDS）用于高效执行 MoE 的动态、负载不均衡的 expert 计算。基于 CUTLASS 2.5 扩展，核心 kernel 设计包括：
    - **Hybrid Blocked-CSR-COO 编码（§5.1.3）**：以 BCSR (Blocked Compressed Sparse Row) 为主要稀疏矩阵格式，同时额外物化每个 non-zero block 的行索引，使得 SDD 操作的并行化无需搜索 row offsets（直接 O(1) 查找 block 坐标）。存储开销可忽略（每个 128×128 block = 16384 个非零值仅需 1 个索引）。
    - **Transpose Indices（§5.1.4）**：为高效支持 block-sparse 矩阵的转置访问（向后传播需要），构造转置后的元数据（等效于 BCSC 编码），包含 column offsets 和转置顺序的 non-zero block 偏移索引数组。无需显式转置非零值（避免数据复制），通过一层间接索引实现转置迭代。类似数据库的 secondary index。
    - **Block Size 选择（§5.1.2）**：基于 CUTLASS 的 tile dimension benchmark，选择 128×128 block size，因为它在 A100 上对所有 tile 配置表现最优（图 5），与 cuBLAS 为 dense Transformer 模型选择的配置一致。
    - **Permutation kernel 融合（§5.2）**：将 token padding（每个 expert 的 token 数填充到 128 的倍数）融合进自定义 permutation kernel。同时在前向开始时构造 block-sparse 矩阵元数据和转置元数据，摊销到后续多次矩阵乘法。
  - 实验比较：
    - Block-sparse kernels vs cuBLAS batched GEMM（§6.3, Figure 9）：18 种 problem configuration（MoE-XS/Small/Medium × 6 operations），平均达到 cuBLAS 98.6% 吞吐量（标准差 4%，最大 104%，最小 91%）。
    - Block-sparse kernels vs Triton Blocksparse（Appendix C, Figure 13）：平均 9× 吞吐量优势（含 Triton Blocksparse 的 sparse matrix preprocessing 开销）。
    - End-to-end：MegaBlocks dMoE vs Tutel dMoE（含动态 capacity factor，§6.1）：1.38×–4.35× 训练加速；MegaBlocks dMoE vs Tutel token-dropping MoE（§6.2）：1.18×–1.38× 加速。

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA A100 SXM4 80GB。单卡 micro-benchmark 和 8-GPU 端到端训练实验。
  - CUDA 11.5 + CUTLASS 2.5。A100 Tensor Cores 用于 FP16 + FP32 accumulation。
  - 8-way expert model parallelism（MoE 层）+ data parallelism（其他层），通过 gradient accumulation 实现 batch size 512 sequences。使用最大不 OOM 的 micro_batch_size（见表 3）。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 CUTLASS 2.5 扩展实现 block-sparse GEMM kernels (SDD, DSD, DDS)，支持所有 transposed/non-transposed 输入组合。
  - 修改了 Megatron-LM 的 MoE layer：将 batched matrix multiplication 替换为 block-sparse matrix multiplication。
  - Custom CUDA kernel 用于构造 sparse matrix topology (make_topology) 和 permutation (padded_gather/padded_scatter)。
  - Micro-benchmark：对 18 种 problem configuration（3 模型 × 6 operations: SDD, DSD, DDS, SDD^T, DS^T D, DSD^T），每种执行 100 次取平均吞吐量。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/stanford-futuredata/megablocks，Apache-2.0
  - 评估原理（kernel 输入到性能输出全过程）：

```
// ===== SDD Kernel (Sparse = Dense × Dense, Figure 11) =====
// 输入: Matrix a (m × k dense), Matrix b (k × n dense),
//       SparseMatrix c (m × n, block-sparse output)
// 输出: SparseMatrix c（写入每个 non-zero block 的计算结果）

__global__ void sdd(Matrix a, Matrix b, SparseMatrix c) {
    // (1) Load block coordinates from hybrid Blocked-CSR-COO
    //     每个 threadblock 处理一个 non-zero block
    int row    = c.row_idxs[blockIdx.x];    // 来自 BCOO 行索引
    int column = c.column_idxs[blockIdx.x]; // 来自 BCSR 列索引

    // (2) Zero accumulator (128×128 tile)
    Tile<128, 128> tile_c(0);

    // (3) Main loop: iterate over k dimension in 128-step
    for (int i = 0; i < k; i += 128) {
        Tile<128, 128> tile_a = LoadTile(a, row, i);
        Tile<128, 128> tile_b = LoadTile(b, i, column);
        tile_c += tile_a * tile_b;  // Tensor Core MMA
    }

    // (4) Store result to sparse output
    StoreTile(tile_c, c);
}

// ===== DSD Kernel (Dense = Sparse × Dense, Figure 12) =====
// 输入: SparseMatrix a (m × k, BCSR), Matrix b (k × n dense)
// 输出: Matrix c (m × n dense)

__global__ void dsd(SparseMatrix a, Matrix b, Matrix c) {
    // (1) Each threadblock computes one 128×128 tile of dense output
    int row    = blockIdx.x;  // output row tile index
    int column = blockIdx.y;  // output column tile index

    // (2) Load BCSR row offset and compute number of non-zeros in this row
    int offset_a = a.row_offsets[row];
    int nnz      = a.row_offsets[row + 1] - offset_a;

    // (3) Zero accumulator
    Tile<128, 128> tile_c(0);

    // (4) Main loop: iterate over non-zero blocks in row
    for (int i = 0; i < nnz; i++) {
        Tile<128, 128> tile_a = LoadTile(a, offset_a, i);
        // (5) Load column index of this non-zero block from a
        //     to determine which row to load from b
        int row_b = a.column_idxs[offset_a + i];
        Tile<128, 128> tile_b = LoadTile(b, row_b, column);
        tile_c += tile_a * tile_b;
    }

    // (6) Store result
    StoreTile(tile_c, c);
}

// ===== Transpose Indices for DSD^T / DDS^T =====
// 当 sparse operand 需要转置时（如 DS^T D）：
// 使用 transpose indices（图 6 中 Transpose Indices 数组）
// 它存储按 transposed 顺序排列的 non-zero block 偏移
// 在 DSD kernel 的 main loop 中：
//    loaded_offset = transpose_indices[offset_a + i]  // 间接索引
//    Tile<128,128> tile_a = LoadTile(a, loaded_offset, ...);
// 避免显式转置整个 sparse matrix（节省内存和时间）

// ===== 从输入到性能输出的全流程 =====
// 1. 输入: token tensor x (num_tokens × hidden_size, FP16)
// 2. Router: indices, weights = router(x)
//    indices: (num_tokens,), 每个 token 分配的 expert ID
// 3. make_topology(indices) → 构造图 6 的稀疏矩阵元数据:
//    - Blocked-CSR-COO: row_offsets, row_idxs, column_idxs
//    - Transpose indices: 转置访问的间接索引
// 4. padded_gather: 按 expert 分组 tokens + padding to 128 倍数
// 5. sdd(x, w1, topology): 每个 non-zero block 启动 1 个 threadblock
//    - 128×128 tile, FP16 Tensor Core MMA (m=128, n=128, k=128)
//    - 输出: block-sparse intermediate (每 expert batch 的结果)
// 6. dsd(intermediate, w2): 每个 dense output tile 启动 1 个 threadblock
//    - 迭代对应 row 的 non-zero blocks
//    - 输出: dense tensor
// 7. padded_scatter + weight scaling → MoE layer output
// 8. 性能度量: TFLOPs = (total math ops) / (elapsed time)
//    vs cuBLAS 的 relative throughput = TFLOPs_kernel / TFLOPs_cuBLAS
```
  - 关键性能结果：
    - 128×128 block size 在 A100 上实测优于其他配置（图 5），对应 128×128 CUTLASS tile dimensions。
    - Metadata 内存开销 <0.1%（得益于大 block size），使得 hybrid CSR-COO 和 transpose indices 的额外存储可忽略。
    - SDD 操作通过 BCOO 行索引避免 row offset 搜索的开销，对高 expert count（稀疏度 >90%）至关重要。
    - DSD^T 和 DDS^T 中 transpose indices 引入的间接访问降低了空间局部性，导致 <10% 吞吐量损失，但由于这两类操作仅占端到端运行时间很小比例，影响极小。
    - Micro batch size 影响：Tutel 因 padding 导致内存占用大，micro_batch_size 被迫缩小 2×–8×（表 3），降低了 GPU 利用率和硬件效率。
