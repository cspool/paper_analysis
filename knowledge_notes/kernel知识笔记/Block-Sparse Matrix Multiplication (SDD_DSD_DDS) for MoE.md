## Block-Sparse Matrix Multiplication (SDD/DSD/DDS) for MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block-sparse matrix multiplication 是 MegaBlocks 用于替代 MoE 中 batched GEMM 的核心计算原语。使用三字母记法（源自 Triton Blocksparse, Tillet et al. 2019）描述稀疏-密集矩阵乘法：每个字符表示输出/左输入/右输入，S=Sparse, D=Dense, T=Transpose。SDD（Sparse = Dense × Dense）即采样密集-密集矩阵乘法（SDDMM），输出为稀疏矩阵；DSD（Dense = Sparse × Dense）和 DDS 是两种不同的稀疏-密集矩阵乘法（SpMM）。在 MoE FFN 前向传播中：第一层 expert 用 SDD（稀疏输出 = 密集 tokens × 密集权重 w1），第二层用 DSD（密集输出 = 稀疏中间结果 × 密集权重 w2）。向后传播需 SDD^T、DS^T D、DSD^T、DD^T S 四种操作。

从kernel调度角度拆解术语：
MegaBlocks SDD kernel 的 CUDA 伪代码（对应图 11）：
```
__global__ void sdd(Matrix a, Matrix b, SparseMatrix c) {
    // (1) 加载 non-zero block 坐标
    int row    = c.row_idxs[blockIdx.x];    // BCOO 行索引
    int column = c.column_idxs[blockIdx.x]; // BCSR 列索引
    // 每个 threadblock 处理一个 128×128 non-zero block

    // (2) 零初始化 accumulator (128×128 tile)
    Tile<128, 128> tile_c(0);

    // (3) Main loop: n_k 维度以 128 步进
    for (int i = 0; i < n_k; i += 128) {
        Tile<128, 128> tile_a = LoadTile(a, row, i);
        Tile<128, 128> tile_b = LoadTile(b, i, column);
        tile_c += tile_a * tile_b;  // Tensor Core MMA (m=128,n=128,k=128)
    }

    // (4) 写结果到 sparse output 的对应 non-zero block
    StoreTile(tile_c, c);
}
```

DSD kernel（对应图 12）：每个 dense output tile 启动 1 个 threadblock，按 BCSR row offsets 迭代对应 row 的 non-zero blocks，从 each non-zero block 的 column_idx 确定加载 b 的哪一行。128×128 block size 基于 A100 的 CUTLASS tile dimension benchmark 选择（图 5）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 基于 NVIDIA CUTLASS 2.5 扩展实现，利用其 tile-based Tensor Core GEMM 基础设施。Kernel 启动策略：SDD → 每 non-zero block 1 个 threadblock；DSD → 每 dense output tile 1 个 threadblock。
- cuSPARSE blocked-ELL 格式要求所有 row 等量 non-zeros（与 MoE 负载不均衡冲突）。Triton Blocksparse 假定稀疏拓扑在迭代间不变（与 MoE 每 iteration 变化的动态路由冲突）。MegaBlocks 的自定义 kernel 专为动态拓扑设计。
- 在 MoE workload 上平均达到 cuBLAS 密集 GEMM 98.6% 吞吐量（标准差 4%，范围 91%-104%，图 9）。
- Hopper GPU (H100) 上推荐使用 Grouped MLP（grouped GEMM）替代 Sparse MLP（block-sparse），因 Hopper 的 grouped GEMM 性能更优。

涉及论文标题：
- MegaBlocks: Efficient Sparse Training with Mixture-of-Experts

---
