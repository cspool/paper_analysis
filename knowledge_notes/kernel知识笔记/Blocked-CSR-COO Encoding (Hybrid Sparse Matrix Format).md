## Blocked-CSR-COO Encoding (Hybrid Sparse Matrix Format)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Blocked-CSR-COO 是 MegaBlocks 设计的混合块稀疏矩阵编码（图 6）。以 Blocked Compressed Sparse Row (BCSR) 为主格式，额外物化每个 non-zero block 的行索引（row_idxs），使 BCSR 同时具备 Blocked Coordinate (BCOO) 格式的随机访问能力。BCSR 天然高效支持按行迭代（DSD/DDS 操作），但 SDD 并行化需要知道每个 output block 的行坐标——纯 BCSR 需要搜索 row_offsets。通过物化 row_idxs（每 128×128 block = 16384 非零值仅需 1 个索引），SDD kernel 中 threadblock 可 O(1) 直接定位其 non-zero block 坐标。

从kernel调度角度拆解术语：
```
// Blocked-CSR-COO 数据结构（图 6）：
// BCSR 部分（主格式，按行压缩）
row_offsets:   [0, 3, 5, 8, ...]  // row i 的 non-zero blocks 起始偏移
column_idxs:   [0, 2, 4, 1, 3, ...]  // 每个 non-zero block 的列索引

// BCOO 部分（额外物化，SDD 并行化需要）
row_idxs:      [0, 0, 0, 1, 1, ...]  // 每个 non-zero block 的行索引
// row_idxs[i] 指示第 i 个 non-zero block 位于 matrix 的哪一行

// SDD kernel 利用 BCOO 行索引:
// 直接 O(1) 获取坐标，无需搜索
row = c.row_idxs[blockIdx.x];
col = c.column_idxs[blockIdx.x];

// 对比：纯 BCSR 的 SDD 需要搜索
// for row in 0..n_rows:
//     if blockIdx.x < row_offsets[row+1]: break
// 在高稀疏度（MoE 中 >90% zeros）下此搜索开销显著
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Metadata 存储开销 <0.1%（因 large block size，128×128 = 16384 个 FP16 值 = 32KB 仅需 4 字节 row index）。
- 同一数据结构的 BCSR 部分用于 DSD（按行迭代），BCOO 部分用于 SDD（坐标访问），无需维护两套格式。
- MoE 场景中稀疏拓扑随每 iteration 变化，make_topology CUDA kernel 每层每 iteration 重建完整元数据（O(num_experts + num_tokens/128)）。

涉及论文标题：
- MegaBlocks: Efficient Sparse Training with Mixture-of-Experts

---
