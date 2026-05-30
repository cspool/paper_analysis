## Transpose Indices (Secondary Index for Sparse Matrix Transposition)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Transpose Indices 是 MegaBlocks 用于高效支持 block-sparse 矩阵转置访问的技术（§5.1.4）。在模型训练的前向+后向传播中，需要混合转置和非转置的 block-sparse 操作（SDD^T, DS^T D, DSD^T, DD^T S）。纯 BCSR 按行存储，在转置顺序下迭代（按列访问）需要搜索所有行来查找目标列中的 non-zero blocks（Buluç et al. 2009）。MegaBlocks 避免显式转置稀疏矩阵（需 O(nnz) 数据复制），而是仅构造转置元数据：等效 BCSC（Blocked Compressed Sparse Column）编码的 column_offsets + 转置顺序的 non-zero block 偏移索引数组（transpose_indices）。kernel 通过 transpose_indices 的间接索引在转置顺序下迭代矩阵，类似数据库的 secondary index。

从kernel调度角度拆解术语：
```
// Transpose Indices 数据结构:
// column_offsets: [0, 2, 5, 7, ...]  // BCSC: 每列 non-zero blocks 的起始偏移
// transpose_indices: [3, 7, 1, 5, 0, ...]  // 转置顺序下的 block 偏移
//   transpose_indices[k] = 原 BCSR 中的第 k 个（转置顺序）non-zero block 的存储偏移

// 在 DSD^T kernel 中使用:
for (int i = 0; i < nnz_in_transposed_row; i++) {
    int blk_offset = a.transpose_indices[col_offset + i];  // 间接索引
    Tile<128,128> tile_a = LoadTile(a, blk_offset);
    // ... 其后与标准 DSD 相同
}
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 存储开销可忽略：仅 column_offsets（~num_experts 个整数）和 transpose_indices（~nnz 个整数），总 metadata <0.1%。
- 与 BCSR-COO 元数据在 make_topology kernel 中同时构造，摊销到 forward+backward 共 6 次 block-sparse 操作。
- 间接访问降低了 DS^T D/DD^T S 操作的空间局部性（<10% 吞吐量损失），但这些 weight gradient 操作仅占端到端训练时间的小部分。

涉及论文标题：
- MegaBlocks: Efficient Sparse Training with Mixture-of-Experts

---
