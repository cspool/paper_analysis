## OP-MCF (Outer-Product-friendly Masked Column-merging Format)

术语是什么？

OP-MCF 是 ASM-SpMM 提出的面向 ARM SME outer-product 执行模型的稀疏矩阵压缩格式。它按 SME vector length 将稀疏矩阵的连续行切分为 row window，在每个 window 内删除空列并将非零位置不重叠的列合并为一个 compressed slot，同时用 bitmask 记录每个原始列内的有效行位置，使 SME predicate register 可以只对有效非零位置参与 outer product 计算。与 GPU Tensor Core 格式（TCF、ME-TCF）的根本区别在于：OP-MCF 消除硬性 block padding，适应 SME 的 predicate-driven vector outer-product 语义。

从 kernel 调度角度拆解术语：

OP-MCF 的格式结构和 SpMM kernel 中对一个 row window 的处理：
```
// OP-MCF 格式结构（四数组）
RowWindowOffset[i]       → row window i 的起始行索引
ColumnOfRowWindow[i]     → row window i 的 compressed column 数量
SparseAtoB[j]            → compressed slot j 对应的原始列索引（指向 B 的行）
ColumnPositionMaskBit[j] → 64-bit bitmask，标识 slot j 内哪些行有有效非零

// SpMM kernel 对一个 row window 的处理
for each row_window_r:
    ZA_tile.clear()                           // 清零目标 ZA tile
    for j in compressed_slots_of(r):
        col_idx = SparseAtoB[j]               // 找到 B 的对应列
        sparse_vals = svld1(Z_reg, A_sparse[j])  // 加载 compressed sparse values
        pred = whilelt(ColumnPositionMaskBit[j])  // bitmask → predicate register
        B_tile = svld1(B[col_idx:col_idx+SVL])   // 加载 dense B tile
        ZA_tile = svmopa(ZA_tile, pred, pred, sparse_vals, B_tile) // outer product
        _svprfw(A_sparse[j+1])                // 预取下一 slot
    svst1_hor(C[row_window_r], ZA_tile)       // 写回输出
```

关键设计决策：
1. **Column compaction**：删除 row window 内的空列（全零列）→ 减少无效 slot 遍历
2. **Masked multi-column merging**：非零位置不重叠的多个原始列可以合并为一个 compressed slot → 提高每 slot 的有效非零密度。mean NNZ per slot 达到约 4-6（CSR 约 1-2）
3. **Bitmask 而非 explicit index**：ColumnPositionMaskBit 为 64-bit（对应 FP64 下 8 行 row window 内的每行），直接转 predicate register，无需 runtime 解压 row index
4. **与 GPU Tensor Core 格式的差异**：TCF 要求 left-aligned tile 和 fixed block padding（2×2/4×4），zero padding 浪费算力；OP-MCF 无对齐约束，predicate 按需屏蔽

术语一般如何实现？如何使用？

OP-MCF 需要一次性格式转换——遍历原始稀疏矩阵，按 SVL 划分 row window，对每个 window 做空列检测、列重叠分析和 bitmask 生成。转换时间复杂度 O(nnz + n_cols_per_window)。适合同一稀疏矩阵被重复执行的场景（GNN inference 中邻接矩阵固定、迭代 solver、超参搜索）。若矩阵频繁变化且复用次数少，格式转换成本可能难摊销。OP-MCF 的实现细节（压缩算法、bitmask 编码）论文通过四数组描述但未开源。

涉及论文标题：
- ASM-SpMM: Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration

---

