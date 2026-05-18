## SpMSpV (Sparse Matrix-Sparse Vector Multiplication)

术语是什么？

SpMSpV (Sparse Matrix-Sparse Vector Multiplication) 是计算 y = A·x 的线性代数原语，其中矩阵 A 和输入向量 x 都是稀疏的（仅少量非零元素）。与 SpMV (Sparse Matrix-Vector multiplication, x 为稠密) 不同，SpMSpV 的计算量由输入向量 x 的非零元数量决定而非矩阵大小，因此当 x 极度稀疏时 SpMSpV 比 SpMV 更高效。SpMSpV 是图分析 (BFS、PageRank、Personalized PageRank) 的核心原语，也是 GraphBLAS、Gunrock、GraphBLAST、GraphMat 等图计算框架的代数基础。此外，SpMSpV 在脉冲神经网络 (SNN) 的 event-driven spike propagation 中也有应用——spike delivery 可自然表示为稀疏矩阵-稀疏向量乘法。

从算法pipeline角度拆解术语：

SpMSpV 有两种执行范式（以矩阵 A 为稀疏矩阵、x 为稀疏向量）：

**Row-major (matrix-driven / CSR / pull) 范式：**
```
Input: A in CSR (row_ptr, indices, values) with N rows;
       dense vector x with value array and bitmask bm
Output: result vector y
1: for all r ← 0 to N in parallel do
2:     start ← row_ptr[r], end ← row_ptr[r+1]
3:     res ← 0
4:     for j ← start to end do
5:         col ← indices[j]
6:         if bm[col] then               // bitmask检查x[col]是否非零
7:             res ← res + values[j] × x[col]
8:     y[r] ← res
```
遍历矩阵所有行，通过 bitmask 跳过 x 的非活跃列。缺点：无论 x 稀疏度如何都遍历所有行，不能充分利用向量稀疏性。

**Column-major (vector-driven / CSC / push) 范式：**
```
Input: A in CSC (col_ptr, indices, values); x in sparse format (idx, val)
Output: result vector y
1: for all active entries i in x in parallel do
2:     col ← idx[i], v_val ← val[i]
3:     start ← col_ptr[col], end ← col_ptr[col+1]
4:     for j ← start to end do
5:         row ← indices[j], mat_val ← values[j]
6:         partial ← mat_val × v_val
7:         write_back(y[row], partial)   // scatter accumulation
```
仅遍历 x 的非零元，每个非零元索引 A 的对应列，计算 partial products 并 scatter 到输出向量 y。缺点：write-back 阶段的多对一 scatter 导致 conflict 和低带宽利用。

**Weighted vs Unweighted SpMSpV：** Weighted SpMSpV 中矩阵 A 和向量 x 均含一般权重（浮点值），要求 multiply-accumulate (乘加)。Unweighted SpMSpV（如 BFS-style）中值为二值，可用 atomicOr 或 output masking 优化，代表性工作包括 TileSpMSpV 和 BerryBees。VDHA 聚焦 Weighted SpMSpV。

术语一般如何实现？如何使用？

GPU 上的 SpMSpV 实现：
- **cuSPARSE**: NVIDIA 官方稀疏库提供 CSR-based SpMV，可通过添加 bitmask value validation 扩展到 SpMSpV
- **Gunrock**: GPU 图分析框架，使用 atomic-based column-major SpMSpV 内核
- **FastSpMSpV**: 提出 sort-reduce 方法避免 atomics，通过全局排序消除 write conflicts
- **Adaptive SpMSpV**: 根据矩阵特征和向量稀疏度在 8 个候选 kernel（row/col-major × atomic/sort × 不同负载均衡策略 + row-major SpMV fallback）间选择
- **VDHA**: 使用 shared-memory hash table 做 local aggregation 减少 global write conflicts，结合 column decomposition 增强 locality
- **HAM-SpMSpV** (CPU): 多核 CPU 上的 masked SpMSpV，使用 pre-bucketing 和 hash-based 算法
- 矩阵存储格式：row-major 使用 CSR (Compressed Sparse Row)，column-major 使用 CSC (Compressed Sparse Column)
- 负载均衡策略：Direct-mapped (直接映射)、Block-mapped (block 内分组)、Global-mapped (全局 prefix-scan 均匀分配)

涉及论文标题：
- VDHA: Vector-Driven Hash Aggregation for Sparse Matrix-Sparse Vector Multiplication on GPUs
- Uni-STC: Unified Sparse Tensor Core

