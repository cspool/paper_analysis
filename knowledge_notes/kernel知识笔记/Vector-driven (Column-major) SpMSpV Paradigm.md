## Vector-driven (Column-major) SpMSpV Paradigm

术语是什么？

Vector-driven (向量驱动) SpMSpV 执行范式（也称 column-major 或 push paradigm）是一种仅遍历稀疏输入向量非零元的 SpMSpV 计算模式。与 row-major (matrix-driven) 遍历矩阵所有行不同，vector-driven 以 x 的每个非零元为起点，索引矩阵对应列，生成 partial products 并 scatter 到输出向量。流程分为两个阶段：**Fetch 阶段**——从 CSC 格式矩阵中加载非零列对应的 indices 和 values；**Write-back 阶段**——将 partial products (mat_val × vec_val) 按 row index scatter 累加到输出向量 y。Vector-driven 范式能充分利用向量稀疏性——计算量与 x 的非零元数成比例而非矩阵大小，在 x 极度稀疏时显著优于 row-major。

从kernel调度角度拆解术语：

Vector-driven SpMSpV 的 GPU kernel 伪代码：

```
// GPU Kernel: Vector-driven SpMSpV
Input: A in CSC (col_ptr, indices, values); x_sparse (idx[], val[]); n_active
Output: y[]

// === Fetch Phase (Load Balancing Variants) ===
// Variant 1: Direct-mapped — 每个活跃列分配给一个 CTA
for each active_idx i in [0, n_active) mapped to CTA tid:
    col = x_sparse.idx[i]
    load column segment: col_ptr[col] to col_ptr[col+1]

// Variant 2: Block-mapped (Gunrock-style) — 多个短列聚合到一个 CTA
CTA groups multiple short columns
for each group assigned to CTA:
    block_prefix_scan over column lengths
    load indices and values for the group

// Variant 3: Global-mapped (merge-based) — 全局 prefix-scan 按 NNZ 均匀分配
for each nonzero in x, record column length
global_inclusive_scan over lengths → total_NNZ
partition total_NNZ evenly across CTAs

// === Write-back Phase ===
for each (row_idx, mat_val) in fetched segment:
    partial = mat_val × x_sparse.val[col]
    // Strategy A: Atomic — 直接 global atomic
    atomicAdd(&y[row_idx], partial)
    // Strategy B: Sort-reduce — buffer + sort + reduce
    buffer.append(row_idx, partial)
    sort(buffer by row_idx); reduce duplicates
    // Strategy C: Hash aggregation (VDHA)
    hash_insert(shared_hash_table, row_idx, partial)
```

负载均衡策略对比：Direct-mapped 无 prefix-scan 开销但负载不均（NaiveSpMSpV）；Block-mapped 多短列聚合到 CTA（Gunrock）；Global-mapped 按 NNZ 均匀分配负载最均衡但有 prefix-scan 开销（merge-based SpMV）。

术语一般如何实现？如何使用？

GPU 实现要点：矩阵需以 CSC 格式存储（col_ptr、row_indices、values 数组）；输入向量以 sparse format (idx[], val[]) 存储；性能瓶颈在 write-back 阶段的 many-to-one scatter pattern 导致的 address contention 和 uncoalesced stores。Adaptive SpMSpV 框架根据矩阵特征和向量稀疏度在多种 vector-driven kernel 与 row-major kernel 间动态选择。适用于图分析（BFS, PageRank）、SNN spike propagation、科学计算中的稀疏线性代数。

涉及论文标题：
- VDHA: Vector-Driven Hash Aggregation for Sparse Matrix-Sparse Vector Multiplication on GPUs

---

