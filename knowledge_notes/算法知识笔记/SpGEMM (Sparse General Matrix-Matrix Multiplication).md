## SpGEMM (Sparse General Matrix-Matrix Multiplication)

术语是什么？

SpGEMM (Sparse General Matrix-Matrix Multiplication) 是计算 C = A × B 的线性代数原语，其中 A 和 B 均为稀疏矩阵，输出 C 也是稀疏矩阵。SpGEMM 是图算法（multi-source BFS 前沿扩展、三角形计数、图粗化）、代数多重网格（Algebraic Multigrid, AMG）的 Galerkin product（计算 R×A×P 的 coarse-grid operator）以及稀疏 DNN（如 GNN 中邻接矩阵的 k 步幂）的核心算子。与 SpMM（A 稀疏 × B dense）不同，SpGEMM 的双输入稀疏性使得：每个 C 非零元素的计算量仅由 A_row 和 B_col 的非零交集决定，执行模型为 Gustavson 算法（CSR-based，对 A 每一行遍历该行非零列→逐列访问 B 的对应行→做 sparse dot product→结果归约到 C 的该行），且 C 的非零模式在计算完成前未知——需通过 symbolic multiplication 预估 C 的 size 和 sparsity pattern。

从算法pipeline角度拆解术语：

Gustavson 算法的 SpGEMM 伪代码：
```
Input: A (CSR), B (CSR)
Output: C (CSR)

// Phase 1: Symbolic (预估 C 的 row_ptr)
C_row_nnz_est[0..N-1] = 0
for i in 0..N-1:
    for k in A.row(i):               // A 第 i 行的所有非零列 k
        for j in B.row(k):           // B 第 k 行的所有非零列 j
            if not marked(i, j):     // 行 i 中 j 列首次出现
                mark(i, j)
                C_row_nnz_est[i]++

// Phase 2: Numeric (计算 C 的实际值)
C_row_ptr = prefix_sum(C_row_nnz_est)
for i in 0..N-1:
    for k in A.row(i):
        val_a = A.val(i, k)
        for j in B.row(k):
            val_b = B.val(k, j)
            C[i][j] += val_a × val_b
```

SpGEMM 的核心难点：
1. **C 非零模式不可预测**：在 numeric 阶段完成前不知道 C 的 sparsity pattern，需要两次 pass 或动态扩容
2. **负载严重不均衡**：A 行长分布可能极度偏斜（power-law），short row 的 C 输出少但 long row 的 sparse dot product 数量以 O(nnz_A_row × avg_nnz_B_row) 爆炸
3. **中间结果膨胀**：每个 A 行可能产生远多于最终 C 实际非零数的中间乘积（称为 intermediate product blowup），需要哈希表或排序去重
4. **访存不规则**：对 B 的访问模式由 A 的非零列索引间接决定，cache miss 率高

在 GNN/科学计算中的典型场景：AMG solver 的 Galerkin product（R×A×P）、GNN 中计算 A²、A³ 以捕捉 k-hop 邻居信息、multi-source BFS 中从多个源点同时扩展 frontier。

术语一般如何实现？如何使用？

GPU SpGEMM 实现策略：
1. **cuSPARSE SpGEMM**：NVIDIA 官方实现，支持 CSR/CSC 输入输出，基于 Hash/排序的 Gustavson 变体
2. **RM-STC**：以 row-row scalar-vector 组合方式，将 SpGEMM 的 Gustavson 算法各 sparse dot-product 映射到固定 shape 的 MAC array 任务
3. **DS-STC**：以 outer-product 方式，将 A 的半列 × B 的半行映射到 MAC array
4. **Uni-STC 方式**：用 BBC 格式统一 A 和 B → TMS 沿 K 维做 bloom filter 判断 C tile 是否非零 → 将 SpGEMM 的计算拆为 4×4×4 T3 任务 → DPG 再拆为 1×1×4 T4 dot-product → SDPU 执行 segmented dot-product 并预合并 partial products
5. **SuiteSparse 方阵子集**：标准 benchmark，Uni-STC 使用 2126 个方阵计算 C=A²

Uni-STC 在 SpGEMM 上相对 RM-STC 和 DS-STC 的平均 speedup 为 1.45x 和 2.40x（64 MAC@FP64），能量效率提升分别为 1.09x 和 3.14x。

涉及论文标题：
- Uni-STC: Unified Sparse Tensor Core

