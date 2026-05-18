## SpMM (Sparse Matrix-Matrix Multiplication)

术语是什么？

SpMM (Sparse Matrix-Matrix Multiplication) 是稀疏矩阵 A（通常表示图邻接矩阵或稀疏特征交互矩阵）与稠密矩阵 B（通常表示节点/特征嵌入矩阵）的乘法操作 C = A × B。SpMM 是科学计算、图分析、GNN（图神经网络）推理和推荐系统中的核心算子。与 SpMV（Sparse Matrix-Vector Multiplication）不同，SpMM 的右侧操作数是多列稠密矩阵，这使得 SpMM 具有更高的算术强度和更多数据重用机会，但也要求格式和 kernel 设计必须平衡稀疏 pattern 的 irregular access 与 dense tile 的 locality 利用。

从算法 pipeline 角度拆解术语：

GNN 推理中 SpMM 的 pipeline 角色（以 GCN 为例）：
```
// GCN Layer 的 SpMM 调用
输入: adjacency_matrix_A[n×n sparse], feature_matrix_W[n×d dense]
输出: C = A × W  // [n×d] dense output

// GCN 推理 pipeline
1. X = input_features                     // [n, d_in]
2. H = Linear(X, W_linear)                // [n, d_hidden]
3. H = SpMM(adjacency_A, H)               // 图卷积聚合：邻接矩阵×特征 → [n, d_hidden]
4. H = ReLU(H)                             // 非线性
5. output = Linear(H, W_output)           // [n, d_out]

// 在多层 GNN 中，SpMM 在每一层重复执行
```

SpMM 的核心计算特征：
```
C[i][j] = Σ_{k: A[i][k] ≠ 0} A[i][k] × B[k][j]
// 每个输出元素 C[i][j] 仅依赖 A 第 i 行的非零元素
// 同一行 i 的所有 j 列共享 A 的非零访问模式 → dense tile B 的数据重用机会
```

与相关算子的区别：
- **SpMV**：B 只有 1 列 → 无 dense tile 重用，算术强度低，主要为 memory-bound
- **SpMM**：B 有多列（如 GNN 中 d_hidden=64/128/256） → 每个非零 A[i][k] 参与 d 次乘法，数据重用的权衡取决于 d 的大小
- **GEMM**：A 和 B 都是 dense → 规则访存模式，可用高度优化的 BLAS kernel
- **SDDMM**：Sampled Dense-Dense Matrix Multiplication → 输出是 sparse，与 SpMM 方向相反

术语一般如何实现？如何使用？

CPU 上的 SpMM 实现策略：
1. **CSR-based SpMM**（如 ArmPL、Eigen、Cholmod）：基于 CSR 格式，对每个非零 A[i][k] 执行 k 行的 vectorized saxpy 更新 C[i][:] ← A[i][k] × B[k][:]。优点：格式简单；缺点：C 的随机写访问导致 cache miss，SME ZA 的 outer-product 能力未利用。
2. **ASM-SpMM 的 SME outer-product SpMM**：通过 OP-MCF 格式将稀疏矩阵转为与 SME vector length 对齐的 row window，每个 window 内用 SME MOPA outer-product 指令计算 sparse_vec ⊗ B_tile 直接累加到 ZA tile。关键优势：利用 predicate mask 消除 zero padding，outer-product 语义天然匹配稀疏计算（sparse vector 作为外积的一个操作数）。
3. **LOOPS 的 hybrid CSR+BCSR**：将矩阵分为 CSR 部分（NEON vector 处理）和 BCSR 部分（SME outer-product 处理），自适应分配避免 zero-propagation 在 SME outer-product 中的算力浪费。

GPU 上的 SpMM 实现策略：
4. **cuSPARSE SpMM**：NVIDIA 官方稀疏库，CSR/COO 等通用格式
5. **Tensor Core SpMM**（TCF、ME-TCF、DTC-LSH、TC-GNN）：将稀疏矩阵切为 fixed-size dense block（2×2/4×4），block 内有足够非零的用 Tensor Core 计算，其余 fallback 到 CUDA core。

典型 benchmark：SuiteSparse Matrix Collection（涵盖不同规模/形状/稀疏度的真实稀疏矩阵），GNN 图数据集（TC-GNN/SNAP/OGB/DGL 的图邻接矩阵）。B 列数通常评估 64/128/256/512/1024。

6. **Swift SpMM**：基于 CSC 格式的 GPU SpMM，通过 sparsity-based column sorting + B row rearrangement + warp-size blocking 实现 dual-input coalesced memory access。将稀疏矩阵按 warpSize=32 分为 regular block（warp 内线程处理连续 32 列→coalesced B 访问）和 irregular block（长列拆分均衡负载），regular kernel 用 segment sum 在 shared memory 中局部归约减少 atomicAdd。在 2757 个 SuiteSparse 矩阵上相对 ASpT/cuSPARSE/RoDe/Sputnik 在 FP64/N=128 下几何平均加速 1.79×/27.02×/3.62×/6.53×。

涉及论文标题：
- ASM-SpMM: Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration
- Swift: High-Performance Sparse-Dense Matrix Multiplication on GPUs
- Uni-STC: Unified Sparse Tensor Core

