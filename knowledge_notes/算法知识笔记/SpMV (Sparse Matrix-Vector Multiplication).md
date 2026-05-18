## SpMV (Sparse Matrix-Vector Multiplication)

术语是什么？

Sparse Matrix-Vector Multiplication (SpMV) 是科学计算、图分析和机器学习中最基础的操作之一，定义为 y = α·A·x + β·y（或简写 y = A·x + y），其中 A 是稀疏矩阵（大部分元素为零），x 和 y 是密集向量。SpMV 是 memory-bound kernel——计算强度极低（每个非零元素仅 2 FLOPs：一次乘法 + 一次加法），性能瓶颈在内存带宽而非计算能力。由于矩阵 A 的稀疏模式高度不规则（非零分布不均、行间 NNZ 方差大），SpMV 在 GPU 上的高效实现极富挑战性：需要克服不规则访存（indirect indexing through column IDs）、行间负载不均衡、以及低计算密度导致的 memory latency 支配。

从算法pipeline角度拆解术语：

SpMV 的 CSR 格式基本算法流程：

```
Input: A (CSR format): row_ptr[N+1], col_idx[NNZ], values[NNZ]
       x: dense vector of size K
Output: y: dense vector of size N

for i in range(N):                      // 每个输出行
    y_i = 0
    for j in range(row_ptr[i], row_ptr[i+1]):  // 该行的非零元素范围
        col = col_idx[j]                 // 非零元素列索引
        val = values[j]                  // 非零元素值
        y_i += val * x[col]             // 乘累加
    y[i] = y_i
```

GPU 上 CSR SpMV 的并行化：将稀疏矩阵的行分配给 thread/warp/block 并行处理。常见策略为 (1) 一行多线程：每行分配多个 thread，每 thread 处理部分非零元素，warp 内用 shuffle/atomic 归约；(2) 多行一线程：每个 thread 处理多行（对短行矩阵有效）；(3) 2D 分解：将非零元素在 2D grid 上分布（如 merge-based SpMV）。

Tensor Core SpMV（Drawloom 方式）：将稀疏矩阵切分为 V-width row strip→填充到 TC block 的 A 矩阵（按列压缩去除零值列）→向量 X 被加载到 TC block 的 B 矩阵→TC MMA 输出沿对角线产生有效 Y 元素。这避免了传统 CSR 的逐非零迭代，通过 TC 硬件批量计算 row strip 内的乘累加。

术语一般如何实现？如何使用？

GPU SpMV 典型实现栈：
1. **cuSPARSE (NVIDIA)**：vendor 优化库，支持 CSR/CSC/BSR/SELL 等多种格式，运行时选最优格式
2. **Tensor Core SpMV**：DASP（m8n8k4固定TC shape）、Drawloom（ArbitWeave自适应任意TC shape+SpTC加速短行）、Spaden（bitmap格式TC）
3. **SuiteSparse Matrix Collection**：标准 benchmark 数据集，涵盖 >4000 个不同领域/规模/稀疏模式的矩阵。22 个代表矩阵（如 pwtk/circuit5M/webbase-1M/in-2004等）常用于论文性能评估
4. **Performance metrics**：GFlops/s（基于 NNZ 的理论 FLOPs = 2×NNZ），speedup over cuSPARSE

SpMV 在 LLM/科学计算中的使用场景：图神经网络（邻接矩阵乘特征向量）、物理仿真（FEM 刚度矩阵乘位移向量）、推荐系统（embedding 交互稀疏矩阵乘向量）。预处理（格式转换）只需执行一次，overhead 可在多次 solver 迭代中摊销。

涉及论文标题：
- Exploiting Efficient Mapping and Pipelined Execution for Accelerating SpMV on Tensor Cores
- Uni-STC: Unified Sparse Tensor Core

