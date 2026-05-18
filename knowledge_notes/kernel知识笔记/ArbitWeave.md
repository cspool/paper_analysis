## ArbitWeave

术语是什么？

ArbitWeave 是 Drawloom 提出的面向 GPU Tensor Core 的 SpMV 稀疏矩阵映射策略，支持将任意稀疏矩阵灵活映射到不同 shape 的 TC 和 SpTC 硬件单元。核心设计：(1) 根据 TC shape 的结构比 V = mma_m / mma_n 将稀疏矩阵切分为 V-width row strip（如 V=2 时每 2 个连续行组成一个 strip），作为映射到 TC block 的基本计算单元；(2) 按每个 row strip 的非零元数量（NNZ）分为 Long Mapping（nnz > T1，独占 TC block）、Medium Mapping（T2 < nnz ≤ T1，多个 strip 聚合到 TC block）、Short Mapping（nnz ≤ T2，利用 2:4 sparsity 映射到 SpTC block）三种策略；(3) 阈值 T1 = (mma_m / V) × mma_k × WarpLoad, T2 = mma_k 由 TC shape 和 warp 负载导出。ArbitWeave 的核心创新在于同时调度 TC 和 SpTC 处理不同稀疏度区域。

从kernel调度角度拆解术语：

ArbitWeave 的映射计算过程（以 m16n8k16 TC, v=2 为例）：

```
// 步骤1: 确定 TC shape 和 strip
mma_m=16, mma_n=8, mma_k=16, V = mma_m/mma_n = 2
row_strip_width = V = 2   // 每2行组成一个strip

// 步骤2: 按NNZ分类每个row strip
for each row_strip in sparse_matrix:
    nnz = count_nonzeros(strip)
    if nnz > T1:
        classify as Long Mapping   // 独占TC blocks
    elif nnz > T2:
        classify as Medium Mapping // 多strip共享TC block
    else:
        classify as Short Mapping  // 映射到SpTC (2:4 sparsity)

// 步骤3: Long/Medium mapping → dense TC
// 将row strip内的非零元按列压缩(去除零值列)
// 对齐到TC block的B矩阵列位置
// 结果Y沿TC output matrix对角线输出
// 压缩后Ecomp = #nonzeros / (mma_m × mma_k)

// 步骤4: Short mapping → SpTC
// 非零元配对为2:4 pattern(每4位置最多2非零)
// 50%压缩+2-bit metadata编码位置
// 向量X按remapped CID重排到SpTC B矩阵
// SpTC硬件跳过零值计算→2x throughput
```

ArbitWeave 相比 DASP 的 m8n8k4 固定映射：在 A100/H100 上选择 m16n8k16（真TC硬件执行→避免ALU fallback），同时通过 Short Mapping 首次用 SpTC 计算极短行（DASP 中 short row 只能 fallback 到 CUDA Cores 计算）。

术语一般如何实现？如何使用？

ArbitWeave 实现为 CUDA C++ kernel 的预处理+运行时映射：预处理阶段扫描矩阵每个 row strip 的 NNZ 决定 Long/Medium/Short category→构建对应的 ZCF 索引结构（longPtr/mediumPtr/shortCid）→kernel launch 时 warp 根据 ZCF 指针索引正确的 TC block 数据→Long row 结果 intra-warp shuffle reduce + inter-warp reduction kernel 归约。Short Mapping 消融实验显示开启 SpTC short mapping 在 M12（econ_fwd500）上提速 1.54×、M17（cop20k_A）上提速 1.70×。

涉及论文标题：
- Exploiting Efficient Mapping and Pipelined Execution for Accelerating SpMV on Tensor Cores

