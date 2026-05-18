## Two-Level Load Balancing (for SpMV on TCs)

术语是什么？

Two-Level Load Balancing 是 Drawloom 为解决 SuiteSparse 数据集中稀疏矩阵行间 NNZ 分布不均导致的 GPU 线程负载不均衡而设计的两级负载均衡策略。(1) Matrix-level：根据 NNZ per row 用两个阈值 T1=(mma_m/V)×mma_k×WarpLoad 和 T2=mma_k 将矩阵行分为 Long/Medium/Short 三类，不同类采用不同 mapping 策略（Long独占TC block、Medium聚合、Short用SpTC）；(2) Warp-level：Long row 生成的 TC blocks 通过 WarpLoad 参数均匀分配至各 warp 执行，Long Mapping 的结果需 inter-warp reduction kernel 归约。

从kernel调度角度拆解术语：

Two-Level Load Balancing 伪代码：

```
// Matrix-level: 按NNZ分类
for row in sparse_matrix:
    nnz = nnz_per_row[row]
    if nnz > T1:
        row_category[row] = LONG
        tc_blocks = ceil(nnz / (mma_m * mma_k))  // 独占TC blocks
    elif nnz > T2:
        row_category[row] = MEDIUM  
        // 多row strip聚合到shared TC block
    else:
        row_category[row] = SHORT  // SpTC

// Warp-level: Long row 的TC blocks分配
// 每个thread block含4 warps (warp_size=32)
WarpLoad = tunable_parameter  // 每warp分配的TC block数
total_long_blocks = sum(tc_blocks for LONG rows)
blocks_per_warp = total_long_blocks / (num_warps)  // 均衡分配

for warp_id in range(num_warps):
    start_block = warp_id * WarpLoad
    end_block = start_block + WarpLoad
    for tc_block in [start_block, end_block):
        compute_tc_mma(tc_block)  // 每个warp处理相近数量的TC blocks

// Long结果归约
if row_category == LONG:
    warp_shuffle_reduce(partial_Y)  // warp内shuffle归约
    atomic_add(global_Y[row_id], partial_Y)  // 或 launch reduction kernel
```

T2=mma_k 的合理性：NNZ < mma_k 的行无法填满一个 TC block 的 k 维度，不适合用 dense TC（浪费算力），故分配给 SpTC Short Mapping。

术语一般如何实现？如何使用？

实现为 CUDA kernel 的 warp 分配逻辑和独立的 reduction kernel（Long row warp 间归约）。WarpLoad 是可调参数，论文中 set to 2。消融实验显示 ZCF 的 Two-Level Load Balancing 将 CSR 的 if-else thread selection 替换为向量化访存，branch 指令减少 50%。

涉及论文标题：
- Exploiting Efficient Mapping and Pipelined Execution for Accelerating SpMV on Tensor Cores

---

