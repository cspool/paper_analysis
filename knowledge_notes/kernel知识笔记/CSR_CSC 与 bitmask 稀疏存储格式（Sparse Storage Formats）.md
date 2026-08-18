## CSR/CSC 与 bitmask 稀疏存储格式（Sparse Storage Formats）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CSR（Compressed Sparse Row）与 CSC（Compressed Sparse Column）是稀疏矩阵的标准压缩存储格式：CSR 用三个数组（values 存非零值、col_indices 存每非零的列号、row_ptr 存每行非零的起始偏移）把 O(nnz) 存储降至 O(nnz+rows)；CSC 是转置版本（列压缩）。bitmask（位掩码）格式对轻度稀疏区域用一位表示一个元素是否有值，索引解码开销低、带宽省。Harmonia 的硬件数据通路（Semi-independent PE rows + DN + MRN）对高稀疏 tile 用 CSR/CSC + 显式坐标列表（explicit coordinate lists），对轻度稀疏区域用 bitmask；进入 PE 数据通路前，非零元素必须做动态对齐（dynamic alignment / intersection）——只让坐标匹配的非零对进入乘法器，用可重构 DN + 轻量 on-row index-matching 逻辑完成，避免把零塞进 MAC（这正是固定数据流加速器在高稀疏下利用率崩塌的原因，如 SIGMA Flex-DPE <10%）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CSR 的 SpMSpM 内核骨架（C=A×B，A 按 CSR 存）：
```
for r in 0..M-1:                      # A 的每个非零行
    for (k, a) in A_row[r]:           # A 行内每个非零 (列号 k, 值 a)
        for (n, b) in B_row[k]:       # B 第 k 行的非零 (列号 n, 值 b) —— 索引匹配
            C[r][n] += a * b          # 只有匹配坐标的非零对被乘
```
Harmonia 的 Row 数据流硬件化该骨架：每个 PE 行驻留一个 A 行，DN 按该 A 行的 nnz 列号选择性路由所需的 B 行片段（B 片段在 on-row BUF 缓冲一次供本行所有 PE 共享），MRN 沿单一 A 行轨迹做行顺序归约——选择性路由直接消除了冗余 B 传输与深度 psum 归并。CSC 对应 InP 的按列取数，bitmask 用于低开销索引对齐。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
标准库实现：cuSPARSE、MKL（mkl_sparse_?_mm）、SuiteSparse 矩阵以 CSR/CSC 分发；GPU 稀疏内核常结合 CSR + 行级平衡或转 COO/bitmap 变体。Harmonia 中的使用方式：格式选择是 tile 粒度决策（与数据流选择联动），全局 SRAM 存 tile 与元数据（含逐 tile nnz，供 pre-execution profiling）；DN 的可编程路由与 on-row 索引匹配在 tile 边界重配置即可适配不同格式，无需改 PE 数据通路。论文未明确说明对格式切换的开销建模细节。评估矩阵来自 SuiteSparse Matrix Collection（bcsstk10.mtx、email.mtx、orani678、rajat19 等，均为标准稀疏矩阵格式分发）。

Ultra-CSR 补充视角（ParetoES，ISCA'26）：Ultra-CSR 是 AccelES（HPCA 2025）提出的 FPGA 友好 CSR 变体——用位掩码（bit-mask）压缩指针/行索引开销，最小化元数据、最大化每 512-bit HBM packet 的非零承载数；ParetoES 沿用它做检索矩阵编码：配合 INT6 量化后每 512-bit 传输 30 个非零（FP32 下约 11 个），带宽效率提升 6×；并新增 Random-CSR（AccelES 提出，动态逐向量访问变体）的簇探测用法。ParetoES 的流水用法：质心与簇子矩阵都预编码为 Ultra-CSR，质心放 HBM 通道头部（每 packet 30 非零、1 packet/cycle 流入 x-decoder）；x-decoder 用位宽 popcount 单周期解析行索引，产出 (x,y,val) 元组直接进乘法器；选中簇按簇局部有序索引组织（簇感知数据布局），随机访问被限制在活跃簇块内、呈流式 burst 访问——这是"全局不规则 → 有界流式"访存重塑的关键一环。

涉及论文标题：
- Harmonia: A Unified Hierarchical Scheduling Framework for Sparse Matrix Multiplication
- ParetoES Hardware-Accelerated Sparse Embedding Similarity via Pareto-Optimal Pruning
