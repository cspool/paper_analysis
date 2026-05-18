## BBC (Bitmap-Bitmap-CSR)

术语是什么？

BBC (Bitmap-Bitmap-CSR) 是 Uni-STC 论文提出的一种统一稀疏矩阵存储格式，通过 CSR 组织 4×4 sparse tile、用两级 bitmap 描述 tile 内非零位置，使 SpMV、SpMSpV、SpMM、SpGEMM 四类稀疏 kernel 共享同一数据结构而无需在线格式转换。BBC 的两级 bitmap 设计为：top-level bitmap 描述 4×4 tile 内 M 维和 N 维是否有非零，bottom-level bitmap 描述每个 4×4 tile 内部具体哪 16 个位置非零。两级 bitmap 使得 TMS 和 DPG 硬件能直接解析生成 task，无需复杂硬件 decoder。

从算法pipeline角度拆解术语：

BBC 格式的数据组织：
```
// top-level CSR 结构
row_ptr[N/4+1]:   // 每 4 行为一个 tile row，记录每个 tile row 的起始位置
col_idx[tile_nnz]: // 非零 tile 的列索引
// 每个非零 tile 内部:
top_bitmap:        // 16-bit，标记 tile 内 4x4 的 M 维和 N 维非零情况
bottom_bitmap[K]:  // 对于 GM＝A×B，沿 K 维每个 tile 配一个 bottom bitmap
val_ptr_lv2:       // 指向 tile 内实际非零数值存储位置
values[]:          // 按 tile 组织存储非零浮点数值
```

BBC 对不同 kernel 的表达：
- **SpMV/SpMSpV**：B 侧为 dense vector x 或 sparse vector x，BBC 仅描述 A 矩阵的稀疏结构，x 作为 operand 直接传入
- **SpMM**：A 稀疏（BBC 描述），B 为 dense matrix（如 64 列），每个 A tile 与 B 的一个 dense tile block 相乘
- **SpGEMM**：A 和 B 均用 BBC 格式描述，C 也用 BBC 作为输出格式，TMS 在 K 维按 bloom filter 判断是否存在非零乘积 tile

相比传统 CSR/CSC 格式的差异：BBC 的 tile-aligned 设计天然匹配 tensor core 的 4×4×4 计算粒度，bitmap 编码比 2:4 structured sparsity 更灵活（不要求固定 50% 稀疏率），且避免了 DS-STC 和 RM-STC 依赖的硬件 decoder（其面积开销大且难以支持多种 kernel）。

术语一般如何实现？如何使用？

BBC 需一次离线构建：输入为 CSR/CSC 格式稀疏矩阵 → 分割为 4×4 tile → 统计每 tile 内非零分布 → 生成 top-level bitmap 和 bottom-level bitmap → 重排非零值数组。构建在 64-core AMD EPYC 7702 CPU <1000ms、NVIDIA A100 GPU <100ms。构建后可用于迭代应用（GNN training、linear solver）一次性摊销 overhead。运行时 warp 通过 stc.load 将 BBC metadata 和 values 直接装载到 Uni-STC 的 Matrix A/B Buffer，无需软件进行格式解码。

涉及论文标题：
- Uni-STC: Unified Sparse Tensor Core

