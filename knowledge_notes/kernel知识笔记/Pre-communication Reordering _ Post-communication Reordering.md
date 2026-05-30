## Pre-communication Reordering / Post-communication Reordering

术语是什么？

Pre-communication reordering 和 Post-communication reordering 是 FlashOverlap 中用于解决 tile 数据地址非连续问题的一对数据重排操作。由于 GEMM tile 的完成顺序（受 block swizzling 影响）与内存地址顺序不一致，且单个 tile 内数据因为 stride 天然非连续，直接对非连续地址调用 NCCL API 无法工作（NCCL 要求发送和接收 buffer 均为连续地址）。Pre-communication reordering 在 GEMM 完成后将数据按执行顺序重排到连续 buffer；post-communication reordering 在通信完成后将数据恢复为原始顺序。

从kernel调度角度拆解术语：

三种通信原语对应的 reordering pattern：

```
// (1) AllReduce: tile-level reordering
// Pre: tiles reordered by wave execution order → contiguous buffer
// Post: mapping_table restores original order during RMSNorm
for each tile t:
    load(C_comm_buffer + mapping_table[t])  // indirect read via mapping
    rmsnorm_result = rmsnorm(tile_data)
    store(C_final + t * tile_size, rmsnorm_result)  // original index

// (2) ReduceScatter: subtile-level reordering  
// Each tile split by row into GPU_num subtiles; k-th subtile → k-th GPU
// Post: AllGather + local row exchange (cyclic permutation) restores order

// (3) All-to-All: subtoken-level reordering
// Each tile split by token (row); dedicated memory pool per destination GPU
// Sub-tokens in each pool reordered by execution order
```
**Annotations**: Pre-communication reordering 开销：tile-level 0.07%/0.35% (A800/RTX4090 GEMM latency)；subtile/subtoken-level 0.67%/0.68%。Post-communication reordering 开销：tile-level 7.46%/8.80% (A800/RTX4090 RMSNorm latency)。Mapping table 大小约为 M×N 的 1.6-12.5%。Post-reordering 虽改变内存访问模式，但因保持局部连续性，memory efficiency 基本保留。

术语一般如何实现？如何使用？

Pre-communication reordering 基于 CUTLASS EVT (Epilogue Visitor Tree)，以 scattering 操作插入 GEMM epilogue——EVT 允许自定义 element-wise 操作的数据访问模式，通过将 write address 改为间接寻址实现。Post-communication reordering 融合到后续 element-wise kernel（如 RMSNorm）中，通过 mapping table 的间接寻址将 read address 从 reordered 映射回 original。开源代码：github.com/infinigence/FlashOverlap。

涉及论文标题：
- Efficient and Adaptable Overlapping for Computation and Communication via Signaling and Reordering
