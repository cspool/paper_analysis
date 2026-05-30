## Signaling-based Computation-Communication Overlapping

术语是什么？

Signaling-based computation-communication overlapping 是 FlashOverlap (EuroSys'26) 提出的一种新型计算-通信重叠机制。其核心思想是：GEMM kernel 在计算过程中通过 lightweight signal（信号）通知通信端已完成的 tile 数据，触发 NCCL 通信，同时 GEMM 继续计算剩余部分，实现 interference-free 的重叠。与此前的 decomposition-based 方法（将 GEMM 切分为子 tensor 后交叠）和 fusion-based 方法（将通信原语融合到 GEMM kernel 内部）不同，signaling-based 方法同时满足三个设计目标：(1) tile-wise overlapping——以 tile 为粒度最大化重叠机会；(2) interference-free computation——不修改 GEMM 的 main loop 和 tiling 策略，保持原始计算性能；(3) communication agnosticism——直接调用标准 NCCL API，无需为不同通信原语（AllReduce、ReduceScatter、All-to-All）重复实现。

从kernel调度角度拆解术语：

FlashOverlap 的 signaling 机制在两 CUDA stream 上的执行流程：

```
// Stream A: GEMM computation
__global__ void gemm_with_signaling(A, B, C_reordered, counting_table, mapping_table) {
    // Main loop: standard CUTLASS GEMM (unchanged)
    for (k = 0; k < K; k += K_TILE) {
        // Load A_tile, B_tile from global memory
        // Compute MMA on Tensor Core
        // Accumulate in registers
    }
    
    // Epilogue: pre-communication reordering + signaling
    tile_idx = blockIdx.x;  // tile completion order ≠ memory order (due to swizzling)
    reordered_idx = mapping_table[tile_idx];  // execution-order-aware remapping
    group_id = tile_idx / tiles_per_group;    // which wave group this tile belongs to
    
    // Scatter tile data to contiguous communication buffer
    store_tile_reordered(C_reordered, accum, reordered_idx);
    
    // Signal: atomically increment counting table for this group
    atomicAdd(&counting_table[group_id], 1);
}

// Stream B: signaling checker + communication
__global__ void signaling_and_communicate(counting_table, C_reordered, group_sizes, P) {
    for (j = 0; j < P; j++) {
        // Spin-wait until all tiles in group G_j are finished
        while (__ldg(&counting_table[j]) < group_sizes[j]) {
            __nanosleep(100);  // backoff to reduce SM contention
        }
        // Trigger NCCL communication for group G_j
        ncclAllReduce(C_reordered + group_offset[j], 
                      C_recv + group_offset[j],
                      group_data_size[j], ncclFloat16, ncclSum, 
                      comm, stream_B);
    }
}
```
**Annotations**: Stream A 中 main loop 完全不变——GEMM 计算不受干扰（interference-free）。Epilogue 中 pre-communication reordering 将 tile 按执行顺序散射到连续地址的通信 buffer。AtomicAdd 在 counting table 中递增对应 group 的计数——开销约 0.07% GEMM latency（A800 tile-level）。Stream B 中 signaling kernel 周期性 spin-wait 查询 counting table——当 group 计数达到目标时立即调用 NCCL API。两 stream 通过 CUDA 硬件调度器并发执行——while Stream B 执行 G_1 的 AllReduce，Stream A 继续计算 G_2 的 GEMM。

术语一般如何实现？如何使用？

基于 CUTLASS 模板 GEMM 实现。Signaling 机制作为一个独立的 GPU kernel 在单独 CUDA stream 中运行，周期性查询 counting table（位于 GPU global memory）。Counting table 大小 = P（wave group 数），初始化为 0。已开源：github.com/infinigence/FlashOverlap，支持 CUDA 12.1+、CUTLASS 3.6.0-3.9.0、NCCL 2.18.3+，GPU 架构 sm80/sm86/sm89。

涉及论文标题：
- Efficient and Adaptable Overlapping for Computation and Communication via Signaling and Reordering
