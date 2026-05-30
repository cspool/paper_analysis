## CUDA CUTLASS

术语是什么？
CUTLASS（CUDA Templates for Linear Algebra Subroutines and Solvers）是 NVIDIA 开发的开源 header-only CUDA C++ 模板库，用于实现高性能矩阵乘法（GeMM）及相关线性代数计算。它将 GeMM 计算分解为层次化的 tile-based 计算结构，从线程级到 warp 级（32 线程）到 thread block 级（CTA）到 kernel 级再到 device 级。CUTLASS 3.x 引入 CuTe 库用于定义和操作多维线程与数据布局，支持 Tensor Core（Volta 到 Blackwell）、混合精度（FP64/FP32/TF32/FP16/BF16/FP8/INT8/INT4 等）以及 warp specialization 等高级特性。本论文使用 CUTLASS 3.1 的 GeMM 和 Conv2D kernel 作为实验基础。

从kernel调度角度拆解术语：
CUTLASS GeMM kernel 的 tile-based 执行流程：
```
// CUTLASS中分块GeMM的典型执行（简化版）
// 输入: A[M×K], B[K×N] → 输出: C[M×N]
// TileA: M_tile×K_tile, TileB: K_tile×N_tile
// Thread Block Grid: (M/M_tile, N/N_tile)

__global__ void cutlass_gemm(A, B, C, M, N, K) {
    // 1. 从global memory加载A tile到shared memory
    //    threadIdx映射到A tile的不同元素，协作加载
    LoadTileToShMem(Ash, A, blockIdx.y * M_tile, K_start, M_tile, K_tile);
    // 2. 加载B tile到shared memory  
    LoadTileToShMem(Bsh, B, K_start, blockIdx.x * N_tile, K_tile, N_tile);
    __syncthreads();
    // 3. 从shared memory加载到寄存器
    // 4. 使用Tensor Core (wmma或mma指令) 执行矩阵乘加
    // 5. 沿K维度迭代，重复1-4
    for k_tile in 0..K step K_tile:
        // pipeline: 在计算当前tile的同时异步加载下一tile
    // 6. 写回C tile到global memory
    StoreTileToGlobal(C, Csh, blockIdx.y * M_tile, blockIdx.x * N_tile);
}
```
cuSync 在 CUTLASS kernel 的 tile 加载和计算之间插入 wait/post 同步点，仅需修改约 25 行代码（0.5%）。

术语一般如何实现？如何使用？
开源地址：github.com/NVIDIA/cutlass。使用方式：include 头文件，定义矩阵类型和 GEMM 配置（ElementA/B/C、Layout、ThreadblockShape、WarpShape、InstructionShape），调用 `cutlass::gemm::device::Gemm` 或直接使用 kernel。CUTLASS profiler 可用于性能调优。cuSync 将 `CuStage` 对象传入 CUTLASS kernel，kernel 内部通过 `stage.tile()` 获取 tile 索引、`stage.wait()`/`stage.post()` 执行同步。CUTLASS 也支持 Implicit GEMM Convolution（将 Conv2D 映射为 GEMM）。

涉及论文标题：
- A Framework for Fine-Grained Synchronization of Dependent GPU Kernels
- FlashAttention-2 Faster Attention with Better Parallelism and Work Partitioning

FlashAttention-2基于CUTLASS 3.x构建其forward和backward attention kernel，利用CUTLASS的TileIterator和Collective抽象实现tiled GEMM（QK^T和PV），结合手写的online softmax（CUDA core: rowmax + MUFU.EX2 + rowsum + rescale）和warp-level work partitioning（split Q across warps而非split-K）。CUTLASS 3.x的CuTe库提供的多维线程/数据布局抽象使得FlashAttention-2能够在不同head dimensions（64/128）和block sizes（{64,128}×{64,128}）间手动tune并生成高效kernel变体。

FlashMoE 在 persistent kernel 内使用 CUTLASS 的 device-side API 实现 in-kernel GEMM——通过 fused `__device__` function 将 GEMM0 (A×W1→GELU→+bias) 和 GEMM1 (C1×W2→identity epilogue) 合并为单一 device-side 调用。这不同于传统的 host-launched cuBLAS kernel：FlashMoE 的 Processor actor 在 persistent kernel loop 内直接调用 CUTLASS device-side GEMM，无需退出 kernel 或 CPU 参与。CUTLASS tile-based MMA 与 FlashMoE 的 (128,64) tile 维度对齐——tile 的 M=128 和 N=64 直接映射为 CUTLASS threadblock tile shape，Tensor Core MMA 指令在 processor thread block 内执行。

涉及论文标题：
- A Framework for Fine-Grained Synchronization of Dependent GPU Kernels
- FlashAttention-2 Faster Attention with Better Parallelism and Work Partitioning
- FlashMoE: Fast Distributed MoE in a Single Kernel
- Efficient and Adaptable Overlapping for Computation and Communication via Signaling and Reordering

FlashOverlap 基于 CUTLASS 模板 GEMM 实现，利用 CUTLASS EVT (Epilogue Visitor Tree) 在 GEMM epilogue 中插入 pre-communication reordering。Main loop 完整保留 CUTLASS profiler 最优配置不变。EVT 通过将 epilogue 的 write address 从线性地址改为间接寻址（`base + mapping_table[tile_idx] * tile_size`），实现 execution-order-aware 的 scattering 操作——开销仅 0.07-0.68% GEMM latency。CUTLASS 的 tile scheduler（含 block swizzling）保持不变，pre-communication reordering 在 epilogue 中解决 swizzling 导致的地址不连续问题。
