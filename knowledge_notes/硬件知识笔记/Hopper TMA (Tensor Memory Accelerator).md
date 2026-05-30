## Hopper TMA (Tensor Memory Accelerator)

术语解释
Tensor Memory Accelerator (TMA) 是 NVIDIA Hopper (SM90) 架构引入的专用硬件数据搬运单元，负责在 global memory 和 shared memory 之间高效传输 tensor 数据，无需占用线程执行资源。TMA 是 Hopper 架构中实现异步计算流水线的关键硬件组件。

术语是什么？
TMA 是独立于 CUDA core 和 Tensor core 的硬件单元，通过 `cp.async.bulk` PTX 指令编程。关键特性：
- 单个线程发起 TMA 操作，硬件完成整个数据传输（1D-5D tensor 均支持），其余 127 个线程可继续计算
- 使用 **mbarrier**（异步 barrier）进行完成同步——线程在 barrier 上等待而非 `__syncthreads()`
- 支持 **multicast**：一次 global memory 读取可广播到同一 cluster 中多个 SM 的 shared memory
- 显著减少寄存器压力（无需手动地址计算）和指令发射开销

从硬件架构角度拆解术语：

在 Comet 的 Hopper GEMM thread block 中，TMA 的工作流程：
```
GEMM Thread Block (Hopper SM):
1. Producer Warp 发起 TMA 请求:
   cp.async.bulk shared_A, global_A[tile_A], descriptor_A
   // TMA 硬件独立执行: global DRAM → L2 → shared memory
   // Producer warp 线程此时空闲，可做其他工作

2. mbarrier 跟踪传输进度:
   mbarrier.arrive_expect_tx(expected_bytes)
   // 硬件跟踪实际到达的字节数

3. Consumer Warp 等待数据就绪:
   mbarrier.try_wait()
   // vs __syncthreads(): 不会阻塞所有线程

4. Consumer Warp 执行 Tensor Core MMA:
   // 使用 ready 的 shared memory 数据
   mma_sync(accumulator, shared_A, shared_B)

5. 软件流水线: producer 在 consumer 计算时发起下一 tile 的 TMA load
```

TMA 与 Comet 的关系：Comet 的 GEMM TB 使用标准 CUTLASS Hopper 实现，内部依赖 TMA + mbarrier 实现高效异步流水线。Comet 的关键设计——将通信 I/O 隔离到独立 TB——正是为了避免 fine-grained NVSHMEM I/O 打破 TMA 的异步流水线。如果通信 I/O 被垂直融合进 GEMM TB，NVSHMEM 的 remote read（数百 cycle 延迟）会阻塞 TMA 流水线，显著降低 GEMM 吞吐。

术语一般如何实现？如何使用？
- 硬件：Hopper (H100/H800/H200) 和 Blackwell (B100/B200) GPU
- 编程接口：`cuda::memcpy_async`（高层）、`cp.async.bulk` PTX（低层）、`cuTensorMapEncodeTiled`（host 端 descriptor 创建）
- 主要使用者：CUTLASS 3.x、cuBLAS、Triton (Gluon API)
- 性能优势：非带宽（峰值带宽与传统 load 相近），而是线程利用率——TMA 释放的线程可执行计算，实现更深的软件流水线
- 对齐要求：shared memory destination 128-byte 对齐，innermost coordinate 16-byte 对齐

涉及论文标题：
- Comet Fine-grained Computation-communication Overlapping for Mixture-of-Experts
