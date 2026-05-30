## Global Memory Semaphore Synchronization (post/wait)

术语是什么？
Global Memory Semaphore Synchronization 是 cuSync 的核心跨 kernel 同步原语。在 GPU global memory（DRAM/HBM）中分配整数 semaphore 数组，producer thread block 通过 `atomicAdd` 递增 semaphore（post），consumer thread block 通过 busy-wait（`while(*sem != expected)`）等待 semaphore（wait）。关键设计：(1) 仅 thread block 的第一个线程执行 busy-wait，其余线程被 `__syncthreads` 阻塞；(2) post 操作前调用 `__threadfence_system()` 确保 global memory 写入对其他 kernel 可见；(3) policy 类决定 semaphore-to-tile 映射。与 CUDA `__syncthreads`（仅限同 block 内）、`cudaDeviceSynchronize`（全局 barrier）不同，这是唯一实现跨 kernel 但非全局的 tile 级同步的方法。

从kernel调度角度拆解术语：
```
// Global Memory Semaphore的post/wait实现
__device__ void post(int* sems, dim2 tile, dim2 grid, Policy p) {
    __syncthreads();                          // block内所有线程完成计算
    if (threadIdx == (0,0,0)) {               // 仅第一个线程操作semaphore
        __threadfence_system();               // 确保global memory写入跨kernel可见
        int idx = p.sem(tile, grid);          // policy决定semaphore索引
        atomicAdd(&sems[idx], 1);             // 原子递增
    }
}

__device__ void wait(int* sems, dim2 tile, dim2 grid, Policy p) {
    if (threadIdx == (0,0,0)) {
        int idx = p.sem(tile, grid);
        int expected = p.value(tile, grid);
        while (atomicLoad(&sems[idx]) != expected);  // busy-wait
    }
    __syncthreads();  // 等待线程完成同步后，所有线程继续
}

// 同步开销上界（论文V-D节）:
// 最坏场景: 2 kernel×1280 TB, 最小计算量(memcpy)
// → overhead 2-3% over StreamSync
```

术语一般如何实现？如何使用？
cuSync 通过 `cudaMalloc` 在 global memory 分配 semaphore 数组，类型为 `int*`。`__threadfence_system()` 是系统级内存屏障（比 `__threadfence()` 强），保证写入对 device 和 host 均可见，是跨 kernel/跨 stream 同步的关键。CUDA 不提供内置的 cross-kernel semaphore，硬件层面的替代包括 GLocks（基于 message passing）和 HQL（L1/L2 cache 队列锁），但 cuSync 选择纯软件方案以保证可移植性。

涉及论文标题：
- A Framework for Fine-Grained Synchronization of Dependent GPU Kernels
