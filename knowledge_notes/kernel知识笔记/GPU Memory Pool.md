## GPU Memory Pool

术语是什么？
GPU Memory Pool（GPU 内存池）是一种预分配 GPU 内存的管理机制，用于减少运行时频繁 cudaMalloc/cudaFree 系统调用的开销。在 HuntKTm 中，task scheduler 在 task 首次内存分配前根据预测的 memory footprint 预分配一个 memory pool。后续所有 allocation 请求若可从 pool 满足，则直接从 pool 返回预分配内存（无需 OS 级别的 GPU memory allocation 系统调用）；deallocation 请求不真正释放内存，而是将内存保留在 pool 中供后续 reuse。pool 中的内存仅在 application 退出时完全释放。

从 kernel 调度角度拆解术语：
GPU memory pool 在 HuntKTm 任务执行中的运转流程：

```
Task 生命周期中的 Memory Pool:

1. Task Scheduler 初始化阶段（runtime，dispatch 后）:
   predicted_footprint ← lazy engine 汇总的 memory requirement
   cudaDeviceGetDefaultMemPool(&pool)  // 获取 CUDA 默认 memory pool
   cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, predicted_footprint)
   // 设置 release threshold：memory usage 低于此 threshold 时不释放
   // 确保 pool 中的内存不会被 CUDA runtime 自动回收

2. Task 执行阶段（lazy engine 顺序执行 deferred operations）:
   cudaMallocAsync(&ptr, size, stream):
     if pool.free_memory >= size:
       直接从 pool 返回预分配内存 → 避免系统调用
     else:
       触发真正的 GPU memory allocation

   cudaFreeAsync(ptr, stream):
     将内存归还 pool → 不真正释放
     // 即使 task 内部多次 alloc/free，pool 保持足够内存

3. Task 退出:
   pool 内所有内存被释放
```

对 kernel 调度的关键影响：
- 减少运行时 memory alloc/free 开销：HuntKTm vs HuntKT（无 memory pool）在单 task 执行中提升 speedup（M1: 3.27×, M2: 3.17× vs Serial）
- 与 memory manager（编译期 liveness analysis）协同：memory manager 减少 peak memory → pool 的 predicted_footprint 更小 → 更多 task 可同时运行
- 消除频繁系统调用导致的 kernel launch 延迟

术语一般如何实现？如何使用？
通过 CUDA 12.0+ 的 Stream-Ordered Memory Allocator API：`cudaMallocAsync` 和 `cudaFreeAsync` 使用设备默认 memory pool。`cudaDeviceGetDefaultMemPool` 获取 pool handle，`cudaMemPoolSetAttribute` 设置 `cudaMemPoolAttrReleaseThreshold` 控制内存释放阈值。HuntKTm 的 lazy engine 将所有 memory allocation 转化为 cudaMallocAsync，并在 task 退出时通过 pool 的 release threshold 防止内存过早归还 OS。此机制与 PyTorch 的 caching allocator 类似，但 HuntKTm 将其集成到通用 GPU 程序的编译-运行时 pipeline 中。

涉及论文标题：
- HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs
