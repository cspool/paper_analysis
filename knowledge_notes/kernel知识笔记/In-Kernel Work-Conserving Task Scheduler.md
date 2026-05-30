## In-Kernel Work-Conserving Task Scheduler

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FlashMoE megakernel 的核心管理组件，由 OS block 中 1 个 warp（32 threads）实现。属性：(1) work-conserving——有 task 就分配；(2) multithreaded——32 threads 并行 sweep doorbells；(3) in-kernel——运行 GPU 上，无需 CPU 干预。Doorbell 是 monotonic counter（非 binary flag），避免丢失 concurrent 更新。taskBound 由 Subscriber atomic increment 动态增加。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
while scheduled < taskBound:
    // 并行 sweep doorbells
    do in parallel: local_counts[tid] = count_pending(doorbells)
    
    // Warp inclusive sum (~5 cycles via __shfl_up_sync)
    WarpInclusiveSum(local_counts, &offset, &total)
    
    // 分发 task 给空闲 processor
    while total > 0:
        repopulate ready_queue
        do in parallel: signal processors about task indices
    
    // 动态更新
    taskBound = WarpBroadcast(AtomicLoad(taskBound))
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Warp shuffle `__shfl_up_sync` (~5 cycles) vs shared memory atomic (更慢)
- 传统 GPU kernel 按预分配数据范围处理；FlashMoE 按 runtime readiness 动态分配
- 类似于 OS 进程调度器的设计哲学，但运行在 GPU warp 上

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel
