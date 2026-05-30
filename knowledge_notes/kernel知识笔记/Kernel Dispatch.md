## Kernel Dispatch

术语是什么？
Kernel Dispatch 是 GPU scheduler 将 CUDA kernel 的 thread block 分配到 SM 上执行的过程。在本文的术语中，一个 kernel 被 dispatched 意味着至少一个 block 被分配到 SM；fully dispatched 意味着所有 block 已被分配。Block 是 dispatch 的基本单位，dispatch 受限于 SM 的资源约束（线程、shared memory、寄存器）和 EE queue 的状态。

从kernel调度角度拆解术语：
Dispatch 的具体流程（整合论文 Rules G2-G3, X1, R1-R3）：

```
函数: dispatch_kernel_from_EE_queue()
输入: EE_queue (FIFO)
输出: blocks assigned to SMs

while EE_queue not empty:
    head_kernel = EE_queue.head
    for each SM in GPU:
        if head_kernel has unassigned blocks:
            block = head_kernel.next_unassigned_block
            if SM.meets_resource_constraints(block):
                // Rules R1-R3
                SM.assign_block(block)
                mark block as assigned
            else:
                continue to next SM  // resource blocking
        else:
            break  // all blocks assigned
    if head_kernel.fully_dispatched:
        EE_queue.dequeue(head_kernel)  // Rule G3
    else:
        break  // head kernel blocked by resources => NO preemption of head (Rule X1)
               // Later kernels in EE queue cannot be dispatched even if resources exist
```

Dispatch 的关键特性：
1. Head-of-line blocking: EE queue 头部 kernel 因资源不足无法 fully dispatched 时，后续 kernel 即使资源满足也不能 dispatch（Rule X1: 非抢占）。
2. Resource blocking: 一个 block 只能分配到一个 SM（不能跨 SM 拆分）。例如，若 kernel 需要 1024 threads/block 且 SM0 有 512 空闲线程、SM1 有 512 空闲线程，该 block 无法分配（因为任何一个 SM 都不满足 1024 线程的要求）。
3. Stream FIFO 与 dispatch 的交互：kernel 在 stream queue 中阻塞（等待前面 kernel 完成）期间不会出现在 EE queue 中，因此不影响其他 stream 的 dispatch。

术语一般如何实现？如何使用？
Kernel dispatch 由 GPU 硬件和 driver 自动完成，对 CUDA 程序员透明。但理解 dispatch 行为对性能调优和实时系统分析至关重要：通过控制 block 的资源需求（thread 数、shared memory、register 数），可以预判 dispatch 的并行度和阻塞情况。CUDA Occupancy API 可以帮助开发者确定最优配置以最大化 SM 利用率。

涉及论文标题：
- GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed

---
