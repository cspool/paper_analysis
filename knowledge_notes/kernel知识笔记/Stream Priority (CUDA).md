## Stream Priority (CUDA)

术语是什么？
CUDA Stream Priority 是 CUDA 提供的流优先级机制。通过 cudaDeviceGetStreamPriorityRange 查询设备支持的优先级范围，通过 cudaStreamCreateWithPriority 创建带优先级的流。在 TX2 上，仅支持两个优先级值：-1（priority-high）和 0（priority-low）。未指定优先级的 stream（priority-none）被视为 priority-low。

从kernel调度角度拆解术语：
论文通过实验（Fig. 6-8）推导出 TX2 上 stream priority 的调度规则（A1-A2）：
- Rule A1: kernel 只能入队与其 stream 优先级匹配的 EE queue（priority-high queue 或 priority-low queue）
- Rule A2: EE queue 头部 kernel 的 block 可被分配，仅当所有更高优先级的 EE queue 为空

这意味着：
1. Priority-high stream 的 kernel 可以"抢占"priority-low stream 的 kernel——不是真正的中断正在执行的 block，而是当有新 block 可被分配时，优先从 priority-high EE queue 选 block。
2. Priority-high 可能导致 priority-low 饥饿（Fig. 6: K1 在 priority-low stream，8 blocks 只执行了 4 个就被 K2(priority-high) 抢占，剩余 4 个需等 K2 和 K3 全部完成后才能执行）。
3. 资源阻塞不影响优先级：若 priority-high kernel 因资源不足无法 dispatch，priority-low kernel 即使资源满足也不能"插队"（Fig. 8）。

伪代码——优先级调度逻辑：
```
function select_block_for_assignment(SM):
    if priority_high_EE_queue is not empty:   // Rule A2
        head_kernel = priority_high_EE_queue.head  // Rule A1
    else if priority_low_EE_queue is not empty:
        head_kernel = priority_low_EE_queue.head
    // Rule X1: only head kernel's blocks are eligible
    if SM has sufficient resources for head_kernel.block:
        assign block of head_kernel to SM
        // Note: blocks from non-head kernels are NOT eligible
```

术语一般如何实现？如何使用？
CUDA Stream Priority 适用于需要区分延迟敏感（priority-high）和吞吐优先（priority-low）工作负载的场景。注意：(1) TX2/Jetson 平台仅支持 2 个优先级级别；(2) 多 process 场景下 stream priority 无效（论文附录 A 发现）；(3) 高优先级可能导致低优先级饥饿，需要谨慎设计以避免无限期延迟。

涉及论文标题：
- GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed

---
