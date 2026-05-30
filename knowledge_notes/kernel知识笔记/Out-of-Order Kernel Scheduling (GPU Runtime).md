## Out-of-Order Kernel Scheduling (GPU Runtime)

术语是什么？
Out-of-Order Kernel Scheduling（GPU 运行时乱序 kernel 调度）是 ACS 提出的核心机制，在 GPU 运行时对顺序发射的 kernel 进行乱序并发调度。与 CPU 乱序指令调度类似，ACS 在固定大小的调度窗口内对已发射但尚未执行的 kernel 进行运行时依赖检查——若一个 kernel 不依赖任何未完成的 kernel，则立即将其发射执行，而非等待排在它前面的所有 kernel 完成。这种机制使原本在单 CUDA stream 中串行的小 kernel 能够并发执行，从而提高 GPU 利用率。与 CPU 乱序执行的关键区别在于：GPU 的瓶颈不是依赖检查延迟（依赖检查仅需 410ns~1640ns），而是 kernel launch/synchronization 的延迟（5-20μs），因此 ACS 通过硬件加速消除后者。

从kernel调度角度拆解术语：
ACS 运行时乱序 kernel 调度的伪代码逻辑：
```
// ACS 运行时调度窗口 (大小 N=32)
SchedulingWindow SW[N];  // 每个slot: kernel_id, upstream_list[N-1], status
InputFIFO input_queue;   // 用户发射的kernel及RW-segments metadata

// === Window Module (CPU线程) ===
while (not_stop):
    // 1. 从输入队列取kernel移入调度窗口
    if SW.has_vacancy() and input_queue.not_empty():
        kernel = input_queue.pop()
        
        // 依赖检查: 比较新kernel的write_segments与窗口中所有kernel的read+write_segments
        upstream = []
        for each k in SW:
            for seg1 in kernel.write_segments:
                for seg2 in (k.read_segments ∪ k.write_segments):
                    if overlap(seg1, seg2):  // 地址范围重叠检测
                        upstream.add(k.id)
        SW.insert(kernel, upstream)  // status = PENDING | READY(if upstream empty)
    
    // 2. 当kernel完成时，更新窗口中所有kernel的upstream列表
    on_kernel_complete(completed_id):
        for each k in SW:
            k.upstream.remove(completed_id)
            if k.upstream.empty():
                k.status = READY         // 所有依赖已满足

// === Scheduler Module (多个CPU线程, 每个绑定1个CUDA stream) ===
while (not_stop):
    if SW.has_ready():
        kernel = SW.pop_ready()
        cudaLaunchKernel(kernel, stream_id)      // 发射到独立stream
        cudaStreamSynchronize(stream_id)          // 等待完成 (ACS-SW)
        SW.notify_complete(kernel.id)             // 通知window module

// ACS-HW变体: 调度窗口在GPU硬件中
// GPU Command Processor中的硬件模块:
// - Scheduling Window SRAM (1KB for N=32)
// - Upstream Load Module: 修正CPU端可能stale的upstream list
// - 硬件自动dispatch ready kernel到kernel dispatch unit
```

在 Deep RL Brax Ant 环境下（RTX 3060, 28 SM），该机制将 GPU 达到的 occupancy 从约 34%（单 stream 串行）提升至接近满载，加速比最高 2.19×（ACS-HW）和 1.87×（ACS-SW）。

术语一般如何实现？如何使用？
ACS-SW 以用户态运行时系统实现：应用通过 `ACS_wrapper` 为每个 kernel 标注 `__read_segments__` 和 `__write_segments__`（起始虚拟地址+大小列表），以及 `get_addresses()` 函数在 kernel launch 前解析虚拟地址。Window module 和 scheduler module 各为独立 CPU 线程，通过共享内存中的调度窗口数据结构通信。ACS-HW 将调度窗口实现为 GPU 命令处理器中的 1KB SRAM（N=32 时），每个 slot 含 8-bit kernel ID + (N-1) 个 8-bit upstream kernel ID（全关联存储）+ 2-bit 状态。论文声明将开源 ACS-SW，当前未找到公开代码仓库。依赖检查算法为 O(segments²) 遍历检查地址范围重叠。

涉及论文标题：
- ACS Concurrent Kernel Execution on Irregular, Input-Dependent Computational Graphs
