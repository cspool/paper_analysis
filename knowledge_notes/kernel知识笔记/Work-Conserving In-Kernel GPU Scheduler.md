## Work-Conserving In-Kernel GPU Scheduler

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Work-Conserving In-Kernel GPU Scheduler 是 FlashMoE 在 persistent kernel 内实现的多线程动态任务调度器 (Algorithm 3)。"Work-conserving" 源自 OS 调度理论——只要有 task 就绪且有 Processor 空闲，立即分配，不延迟等 batch。由 1 个 warp (32 threads) 在 OS block 内运行，通过并行 sweep doorbells + warp-level inclusive sum 实现高吞吐调度。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Algorithm: while (scheduled < task_bound) { do_in_parallel { sweep doorbells → tqState → lt }; qS, task_tally = WarpInclusiveSum(lt); while (task_tally > 0) { repopulate ready_queue; do_in_parallel { signal processor ready_queue[qS[t]] }; } task_bound = AtomicLoad(global_task_bound); }。InterruptSubscribers(); InterruptProcessors()。Work-conserving: readiness-based 非静态顺序，no batching，continuous polling，dynamic task_bound。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

CPU-based 调度是 non-work-conserving (须等 kernel 返回才 launch 下一个)；FlashMoE in-kernel Scheduler 有全局 visibility (所有 Processor 和 queue 在同一 kernel)，可 work-conserving。

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel
