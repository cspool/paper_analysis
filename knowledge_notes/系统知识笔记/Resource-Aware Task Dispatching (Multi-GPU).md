## Resource-Aware Task Dispatching (Multi-GPU)

术语是什么？
Resource-Aware Task Dispatching 是 HuntKTm 的 task scheduler 中的运行时调度算法，将 task 动态分配到 multi-GPU 系统中的合适设备。与简单的 round-robin 或 first-fit 分配不同，它同时考虑三个维度的 GPU 资源：memory（free memory 是否满足 task 需求）、compute（从 threads/registers/shared memory 三个子维度评估 SM 可用量）、和 hardware queue（可用的 concurrent stream 数量）。

从系统架构角度拆解术语：
Resource-aware task dispatching 的调度算法（Algorithm 1）：

```
Input: 可用 GPU 列表 G, pending task queue Q, 待调度 task T (及其资源需求)
Output: 目标 GPU g_target

Function TaskSchedule(G, Q, T):
  g_best ← None, score_best ← 0
  for each g in G:
    if g.free_memory < T.memory_requirement or
       g.available_hw_queues < T.stream_count:
      continue  // 内存或 hardware queue 不足，跳过

    // 三维 SM 可用量评估
    score_threads ← (g.available_threads + T.thread_requirement) / g.total_threads
    score_regs   ← (g.available_regs + T.reg_requirement) / g.total_regs
    score_shmem  ← (g.available_shmem + T.shmem_requirement) / g.total_shmem
    score ← g.total_sms - min(score_threads, score_regs, score_shmem)

    if score > score_best:
      g_best ← g, score_best ← score

  if g_best is not None:
    g.free_memory -= T.memory_requirement  // 预留资源
    return g_best
  else:
    T.suspend()  // 挂起到 pending queue，等待资源释放时重试
    return None
```

核心设计原则：
- 三维评估（threads/registers/shared memory）：防止单一资源维度成为瓶颈（如 GEMM kernel 可能受限于 registers 而非 threads）
- 负载均衡：优先选择可用 SM 最多的 GPU
- 非抢占式：一旦 task 分配到 GPU，不中途迁移
- pending queue 重试：无满足条件的 GPU 时挂起 task，资源释放时重新调度

术语一般如何实现？如何使用？
Task scheduler 在运行时维护每个 GPU 的资源状态（free memory、available SMs per resource type）。Lazy engine 在 task 到达 cudaTaskSchedule 点时汇总资源需求并通过共享内存发送给 task dispatcher。Task dispatcher 调用 Algorithm 1 选择目标 GPU，返回 GPU ID 给 lazy engine。Memory pool 预分配（减少运行时 alloc/free 开销）和 NVIDIA persistence mode（减少 GPU 初始化开销）被启用。与 CASE 的关键区别：HuntKTm 的调度考虑混合 kernel-level + task-level 并发带来的 multi-stream 资源需求（每个 task 可能占用多个 hardware queue）。

涉及论文标题：
- HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs
