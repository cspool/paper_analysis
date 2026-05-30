## Hybrid Scheduling (Kernel-Level + Task-Level GPU Concurrency)

术语是什么？
Hybrid Scheduling（混合调度）是 HuntKTm 提出的 GPU 并发执行策略，将 kernel-level 并发（单个应用内的多 kernel 通过多 CUDA stream 并发）和 task-level 并发（多个独立应用通过 GPU space-sharing 并发）协同组合。核心洞察：仅靠 kernel-level 并发（如 CKE）或 task-level 并发（如 MPS）都无法充分饱和 GPU 硬件资源——即使 GPU utilization 显示 100%，SM occupancy 可能仍低于 10%（memory-intensive kernel 场景）。Hybrid scheduling 通过两个层面的并发叠加来更充分地利用硬件资源。

从系统架构角度拆解术语：
Hybrid scheduling 在系统架构中的执行流程：

```
系统架构中的 Hybrid Scheduling:

编译期:
  每个 Task（GPU 程序）→ Stream Scheduler（kernel-level 并发）
  → 输出 multi-stream 可执行程序（多 hardware queue 的 kernel 执行计划）
  → Resource Analyzer 提取资源需求

运行时:
  Task Arrival → Task Dispatcher 的 pending queue
  ┌─ Task Dispatcher 循环 ──────────────────────────────────────┐
  │ 1. Dequeue task from pending queue                           │
  │ 2. 遍历 available GPUs:                                      │
  │    - 检查 free memory ≥ task.memory_requirement              │
  │    - 检查 available hardware queues ≥ task.stream_count      │
  │    - 评估三维 SM 可用量: threads, registers, shared memory   │
  │    - 选择可用 SM 最多的 GPU（负载均衡）                       │
  │ 3. 若无 GPU 满足条件 → task 挂起到 pending queue             │
  │ 4. 若有 GPU 满足条件 → dispatch task → GPU_i                 │
  │ 5. GPU_i 上: task 的多 stream kernel 通过 MPS 与其他 task    │
  │    的 kernel 在同一 GPU 上 space-sharing                     │
  └─────────────────────────────────────────────────────────────┘
```

关键设计：
- kernel-level 并发（Stream Scheduler）：编译期自动将单 task 的 kernel 分配到多个 hardware queue
- task-level 并发（Task Scheduler）：运行期根据资源需求和可用性将 task dispatch 到 GPU
- Memory Manager：编译期降低 task 的 memory footprint，使更多 task 可同时运行
- MPS 启用：支持跨进程 kernel 在同一 GPU 上 space-sharing
- 限制 per-GPU hardware queue 数 ≤ 32（匹配 CUDA runtime 的最大连接数）

与仅 kernel-level（Taskflow）或仅 task-level（CASE）的区别：
- CASE: 仅 task-level 并发 + 静态资源分析，无 kernel 内部并发优化
- Taskflow: 仅 kernel-level 并发（需手动声明依赖），无 task-level 调度
- HuntKTm: 混合两者 + 内存管理

术语一般如何实现？如何使用？
HuntKTm 在 LLVM 编译框架中实现 stream scheduler（kernel-level），在运行时实现 task scheduler（task-level）。系统吞吐量较 CASE 提升 33.2%（4×A100），较 Taskflow 提升 13.8%。硬件资源利用率方面：FP32 utilization 提升 3.54×，memory bandwidth 提升 2.83×，SM occupancy 提升 2.47×（vs SA baseline）。在 memory-constrained 场景（20GB/GPU），HuntKTm 较 CASE 提升 61.8%。

涉及论文标题：
- HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs

---
