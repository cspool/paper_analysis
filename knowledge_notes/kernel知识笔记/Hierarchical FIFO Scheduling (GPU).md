## Hierarchical FIFO Scheduling (GPU)

术语是什么？
Hierarchical FIFO Scheduling（层次化 FIFO 调度）是本文发现并命名的 TX2 GPU scheduler 的调度策略。它是一种多级 FIFO 队列结构：多个 stream queue（每 stream 一个 FIFO）→ EE queue（per-address-space FIFO）→ SM assignment。每一级都使用 FIFO 顺序，但存在因资源约束和 stream 间依赖导致的 blocking delay，使得调度不完全是 work-conserving。

从kernel调度角度拆解术语：
调度的完整流程（论文 Rules G1-G4, X1, R1-R3, C1-C4, N1-N2, A1-A2）：

```
Level 1 — Stream Queues (per stream):
  In:  CUDA API calls (cudaLaunchKernel, cudaMemcpyAsync)
  Out: Kernel → EE queue (G2); Copy → CE queue (C1)
  Ordering: FIFO within each stream
  Blocking: Operations wait in stream queue until previous ops complete (G4/C4)

Level 2 — EE Queue (per address space):
  In:  Kernels from heads of stream queues (G2)
  Out: Block assignment to SMs (X1, R1-R3)
  Ordering: FIFO (launch-time order across streams)
  Non-preemptive: Only head kernel's blocks eligible (X1)

  With stream priority (A1-A2):
    Two EE queues: priority-high and priority-low
    High-priority queue must be empty before low-priority blocks assigned

Level 2b — CE Queue:
  In:  Copy ops from heads of stream queues (C1)
  Out: Assignment to CE (C2-C3)
  Ordering: FIFO
  Non-preemptive: Only head copy assigned at a time

Level 3 — SM Assignment:
  In:  Blocks from head of EE queue
  Constraints: threads ≤ 2048/SM, shmem ≤ 64KB/SM, registers ≤ 65536/SM
  Assignment: Greedy — assign eligible block to any SM with sufficient resources
```

论文指出这种调度"具有可分析的响应时间边界"——类似于多处理器上的 FIFO 调度已被证明具有可分析的 tardiness bounds [13]，因此 TX2 的 GPU scheduler 可能适用于实时可调度性分析。

术语一般如何实现？如何使用？
这种调度策略是 NVIDIA GPU 的硬件+驱动实现，对 CUDA 开发者透明。了解其行为对实时系统设计至关重要：(1) 使用多 stream 而非单 stream 可提高并发；(2) 避免 NULL stream 的隐式同步；(3) 资源约束导致的 blocking 不可避免，但可预测；(4) 在 task 共享地址空间场景下调度更可预测（vs process 独立地址空间场景）。

涉及论文标题：
- GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed

---
