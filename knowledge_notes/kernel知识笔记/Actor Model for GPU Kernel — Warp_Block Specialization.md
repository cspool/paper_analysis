## Actor Model for GPU Kernel — Warp/Block Specialization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Actor Model 是并发计算模型（Carl Hewitt, 1973）。FlashMoE 将其移植到 GPU kernel 内部：将 GPU thread block 和 warp 特化为三种独立 actor 角色——Processor（N-1 个 block，执行 GEMM/element-wise）、Scheduler（1 个 warp，多线程 work-conserving 调度）、Subscriber（3 个 warp，解码 remote packet 为 task descriptor）。每个 actor 通过共享/全局内存交换消息（doorbell 信号 + task descriptor），以非阻塞、松耦合方式并发。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
GPU SM 分配:
┌────────────────────────────────────────────┐
│ SM_0: Proc 0  │ SM_1: Proc 1  │ ...       │
│ SM_{N-1}: OS Block (Scheduler + Subscriber)│
│   Warp 0: Scheduler (32 threads 并行 sweep)│
│   Warp 1-3: Subscriber (并发 decode flags) │
└────────────────────────────────────────────┘

Actor 间通信:
- Scheduler ↔ Subscriber: shared memory (同 block 内)
- Scheduler → Processor:   global memory doorbell + task queue
- Remote GPU → Subscriber: NVSHMEM one-sided put + signal

Scheduler 循环 (Algorithm 3):
while scheduled < taskBound:
    do in parallel: sweep doorbells → tqState
    WarpInclusiveSum(counts, offset, total)
    while total > 0:
        repopulate ready_queue
        do in parallel: signal processors
    taskBound = AtomicLoad(taskBound)  // dynamic

Subscriber 循环 (Algorithm 4):
while interrupt == False:
    do in parallel: atomically claim dispatch flags
        if set: decode → GEMM0 tasks → notify scheduler
    do in parallel: atomically claim combine flags
        if set: decode → combine tasks → notify scheduler
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- OS block 仅占 1/N 资源做管理，N-1 个 block 全力计算
- Scheduler work-conserving + multithreaded：有 task 就分配，32 threads 并行 sweep
- Subscriber 用 3 warps（非 1）并行处理 dispatch + combine 信号
- 传统 GPU kernel 所有 block 执行相同对称代码；FlashMoE 将 GPU 视为分布式系统，block/warp 是独立"处理节点"

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel
