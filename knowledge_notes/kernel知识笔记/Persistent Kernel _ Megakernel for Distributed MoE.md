## Persistent Kernel / Megakernel for Distributed MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Persistent Kernel（持久内核）是一种仅需一次 CPU launch、在 GPU 上持续运行直到完成全部计算任务的 GPU kernel 设计范式。FlashMoE 将这一概念推向极致，构建了一个 Megakernel（巨型融合内核），将分布式 MoE operator 的全部阶段——Gate routing、Token Dispatch、Expert FFN (2×GEMM)、Expert Combine、跨 GPU 通信——融合为单一持久 kernel。传统实现（DeepSpeed-MoE、Megatron-LM）每个 MoE layer 需要 33-550 次独立 kernel launch（Table 1），每次 launch 由 CPU 串行调度，产生 CPU-GPU 同步间隙和非确定性 kernel start time。FlashMoE 的 megakernel 仅需 1 次 launch，kernel 内 GPU 自主管理所有任务调度和执行。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// === 单次 CPU launch (仅1次!) ===
LaunchKernel<<<N_blocks, 128_threads>>>(A, X, O, N);

// === Kernel 内部 (GPU 自主执行) ===
// Phase 0: FusedGate (所有 block 参与)
T_φ, G_φ ← FusedGate(A)

// Phase 1: Role Assignment
if blockId + 1 < N:
    Dispatch(T_φ, A)               // 准备 dispatch packet
    while interrupt == False:      // 持久循环!
        awaitTask_from_Scheduler() // 等待 task assignment
        switch task.Type:
            case GEMM0: fGET_GEMM0()
            case GEMM1: fGET_GEMM1()
            case Combine: combine()
else:
    if warpId == 0: scheduler_loop()
    else: subscriber_loop()

// Phase N: Interrupt → 所有 actor 退出 → kernel 返回
```

Kernel 在整个 MoE layer 执行期间持续活跃（不返回 CPU），通过 while 循环和 doorbell 信号机制维持。Scheduler 持续 sweep doorbells → 发现新 task → 分配 → processor 执行 → 结果写回 → notify subscriber → subscriber 解码下一轮 task。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- FlashMoE kernel 实现指标（Table 4）：6820 LOC, 0 B stack frame, 0 spill, 46 KB shared memory/block, 255 registers/thread, 2 max active blocks/SM, 53s compilation, 29 MB binary
- 单一 kernel 的核心优势：消除 launch 间隙 → SM utilization 93.17%（vs DeepEP 14%）；消除 kernel boundary global memory round-trip；确保确定性的 GPU-native 调度时序
- 与 CUDA Graphs 的区别：CUDA Graphs 仅消除 CPU launch overhead（仅适用 static workload），不解决 kernel boundary round-trip 或 host-device synchronization

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel
