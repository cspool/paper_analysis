## Warp / Warp Scheduler

术语是什么？
Warp 是 NVIDIA GPU 中最小的线程调度单元，固定由 32 个线程组成，以 SIMT (Single Instruction, Multiple Thread) 模式执行。Warp Scheduler 是 SM 内部的硬件单元，负责在每个时钟周期从驻留在 SM 上的 warp 中选择可执行的 warp 并发射指令。TX2 的每个 SM 有 4 个 Warp Scheduler，每个可以发射 2 条指令/周期。

从硬件架构角度拆解术语：
在 Pascal 架构（TX2 使用）中，每个 SM 的 4 个 Warp Scheduler 各管理一组静态 warp。当一个 thread block 被分配到 SM 后，block 内的线程被划分为 warp（例如 256 线程的 block = 8 个 warp）。Warp Scheduler 利用 stall（如等待 global memory 访问）时立即切换到另一 warp，实现延迟隐藏（latency hiding）。TX2 每 SM 最多 2048 线程 = 64 warp 同时驻留。Warp Scheduler 可以在同一周期内调度来自不同 kernel 的不同 warp——因此不同 stream 的 kernel 可以在同一 SM 上真正并发执行。

术语一般如何实现？如何使用？
Warp 调度对 CUDA 程序员透明，但理解 warp 行为对性能优化至关重要：warp divergence（同一 warp 内线程走不同分支）导致性能损失；coalesced memory access（warp 内线程访问连续地址）最大化带宽。CUDA 提供 warp-level 原语（如 __shfl_sync, __ballot_sync）用于 warp 内通信。在实时系统分析中，warp 调度是 block 执行时间的微观决定因素。

涉及论文标题：
- GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed

---
