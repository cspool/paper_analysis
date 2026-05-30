## Thread Block Wave Execution

术语是什么？
Thread Block Wave Execution 是 GPU 上 CUDA kernel 中 thread block 的调度执行模式。当一个 kernel 发射的 thread block 数量超过 GPU 可同时执行的数量（SM 数 × occupancy）时，thread block 会分批执行，每一批称为一个 wave。Wave 数 = ceil(Thread Blocks / (Number_of_SMs × occupancy))。前几波是 full wave（所有 SM 都被占用），最后一波可能是 partial wave（只有部分 SM 被占用）。NVIDIA 未公开 CUDA 调度 thread block 到 SM 的具体机制。

从kernel调度角度拆解术语：
Wave 调度在 GPU 上的执行流程：
```
给定: 80 SM, occupancy = 1 TB/SM, grid = [1, 48, 4] → 192 thread blocks
- Waves = ceil(192 / (80 × 1)) = 3
- Full waves: 2 waves × 80 TB = 160 TB (SM util: 100%)
- Partial wave: 192 - 160 = 32 TB → 32 SM busy, 48 idle (40% utilization)
- 平均利用率 = (2 × 100% + 1 × 40%) / 3 = 80%

当两个依赖kernel串行执行时，利用率问题叠加:
Producer: 3 waves (80, 80, 32), Consumer: 3 waves (80, 80, 32)
总waves: 6, 平均利用率: (80+80+32+80+80+32)/(6×80) = 384/480 = 80%

cqSync减少总waves: Producer和Consumer的independent TB可在同一wave混合执行
→ 从各自3+3=6 waves 降为约4.8 waves → 利用率提升
```

术语一般如何实现？如何使用？
Wave 执行是 GPU 硬件调度器的固件行为，程序员无法直接控制。但可以通过以下方式间接影响：(1) 调整 grid size（thread block 数量）；(2) 调整 occupancy（通过寄存器使用、共享内存分配）；(3) 使用 CUDA MPS（Multi-Process Service）让多个 kernel 的 thread block 混合调度；(4) cuSync 通过细粒度同步使不同 kernel 的 independent thread block 在同一 wave 中混合执行。理解 wave 行为对性能优化至关重要——当 grid size 刚好是 SM 数×occupancy 的整数倍时，不存在 partial wave，GPU 利用率最高。

涉及论文标题：
- A Framework for Fine-Grained Synchronization of Dependent GPU Kernels
- Efficient and Adaptable Overlapping for Computation and Communication via Signaling and Reordering

FlashOverlap 利用 GEMM 的 wave pattern 实现 computation-communication overlap。每个 wave 内的所有 tile 几乎同时完成（完成时间差 < 5% wave duration），因此以 wave 为粒度（而非 tile 为粒度）触发通信，可在不损失 overlapping opportunity 的前提下获得更好的带宽利用率。Wave 数 T = tile_num / (SM_num - comm_SM_num)，其中 comm_SM_num 为 NCCL 通信占用的 SM 数。Wave group 将连续多个 wave 合并为一个通信单元以进一步优化带宽利用。
