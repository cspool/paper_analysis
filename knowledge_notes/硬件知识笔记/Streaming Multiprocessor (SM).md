## Streaming Multiprocessor (SM)

术语是什么？
Streaming Multiprocessor (SM) 是 NVIDIA GPU 中的核心计算单元，每个 SM 包含多个 CUDA Core、Warp Scheduler、寄存器文件、共享内存和 L1 Cache。在本文研究的 NVIDIA Jetson TX2 上，GPU 有 2 个 SM，每个 SM 包含 128 个 CUDA Core（运行在 1.3GHz），共享 512KB L2 Cache。每个 SM 同时最多支持 2048 个线程、64KB shared memory、65536 个寄存器。Thread Block 被分配到 SM 上执行，一个 block 内的所有线程始终在同一 SM 上运行。

从硬件架构角度拆解术语：
在 TX2 的 Pascal GPU 中，每个 SM 内部有 4 个 Warp Scheduler。SM 接收从 EE queue dispatch 的 thread block，然后将其 rasterize 为 warp（每组 32 线程）分配给各 Warp Scheduler。Warp Scheduler 利用 stall（如等待内存访问）时立即切换到另一个 warp，以此隐藏内存延迟。TX2 的每个 SM 可同时驻留最多 2048 个线程（64 warps），每个 SM 的 shared memory 上限为 64KB，register file 为 256KB。当 GPU scheduler 将一个 kernel 的 block 分配到 SM 时，必须满足：该 SM 上的总线程数 ≤ 2048 (Rule R2)、总 shared memory ≤ 64KB (Rule R3)、总寄存器数 ≤ 65536。这些资源约束直接影响 kernel dispatch 的调度决策。

术语一般如何实现？如何使用？
在 NVIDIA GPU 架构中，SM 从 Kepler 到 Blackwell 持续演进。Pascal 架构中的 SM 有两种变体：GP100（每 SM 64 CUDA Core + 2 Warp Scheduler）和 GP104/TX2 使用的变体（每 SM 128 CUDA Core + 4 Warp Scheduler）。编程时开发者通过配置 kernel launch 参数（<<<numBlocks, threadsPerBlock, sharedMemPerBlock>>>）间接控制 SM 资源使用。CUDA Occupancy Calculator 可帮助确定最优的 block/thread 配置以最大化 SM 占用率。在实时系统中，了解 SM 资源约束对预测 kernel 执行时间至关重要——如本文所示，一个需要 1024 threads/block 的 kernel 每 SM 最多只能分配 2 个 block。

涉及论文标题：
- GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed

---
