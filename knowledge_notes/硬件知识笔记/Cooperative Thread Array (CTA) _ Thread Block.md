## Cooperative Thread Array (CTA) / Thread Block

术语是什么？
Cooperative Thread Array (CTA) 是NVIDIA GPU中线程的组织和调度单位，即通常所说的"thread block"。一个CTA内的所有线程在同一个Streaming Multiprocessor (SM) 上并发执行，共享该SM的shared memory和寄存器文件。CUDA kernel launch时通过grid维度（CTA总数）和block维度（每CTA的线程数）指定并行配置。GPU SM调度器将CTA分配到有可用资源的SM上——每个SM可同时驻留多个CTA（受限于线程数、shared memory和寄存器上限）。CTA是GPU上GEMM并行分解的基本单位：在data-parallel GEMM中，每个CTA负责计算一个output tile；在Stream-K中，每个CTA负责一段连续的MAC-loop迭代范围。

CTA内线程可以进一步划分为warp（32线程组），通过__syncthreads()进行CTA级同步。CTA之间无法直接同步（CUDA编程模型限制），但可通过global memory的atomic操作或cooperative groups实现有限形式的inter-CTA协调。

从硬件架构角度拆解术语：
CTA到硬件的映射过程：

```
Kernel Launch: <<<gridDim, blockDim, sharedMemPerBlock, stream>>>
  gridDim: CTA总数 = ceil(m/BLK_M) × ceil(n/BLK_N) (data-parallel)
  blockDim: 每CTA线程数 (e.g., 256)
  sharedMemPerBlock: 每CTA动态shared memory分配

GPU SM调度器:
  1. 读取EE queue中的kernel
  2. 检查每个SM的资源可用性:
     - maxThreadsPerSM (e.g., 2048 for Pascal → A100/H100)
     - maxSharedMemPerSM (e.g., 64KB Pascal, 164KB A100, 228KB H100)
     - maxRegsPerSM (e.g., 65536)
  3. 将CTA分配给满足条件的SM
  4. SM上的Warp Scheduler将CTA的warp分配执行

CTA在Stream-K中的特殊用法:
  - 每CTA执行连续MAC-loop迭代范围，可跨越tile边界
  - 通过global memory temporary storage进行inter-CTA partial sum交换
  - Signal(flags) / Wait(flags) 实现轻量级inter-CTA同步
  - Synchronization overhead O(p)（p=SM数），而非O(tiles)
```

术语一般如何实现？如何使用？
CUDA编程Model自2006年引入CTA/thread block概念以来保持一致。开发者通过<<<gridDim, blockDim>>>配置CTA的数量和大小。CTA内的shared memory通过__shared__关键字声明。Stream-K论文将CTA用作iteration-balancing的并行单位：将GEMM的total_iters均匀分配给g个CTA，不同于data-parallel将output tile分配给CTA。CTA的硬件资源约束（shared memory、寄存器、最大线程数）影响blocking factor的选择——Stream-K使用能够达到99%峰值TFLOPs的最小tile size（FP64: 64×64×16, FP16→32: 128×128×32, A100），以在per-CTA资源约束下最大化效率。

涉及论文标题：
- Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU
- GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed

---
