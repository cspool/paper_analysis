## Thread Block (as Schedulable Entity)

术语是什么？
Thread Block（线程块）是一组可并发执行的 GPU 线程集合，是 GPU scheduler 进行调度决策的基本单位（schedulable entity）。一个 CUDA kernel 在 launch 时指定 block 数量和每个 block 的线程数。同一个 block 内的所有线程在同一 SM 上执行，可以通过 shared memory 和 __syncthreads() 进行协作。block 可以按任意顺序执行。

从kernel调度角度拆解术语：
在本文揭示的 TX2 GPU scheduler 中，"blocks are schedulable entities" 是核心发现。调度规则的关键：只有 EE queue 头部 kernel 的 block 才可被分配（Rule X1）；每个 block 的分配需满足目标 SM 的资源约束（Rules R1-R3: threads ≤ 2048/SM, shared memory ≤ 64KB/SM, registers ≤ 65536/SM）。不同 kernel 的 block 可在同一 SM 上同时执行（前提是都满足资源约束且符合 EE queue 优先级规则）。

具体计算过程——以论文 Fig. 3 的 K1 为例：
```
K1: 6 blocks × 768 threads/block, 0 shared memory/block
TX2 SM constraints: 2048 threads/SM, 64KB shmem/SM

Round 1 dispatch (Fig. 4(a)):
  SM0: 2 blocks of K1 assigned (2×768 = 1536 ≤ 2048 threads ✓)
  SM1: 2 blocks of K1 assigned (2×768 = 1536 ≤ 2048 threads ✓)
  Remaining: 2 blocks of K1 waiting (not enough thread resources)

Round 2 dispatch (Fig. 4(b)):
  After first 4 blocks complete, SM0 and SM1 each have 2048 threads freed
  SM0: 1 remaining block of K1 + 2 blocks of K4 assigned (768 + 2×256 = 1280 ≤ 2048 ✓)
  SM1: 1 remaining block of K1 + 2 blocks of K4 assigned (same)

K5 cannot be dispatched even though threads available:
  K5 requires 32KB shmem/block
  Each SM: 2 blocks of K4 × 32KB = 64KB = SM limit → no room for K5 (Rule R3)
```

术语一般如何实现？如何使用？
在 CUDA 编程中，kernel launch 的 `<<<gridDim, blockDim, sharedMem, stream>>>` 语法定义了 block 的配置。Block 大小的选择需要权衡 occupancy（SM 上同时驻留的 block/warp 数量）和资源使用（寄存器、shared memory）。在实时系统中，block 是 GPU 抢占的粒度边界——如本文发现 Pascal 架构支持指令级 preemption，但 block 仍是资源分配和调度的基本单位。

涉及论文标题：
- GPU Scheduling on the NVIDIA TX2: Hidden Details Revealed

---
