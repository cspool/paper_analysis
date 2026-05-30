## Limiting Resource (SM Thread Block Scheduling)

术语解释
Limiting Resource 是在 NVIDIA GPU most-room policy 下，决定一个 SM 最多能容纳多少新 kernel thread block 的瓶颈资源维度。当 thread block scheduler 计算各 SM 能容纳的 block 数量时，取所有 SM 资源维度（threads、shared memory、registers、blocks/SM 上限、warps/SM 上限）的最小容纳值作为该 SM 的可用容量。Limiting resource 的变化（如 thread/block 从 33→32）可能完全改变 block 的 placement，导致性能的巨大差异。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

SM 的每个资源维度有其硬件上限，thread block 消耗的资源量由其 launch configuration 决定。Limiting resource 是以下计算中得出最小值的资源维度：
`blocks_fit = min(floor(sm.max_threads - sm.used_threads) / B.threads_per_block), floor((sm.max_shmem - sm.used_shmem) / B.shmem_per_block), floor((sm.max_regs - sm.used_regs) / B.regs_per_block), sm.max_blocks - sm.current_blocks, sm.max_warps - sm.current_warps)`

在 most-room policy 中，各 SM 的 blocks_fit 值可能不同（因已 resident block 的尺寸差异），scheduler 选 blocks_fit 最大的 SM放置下一个 block。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Limiting Resource 变化触发不同 placement 的具体例子（Turing RTX 2080 Ti）：

```
Kernel A: 67 blocks × 512 threads, 占满 SM0-SM66, SM67 空
Kernel B version 1: 8 blocks × 33 threads
  - Limiting resource = threads
  - SM67 (1024 free threads): floor(1024/33) = 31 blocks of B
  - SM0-SM66 (512 free threads each): floor(512/33) = 15 blocks of B
  → Most-room 选 SM67 (31 > 15)
  → 全部 8 个 B block → SM67 → Concurrent-Isolated

Kernel B version 2: 8 blocks × 32 threads
  - Limiting resource = blocks/SM (16 max)
  - SM67: floor(1024/32)=32, but min(..., 16-0=16)=16 blocks of B
  - SM0-SM66: floor(512/32)=16, but min(..., 16-1=15)=15 blocks of B
  → Most-room 选 SM67 (16 > 15) → B_0→SM67
  - SM67 现在: blocks/SM=16-1=15, 与 SM0-SM66 平票
  → Tie-breaking → B_1→SM0 (even-then-odds ordering)
  → B blocks 分散到 8 个 SM → Concurrent-Colocated
```

结果：1 thread/block 的变化 → limiting resource 从 threads 变为 blocks/SM → 完全不同的 block placement → 对于 transfer-bandwidth-dependent kernel，Kernel B 的性能退化从 concurrent-isolated 下的 2.73X 增至 concurrent-colocated 下的 3.58X。

各种 GPU 架构的硬件资源上限（Table 2）：
- Pascal (GTX 1080): 2048 threads/SM, 32 blocks/SM, 64 warps/SM
- Volta (V100): 2048 threads/SM, 32 blocks/SM, 64 warps/SM
- Turing (RTX 2080 Ti): 1024 threads/SM, 16 blocks/SM, 32 warps/SM

论文声明可能存在其他未识别的 limiting factor（因 scheduler 是黑盒）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Limiting resource 是硬件层面的行为结果，非可编程概念。应用开发者可通过以下方式间接利用此理解：(i) 调整 block 配置（threads/block、shared memory/block）来改变 limiting resource，从而控制 concurrent workload 中的 placement；(ii) 使用 `cudaOccupancyMaxPotentialBlockSize` API 获取特定 kernel 在目标 GPU 上的 optimal block 配置（但该 API 返回的是单 kernel 场景下的 optimal occupancy，不考虑 concurrent workload 下的 most-room policy 行为）；(iii) GPU 模拟器可基于 hardware spec 实现 most-room policy 的 limiting resource 计算，提高 concurrent kernel workload 的模拟精度。

涉及论文标题：
- Demystifying the Placement Policies of the NVIDIA GPU Thread Block Scheduler for Concurrent Kernels
- Characterizing Concurrency Mechanisms for NVIDIA GPUs under Deep Learning Workloads
