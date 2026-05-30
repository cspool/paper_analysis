## Thread Block Scheduler (NVIDIA GPU)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Thread Block Scheduler 是 NVIDIA GPU 硬件中负责将 thread block 分配到 SM 进行执行的调度器。当一个 kernel 被发射到 GPU 时，其 thread block 不直接执行——它们首先进入调度队列，由 thread block scheduler 在资源可用时逐个分配到 SM。Scheduler 使用两个 policy：(i) Leftover Policy 决定**when/which** block 被调度（只有队列头 kernel 的 block 可被调度，不可抢占）；(ii) Most-Room Policy 决定**where** 放置该 block（选能容纳最多 block 的 SM）。一旦 block 被分配到 SM，SM 内部的 warp scheduler 负责将 warp（32-thread group）调度到执行核心。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Thread Block Scheduler 在 GPU kernel 执行流程中的位置：

```
CUDA Application (CPU)
    │
    │  cudaLaunchKernel(grid, block, stream)
    ▼
CUDA Runtime (CPU → GPU via MMIO)
    │
    │  将 kernel launch packet 写入 command queue
    ▼
GPU Command Processor
    │
    │  解码 packet, 将 kernel 放入执行队列
    ▼
Thread Block Scheduler ───────────────────────────────────┐
    │                                                      │
    │  While (queue not empty):                            │
    │    1. Leftover Policy: 只看队头 kernel              │
    │       - 队头 kernel 的所有 block 调度完之前          │
    │       - 其他 kernel 的 block 不被调度               │
    │    2. Most-Room Policy: 选择 SM                      │
    │       - 基于各 SM 当前资源可用性                     │
    │       - 选能容纳当前 kernel 最多 block 的 SM         │
    │       - 每次分配 1 个 block                          │
    │    3. Assign block → SM, update SM resource state   │
    ▼                                                      │
SM[0..N-1]                                                │
    │                                                      │
    │  Warp Scheduler (per SM):                            │
    │  - 将 block 的 warp 调度到执行核心                   │
    │  - Warp 间交错执行以隐藏延迟                         │
    ▼                                                      │
Execution Cores (CUDA cores, Tensor cores, etc.)
```

关键约束：Scheduler 不能抢占 kernel（block 不可被中途暂停），无跨 kernel 的依赖感知（不关心 kernel 间数据依赖关系）。Leftover policy 意味着并发主要发生在多个小 kernel（所有 block 能一次全部调度）之间——大 kernel（block 数超过 SM 可容纳数）会独占 GPU。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Thread block scheduler 是 NVIDIA GPU 固件（GPU System Processor / GSP firmware）的闭源实现，用户不可编程控制。唯一的影响方式是间接的：(i) 调整 block 的 resource requirement（threads/block、shared memory/block）来改变 limiting resource 和调度结果；(ii) 利用 CUDA MPS（Multi-Process Service）进行 SM 级分区；(iii) 使用 libsmctrl 等底层库修改 stream SM mask。论文使用 `smid` 和 `globaltimer` 从外部观察行为，未逆向工程 scheduler 固件本身。

涉及论文标题：
- Demystifying the Placement Policies of the NVIDIA GPU Thread Block Scheduler for Concurrent Kernels
- Characterizing Concurrency Mechanisms for NVIDIA GPUs under Deep Learning Workloads
