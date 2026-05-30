## Spatial Multiplexing / Temporal Multiplexing (GPU Concurrency Strategies)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Spatial Multiplexing 和 Temporal Multiplexing 是 GPU 并发执行多应用的两种基本策略：
- **Spatial Multiplexing**：将 GPU 的 SM（空间资源）划分给不同应用，多应用 **同时在** 不同 SM（或同一 SM 的不同 resource partition）上并行执行。代表机制：Priority Streams（同进程）、MPS（跨进程）。优势是提高资源利用率（非限制性资源不再空闲），代价是 SM 内资源竞争（L1 cache/functional unit/warp scheduler contention）可能导致 performance degradation 和不可预测性。
- **Temporal Multiplexing**：通过时间分片将 GPU 整体轮流分配给不同应用，**任何时候只有一个应用**占有 GPU。代表机制：Time-Slicing（跨进程）。优势是隔离性好、predictable（无 SM 内竞争），代价是利用率低（空闲资源无法被另一应用使用）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

两种策略的调度示意：

```
// Spatial Multiplexing (MPS)
Time ------>
SM0:  [Train_B0][Train_B1][Inf_B0][Train_B2]...  ← 同一 SM 上 colocation
SM1:  [Train_B3][Inf_B1][Train_B4][Inf_B2]...    ← resource contention
...
SM81: [Inf_B3][Train_B99][Inf_B4]...

所有 SM 同时执行，利用率高但性能不可预测

// Temporal Multiplexing (Time-Slicing)
Time ------>
GPU:  [|-- Process A (2ms) --|-- Process B (2ms) --|-- Process A --|...]
      ↑ 所有 82 SMs 专属 A              ↑ 所有 82 SMs 专属 B
      无 colocation, predictable         无 spatial sharing, 利用率低
```

本文的关键发现：
- Spatial + Temporal 结合（即 fine-grained preemption 配合 spatial multiplexing）可能优于两者单独使用，这在 Jain et al. 的初步工作中也有体现（Dynamic Space-Time Scheduling）。
- DL workload 特性（sequential kernel launches, fluctuating resource requirements, stochastic inference arrivals）使得纯 spatial（MPS, priority streams）或纯 temporal（time-slicing）都难以同时达到高 utilization + 低 predictable latency。
- 论文 O7-O10 论证 fine-grained preemption 是启用灵活 space-time scheduling 的必要硬件能力。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：
- Spatial: CUDA streams (intra-process), MPS (cross-process), MIG (hardware partition on A100/A30), libsmctrl (fine-grained SM mask control), GreenContext (CUDA 12.4+, in-process SM partition).
- Temporal: Default time-slicing (cross-process, Linux), Jetson 平台支持配置 time slice 长度和频率。
- 混合: 论文提出的 fine-grained block-level preemption + contention-aware placement + MPS thread limiting = 动态 space-time scheduling。现有最接近的实现是 Bullet (OSDI 2024) 的 spatial-temporal orchestration 使用 MPS + libsmctrl + 动态重分区。

涉及论文标题：
- Characterizing Concurrency Mechanisms for NVIDIA GPUs under Deep Learning Workloads
