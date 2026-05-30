## Streaming Multiprocessor (SM, NVIDIA GPU Compute Unit)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Streaming Multiprocessor (SM) 是 NVIDIA GPU 的核心计算单元，每个 SM 包含：计算核心（CUDA Cores/INT32/FP32/Tensor Cores）、warp scheduler（调度 warp 指令发射）、register file（寄存器文件）、L0 指令 cache、L1 data cache/shared memory（可配置分区）、constant cache 等。SM 是 thread block 的执行载体——当一个 thread block 被 thread block scheduler 分配到 SM 后，block 的所有 warp 在该 SM 上分时执行，直到全体完成。一个 SM 可同时容纳多个 thread block（resident blocks），数量受硬件资源上限约束（max threads、max blocks、max registers、max shared memory per SM）。SM 之间通过 L2 cache 和 crossbar 互联访问 HBM，每 SM 有独立的指令发射和数据路径。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

SM 在 GPU 执行层级中的位置（Ampere 架构，RTX 3090 为例）：

```
GPU (RTX 3090, GA102)
├── GPC (Graphics Processing Cluster) ×7
│   └── TPC (Texture Processing Cluster) ×... 
│       └── SM ×82 total (across all GPCs)
│           ├── CUDA Cores (FP32/INT32)
│           ├── Tensor Cores (3rd gen, FP16/INT8/...)
│           ├── RT Cores
│           ├── Register File: 64KB (256KB per SM)
│           ├── L1 Data Cache / Shared Memory: 128KB
│           │   (configurable partition, up to 100KB shared memory)
│           ├── L0 Instruction Cache
│           ├── Warp Scheduler ×4
│           │   (each schedules warps from resident blocks)
│           ├── Dispatch Unit ×4 (每周期每 scheduler 1 warp)
│           ├── Load/Store Unit ×32
│           └── Special Function Unit (SFU) ×4
│
├── L2 Cache: 6144KB (shared among all SMs via crossbar)
├── HBM / GDDR6X Controllers: 12 × 32-bit → 384-bit bus
└── GDDR6X DRAM: 24GB, 936 GB/s bandwidth
```

SM 上并发执行的具体例子（本文场景）：
```
SM0 资源状态（RTX 3090 硬件上限）:
  max_threads:   1536
  max_blocks:    16
  max_regs:      65536 (64KB)
  max_shmem:     102400 bytes (100KB, 配置为 max shared memory mode)

Training kernel block (256 threads, 32 regs/thread, 0 shared mem):
  threads per block: 256
  regs per block:    256 × 32 = 8192
  blocks fit on SM0: min(1536/256=6, 65536/8192=8, 16) = 6

Inference kernel block (64 threads, 80 regs/thread, 0 shared mem):
  threads per block: 64
  regs per block:    64 × 80 = 5120
  blocks fit on SM0: min(1536/64=24, 65536/5120=12, 16) = 12

如果 SM0 已有 5 个 training blocks:
  used_threads: 5 × 256 = 1280, remaining: 256
  used_regs: 5 × 8192 = 40960, remaining: 24576
  remaining blocks: 1 training block (256 threads, 8192 regs)
  → 移掉 1 个 training block 可放 4 个 inference blocks
  → thread usage 不变（1280 → 256 + 4×64 = 512... 不对，应该是:
  移掉 1 training: free 256 threads + 8192 regs
  4 inference: 256 threads + 20480 regs → OK!
  原来: 5 training blocks = 1280 threads, 40960 regs
  新: 4 training + 4 inference = 1024+256=1280 threads, 32768+20480=53248 regs
  → 更多 register 被利用!
```

本文的核心论点：Simple thread-based utilization metric（如 nvidia-smi 的 "GPU utilization %"）不能区分上述的 "5 training blocks"（under-utilized registers）和 "4 training + 4 inference"（better register utilization）——两者都是 100% thread utilization。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

SM 是硬件单元，用户无法直接控制其行为。可编程的间接影响：(i) block 配置（threads/block、shared memory/block）决定 SM 上容纳的 block 数量；(ii) `cudaOccupancyMaxPotentialBlockSize` API 返回 optimal block size；(iii) CUDA MPS 可限制 per-client 的 thread usage 间接控制 SM 的资源分配；(iv) 利用 SM 间资源隔离（MIG、GreenContext）对 SM 进行分区；(v) 通过 `smid` 寄存器（PTX inline assembly）在 kernel 内读取当前执行的 SM ID（用于 profiling/debugging）。

涉及论文标题：
- Characterizing Concurrency Mechanisms for NVIDIA GPUs under Deep Learning Workloads
- Demystifying the Placement Policies of the NVIDIA GPU Thread Block Scheduler for Concurrent Kernels
