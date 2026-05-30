## Concurrent Kernel Execution (CKE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Concurrent Kernel Execution (CKE, 并发 Kernel 执行) 是指来自不同 CUDA stream 的多个 kernel 在同一 GPU 上的同一时刻并行执行。由于 GPU thread block scheduler 使用 leftover policy + most-room policy 决定 block 的调度时机和放置位置，CKE 的实际发生条件为：队头 kernel 的所有 block 已被调度到 SM 但仍有空闲资源 → 队列中下一个 kernel 的 block 开始被调度并与前一 kernel 的 block 在同一时间执行。CKE 被广泛认为是提升 GPU 利用率的关键手段——通过同时运行计算密集型和内存密集型 kernel 来填满 GPU 的 compute 和 memory bandwidth 资源。CUDA stream（每个 stream 是命令序列）和 MPS（Multi-Process Service）是 CKE 的主要用户接口。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

CKE 在 most-room policy 影响下的两种关键场景：

```
场景1: Concurrent-Isolated (两个 kernel 的 block 在不同 SM 上)
  SM0: [A_0][A_0][...]  ← 仅 Kernel A
  SM1: [A_1][A_1][...]
  ...
  SM66: [A_66][A_66][...]
  SM67: [B_0][B_1]...[B_7]  ← 仅 Kernel B

  条件: B 的 limiting resource 使空 SM 能容纳最多 B block
        → Most-room 将全部 B block 分配到空 SM
  效果: 两 kernel 各自独占 L1 cache 和 functional units
        性能与串行执行相同（无竞争）

场景2: Concurrent-Colocated (两个 kernel 的 block 在同一 SM 上)
  SM0:  [A_0][B_0]  ← 混合执行
  SM2:  [A_2][B_1]
  SM4:  [A_4][B_2]
  ...
  SM67: [B_7]

  条件: B 的 limiting resource 变化 → Most-room 将 B block
        分散到已有 A block 的 SM 上
  效果: L1 cache contention, functional unit contention
        性能退化从 1.24X 到 96.1X（取决于 kernel 类型）
```

关键发现：Most-room policy 不区分 isolated vs colocated 对性能的影响——它仅看"能容纳多少 block"而不考虑 colocation 导致的资源竞争。这使得 CKE 的性能难以预测。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 CUDA 中实现 CKE 的标准方式：(1) 创建多个 CUDA stream；(2) 将不同 kernel 发射到不同 stream；(3) `cudaStreamSynchronize` 或 `cudaDeviceSynchronize` 等待完成。更高级控制：(i) CUDA MPS 分配独立 SM partition；(ii) libsmctrl 设置 stream SM mask 精确控制 kernel 在哪些 SM 上执行；(iii) CUDA stream priority 提供 hint 但非硬性保证。论文指出 CKE 的性能预测困难——取决于 most-room policy placement、资源竞争类型（L1/functional unit/memory BW/PCIe）、以及 kernel launch timing 等外部因素。

涉及论文标题：
- Demystifying the Placement Policies of the NVIDIA GPU Thread Block Scheduler for Concurrent Kernels
- Characterizing Concurrency Mechanisms for NVIDIA GPUs under Deep Learning Workloads
