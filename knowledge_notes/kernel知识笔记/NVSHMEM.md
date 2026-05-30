## NVSHMEM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

NVSHMEM 是 NVIDIA 基于 OpenSHMEM 标准的 GPU 加速通信库，为 GPU 集群提供 Partitioned Global Address Space (PGAS) 编程模型。核心原语是 one-sided put/get 操作——发送端 GPU 直接向远程 GPU 的对称内存区域写入数据，无需远程端显式参与。通过 IBGDA 实现 GPU 直接访问 RDMA-capable 网络，消除传统 two-sided MPI 通信中的 CPU 中转。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// 初始化对称内存
int *buf = (int*)nvshmem_malloc(sizeof(int) * N);
// buf 在所有 PE (Processing Element = GPU) 上有对称地址

// One-sided put: GPU→GPU direct write
nvshmem_int_put(dest, source, count, peer);
// One-sided get: GPU→GPU direct read
nvshmem_int_get(dest, source, count, peer);

// Non-blocking + 同步
nvshmem_putmem_nbi(dest, src, size, peer);  // non-blocking
nvshmem_fence();   // 确保同一 PE 的操作 ordering
nvshmem_quiet();   // 等待该 PE 的所有 outstanding 操作完成
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- One-sided 语义使 GPU 可自行发起 RDMA 传输，减少延迟 jitter——DeepEP 核心依赖此特性
- 需要显式对称内存管理（nvshmem_malloc）和 barrier/fence/quiet 同步
- 与特定硬件（InfiniBand、IBGDA）紧密耦合
- FUSCO 选择基于 NCCL 而非 NVSHMEM，以保持跨网络（TCP/IP、RoCE、IB）portability

涉及论文标题：
- FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion
- FlashMoE: Fast Distributed MoE in a Single Kernel
- JANUS: Disaggregating Attention and Experts for Scalable MoE Inference
