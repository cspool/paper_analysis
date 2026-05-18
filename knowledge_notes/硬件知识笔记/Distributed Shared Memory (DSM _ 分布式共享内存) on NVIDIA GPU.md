## Distributed Shared Memory (DSM / 分布式共享内存) on NVIDIA GPU

术语是什么？通过联网搜索让回答具体和精准。

Distributed Shared Memory (DSM) 是 NVIDIA H100 (Hopper) GPU 架构引入的硬件特性，通过 SM-to-SM NoC 将同一 Thread Block Cluster 内多个 SM 的 shared memory（L1 cache）互联，形成一个更大的片上内存池。DSM 在 GPU 内存层次中位于 L1.5 cache 层——高于 register(reg)和单 SM 的 shared memory(SMEM/L1)，低于 L2 cache 和 HBM global memory。物理上，H100 允许在一个 cluster 内最多 16 个 SM 通过 crossbar 互联共享 SMEM，因此 DSM 容量理论上限约为 16 × 228KB = 3.6MB（实际受限于 cluster size 配置）。DSM 的带宽和延迟随 cluster size 变化：cluster size 越小（如 2 SM），DSM 带宽最高且延迟最低；cluster size 越大（如 16 SM），DSM 带宽降低但仍高于 HBM 带宽，延迟始终低于 global memory。DSM 本身是硬件 SM-to-SM 互联的软件抽象名称，CUDA 编程指南中称为 Distributed Shared Memory（cuda-c-programming-guide section 4.6.10）。

从硬件架构角度拆解术语：

DSM 在 H100 GPU 上的内存层次和运转流程（以 FlashFuser fused GEMM kernel 为例）：

```
H100 Memory Hierarchy (从近核到远核):
  L0:  Register File (reg)         — 每 thread 可见, ~256KB/SM
  L1:  Shared Memory (SMEM)        — 单 SM 内所有 threads 可访问, ~228KB/SM
  L1.5: Distributed Shared Memory  — cluster 内多 SM 通过 SM-to-SM NoC 互联
  L2:  L2 Cache                    — 所有 SM 共享, ~50MB
  HBM: Global Memory (HBM3)        — 3TB/s bandwidth, 80GB

DSM 数据流 (FlashFuser GEMM chain):
  SM 0                        SM 1 (同一 cluster)
  ┌─────────────────┐        ┌─────────────────┐
  │ GEMM0: A×B→C(0) │        │ GEMM0: A×B→C(1) │
  │ partial C_0,0(0) │        │ partial C_0,0(1) │
  └───────┬─────────┘        └───────┬─────────┘
          │ dsm_all_exchange         │
          └────────┬─────────────────┘
                   ▼ (SM-to-SM NoC 直接传输)
          cluster 内 All-Reduce → 完整 C_0,0 tile
          (不经过 L2/HBM，片上直接交换)
                   
  SM 0: GEMM1 C_0,0×D→E partial
  SM 1: GEMM1 C_0,0×D→E partial
          │ dsm_shuffle (ring communication)
          ▼
  dsm_reduce_scatter → E tile → store to HBM
```

关键硬件特性：(1) DSM 是通过 SM-to-SM NoC 实现的直接片上数据交换路径，避免传统 global memory round-trip（"write C to HBM → read C from HBM"）；(2) 只有同一 cluster 内的 SM 可以通过 DSM 交换数据，不同 cluster 之间必须通过 L2/HBM；(3) DSM 通信需要硬件同步机制——FlashFuser 使用 mbarrier many-to-many synchronization 而非默认的 all-to-one cluster-sync，以实现仅同步必要 CTA 子集的更灵活通信模式。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

DSM 在 CUDA 中通过以下方式使用：
1. **Thread Block Cluster 声明**：`__cluster_dims__(2,4,2) __launch_bounds__(N)` 声明 kernel 使用 cluster launch，SM 的 shared memory 自动可被 cluster 内其他 SM 通过 DSM 访问
2. **SMEM 地址映射**：SM 的 shared memory 被映射到 cluster-wide DSM 地址空间，其他 SM 可通过 `mapa`/`mapa.shared::cluster` PTX 指令访问
3. **TMA 配合 DSM**：`cp.async.bulk.tensor.2d.shared::cluster.global`——TMA 可以在 global→shared 传输中使用 `shared::cluster` 修饰符，将数据 multicast 到 cluster 内多 SM
4. **同步原语**：DSM 传输需要 cluster-level barrier 同步——CUTLASS 默认使用 all-to-one cluster-sync，FlashFuser 使用 mbarrier 实现 many-to-many sync 以支持更灵活的 group-wise communication

DSM 的限制：(1) 仅在 H100 (Hopper) 及更新架构支持，A100 不支持；(2) cluster size 最大 16，且 cluster shape 的 product 不能超过硬件上限；(3) DSM bandwidth 和 latency 随 cluster size 变化，需要分析性模型选择最优 cluster size；(4) DSM 需要显式同步和 fencing，编程复杂度高于单 SM SMEM。

涉及论文标题：
- FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection

