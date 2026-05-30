## Distributed Shared Memory (DSM / 分布式共享内存) on NVIDIA GPU

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Distributed Shared Memory (DSM) 是 NVIDIA H100 (Hopper) GPU 架构引入的硬件特性，通过 SM-to-SM Crossbar NoC 将同一 Thread Block Cluster 内多个 SM 的 shared memory（L1 cache）互联，形成一个更大的片上内存池。DSM 在 GPU 内存层次中位于 L1.5 cache 层——高于 register (reg) 和单 SM 的 shared memory (SMEM/L1)，低于 L2 cache 和 HBM global memory。H100 允许在一个 cluster 内最多 16 个 SM 通过 crossbar 互联共享 SMEM，DSM 容量理论上限约 16 × 228KB = 3.6MB。DSM bandwidth 和 latency 随 cluster size 变化：cluster size 越小（如 2 SM），DSM bandwidth 最高（~8TB/s）且 latency 最低（~20ns）；cluster size 越大（如 16 SM），bandwidth 降低（~4TB/s）但仍高于 HBM bandwidth（3.35TB/s），latency 始终远低于 global memory（~280ns）。

从硬件架构角度拆解术语：
H100 Memory Hierarchy 和 DSM 在 FlashFuser fused GEMM kernel 中的数据流：
```
SM 0 (同一 cluster)              SM 1 (同一 cluster)
┌──────────────────┐           ┌──────────────────┐
│ GEMM0: A×B→C(0)  │           │ GEMM0: A×B→C(1)  │
│ partial C_0,0(0) │           │ partial C_0,0(1) │
└───────┬──────────┘           └───────┬──────────┘
        │ dsm_all_exchange (SM-to-SM NoC) │
        └──────────────┬─────────────────┘
                       ▼
          Cluster内 AllReduce → 完整 C_0,0 tile
          (片上直接交换，不经过 L2/HBM)
                       ▼
  SM 0: GEMM1 C_0,0×D→E partial
  SM 1: GEMM1 C_0,0×D→E partial
        │ dsm_shuffle (ring communication) → Store via dsm_reduce_scatter
```

DSM 的硬件约束：(1) 只有同一 cluster 内的 SM 可通过 DSM 交换数据，不同 cluster 之间必须通过 L2/HBM；(2) cluster size product 不能超过 16 (H100 hardware limit)；(3) DSM bandwidth 和 latency 随 cluster size 变化——选择最优 cluster size 需要 analytical model 权衡并行度和通信开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CUDA 中使用 DSM 的方式：(1) `__cluster_dims__(a,b,c)` 声明 kernel 使用 cluster launch，shared memory 自动可通过 DSM 访问；(2) SMEM 地址映射到 cluster-wide DSM 地址空间，其他 SM 可通过 `mapa.shared::cluster` PTX 指令访问；(3) TMA 配合 DSM——`cp.async.bulk.tensor.2d.shared::cluster.global` 使用 `shared::cluster` 修饰符将数据 multicast 到 cluster 内多 SM。FlashFuser 使用 mbarrier many-to-many sync 实现灵活的 group-wise DSM communication。DSM 仅在 H100+ (Hopper) 架构支持。

涉及论文标题：
- FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection
