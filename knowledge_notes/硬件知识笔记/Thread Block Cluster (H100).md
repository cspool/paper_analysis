## Thread Block Cluster (H100)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Thread Block Cluster 是 NVIDIA H100 (Hopper CC 9.0) 引入的编程模型扩展，允许一组 thread blocks (CTA) 作为一个 cooperative group 在物理上相邻的 SM 上执行，并通过 Distributed Shared Memory (DSM) 直接交换数据。与传统的 thread block 独立执行模型不同，cluster 内的多个 CTA 可以：(1) 通过 SM-to-SM NoC 直接访问彼此的 shared memory；(2) 使用 mbarrier 进行 cluster-scope 同步；(3) 通过 TMA multicast 同时从 HBM 预取数据。最大 cluster size 为 16 blocks (8 for CUDA C++ API)。

从硬件架构角度拆解术语：
FlashFuser 中 cluster 的四维参数化：(clsm, clsn, clsk, clsl) 分别对应 M/N/K/L 四个 GEMM 维度的 spatial partition degree。每个 CTA 负责一个 blk_m×blk_n 的 output tile。clsk 决定 K 维有多少个 CTA 并行计算 GEMM0 后需要通过 DSM all-exchange 做 accumulation；clsl 决定 L 维的 parallel degree 影响 shuffle group 大小 (clsshuffle = clsl/clsk)；clsn 决定 N 维的 parallel degree 影响 reduce group 大小。product 受硬件上限 16 约束。不同 cluster config 产生不同的数据交换模式——FlashFuser 的 search engine 在 5^4 种 cluster config 中搜索最优方案。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CUDA 中通过 `__cluster_dims__(x,y,z)` annotation 声明 cluster launch configuration，`grid_rank`/`block_rank` API 获取 cluster 内位置。CUTLASS 3.x 提供 cluster launch abstraction。FlashFuser 在 CUTLASS 基础上扩展了 cluster 内 group-wise synchronization (mbarrier)，支持仅同步参与特定 shuffle/reduce 的 CTA 子集而非整个 cluster。

涉及论文标题：
- FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection
