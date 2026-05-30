## dsm_comm Primitive（DSM通信原语）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
dsm_comm primitive 是 FlashFuser 提出的高层 DSM 通信抽象，用于在 fused GEMM kernel 中描述 cluster 内 SM 之间的数据交换。它将 H100 Thread Block Cluster 的 SM 划分、数据流方向和通信模式统一编码为四种可组合的基本操作：(1) dsm_all_exchange——cluster 内沿 K 维 AllReduce（标准 FFN）或 Mul（Gated FFN）聚合 partial sum 产生完整中间 tile；(2) dsm_shuffle——Shuffle Group 内 ring communication 交换中间 tensor 切片；(3) dsm_reduce_scatter——cluster 内 scatter-reduce 聚合 partial output；(4) inter_cluster_reduce——基于 TMA cp.reduce.async.bulk 的跨 cluster 原子归约。原语的核心参数由 cluster size (clsm, clsn, clsk, clsl) 派生：clsshuffle = clsl/clsk（参与 shuffle 的 block 数），clsreduce = clsn/clsshuffle（参与 reduce 的 shuffle group 数）。

从kernel调度角度拆解术语，以标准 FFN (A×B=C, C×D=E) 的 cluster size (2,4,2,4) 为例：

```
// GEMM0 Phase: K-dim spatial partition → partial C
// clsk=2 → 2 blocks 并行沿 K 维 compute
Block(0,0): C_0,0(0) = Σ(A_0,i × B_i,0) for i=0..K/2
Block(0,1): C_0,0(1) = Σ(A_0,i × B_i,0) for i=K/2..K

// dsm_all_exchange: intra-cluster AllReduce along K-dim
dsm_all_exchange(group=[Block(0,0), Block(0,1)], op=Add)
  → C_0,0 = C_0,0(0) + C_0,0(1)  // 完整 C tile 驻留 DSM

// GEMM1 Phase: dsm_shuffle in Shuffle Group
clsshuffle = clsl/clsk = 4/2 = 2 blocks per shuffle group
dsm_shuffle(group=ShuffleGroup_0, pattern=ring_communication)
  Block(0,0): 接收 C_0,0 → 计算 E_0,0 = C_0,0 × D_0,0
  Block(0,1): 接收 C_0,0 → 计算 E_0,1 = C_0,0 × D_0,1

// Store Phase: dsm_reduce_scatter + inter_cluster_reduce
dsm_reduce_scatter(group=ReduceGroup, op=Add)
  → Block(0,0): responsible for E_0,0
  → Block(0,1): responsible for E_0,1
inter_cluster_reduce(E_tile, op=Add)  // via TMA
```

Gated FFN 变体：(1) spatial partitioning (clsk=2, 两 GEMM branch 分配到不同 block group) 最大化并行度；(2) sequential execution within single Block 最小化 DSM 通信开销。dsm_all_exchange 从 Add 变为 Mul 操作完成 SiLU 分支产出与另一 GEMM 分支的 element-wise multiply。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现基于：(1) TMA `shared::cluster` 地址空间进行 SM-to-SM 数据搬移；(2) mbarrier many-to-many sync——每个 dsm_comm 操作仅同步参与操作的 CTA 子集，而非全部 cluster；(3) CUTLASS kernel 模板的三阶段插入——prologue 中初始化 DSM semaphore/mbarrier，mainloop 中注入 dsm_all_exchange + dsm_shuffle，epilogue 中执行 dsm_reduce_scatter + inter_cluster_reduce；(4) ring communication 实现 shuffle——CTA i 发送 C tile slice 给 CTA i+1，同时从 CTA i-1 接收需要的 slice。

涉及论文标题：
- FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection
