## dsm_comm Primitive（DSM通信原语）

术语是什么？通过联网搜索让回答具体和精准。

dsm_comm primitive 是 FlashFuser 论文提出的高层 DSM 通信抽象，用于在 fused GEMM kernel 中描述 cluster 内 SM 之间的数据交换模式。它将 H100 Thread Block Cluster 的 SM 划分、数据流方向和通信模式（reduce/shuffle/multiply）统一编码为可组合的原语。整个原语体系包括四种基本操作：(1) dsm_all_exchange——cluster 内沿 K 维 All-Reduce 聚合 partial sum 以产生完整中间 tile；(2) dsm_shuffle——Shuffle Group 内 ring communication 交换中间 tensor 切片；(3) dsm_reduce_scatter——cluster 内 scatter-reduce 聚合 partial output；(4) inter_cluster_reduce——基于 TMA cp.reduce.async.bulk 的跨 cluster 原子归约。对 Gated FFN，dsm_all_exchange 从 Add 变为 Mul 操作。primitive 的核心参数由 cluster size 四维参数 (clsm, clsn, clsk, clsl) 决定，派生两个关键变量：clsshuffle = clsl / clsk（参与 shuffle 的 block 数）和 clsreduce = clsn / clsshuffle（参与 reduce 的 shuffle group 数）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

以标准 FFN 两段 GEMM (A×B=C, C×D=E) 的 cluster size (2,4,2,4) 为例：

```
// ===== Cluster 配置 =====
clsm=2, clsn=4, clsk=2, clsl=4
clsshuffle = clsl/clsk = 4/2 = 2  (每组 2 个 block 参与 shuffle)
clsreduce = clsn/clsshuffle = 4/2 = 2  (2 个 shuffle group 参与 reduce)

// ===== GEMM0 Phase (A[M×K] × B[K×N] → C[M×N]) =====
// clsk=2 → K 维 spatial partition 到 2 个平行 block
// Block(0,0): C_0,0(0) = Σ(A_0,i × B_i,0) for i=0..K/2
// Block(0,1): C_0,0(1) = Σ(A_0,i × B_i,0) for i=K/2..K

// ===== dsm_all_exchange (沿 K 维 All-Reduce) =====
// 同一 cluster 内参与 K 维 partition 的 block 交换 partial sum
dsm_all_exchange(group=[Block(0,0), Block(0,1)], op=Add)
  → C_0,0 = C_0,0(0) + C_0,0(1)
// C_0,0 留在 DSM 中，不写 global memory

// ===== GEMM1 Phase (C[M×N] × D[N×L] → E[M×L]) =====
// dsm_shuffle: Shuffle Group 内交换 C 切片
// Shuffle Group 0: {Block(0,0), Block(0,1)} — 共享 C row 0
// Shuffle Group 1: {Block(1,0), Block(1,1)}
dsm_shuffle(group=ShuffleGroup_0, pattern=ring_communication)
  // 每个 Block 需要完整 C row 才能与 D 的不同 tile 相乘
  Block(0,0): C_0,0,C_0,2,... → 用于计算 E_0,0
  Block(0,1): C_0,0,C_0,1,... → 用于计算 E_0,1

// ===== GEMM1 计算 =====
Block(0,0): E_0,0(0) = C_0,0 × D_0,0
Block(0,1): E_0,1(0) = C_0,0 × D_0,1

// ===== Store Phase =====
// dsm_reduce_scatter: cluster 内 scatter-reduce
dsm_reduce_scatter(group=ReduceGroup, op=Add)
  → Block(0,0): 负责写回 E_0,0 = E_0,0(0) + E_0,0(1)
  → Block(0,1): 负责写回 E_0,1 = E_0,1(0) + E_0,1(1)

// inter_cluster_reduce: 跨 cluster 原子归约
if (多 cluster 贡献同一输出 E tile):
  inter_cluster_reduce(E_tile, op=Add)
  // 通过 TMA cp.reduce.async.bulk 异步原子 reduce
```

Gated FFN 变体：两个 Up-FFN GEMM 分支并行执行→dsm_all_exchange 从 Add 变为 Mul→将 SiLU 分支和另一 GEMM 分支的结果 element-wise 乘。可空间划分(clsk=2, 两分支到不同 block group)最大化并行或顺序执行最小化 DSM 通信。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

dsm_comm primitive 的实现基于 NVIDIA H100 的以下硬件/软件机制：
1. **数据移动**：通过 TMA（Tensor Memory Accelerator）实现 SM-to-SM 的细粒度数据交换。TMA 的 cluster 内地址空间支持 `shared::cluster` 修饰符直接访问其他 SM 的 shared memory
2. **同步**：使用 mbarrier many-to-many synchronization 而非 CUTLASS 默认的 all-to-one cluster-sync。mbarrier 可只同步参与特定 shuffle/reduce 的 CTA 子集
3. **代码生成**：FlashFuser 后端在 CUTLASS kernel 结构的三个位置插入 dsm_comm 操作——prologue 初始化 DSM semaphore/barrier；mainloop 插入 DSM mul/shuffle（GEMM accumulation 完成后）；epilogue 执行 DSM reduce + global memory store
4. **Ring Communication**：SHUFFLE 使用 ring communication 模式——各 CTA 发送本 CTA 的 C tile 切片给下一个 CTA，同时从上一个 CTA 接收需要的切片
5. **配置灵活性**：cluster size (clsm, clsn, clsk, clsl) 可配置以适应不同 problem size（尤其是小尺寸或不可整除 case）

涉及论文标题：
- FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection

