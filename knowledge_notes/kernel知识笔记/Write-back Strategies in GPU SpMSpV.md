## Write-back Strategies in GPU SpMSpV

术语是什么？

GPU SpMSpV 的 write-back 阶段有三种基本策略处理 partial products 到输出向量的 many-to-one scatter accumulation：

1. **Atomic write-back**: 每个 partial product 通过 `atomicAdd(&y[row_idx], partial)` 直接累加。优点：实现简单、无额外内存。缺点：many-to-one scatter 导致 severe address contention，uncoalesced stores 浪费带宽（A100 上 sparsity=0.1 时 ~270 GB/s，仅 17% peak），write-back 占 runtime >30%。

2. **Sort-based write-back** (FastSpMSpV): buffer 所有 (row_idx, val) pairs → global sort → sequential reduce。优点：完全避免 atomics。缺点：sort 阶段极慢（~43.3 GB/s），占 runtime >70%，需大临时 buffer。

3. **Hash-based write-back** (VDHA): CTA-private shared-memory hash table → local aggregation → bucket-order flush。优点：减少 conflicts（atomic-unit utilization 22.99%→12.82%），改善 coalescing（γ 0.744→2.607），hash cost 可通过 pipeline 隐藏。缺点：额外 SMEM 消耗，缺乏 locality 的矩阵收益有限。

4. **Segment-sum write-back** (Swift): warp-level shared-memory segment sum → 仅 segment 起点的线程做 atomicAdd。利用 positionIdx/offsetIdx 辅助索引，在 shared memory 中将同 row_idx 的 partial 预合并。优点：无需 hash probing（确定性索引，无 collision/fallback），segment sum 在 regular block 中极高效。缺点：依赖列排序预处理生成辅助索引；irregular block 不可用，仍需 direct atomicAdd。

从kernel调度角度拆解术语：

三种策略对比（it-2004, sparsity=0.1, A100）：

| 策略 | Bandwidth | Runtime占比 | Memory开销 |
|------|-----------|------------|-----------|
| Atomic | ~270 GB/s (A100, sparsity=0.1) | >30% | 无额外buffer |
| Sort | ~43 GB/s | >70% | 临时buffer存储全部pairs |
| Hash | improved | reduced | 16KB SMEM/CTA + flush buffer |
| Segment Sum | improved (确定性归约) | reduced (仅segment起点写回) | positionIdx/offsetIdx + SMEM buffer (~128B/warp) |

VDHA motivation benchmark：随着密度增加，atomic bandwidth 下降（sparsity 0.2→251 GB/s），sort bandwidth 始终~45 GB/s。Hash-based 通过 local aggregation 取两者之长：避免 sort 全局开销 + 减少 atomic conflict 次数。

术语一般如何实现？如何使用？

选择指导：Atomic 适用 conflict 少的矩阵或极度稀疏向量；Sort 适用需完全避免 atomics 场景但通常不如 hash；Hash 适用有 locality 的矩阵（small-world graphs），需足够 SMEM 和 fallback。Segment sum 适用列结构规整的矩阵（已做列排序+regular blocking），无需处理 hash collision 但依赖预处理。Adaptive SpMSpV 使用 ML-based kernel selector 在 atomic/sort/row-major 间选择；VDHA 提供 decision tree predictor（91.3% accuracy）判断 hash 是否更优。

涉及论文标题：
- VDHA: Vector-Driven Hash Aggregation for Sparse Matrix-Sparse Vector Multiplication on GPUs
- Swift: High-Performance Sparse-Dense Matrix Multiplication on GPUs

