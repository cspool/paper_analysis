## Fused Hybrid Attention Kernel (via Block Sparse Attention) (融合混合注意力Kernel)

术语是什么？

Fused Hybrid Attention Kernel 是 Elastic Attention 中用于在同一 kernel launch 中同时计算 Full Attention (FA) heads 和 Sparse Attention (SA) heads 的 GPU kernel，基于 Block Sparse Attention (BSA) Kernel（Guo et al., 2024, mit-han-lab）。与传统 Serial Dispatch（先 split tensor → 两个独立 kernel → merge）不同，Fused Kernel 将 routing decisions 直接传入 kernel 作为 metadata，kernel 内部通过 thread-block level branching 判断每个 head 的类型并执行对应 attention logic。

从kernel调度角度拆解术语。

```
# Serial Dispatch (Baseline)
r = Router(x_K)
I_full = where(r == 0); I_sp = where(r == 1)
Q_full = Q[:, I_full]; Q_sp = Q[:, I_sp]
O_full = FlashAttn(Q_full, K, V)        # kernel launch 1
O_sp = SparseAttn(Q_sp, K, V)           # kernel launch 2
O[:, I_full] = O_full; O[:, I_sp] = O_sp  # merge

# Fused Kernel (Elastic Attention via BSA)
r = Router(x_K)
m = Map(r)  # {h: FULL|SPARSE} metadata
O = BSA_Kernel(Q, K, V, m)             # single kernel launch
# Inside kernel (grid: Batch × Heads × SeqBlocks):
#   par for h in range(H):
#     block_type = m[h]
#     if block_type == FULL:
#       O[h] = FullAttnTile(Q[h], K, V)
#     else:
#       sp_indices = {sink=128, recent=2048, selected}
#       O[h] = SparseAttnTile(Q[h], K[sp_indices], V[sp_indices])
```

术语一般如何实现？如何使用？

基于 Block Sparse Attention Kernel（https://github.com/mit-han-lab/Block-Sparse-Attention）。配置：block_size=64, chunk_size=16384, sink_size=128。相比 Serial Dispatch 消除两种 overhead：(1) Memory overhead——不再需要 allocate/copy 非连续 tensor fragment（Q_full/Q_sp split）；(2) Kernel Launch & Scheduling overhead——单次 launch 避免 workload fragmentation。Grid 完整性（Batch×Heads×SeqBlocks）允许 GPU scheduler 最优分布 sequence blocks。当序列长度足够大时，sequence-dim parallelism 主导，加速效果显著。Router 额外延迟仅 ~0.196ms（不随 seq_len 增长）。代码：https://github.com/LCM-Lab/Elastic-Attention。

涉及论文标题：
- Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers
- InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU (Triton-based BSA kernel with FlashAttention-style prefill + FlashDecoding-style decoding + PagedAttention block KV management, combined with per-stage mask caching to reduce decoding BSA to ~2.2% of total attention latency)

---
