## Sparse Attention Mask Caching (稀疏注意力Mask缓存)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Sparse Attention Mask Caching 是 InfiniteHiP 用于降低 decoding 阶段剪枝开销的优化策略。核心观察：在连续 decoding step 中，相邻 query token 的 attention pattern 高度相似（temporal locality），因此不需要每步都重新计算所有剪枝 stage 的稀疏 mask。通过为每个剪枝 stage 独立维护 mask 缓存和 refresh interval（n_refresh^(i)），仅周期性地更新 mask，可大幅降低 decoding 延迟。

从算法pipeline角度拆解术语：

```
// Decoding loop with mask caching
c^(i) = 0 for i = 1..N  // stage counters

For each decoding step:
  For each layer l = 1..L:
    For each stage i = 1..N:
      if c^(i) % n_refresh^(i) == 0:
        I^(l,i) = RunPruningStage(q_l, K, I^(l,i-1))  // 重新计算 mask
        // 记录 cache miss → 从 CPU UVM 加载缺失 key
      // else: 复用缓存的 I^(l,i)（跳过剪枝计算）
    O_l = BlockSparseAttention(q_l, K, V, I^(l,N))  // BSA with cached mask
  c^(i) = (c^(i) + 1) % n_refresh^(i)  // 更新所有 counter
```

**三种 refresh 配置及其效果（256K context decoding latency per token）**：
- Default: n_refresh = (16, 8, 4) → All cached: 110 µs/token → mask hit ratio Stage1 71.67%, Stage1&2 98.75%
- Fast: n_refresh = (32, 16, 8) → lower refresh frequency, 更低的平均延迟
- Flash: n_refresh = (96, 24, 8) → Stage1 几乎从不重算 → 最高 throughput（3M context 23.8 tok/s on L40S）

术语一般如何实现？如何使用？

实现要点：(1) 每个 stage 维护独立的 mask indices I^(l,i) 和 counter c^(i)；(2) 第一 stage（最昂贵，O(T_kv)）的 refresh interval 最大（16/32/96），因为其 mask 变化最慢；(3) 后续 stage（更便宜，O(constant)）的 refresh interval 较小；(4) 可在解码速度（更大的 interval）和 mask 精度（更小的 interval）间 trade off——论文显示增大 interval 对 NLU 性能影响极小（LongBench/∞Bench 中 3K-fast 和 3K 差异 <1%）。

涉及论文标题：
- InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU
