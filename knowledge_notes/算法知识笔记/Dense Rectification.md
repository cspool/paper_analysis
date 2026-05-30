## Dense Rectification

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Dense Rectification（密集校正）是 ReSA 论文的核心机制：在稀疏解码每生成 f 个 token 后，将这 f 个 token 通过一次并行 dense attention forward pass 重新编码，刷新 KV cache 中对应的条目，将稀疏误差的累积范围限制在最近 f 步以内。其设计关键：(a) 批量并行——f 个 token 拼成 mini-batch，一次 dense forward 同时重编码；(b) 同步刷新 block key cache（block descriptors），否则新稀疏解码的 block selection 基于过时描述符会加剧误差；(c) 频率 f 权衡质量/效率——f=32 近 dense 精度，f=128 保留大部分增益但 overhead 更低。

从算法pipeline角度拆解术语：

```
Algorithm: Rectified Sparse Decoding
Input: P(prompt), M(model), f(frequency), T(max steps)

K, B = Prefill(P)  // dense prefill
for i = 1 to T:
    t = SparseForward(G[i-1], K, B)  // GBSA with block selection
    G.append(t); K.update(t); B.update(t)
    if i % f == 0:
        K, B = DenseForward(G[i-f:i], K, B)  // batch rectification
```

Memory access 成本：Avg(mem) = mem(KV cache) × (1/b + p + 1/f)，三项分别对应 block descriptor scan、sparse attention、rectification 摊销。256K context 下 rectification 占 attention 总延迟 32.7%。

术语一般如何实现？如何使用？

需支持在同一 session 中交替使用 sparse 和 dense attention kernel，dense forward 仅作用于最近 f 个 token。ReSA kernel 基于 TileLang + Flash Decoding split-execution。与 speculative decoding 的区别：rectification 无条件接受 sparse decoding 产生的所有 token 并用 dense 刷新 KV，避免 per-token accept/reject latency 惩罚（ReSA Table 3: 平均 1.92× faster than self-speculation）。天然兼容 continuous batching 和 chunked prefill。

涉及论文标题：
- Rectified Sparse Attention
