## Observation Window in KV Cache Eviction

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Observation Window 是 KV cache eviction 方法中用于计算 eviction metrics 的 queries 范围。在 LLM 推理中，prefill 阶段处理完整 input prompt 后生成 KV cache。为了决定 evict 哪些 KVs，需要评估每个 KV 的"重要性"——这通过聚合该 KV 收到的 attention 来实现。Observation Window 限制了聚合哪些 queries 的 attention：是聚合全部 queries（从第一个到最后一个 prompt token），还是仅聚合最后的 w 个 queries。

两种主要策略：(1) **Full observation**（H2O, KVC-full）：聚合所有过去 queries，$\sum_{i=1}^L A_{h,i,j}$，O(L²) 计算复杂度；(2) **Limited observation window**（SnapKV, KVC-w）：仅聚合最后 w 个 queries，$\sum_{i=L-w}^L A_{h,i,j}$，O(L) 计算复杂度，且可避免写完整 attention matrix 到 global memory（兼容 FlashAttention）。

KV-Compress 设计了两个变体——KVC-full（全部 queries，排除 local window v=10）和 KVC-w（window w=8 + max-pooling p=7），发现：(1) KVC-full 在多数 subtask 上表现最好，但计算开销大（quadratic scaling），且在某些任务（SAMSum）上严重退化；(2) KVC-w-8 在整体上是更实用的选择。

从算法pipeline角度拆解术语：

```
# Full observation (H2O-style, KVC-full):
for query i in 1..L:
    for key j in 1..i:  # causal
        M[j] += A[i,j]  # 所有 queries

# KVC-full with excluded queries (Equation 19):
for query i in (j+v)..L:  # skip v local queries after key j
    M[j] += (A[i,j])^2

# Limited observation window (SnapKV-style, KVC-w):
for query i in (L-w)..L:  # only last w queries
    for key j in 1..i:
        M[j] += A[i,j]
# then max-pool (window along key dim, size p)

# KV-Compress KVC-w-8 (Equation 10 with squared attention):
for query i in L-8..L:
    for key j in 1..i:
        M[j] += (A[i,j])^2
M = max_pool(M, kernel_size=7)
```

术语一般如何实现？如何使用？

Observation window 策略的选择影响三个维度：(a) 计算开销——full observation O(L²) vs limited O(L)；(b) 指标质量——full observation 理论上信息更全，但近期 queries 的 attention pattern 可能更好预测 decoding 阶段的 attention；(c) 与 FlashAttention 的兼容性——limited window 可避免物化完整 attention matrix。

KV-Compress 实验表明：w=8 优于 w=32（Mistral 实验），较小的 window 更聚焦于与 decoding 行为相似的结尾 queries。max-pooling（p=7）沿 key 维度平滑 metric，保留 heavy-hitter 附近的 context KVs。

适用场景：所有基于 attention score 聚合的 KV eviction 方法。对于 prompt 结构为 "长文档 + 短问题" 的场景，结尾 queries（问题区域的 attention）能很好预测 decoding 阶段的 attention pattern——此时 limited window 有效。对于需要全局信息检索的任务，KVC-full 可能更优，但计算开销大。

涉及论文标题：
- SnapKV: LLM Knows What You are Looking for Before Generation
- KV-Compress__Paged_KV-Cache_Compression_with_Variable_Compression_Rates_per_Attention_Head
- TriAttention: Efficient Long Reasoning with Trigonometric KV Compression （TriAttention 批判了 post-RoPE 下的 Observation Window 的根本局限：因 RoPE 旋转使 query 朝向随位置变化，仅最近约 25 个 query 有效，窗口无法通过增大改善。TriAttention 回到 pre-RoPE 空间，利用 Q/K 聚集（Q/K Concentration）和三角函数级数预测 attention，完全绕过观察窗口）

---
