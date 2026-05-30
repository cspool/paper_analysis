## Query-Agnostic KV Cache Eviction (查询无关的 KV Cache 淘汰)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Query-Agnostic KV Cache Eviction 是一类在 prefill 完成后不依赖任何 query 信息即可决定哪些 KV pairs 保留/淘汰的压缩范式。与 query-aware 方法（如 SnapKV、PyramidKV，利用 trailing window 中的 query token 计算 attention-based 重要性分数）相反，query-agnostic 方法的评分仅基于 context 自身，压缩后的 KV cache 可跨任意 query 复用，无需重复 prefill。

KVzip 是该范式的代表性方法，核心 insight：Transformer 天然作为 encoder-decoder——将 context 编码进 KV pairs（类比 Zip 压缩）；使 LLM 模拟重建原始上下文时，接收高 attention 的 KV pairs 恰好也是下游任务所需的关键信息。DuoAttention 的 context-independent head-level eviction 也属于 query-agnostic 范畴，但 DuoAttention 需数小时 8-GPU 优化 head scores，KVzip 仅需数次 forward pass 一分钟内完成。

从算法pipeline角度拆解术语：

**Query-Agnostic vs Query-Aware 对比**：

```
// === Query-Aware (SnapKV) ===
KV_c = Prefill(context || query_window)  // query 参与 prefill
scores = pool(softmax(Q_query_window @ K_context^T))
KV_compressed = topk_filter(KV_c, scores, budget)
// 问题：压缩 cache 对当前 query 过拟合，新 query 需重新 prefill

// === Query-Agnostic (KVzip) ===
KV_c = Prefill(context)                  // 仅 context，不含 query
input = "Repeat the previous context:" + context
scores = max_cross_attn(Forward(input, use_cache=KV_c))
KV_compressed = topk_filter(KV_c, scores, budget)
// 结果：压缩 cache 可跨任意 query 复用，单次 prefill 服务所有 query
```

术语一般如何实现？如何使用？

适用于 KV cache 可离线准备的场景：个性化对话代理（保留用户指令和对话历史）、企业级预计算文档 KV cache 检索、固定知识库多轮问答等。与 FlashAttention-2 兼容，通过 chunked scoring 扩展到长上下文（O(m·n_c) 线性复杂度）。支持与 KV cache 量化（QServe W8A8KV4）正交集成。代码开源：https://github.com/snu-mllab/KVzip。

LagKV 也是 query-agnostic 范式的一种实现，但其评分机制完全不同：不依赖上下文重建的反向 attention，而是利用 token-wise locality 和 lag-relative 归一化——用下一分区的 K/V 统计量归一化当前分区后计算 channel-wise std 作为重要性分数。这种方法进一步消除了 KVzip 仍需多次 forward pass 的成本，仅需一次 forward 即可完成压缩。

涉及论文标题：
- KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction
- LagKV: Lag-Relative Information of the KV Cache Tells Which Tokens Are Important

---
