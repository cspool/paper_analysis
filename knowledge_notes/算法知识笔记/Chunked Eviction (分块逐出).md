## Chunked Eviction (分块逐出)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Chunked Eviction 是将 post-fill eviction 方法适配到 chunked pre-filling 场景的技术：在每个 pre-fill chunk 处理后立即执行 KV eviction，而非等所有 chunk 处理完毕。两种变体：Naive（chunk 自身末尾 k token 计算重要性）和 Patched（拼接 prompt tail 提供全局重要性信号）。Chunked Eviction 显著降低 KV Footprint——因 KV 在 pre-fill 过程中即被提前移除。

从算法pipeline角度拆解术语。

**Naive Chunked Eviction**：
```
for each chunk c:
    K_c, V_c = forward(chunk_tokens[c])
    scores = attention_score(last_k_tokens_of_chunk, K_c)  // 局部信号
    keep_idx = top_k(smoothed_scores, p × len(K_c))
    K_cache.append(K_c[keep_idx])  // 非 keep 的 KV 被 evict
```

**Patched Chunked Eviction**：
```
for each chunk c:
    X_patched = concat([chunk_tokens[c], prompt[-k:]])
    K_patched, V_patched = forward(X_patched)
    scores = attention_score(prompt_tail, K_patched)  // 全局信号
    keep_idx = top_k(smoothed_scores, p × len(K_c))
    K_cache.append(K_c[keep_idx])
    // 丢弃 patched token 的 KV（最后 chunk 除外）
```

术语一般如何实现？如何使用？

GQA 场景下需在 KV group 内 mean-pool attention 后统一选择，避免为每个 query head 独立选择 → 8× 内存节省（Llama-3.1-8B GQA）。Patched PyramidKV + mean-pool 在 RAG（<34% KV footprint）和 LongQA（<35%）上取得所有方法中最优。代码开源：https://github.com/princeton-pli/PruLong

涉及论文标题：
- Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs

---
