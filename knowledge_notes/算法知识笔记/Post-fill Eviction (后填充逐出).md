## Post-fill Eviction (后填充逐出)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Post-fill Eviction 是 KV cache 压缩方法的一类：在 pre-fill 阶段完全结束后才基于 attention scores 启发式一次性选择并逐出 KV。代表工作：H2O（累积 attention score）、PyramidKV（pyramidal budget）、SnapKV（最后 token attention）、FastGen。根本缺陷：pre-fill 阶段保留所有 KV → 高峰值内存 → KV Footprint 几乎无 reduction。在 chunked pre-filling 成为标准实践（SGLang 默认 8192 chunks）后问题更突出。PruLong 论文通过 Chunked Eviction 解决了这一缺陷。

从算法pipeline角度拆解术语。

```
// 标准 Post-fill Eviction 流程
for token in prompt:  // Pre-fill: 全部保留
    K_cache.append(W_K(token)); V_cache.append(W_V(token))
// Pre-fill 后一次性 evict
scores = moving_average(Σ last_k_queries softmax(Q @ K^T / √d))
keep_idx = top_k(scores, budget)
K_cache = K_cache[keep_idx]; V_cache = V_cache[keep_idx]
// Decoding: 使用压缩后的 cache
while not EOS: ...
```

术语一般如何实现？如何使用？

Hook 方式集成到 HuggingFace Transformers attention 层。PyramidKV: https://github.com/FYYFU/PyramidKV；SnapKV: https://github.com/FasterDecoding/SnapKV。论文将其适配为 Chunked Eviction 后大幅降低 KV Footprint。

涉及论文标题：
- Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs

---
