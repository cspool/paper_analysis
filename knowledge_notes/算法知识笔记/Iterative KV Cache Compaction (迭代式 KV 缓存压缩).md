## Iterative KV Cache Compaction (迭代式 KV 缓存压缩)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Iterative KV Cache Compaction 是 LaCache 提出的支持无限长度连续生成的 KV cache 管理策略。当 KV cache 达预设容量上限后，对已压缩 cache 再次应用压缩算法，逐次释放空间。随迭代次数增加，老 token 经历更多轮压缩（被更激进淘汰），新 token 保留更多。内存复杂度 O(1)（constant cache size）。

与 StreamingLLM sliding window 的关键区别：不是简单丢弃最早 token，而是通过 cascaded 压缩实现渐进信息衰减——老 token 在多轮 ladder eviction 中逐步失去各层覆盖，而非一次性消失。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
for each decode step:
    append new KV to cache
    if cache_size >= budget:
        for layer l in 1..L:
            # 对已压缩 cache 再 apply ladder pattern
            K_cache[l] = ladder_evict(K_cache[l], S, O)
            V_cache[l] = ladder_evict(V_cache[l], S, O)
            # 老 token 在 ladder 左端被淘汰，新 token 在右端保留
    output = attention(Q_new, K_cache, V_cache)
```

术语一般如何实现？如何使用？

与 ladder-shaped pattern 联合部署，每步 decode 后检查 cache size 并触发。PG19 实验：LaCache 连续生成 10M+ tokens 且 PPL 保持稳定；Full KV cache 在 160K tokens 即 OOM。实现极简——仅在 HuggingFace Transformers attention 层中增加一个 cache size check 和 ladder eviction 触发。代码开源：https://github.com/GATECH-EIC/LaCache。

涉及论文标题：
- LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models

---
