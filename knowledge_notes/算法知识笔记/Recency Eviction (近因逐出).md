## Recency Eviction (近因逐出)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Recency Eviction 是基于时间近因的 KV cache 压缩策略：仅保留最近 W 个 token（local window）和前 S 个 attention sink token，其余 KV 全部 evict。KV cache 大小固定为 W+S（不随序列长度增长）。代表工作：StreamingLLM（发现 attention sink + streaming heads）、DuoAttention（仅部分 heads 设为 streaming）、MoA（natural text 训练 head specialization）、PruLong（改进的 head specialization）。优势：decoding 阶段 memory 恒定、与 chunked pre-filling 天然兼容、显著降低 KV Footprint。代价：可能遗忘远处重要信息。

从算法pipeline角度拆解术语。

```
// 纯 Recency Eviction（StreamingLLM 风格）
K_cache, V_cache = [], []  // 固定最大 W+S
for token t:
    k, v = W_K(t), W_V(t)
    K_cache.append(k); V_cache.append(v)
    if len > W+S:
        K_cache = concat([K_cache[:S], K_cache[-W:]])  // 保留 sinks + local
        V_cache = concat([V_cache[:S], V_cache[-W:]])

// 混合模式（PruLong/DuoAttention 风格，仅 streaming heads）
if z_lh == 0:  // streaming head
    K_attn = concat([K_cache[:S], K_cache[-W:]])
else:  // retrieval head: full cache
    K_attn = K_cache
```

术语一般如何实现？如何使用？

Attention mask 修改或 KV cache 截断实现。PruLong 默认 W=1024, S=128（对 128K 上下文有效）。StreamingLLM：https://github.com/mit-han-lab/streaming-llm。PruLong：https://github.com/princeton-pli/PruLong。

涉及论文标题：
- Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs

---
