## TOVA / Token Omission Via Attention (基于末端注意力的 KV Cache 动态逐出)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

TOVA（Oren et al., 2024）是一种 KV cache eviction 方法，核心思想：使用**最后一个 token 的 attention scores** 评估所有历史 token 的重要性，保留 attention 最高的 top-k 个 token。其理论基础是论文核心发现 "Transformers are Multi-State RNNs"——transformer decoder 可视为多状态 RNN，attention 机制维护有限数量的隐藏状态（保留的 token 的 KV pairs）。

与 H2O 使用累积 attention（跨多步历史）不同，TOVA 每步只使用最新 attention scores 做淘汰决策，无需维护 per-head 累积数组，计算和内存开销更低。与 TreeKV 的循环淘汰范围不同，TOVA 每次在全局选 top-k，产生强烈区域偏差（Figure 1 显示被选 token 集中于少数注意力高峰区域）。

从算法pipeline角度拆解术语。

**TOVA Decoding 流程**：

```
参数: cache_size, sink_size, recent_size

for t in 1..T:
    q, k, v = x[t] @ W_Q, x[t] @ W_K, x[t] @ W_V
    K_cache.append(k); V_cache.append(v)
    a = softmax(q @ K_cache^T / sqrt(d))

    if len(K_cache) > cache_size:
        mid_scores = a[0, sink_size:-recent_size]
        topk_idx = topk(mid_scores, k=cache_size - sink_size - recent_size)
        keep = [0:sink_size] + topk_idx + [-recent_size:]
        K_cache, V_cache = K_cache[keep], V_cache[keep]
```

**Annotations**: 与 H2O 的 `attn_accum += scores` 跨步累积不同，TOVA 的 scoring 不跨步——每步用最新的 `a[0, :]` 独立评估。Sink tokens（前几个）+ recent tokens（后几个）固定保留，仅中间区域参与动态淘汰。由于不跨步累积，老 token 的"历史高分"无法保护其不被淘汰（优劣参杂——减少偏差但可能误淘汰曾重要的 token）。

术语一般如何实现？如何使用？

论文 "Transformers are Multi-State RNNs" (arXiv 2401.06104)。在 Llama-2-7B 上 PG19 perplexity: 4k context PPL 7.00 (TOVA) vs 7.06 (H2O) vs 6.84 (Full)。16k 时 TOVA 7.15 vs TreeKV 6.91（3.6% 差距），显示全局贪心在长序列下的局限性。TOVA 的简化设计使其计算高效于 H2O（无累积维护），但 TreeKV 在超长序列（10M）和复杂上下文任务上远优。实践中 TOVA 与 FlashAttention 兼容性问题与 H2O 类似——需要多 pass attention 获取 attention scores。

涉及论文标题：
- TreeKV: Smooth Key-Value Cache Compression with Tree Structures
