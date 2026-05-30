## Heavy Hitter Oracle (H2O)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Heavy Hitter Oracle (H2O) 是 Zhang et al. (NeurIPS 2023) 提出的 KV cache 逐出方法。核心观察：attention 分数高度不均匀，少数 token（Heavy Hitters）贡献大部分 attention score。算法：每步 decode 后保留最近 w 个 token + 历史中累积 attention scores 最高的 (b-w) 个 token。

评分方式：score_k = Σ_q A_{q,k}（所有 query 对该 key 的累积注意力）。所有层使用统一 budget。局限性：token 级离散选择破坏语义连贯性，所有任务统一策略，不区分层间注意力密度差异。

伪代码：
```
for each decode step:
    A = softmax(Q @ K_cache^T / sqrt(d_k))
    scores += sum(A, dim=query)
    keep = topk(scores, budget-w) ∪ recent_tokens
    K_cache, V_cache = K_cache[keep], V_cache[keep]
```

术语一般如何实现？如何使用？

集成于 HuggingFace Transformers，与 FlashAttention 兼容。PyramidKV 仓库提供统一实现：https://github.com/Zefan-Cai/PyramidKV。WindowKV 将 H2O 作为 baseline 对比，在 KV cache=2048 下 LongBench avg：H2O 31.34 vs WindowKV 32.75 (Qwen2.5)，H2O 41.08 vs WindowKV 41.35 (LLaMA3)。

涉及论文标题：
- WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference

---
