## Encoding Phase One-Eviction (编码阶段一次性淘汰)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Encoding Phase One-Eviction 是 NACL 的 KV Cache 淘汰范式创新。传统方法（H2O、MSRNN）在 generation 阶段每步贪心淘汰 O(p+T)。NACL 将淘汰移至 encoding 阶段一次性完成：利用完整 attention matrix A ∈ R^{p×p} 做全局最优淘汰，压缩 KV cache 用于全部 generation。Generation 阶段仅每 m 步轻量维护。Long-context 下 T ≪ p，复杂度从 O(p+T) 降至 O(1)。

从算法pipeline角度拆解术语：

传统 step-by-step（H2O）:
```
for each token t: K_cache.append(K_t,V_t); scores += attn_scores; K_cache = topK(scores, C)
复杂度: O(p+T) per eviction
```

NACL one-eviction:
```
Encoding: S = F_score(A_full, C); K_cache = K_prompt[S]
Generation: for each t, if t%m==0: light eviction    # 轻量维护
复杂度: O(1) (T ≪ p)
```

消融：移除 global eviction → -1.3% short-text, -1.5% long-text。

术语一般如何实现？如何使用？

在 prefill 完成后、generation 前插入一次性淘汰 hook。需 access encoding 阶段的 attention matrix（通过 FlashAttention-2 LSE 重算或仅 proxy tokens 重算）。NACL Algorithm 1 描述完整两阶段流程。

涉及论文标题：
- NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time

---
