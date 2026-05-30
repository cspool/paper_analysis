## Output-aware Pruning for KV Cache (输出感知的KV缓存剪枝)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Output-aware Pruning 是一种使用 scoring metric 作为 proxy 估计 KV cache 中每个元素对 attention output 贡献的剪枝技术。核心思想：不单独看元素本身的 magnitude，而是将元素与对应输入（query 或 attention score）相乘，利用矩阵乘法链式关系评估对最终 output 的实际贡献。

Mustafar 中的两种形式：

1. **Key cache output-aware**: S = |K| ⊙ broadcast(Σ_{t} |Q_t|)，累加 32 个 query 的 L1 绝对值与 Key element-wise 乘积。Q×K attention 中每个 K 元素贡献正比于 |K_j| × |Q_j|。

2. **Value cache output-aware**: 
   - Per-channel: S = |V| ⊙ broadcast(Σ|α_t|)，需 attention scores（与 FlashAttention 不兼容——FA 不物化完整 attention matrix）。
   - Per-token: 等价于 magnitude-based——同 token 内所有 V 元素乘以同一 α_i，排序不变。

从算法pipeline角度拆解：

```
# Key cache output-aware pruning score
Q_accum = sum(|Q_t| for t in 0..31)     # shape [d]
for i in 1..T-W:
    score_K[i] = |K_cache[i]| * Q_accum  # element-wise
    mask_K[i] = topk_mask(score_K[i], d*(1-s))

# Value cache per-channel output-aware (需attention scores)
Attn_accum = sum(|attn_scores[t]| for t in 0..31)  # shape [T]
for c in 1..d:
    score_V[:,c] = |V_cache[:,c]| * Attn_accum     # per-channel
```

术语一般如何实现？如何使用？

Key cache 推荐 magnitude-only（精度已足够）；Value cache 推荐 per-token magnitude（自动等价 output-aware，无需 attention scores 且与 FlashAttention 兼容）。计算开销约 O(Td) per head，通常小于 2% of attention compute。

涉及论文标题：
- Mustafar__Promoting_Unstructured_Sparsity_for_KV_Cache_Pruning_in_LLM_Inference

---
