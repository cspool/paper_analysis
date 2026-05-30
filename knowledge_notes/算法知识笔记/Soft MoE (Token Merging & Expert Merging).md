## Soft MoE (Token Merging & Expert Merging)

术语解释
Soft MoE 是一类保持完全可微的 MoE 变体，避免离散的 top-k 专家选择。分为 Token Merging（Puigcerver et al. 2023，仅 vision）和 Expert Merging（SMEAR 2023 → Lory 2024，支持自回归 LM）。

术语是什么？
离散门控的主要问题是不可微且负载不均。Soft MoE 通过软合并避免这些问题：
1. **Token Merging**: 计算所有 tokens 的加权平均（权重依赖于 token 和专家），每个专家处理一个"合并 token"
2. **Expert Merging (SMEAR/Lory)**: 对所有专家参数做加权平均得到单一"合并专家"，然后执行单次 FFN 前向

关键公式（Expert Merging）：y = FFN(x; Σ_i e_i · θ_i)，替代 y = Σ_i e_i · FFN(x; θ_i)

从算法pipeline角度拆解术语。
```
# Expert Merging (SMEAR/Lory)
def soft_merged_moe(x, experts, router):
    logits = router(x)                                  # [batch, N]
    weights = softmax(logits, dim=-1)                   # [batch, N]
    # Merge experts in parameter space
    W1_merged = sum(w[b,i] * experts[i].W1 for i, b)
    W2_merged = sum(w[b,i] * experts[i].W2 for i, b)
    h = activation(x @ W1_merged.T)                      # [batch, d_ffn]
    y = h @ W2_merged.T                                  # [batch, d_model]
    return y
```

术语一般如何实现？如何使用？
- Lory: 首个扩展到自回归 LM 预训练（150B tokens, 32 experts），+13.9% perplexity vs dense
- 因果分段路由：用上一段 hidden state 计算本段路由权重，保持自回归性
- 专家自然学习到 domain-level specialization
- 局限：token 级离散路由 MoE 仍有性能优势

涉及论文标题：
- A Survey on Mixture of Experts in Large Language Models

---
