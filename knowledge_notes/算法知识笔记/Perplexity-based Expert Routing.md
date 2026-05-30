## Perplexity-based Expert Routing

术语是什么？
Perplexity-based (PPL) Expert Routing 是 MergeME 提出的无需训练的 MoE 路由启发式。利用 perplexity 衡量各 expert 对输入的不确定度：PPL(x|θᵢ) = exp(−1/t · Σ log P(xⱼ|x_{<j}, θᵢ))。PPL 低 → confidence 高 → 路由权重大：α = SoftMax(top-K(1/PPL₁, ..., 1/PPLₗ))。仅需一次额外 forward pass（远小于 inference 时 generate 多 token 的多次 forward）。MergeME Table 2 验证 PPL 路由能有效导向领域专家（GSM8K → Math 43%, HumanEval → Code 43%）。Table 3: separate attention + PPL routing avg 8.08 vs merge attention + PPL routing 7.32 vs Dare Dense 7.11。

从算法pipeline角度拆解术语：
```
输入: prompt x (t tokens), experts [θ₁,...,θₗ]
for each expert i:
    PPL_i = exp(-1/t * Σ log P(x_j | x_{<j}, θ_i))
    conf_i = 1 / PPL_i
α = SoftMax(top-K(conf_1, ..., conf_l))     // routing weights
output = Σ α_i · expert_i.forward(x)          // 加权组合
```

术语一般如何实现？如何使用？
- 一次 no_grad forward pass 计算 log_softmax → PPL。开销 ≈ O(1) forward vs O(generate_tokens) forward。
- 局限性：(a) 跨领域输入可能选错 expert；(b) 所有 expert PPL 接近时区分度差。
- PPL 路由优于 Task Vector Routing（Table 3: 8.08 vs 7.05）。

涉及论文标题：
- MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs

---
