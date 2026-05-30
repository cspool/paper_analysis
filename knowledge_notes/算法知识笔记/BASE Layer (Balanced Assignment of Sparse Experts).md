## BASE Layer (Balanced Assignment of Sparse Experts)

术语解释
由 Lewis et al. (ICML 2021) 提出，将 MoE 的 token-to-expert 分配建模为线性分配问题（Linear Assignment），在约束条件下最大化 token-expert affinity 总和，保证每个专家处理等量 tokens。

术语是什么？
给定 affinity S ∈ R^{T×N}，约束每个 expert 恰好 B 个 tokens，每个 token 恰好 1 个 expert，目标 max Σ S_{i, assigned(i)}。解法：Hungarian algorithm (O(T^3)) 或 Sinkhorn (S-BASE)。

```
assignment = linear_sum_assignment(-S)     # [T] -> expert_idx
# 每个专家恰好 B tokens，每个 token 恰好 1 expert
# S-BASE: Sinkhorn normalization → soft assignment → harden during training
```

术语一般如何实现？如何使用？
- 严格保证负载均衡（数学约束），不需要 auxiliary loss
- S-BASE 通过 Sinkhorn 迭代提供可微训练版本
- 局限：每个 token 仅 1 个 expert（vs top-K 可使用多个）
- 适用场景：对负载均衡有严格要求的分布式训练

涉及论文标题：
- A Survey on Mixture of Experts in Large Language Models
