## Expert Sampling Strategy (Threshold / TopK / Threshold-TopK for MoE)

术语解释
DS-MoE 推理阶段从 dense training 的"全激活"切换到 sparse inference 的"部分激活"的策略选择方法。三种方法在 sparsity-性能 trade-off 和部署实用性之间提供不同平衡点。

术语是什么？
三种 expert sampling 策略定义了推理时如何从 N 个 expert 中选择 K 个激活：

1. **Threshold**（自适应阈值）：对每个 token 独立计算归一化概率 p_norm_i = S_i · N。选择所有 p_norm_i > ε 的 expert。优点：per-token 自适应——难度高的 token 自然激活更多 expert，简单 token 激活更少。缺点：batch 中不同 token 激活不同数量的 expert，需要 padding 或复杂调度，不利于 GPU 并行和 batch inference。

2. **TopK**（固定数量）：每层每 token 激活固定 K 个 expert（最高分的 K 个）。优点：batch 内所有 token 激活相同数量 expert，GPU 调度简单高效，适合生产部署。缺点：无法自适应——简单 token 浪费计算，复杂 token 可能不够。

3. **Threshold-TopK**（混合策略）：先用 Threshold 计算每个 token 应激活的 expert 数，再取 batch 内平均值作为统一 K 值进行 TopK 选择。优点：兼顾自适应（per-batch 调整）和 batch 效率（所有 token 统一 K）。缺点：需要先 forward 一次 Router 再决定 K 值。

从算法pipeline角度拆解术语：
```
# 1. Threshold Sampling
S = Softmax(h(X))                   # [B, N]
p_norm = S * N                       # normalized [B, N]
active_mask = p_norm > epsilon       # [B, N] bool
A = where(active_mask)               # variable-length per token
O = weight_sum([E_i(X) for i in A], [S[A[idx]] for idx])  # irregular computation

# 2. TopK Sampling (deployment-friendly)
S = Softmax(h(X))                    # [B, N]
A = topK(S, K)                       # [B, K], same K for all tokens
O = ParallelLinear(X, A, all_expert_weights)  # regular, efficient

# 3. Threshold-TopK (hybrid)
S = Softmax(h(X))                    # [B, N]
p_norm = S * N
per_token_K = sum(p_norm > epsilon, dim=-1)  # [B]
avg_K = round(mean(per_token_K))     # scalar
A = topK(S, avg_K)                   # [B, avg_K]
O = ParallelLinear(X, A, all_expert_weights)
```

术语一般如何实现？如何使用？
- **DS-MoE 默认**：使用 ε=0.48 的 Threshold 策略进行评估（追求最优 PPL 权衡）。部署时使用 TopK 或 Threshold-TopK。
- **性能比较**：DS-MoE-3B WikiText PPL → Threshold (best) > Threshold-TopK (practical) > TopK (deploy-friendly) at same active params
- **实际部署建议**：attention 层使用 dense（sparsity<40% 时 sparse overhead > dense），MLP 层使用 ParallelLinear + TopK
- **K 值选择**：DS-MoE-3B 使用 K=6, DS-MoE-6B 使用 K=4；更大模型可承受更高 sparsity（更小 K）

涉及论文标题：
- Dense Training, Sparse Inference Rethinking Training of Mixture-of-Experts Language Models
