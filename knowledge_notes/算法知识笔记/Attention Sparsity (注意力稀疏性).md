## Attention Sparsity (注意力稀疏性)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Attention Sparsity 指 Transformer 的自注意力分数矩阵 $A \in \mathbb{R}^{l \times l}$ 中只有极少数 token 对具有显著权重（非零），大部分 token 对的 attention score 接近于零的现象。这是 softmax 指数归一化后的固有特性：softmax 将指数归一化后的分数集中到极少数"热点"。KIVI 论文中 Llama-2-13B 的 attention sparsity 高达 84.3%，即超过 84% 的 attention 权重接近零。

Attention sparsity 是 KIVI per-token value cache 量化有效性的理论基础：由于 $t_O = \sum_j A_{ij} [X_V]_{j*}$，attention output 只是少数重要 token 的 value cache 加权组合。Per-token 量化将误差限制在每个 token 内，quantizing 不重要 token 不影响重要 token 的精度，因此 per-token value 量化误差远小于 per-channel（约 15×）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Attention sparsity 与 value cache 量化的关系：

```
# 设 attention score 稀疏（仅少数 token 显著）
A = [[0.01, 0.02, 0.85, 0.01, 0.02, ...]]  # token 2 是heavy-hitter

# Per-token value quantization:
# value cache token j → 被量化为 Q([X_V]_j)
# 误差: ε_j = [X_V]_j - Q([X_V]_j)
# end-to-end error: Δ = Σ_j A_j * ε_j
# 由于A_j≈0 for j≠heavy-hitter, Δ ≈ A_heavy * ε_heavy
# → 仅heavy-hitter token的量化误差被放大

# Per-channel value quantization:
# 跨token共享量化参数 → token间误差混合
# heavy-hitter value 受其他token量化参数影响 → 精度崩塌
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Attention sparsity 被用于多种优化：H2O 基于 sparsity 只保留 heavy-hitter token 的 KV Cache；Scissorhands 利用 sparsity 的持久性 evict 不重要 token；KIVI 利用 sparsity 解释 per-token value 量化有效性的理论基础。sparsity 通常通过 `torch.topk(A, k)` 或阈值过滤 `A > threshold` 检测 heavy-hitter。

涉及论文标题：
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

---
