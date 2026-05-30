## Hybrid Sparse-Dense Attention Architecture

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Hybrid Sparse-Dense Attention Architecture 是一种将少量 dense attention head 与大量 sparse attention head 组合同一层中的架构设计。在 MoSA 中，hybrid 模型保留少量 dense head（实验确定最优为 4 个），其余 head 替换为 MoSA sparse head。Dense head 计算全部 T 个 token 的标准 attention（T×T），sparse head 仅处理 k << T 个 token（k×k）。

核心动机：纯 sparse attention 存在 router-attention 联合训练的稳定性问题——训练初期 router 随机选择，attention 学不到有用模式，导致 router 无梯度信号，形成恶性循环。少量 dense head 提供稳定的全局梯度流和语义信息，帮助 router 收敛到有意义的 token selection。

ablated 结论：(1) 0 dense head → 性能崩溃（Tiny 模型 ρ=16 时 ppl 从 22.46 升至 29.76）；(2) optimal dense head count = 4，与 sparsity ρ 无关；(3) >4 dense head → 占用 FLOP budget 导致可用的 MoSA head 减少，perplexity 回升。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Hybrid MoSA Layer
Input: X ∈ R^{T×h}

# Dense Heads (H_dense = 4, standard MHA)
Y_dense = 0
for i in 1..H_dense:
  Q, K, V = X @ W^Q_i, X @ W^K_i, X @ W^V_i  # all T tokens
  A = softmax(Q @ K^T / √h' + M_causal) @ V
  Y_dense += A @ W^O_i

# MoSA Sparse Heads (H_mosa, k = T/ρ tokens per head)
Y_sparse = 0
for i in 1..H_mosa:
  r = σ(X @ W^r_i); r_topk, I = TopK(r, k)
  X^s = X[I]
  Q, K, V = X^s @ W^Q_i, X^s @ W^K_i, X^s @ W^V_i
  M[a,b] = 0 if I[a] >= I[b] else -∞
  A = softmax(Q @ K^T / √h' + M) @ V
  X^o = diag(r_topk) @ A @ W^O_i
  Y_sparse = scatter_add(Y_sparse, X^o, I)

Output: Y = Y_dense + Y_sparse
```

**FLOP-matching 规则**：max H_mosa s.t. H_dense·FLOP_dense + H_mosa·FLOP_mosa ≤ H_baseline·FLOP_dense

术语一般如何实现？如何使用？

实现：修改 Transformer layer，同时实例化 dense heads（标准 MHA 逻辑）和 MoSA heads（router + sparse attention）。Optimal dense head count 通过 ablating ρ=4 和 ρ=16 在不同 dense head count (0-9) 下的 perplexity 确定——结果一致为 4，说明稳定化效果与 sparsity 无关。KV-cache: KV_total = T·H_dense + k·H_mosa，在 perplexity-matched 设定下 KV-cache 减少 51-70%。训练：与标准 transformer 相同（Adam, lr=0.00025, gradient clipping 0.25, warmup 4k steps）。

涉及论文标题：
- Mixture of Sparse Attention: Content-Based Learnable Sparse Attention via Expert-Choice Routing
