## Top-k Gating for Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Top-k Gating for Attention 是将 MoE 中 top-k gating 机制应用于 attention 的 block 选择技术。每个 query token 计算与所有 KV blocks 的 affinity score 后，通过 top-k 选择最相关的 blocks 进行 attention。与 MoE 中 expert routing 的区别：这里路由的是 attention context（KV blocks），而非 FFN experts；gating 是 parameter-free 的（使用 query-key 内积而非可学习权重矩阵）。

从算法pipeline角度拆解术语：
```
s_i = ⟨q, mean_pool(K[I_i])⟩  # affinity score, query-to-block relevance
g_i = 1 if s_i ∈ Topk({s_j | j∈[n]}, k) else 0  # binary gating
I = ∪_{g_i>0} I_i  # selected KV indices for this query
```
Critical design elements:
- **Causality enforcement**: s_i = -∞ for future blocks (pos(q) < i·B)
- **Current block as shared expert**: 强制 g_i = 1 for query's own block, 类比 MoE shared expert
- **Parameter-free**: 不引入可训练参数，gating 仅依赖 Q 和 block-mean-pooled K

MoBA 证明 sliding window attention 和 attention sink 都是 top-k gating 的特例——gating 固定选择最近 blocks（SWA）或首尾 blocks（Sink）。

术语一般如何实现？如何使用？

实现为矩阵操作：\(S = Q @ K̄^T\)（batch matmul），\(G = \operatorname{topk}(S + M, k)\)（GPU topk）。在 MoBA 中使用，block_size 和 top-k 是主要超参。典型配置：训练用 B=512/k=3（short context），推理用 B=4096/k=12（1M context）。额外计算开销 <1% attention FLOPs。

涉及论文标题：
- MoBA: Mixture of Block Attention for Long-Context LLMs
