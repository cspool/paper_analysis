## Query-to-Block Dispatch (MoE-style Token Dispatch for Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Query-to-Block Dispatch 是 MoBA 中借鉴 MoE token dispatch 机制实现的 attention 计算调度技术。核心操作：根据 top-k gating 的结果（稀疏矩阵 G ∈ {0,1}^{N×h×n}），将 queries 按它们被分配到的 KV blocks 重新分组排列，使得同一 KV block 的 queries 被连续放置，便于后续 FlashAttention varlen 高效处理。

从kernel调度角度拆解术语：
```
输入：Q ∈ R^{N×h×d}, G ∈ {0,1}^{N×h×n}  # G[q, head, block] = 1 if selected
输出：Q^m (packed queries grouped by block), cu_seqlens (segment boundaries)

Algorithm (index_select_moba_attn_block):
  for each KV block i in 1..n:
      for each head:
          mask = G[:, head, i]  # [N], boolean
          selected_queries = Q[mask, head, :]  # variable count per block
          append selected_queries to Q^m
          record segment boundary in cu_seqlens

实际实现使用 scatter/gather + prefix sum:
  counts = sum(G, dim=0)  # [h, n] query counts per head per block
  offsets = cumsum(counts)  # prefix sum for packed layout
  for each (head, block) pair:
      scatter Q indices according to G into Q^m at offsets[head, block]
```
类似于 MoE 中 `token_dispatch` 将 tokens 路由到 experts，这里将 queries 路由到 KV blocks。

术语一般如何实现？如何使用？

在 MoBA 中实现为 `index_select_moba_attn_block(Q, K̃, Ṽ, G)` 函数。关键优化：
- 使用 PyTorch `nonzero()` + `index_select()` 或自定义 CUDA kernel 实现高效 gather
- cu_seqlens 直接从 counts 累加得到（O(n) 而非 O(N)）
- 当前 block attention 通过 `get_self_attn_block` 单独处理（不需要 dispatch，每个 query 固定属于自己所在的 block）
- 对于 GQA 模型，按 KV head 维度 dispatch（而非 query head），减少 dispatch groups

涉及论文标题：
- MoBA: Mixture of Block Attention for Long-Context LLMs

---
