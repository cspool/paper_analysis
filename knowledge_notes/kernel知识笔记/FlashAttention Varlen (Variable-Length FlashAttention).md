## FlashAttention Varlen (Variable-Length FlashAttention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

FlashAttention Varlen 是 FlashAttention 的变长序列版本，支持在一次 kernel call 中对不同长度的序列组分别计算 attention，避免 padding 带来的无效计算。输入使用 `cu_seqlens`（cumulative sequence lengths）指定各序列的边界，Q/K/V tensors 为 packed 形式（所有序列拼接为一个 tensor）。

在 MoBA 中，varlen 的作用至关重要：经过 top-k gating 后，不同 KV block 被分配的 query 数量不同——有些 block 被很多 queries 选中（热门 block），有些被很少选中（冷门 block）。FlashAttention varlen 允许对不同 (query_group, kv_block) 对分别使用适配其实际长度的 FlashAttention 计算，而非 padding 到相同长度。

从kernel调度角度拆解术语：
```
输入：
  Q^m ∈ R^{total_queries × h × d}  # packed queries
  K̃^m ∈ R^{total_keys × h × d}     # packed keys
  Ṽ^m ∈ R^{total_keys × h × d}     # packed values
  cu_seqlens_q = [0, n1, n1+n2, ..., total_queries]  # query segment boundaries
  cu_seqlens_k = [0, m1, m1+m2, ..., total_keys]      # key segment boundaries
  max_seqlen_q, max_seqlen_k

CUDA kernel:
  grid = (num_segments * num_heads, ceil(max_seqlen_q / 32), 1)
  for each thread block:
      seg_id = blockIdx.x / num_heads
      q_start, q_end = cu_seqlens_q[seg_id], cu_seqlens_q[seg_id+1]
      k_start, k_end = cu_seqlens_k[seg_id], cu_seqlens_k[seg_id+1]
      Q_tile = Q[q_start:q_end]  # variable-length segment
      K_tile = K[k_start:k_end]  # corresponding KV block
      V_tile = V[k_start:k_end]
      # Standard FlashAttention tiling on this segment
      O_seg = flash_attn_tiling(Q_tile, K_tile, V_tile, causal=False)
```
关键：每个 thread block 从 cu_seqlens 推导自己的 Q/K 范围，无需 padding。causal=False（因为 causal 约束已在 block-level routing 中保证）。

术语一般如何实现？如何使用？

FlashAttention-2 提供 `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal)` API。在 MoBA 中使用两个 varlen call：
- Self-attention: `causal=True`（当前 block 内需要 causal mask）
- MoBA attention: `causal=False`（历史 block routing 已保证因果性）

也广泛应用于 vLLM/PagedAttention 的 continuous batching、sequence packing 等长上下文推理场景。FlashAttention-3 在 H100 上对 varlen 做了进一步优化（dynamic split selection）。

涉及论文标题：
- MoBA: Mixture of Block Attention for Long-Context LLMs
