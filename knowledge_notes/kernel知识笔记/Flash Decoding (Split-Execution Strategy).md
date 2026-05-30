## Flash Decoding (Split-Execution Strategy)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Flash Decoding 是 FlashAttention 团队提出的长序列 decoding 优化 kernel，核心思想是将 attention 计算沿序列长度维度拆分（split）到多个 SM 上并行执行，最后通过 log-sum-exp reduction 合并各 SM 的 partial results。与 FlashAttention 的区别：FlashAttention 针对 training/prefill 场景的单 query 长序列优化（batch=1, long KV），而 Flash Decoding 针对 decoding 场景的单 KV 多 query 优化（batch>1, fixed KV cache）。

Split-Execution 的具体策略：将 KV cache 沿序列维度切成 num_splits 份，每个 SM 负责一份 KV 的 partial attention，独立计算 online softmax 的 (m_i, l_i, acc_i)，最后通过跨 SM 的 log-sum-exp reduction 合并出最终的 attention output。

从kernel调度角度拆解术语：

```
// Flash Decoding with block-sparse integration (ReSA Appendix A)
// Grid = (num_splits, num_kv_heads, batch_size)
for each (split_idx, kv_head_idx, batch_idx) in grid:
    // 1. Load query vectors
    q ← load_query(GQA_group)  // intra-GQA shared query
    
    // 2. Partition selected blocks across splits
    partial_blocks = partition(selected_block_indices, num_splits)[split_idx]
    
    // 3. Initialize online softmax accumulators
    mi ← -∞, li ← 1.0, acc ← 0
    
    // 4. Sparse attention over local blocks
    for block_id in partial_blocks:
        k ← load_keys(block_id)     // contiguous block: b tokens
        v ← load_values(block_id)
        qk ← (q @ k^T) × sm_scale
        qk[invalid_pos] ← -1e6
        // Online softmax update
        mi_new ← max(mi, row_max(qk))
        li_new ← li × exp(mi - mi_new) + row_sum(exp(qk - mi_new))
        acc ← acc × (li / li_new) × exp(mi - mi_new) + softmax(qk) @ v
        mi, li ← mi_new, li_new
    
    // 5. Store partial results
    out_partial[split_idx] ← acc
    logsum_partial[split_idx] ← mi + log(li)

// 6. Combine across splits (log-sum-exp reduction)
out = combine_logsumexp(out_partial, logsum_partial)  // cross-SM reduction
```

与标准 Flash Decoding 的区别：内层循环仅遍历 selected_block_indices 而非全部 KV blocks，这是 block-sparse attention 的关键性能来源。

术语一般如何实现？如何使用？

FlashAttention 官方库（https://github.com/Dao-AILab/flash-attention）提供 `flash_attn_with_kvcache` API。ReSA 在此基础上增加 block-sparse support：在 Flash Decoding 的 split-execution pipeline 中，partition 步骤按 block indices 而非连续 range 拆分。TileLang 实现该 variant，核心修改仅约 200 LOC（在 Flash Decoding 伪代码基础上增加 block_indices 的 partition 和 gather 逻辑）。

涉及论文标题：
- Rectified Sparse Attention
