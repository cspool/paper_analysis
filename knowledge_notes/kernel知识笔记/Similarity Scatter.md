## Similarity Scatter

术语是什么？通过联网搜索让回答具体和精准。
Similarity Scatter是Focus SIC中的streaming重建操作，与Similarity Gather配对构成gather-scatter循环。Scatter在GEMM对concentrated (deduplicated) vectors执行计算后，根据上一层Similarity Gather产生的similarity map将compact vectors的partial sums复制/分发回所有原始token位置，在output-stationary buffer中累加得到正确的full-size output tile。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Similarity Scatter的per-GEMM-iteration执行伪代码：
```
# Input: concentrated input tile: p vectors × K (p < 1024)
#        weight tile: k × n (k=32, n=32)
#        similarity_map[m] from previous Similarity Gather

def similarity_scatter(compact_input[p][K], weight[k][n], sim_map[m]):
    # GEMM execution on compact vectors (output-stationary outer loop)
    output_tile = zeros(m, n)  # m=1024, n=32 — full size accumulator
    for k_start in 0..K-1 step k:  # outer loop: ⌈K/k⌉ iterations
        # Inner loop: weight stationary GEMM
        partial_sum = compact_input[:, k_start:k_start+k] @ weight[k_start:k_start+k, :]
        # partial_sum: p × n (p compact vectors, each n=32 dims)

        # Scatter: replicate compact vector results to original positions
        for orig_idx in 0..m-1:
            compact_idx = sim_map[orig_idx]
            # accumulate: each original position gets its representative's partial
            output_tile[orig_idx, :] += partial_sum[compact_idx, :]

    # After all K iterations: output_tile contains correct full results
    # Then invoke Similarity Gather for next-layer compression
    return output_tile
```
Key: Scatter使用2a-wide accumulator (64 when a=n=32)，支持concurrent accumulation of reconstructed + streaming outputs。因为不同sub-tile可能有不同的compact vector subsets（每个subset代表多个original tokens），直接accumulation会因semantic aliasing产生错误——Scatter通过similarity map的index-based replication解决此问题。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Similarity Scatter在Focus中实现为hardware accumulation logic：2a-wide (64-wide) accumulator array → index mapper (lookup sim_map per original position) → replication logic (broadcast compact partial sum to mapped positions) → output-stationary buffer (512KB for full m×n tile)。Scatter的reconstruction overhead negligible（index lookup + parallel accumulation），不require additional memory allocation。Scatter与GEMM inner loop pipeline重叠：每个inner loop iteration生成partial sums后立即scatter→accumulate，不等待所有iterations完成。开源RTL见https://github.com/dubcyfor3/Focus。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

