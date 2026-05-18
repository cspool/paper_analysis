## GEMM Tiling with Streaming Concentration

术语是什么？通过联网搜索让回答具体和精准。
GEMM Tiling with Streaming Concentration是Focus的核心运行时调度策略，将systolic array的标准GEMM tiling（m=1024, n=32, k=32）与on-chip streaming concentration紧密对齐。每个GEMM tile的输出不直接写回DRAM，而是立即stream到SIC做vector-level similarity detection和deduplication，仅deduplicated vectors + similarity map写回DRAM。后续GEMM对compact vectors执行计算，通过similarity map做scatter reconstruction。这种tile-local compression-while-computing模式使compression与GEMM pipeline完全融合，消除global token-wise方法的DRAM往返。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
GEMM Tiling with Streaming Concentration的完整tile-level pipeline：
```
# Focus accelerator GEMM tiling + concentration pipeline

for each layer:
    # === Attention layers: SEC semantic pruning ===
    if is_attention_layer:
        # GEMM: QK^T tiling (standard)
        S_tile = systolic_array(Q_tile, K_tile^T)  # output: m × (M+T)
        S_softmax = SFU.softmax(S_tile)  # special function unit

        # SEC streaming: importance + top-k (overlapped with next GEMM)
        importance = SEC.importance_analyzer(S_softmax[T:, :M])  # T×M → 1×M
        top_k_indices = SEC.streaming_bubble_sort(importance, k)

        # Pruned P×V GEMM: only load retained tokens
        P_tile = systolic_array(Q_retained, K_retained^T)  # smaller M
        V_output = systolic_array(P_tile, V_tile)

    # === FC layers: SIC scatter-gather ===
    if is_fc_layer:
        # Previous layer wrote: compact_vectors[p][n] + similarity_map[m]
        # Current layer GEMM on compact vectors:
        for k_start in 0..K-1 step k:  # outer loop (output stationary)
            # inner loop (weight stationary):
            partial = systolic_array(compact_input[:, k_start:k_start+k],
                                     weight[k_start:k_start+k, :])
            # SIC Scatter (in-place, concurrent with accumulation):
            for orig_idx in 0..m-1:
                compact_idx = sim_map[orig_idx]
                output_tile[orig_idx] += partial[compact_idx]
                # 2a-wide accumulator for concurrent scatter

        # After all K iterations: output_tile is full and correct
        # SIC Gather: compress output tile before DRAM writeback
        compact_next, sim_map_next = SIC.similarity_gather(output_tile)
        write_to_dram(compact_next, sim_map_next)  # for next FC layer
```
关键优势：reduced input vectors (p < 1024) → lower GEMM workload → 每次FC GEMM的compute savings累计可达~5.0×。DRAM traffic reduction：仅compact vectors + similarity map (small int metadata) 写回，vs baseline全量tokens，达4.9× reduction。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
此tiling+concentration pipeline在Focus的SCALEsim-v2-based cycle-accurate simulator中建模，接收PyTorch算法实现生成的layer-wise sparse traces（每tile的active/inactive indices, similarity map, compact vector count）作为输入。硬件RTL实现中，GEMM controller与SIC的gather/scatter logic通过ready/valid handshake同步：GEMM tile ready → trigger SIC gather → DRAM write → next layer's GEMM reads compact data → scatter during accumulation → after tile complete → trigger gather again。开源实现含完整simulator和RTL：https://github.com/dubcyfor3/Focus。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

