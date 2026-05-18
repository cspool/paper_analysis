## Similarity Gather

术语是什么？通过联网搜索让回答具体和精准。
Similarity Gather是Focus SIC中的streaming压缩操作，在systolic array的每个GEMM tile输出后立即执行，将m×n tile中的vectors做vector-level similarity检测和deduplication，最终仅将deduplicated vectors和similarity map写回DRAM。它是SIC的"压缩"阶段（Scatter是"恢复"阶段），两者成对构成gather-scatter循环贯穿所有FC层。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Similarity Gather的per-tile执行伪代码：
```
# Input: GEMM output tile O[m][n], m=1024, n=32 (a=n=32)
#        SEC offset encoding for position recovery
# Output: compact_vectors[p][n] (p <= m), similarity_map[m]

def similarity_gather(tile_O[m][n], offsets[m]):
    # Step 1: Position recovery + FHW layout
    positions = restore_fhw(offsets)  # (f, r, c) per vector
    vectors = reorder_to_fhw(tile_O, positions)

    # Step 2: L2-norm precompute (each vector = 32-dim)
    l2_norms[0..m-1] = [sqrt(sum(v[d]^2 for d in 0..31)) for v in vectors]

    # Step 3: Block-wise similarity matching
    similarity_map = [-1] * m  # -1 = unique, else points to representative
    compact_idx = 0
    compact_vectors = []

    for each 2×2×2 block (stride 1):
        # 8 vectors per block: 4 from frame A, 4 from frame B
        key_idx = max(block)  # highest-index vector as key
        key_vec = vectors[key_idx]
        key_norm = l2_norms[key_idx]

        if similarity_map[key_idx] == -1:  # key not yet matched
            similarity_map[key_idx] = compact_idx
            compact_vectors.append(key_vec)
            compact_idx += 1

        for other_idx in block \ {key_idx}:
            if similarity_map[other_idx] != -1: continue  # already processed
            dot_prod = sum(key_vec[d] * vectors[other_idx][d] for d in 0..31)
            cos_sim = dot_prod / (key_norm * l2_norms[other_idx])
            if cos_sim > 0.9:  # similarity threshold
                similarity_map[other_idx] = similarity_map[key_idx]
                # other_idx reuses key's compact index
            else:
                similarity_map[other_idx] = compact_idx
                compact_vectors.append(vectors[other_idx])
                compact_idx += 1

    # Step 4: Writeback
    write_to_dram(compact_vectors)  # p vectors, p <= 1024
    write_to_dram(similarity_map)   # 1 × m
```

Timing分析：matcher最多需要 `8 × m = 8 × 1024 = 8192 cycles` per tile，而GEMM需要 `K/b × m = 3584/32 × 1024 = 114,688 cycles`。Matcher overhead < 7% of GEMM time，不在critical path。仅当K < 256时matcher接近critical path，此时可部署多matcher并行。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Similarity Gather在Focus中实现为hardware module：convolution-style layouter (地址生成+bank mapping) → L2-norm buffer (1×m FP32 values) → dot-product unit (32-cycle for 32-dim) → comparator (threshold 0.9) → compact output buffer (stores deduplicated vectors) + similarity map buffer (1×m int indices)。Gather operation在GEMM tile output streaming完成时触发，完成后触发DRAM write。下一层GEMM读取compact vectors + similarity map，执行Similarity Scatter恢复full output。开源实现含algorithm/simulator/rtl：https://github.com/dubcyfor3/Focus。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

