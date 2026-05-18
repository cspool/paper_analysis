## Similarity Map

术语是什么？通过联网搜索让回答具体和精准。
Similarity Map是Focus SIC中的核心metadata结构，大小为1×m（m=1024 per tile），记录每个original vector在compact buffer中的代表vector index。例如，若token 32与token 31的cosine similarity > 0.9，则similarity_map[32] = similarity_map[31] = index_of_token_31。该map使下游GEMM的Similarity Scatter能正确地将compact vector的partial sums复制回所有原始token位置，保证concentration后的功能正确性。Similarity Map是Focus实现lossless vector-level compression的关键enabler。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Similarity Map的生成和使用：
```
# Generation (in Similarity Gather):
similarity_map = [-1] * 1024  # -1 = unassigned
compact_idx = 0
for each 2×2×2 block:
    key_idx = max(block)
    if similarity_map[key_idx] == -1:
        similarity_map[key_idx] = compact_idx
        compact_idx += 1
    for other_idx in block \ {key_idx}:
        if cos_sim(key, vectors[other_idx]) > 0.9:
            similarity_map[other_idx] = similarity_map[key_idx]  # reuse
        else:
            similarity_map[other_idx] = compact_idx
            compact_idx += 1

# Usage (in Similarity Scatter):
# partial_sum[p][n]: results for p compact vectors
for orig_idx in 0..1023:
    compact_idx = similarity_map[orig_idx]
    output[orig_idx] += partial_sum[compact_idx]
```
Similarity Map存储为1×m的int array，每entry为representative index（最多log2(m) bits）。Memory overhead极小：1024 × 10 bits ≈ 1.28KB per tile。写回DRAM时与deduplicated vectors一同写入，读回时一同读出。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Similarity Map在Focus硬件中存储为1×m on-chip buffer（~1.28KB per tile）。在Similarity Gather阶段由similarity collection logic写入，在Similarity Scatter阶段由index mapper读出。Map的index lookup为single-cycle random access（地址=original_idx→读出representative_idx）。Scatter阶段用此index从compact partial sums中选取对应结果并广播到original output position进行累加。开源实现见https://github.com/dubcyfor3/Focus。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

