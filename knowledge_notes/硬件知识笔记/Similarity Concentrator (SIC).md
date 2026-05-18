## Similarity Concentrator (SIC)

术语解释
Similarity Concentrator (SIC) 是Focus架构中负责vector-level冗余消除的硬件模块，在systolic-array accelerator的FC/FFN/projection GEMM层中执行streaming similarity detection、deduplication和reconstruction。

术语是什么？通过联网搜索让回答具体和精准。
SIC是Focus Unit的一个子模块，在hardware accelerator的FC层pipeline中对GEMM tile输出做vector-level redundancy elimination。它包含两个operation phase：(1) Similarity Gather：在GEMM tile输出后立即做on-chip compression——convolution-style layouter将tile vectors按FHW坐标重组为2×2×2时空block→similarity matcher做block内cosine similarity matching (dot-product + L2-norm)→matched vectors仅记录代表index→deduplicated vectors和similarity map写回DRAM；(2) Similarity Scatter：在后续GEMM对concentrated vectors执行计算后→根据similarity map将partial sums复制/分发回原始token位置→2a-wide accumulator并发累加→tile完成后再次Similarity Gather压缩。SIC实现gather-scatter循环贯穿所有FC层，保证功能正确性的同时最大化计算和带宽节省。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
SIC在Focus accelerator的硬件执行流程：
```
-- Similarity Gather Phase --
1. GEMM tile output (m=1024, n=32, a=n=32) stream into SIC
2. Convolution-style layouter (16KB buffer for 256-vector window):
   - Recover token FHW position via SEC offset encoding
   - Map vectors to banks using: Bank = f%2×4 + r%2×2 + c%2
   - 8 vectors per 2×2×2 block mapped to 8 distinct banks → conflict-free read
3. Similarity matcher:
   - Precompute L2-norm per vector → store in norm buffer (1×m)
   - For each block: key = highest-index vector, compare with 7 others
   - Cosine sim = dot(a,b) / (L2[a] * L2[b]) → if > 0.9: mark match
   - Matcher needs ≤8×m cycles per tile (vs GEMM K/b×m = 112×m cycles)
4. Similarity collection:
   - Unique vectors → compact output buffer
   - Similarity map (1×m) records: original_idx → representative_idx
   - Write deduplicated vectors + similarity map to DRAM

-- Similarity Scatter Phase --
5. GEMM on concentrated vectors: input p×K (p<1024) × weight K×n
   → outer loop output-stationary, inner loop weight-stationary
6. Per inner loop iteration: partial sums streamed to temp buffer
7. Scatter: using similarity map, replicate each compact vector's partial sum
   to all original token positions it represents
8. 2a-wide accumulator (64 when a=32): concurrent accumulation of
   reconstructed vectors
9. After ⌈K/k⌉ outer loop iterations: full output tile → Similarity Gather again
```
SIC总面积仅占Focus Unit的0.8%（~0.026 mm² in 28nm），matcher <1% of systolic array area。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SIC用SystemVerilog实现为Focus Unit的子模块。Similarity matcher复用或借鉴Special Function Unit (SFU)的现有逻辑（SFU通常用于RMSNorm和SoftMax），但为公平评估，论文将matcher作为独立模块计入area和energy。Convolution-style layouter使用16KB buffer存储256-vector sliding window，提供position recovery、bank mapping和conflict-free access。Module可配置参数：m=1024 (tile height), n=32 (vector length = tile width), block size=2×2×2, threshold=0.9。开源RTL见https://github.com/dubcyfor3/Focus。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

