## Column Decomposition with Reordering

术语是什么？

Column decomposition with reordering (列分解与重排序) 是 SpMSpV 预处理阶段的 locality enhancement 技术。真实世界图（社交网络、web graphs）的列长分布极度偏斜：少数 long column 包含大多数非零元。例如 it-2004 中 1.4% 的列（≥256 NNZ）占总 NNZ 的 70%+，平均长度 1403 vs 全局均值 27.9。Long column 内部 row indices 不重叠——hash table 无法在列内提供聚合收益。跨列 segments 可能共享重叠 row ranges。Column decomposition 将 long column 切分为固定大小 segments，Reordering 按 segment 首 row index 排序增强跨列 locality。

从kernel调度角度拆解术语：

VDHA Vector Processing 伪代码：

```
// Column classification + splitting + reordering
segments = []
for each active (col, vec_val) in x_sparse:
    start = col_ptr[col], len = col_ptr[col+1] - start
    if len < LEN_THRES:               // short column, LEN_THRES=128
        segments.append((start, len, vec_val))
    else:                              // long column → split
        for i in [0, ceil(len/SPLIT_SIZE)):   // SPLIT_SIZE=256
            seg_start = start + i * SPLIT_SIZE
            seg_len = min(len - i*SPLIT_SIZE, SPLIT_SIZE)
            segments.append((seg_start, seg_len, vec_val))

// Reorder: 仅排序 long-column segments 的 metadata
long_segs = [s for s in segments if from long column]
sort(long_segs by A.indices[s.start])     // O(S log S), S << N
short_segs = [s for s in segments if short column]
segment_queue = short_segs + long_segs    // block-mapped to CTAs
```

关键参数：LEN_THRES=128 区分 short/long column；SPLIT_SIZE=256 使每个 segment 适应 CTA 256 threads + hash table。Reordering 成本 O(S log S) 远小于排序所有 nonzeros O(N log N)。Short columns 不重排序——数量可能接近输入向量长度，排序开销过大。

在 it-2004 上的效果（T=2048, density=100%）：ρ 51.0%→89.8%，γ 0.744→2.607，atomic-unit utilization 22.99%→12.82%。

术语一般如何实现？如何使用？

CPU 预处理步骤（轻量级，可在线执行）。仅排序 segment metadata 而非 nonzeros 本身。受 RoDe (row decomposition-based SpMV) 启发——将类似思想从行应用到列并增加重排序。适用于偏斜列长分布的矩阵（power-law），对角线规则矩阵收益有限。

涉及论文标题：
- VDHA: Vector-Driven Hash Aggregation for Sparse Matrix-Sparse Vector Multiplication on GPUs

---

