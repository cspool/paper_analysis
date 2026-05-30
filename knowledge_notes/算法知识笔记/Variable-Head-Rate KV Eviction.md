## Variable-Head-Rate KV Eviction

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Variable-Head-Rate KV Eviction 是一种 KV cache 压缩策略，允许每个 attention head 以不同的压缩率（eviction rate）保留不同数量的 KVs，而非所有 head 统一 evict 相同数量的 KVs。该概念由 Ada-SnapKV (Feng et al., 2024) 首次从算法角度提出——按 per-head eviction metric 跨 head 进行 eviction 选择（cross-head eviction），在 LongBench 上证明了可变 per-head 压缩率相比 uniform compression 的精度优势。

然而，Ada-SnapKV 的方案在现有推理框架中只能增加 cache 碎片化而无法实际减少物理内存占用：因为现有 PagedAttention 中所有 heads 的 KVs 打包在同一 cache block 中，evict 一个 head 的 KVs 而不 evict 其他 heads 的 KVs 不释放整个 block。

KV-Compress 通过修改 PagedAttention 的 block 布局（per-head per-layer 独立 block）使 variable-head-rate eviction 可以实际释放物理内存——每个 head 有独立的 block table，evict 某 head 的 block 可直接释放该 block 的物理内存。同时扩展到 variable per-layer rate。

从算法pipeline角度拆解术语：

**Variable-Head-Rate Eviction 的 eviction 选择过程**：
```
输入：M ∈ R^{H_kv × L}（per-head per-token eviction metrics）
参数：target total KVs after compression T（等价于 max-cache-size C）

# Step 1: 跨 head 展平 metrics
M_flat = M.reshape(-1)  # [H_kv * L]
# 每个元素对应一个特定 (head, token_position) 的 KV

# Step 2: 全局排序（variable-head-rate 的关键）
sorted_idx = argsort(M_flat)  # 按 metric 升序排列
evict_idx = sorted_idx[:H_kv*L - T]  # 最低 metric 的 KVs

# Step 3: per-head eviction count 自动由全局排序决定
for h in 0..H_kv-1:
    evict_count[h] = count(evict_idx where head == h)
    # 高 attention 集中 head → 少 evict → 保留更多 KVs
    # 低 attention 分散 head → 多 evict → 保留更少 KVs
```

**Uniform vs Variable-Head-Rate 对比**：
```
# Uniform rate (H2O, SnapKV, PyramidKV):
for h in 0..H_kv-1:
    evict_count[h] = E / H_kv  # 所有 head 相同

# Variable-head-rate (Ada-SnapKV, KV-Compress):
# evict_count 由全局排序自动分配
# 高 attention 集中的 head 自然保留更多 KVs
```

术语一般如何实现？如何使用？

在 KV-Compress 中，variable-head-rate eviction 通过两步排序实现：(1) 先在每个 head 内排序 metrics 获得 per-head per-eviction-block 的最大 metric；(2) 再跨 head 排序候选 block evictions。最终按总 evict block 预算 E_s 选择跨 head 的 block eviction 方案。跨 head 的 eviction 分配完全由 metric 值驱动，不需要手动设定 per-head 预算。

适用场景：需要高压缩率的长上下文推理场景。当不同 attention head 的功能差异显著时（一些 head 关注全局语义检索，一些关注局部 token 关系），variable-head-rate 相比 uniform 的优势更大。KV-Compress 在 Llama-3.1-8B LongBench 上以 1/4 KVs 超越所有 uniform-rate baselines。

涉及论文标题：
- KV-Compress__Paged_KV-Cache_Compression_with_Variable_Compression_Rates_per_Attention_Head

---
