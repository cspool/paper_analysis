## Query-Group Compression (GQA-aware KV Cache Compression)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Query-Group Compression 是 KV-Compress 提出的针对 GQA 模型的 KV cache 压缩策略。传统 KV cache 压缩方法（SnapKV、PyramidKV、H2O 等）在 GQA 模型上先将 KV cache repeat 到 query head 数量（例如从 8 KV heads repeat 到 32 query heads），再在重复后的 cache 上执行压缩。这导致：(1) cache 中 3/4（r-1/r）的 KVs 是重复数据；(2) 压缩率需超过 query-to-KV-head ratio r 才能带来超过 GQA 本身的额外压缩效果。

Query-Group Compression 直接在非 repeat 的 GQA KV cache（shape H_kv × L × d）上执行压缩。关键修改是将 eviction metric 的聚合范围改为每个 key 所属的 query group 内的所有 queries：$M_{h_k,j} = \sum_{h \in H_k} \sum_i (A_{h,i,j})^2$，其中 $H_k = \{h: r \cdot h_k \le h < r \cdot (h_k + 1)\}$。这对应 Equation 9-10 的推广。

对于 Llama-3/Mistral 模型 (r=4)，同样 max-cache-size C 下，KV-Compress 实际持有 1/4 的 KVs，相当于 4x 额外有效压缩率。

从算法pipeline角度拆解术语：

**GQA Query-Group Compression 的计算流程**：
```
输入：GQA model with H_kv KV heads, r query heads per KV head
参数：observation window w, pooling size p（或 excluded query window v for full mode）

for each layer and each KV head h_k in 1..H_kv:
    # 定义该 KV head 对应的 query group
    H_k = {h : r*h_k <= h < r*(h_k + 1)}  # r 个 query heads

    # 聚合该 group 内所有 query heads 的 squared attention
    for each query head h in H_k:
        for each query i in observation range:
            for each key j in causal range (j <= i):
                M_{h_k, j} += (A_{h, i, j})^2

    # 可选 max-pooling（KVC-w 变体）
    M_{h_k, j} = max_{t in [j-p/2, j+p/2]} M_{h_k, t}

# 在非 repeat 的 KV cache 上执行 eviction：
# 排序 M_{h_k, :} → 选择 top-C KVs per head → 释放其余
```

**与 baseline 方法的区别**：
```
# Baseline (SnapKV/PyramidKV naive GQA):
K_cache_raw = K_cache[:, :H_kv, :]          # [L, 8, d]
K_cache_repeat = repeat(K_cache_raw, r)       # [L, 32, d] — 3/4 duplicates!
metrics = compute_attention_scores(K_cache_repeat)  # on repeated cache
evict_KVs(K_cache_repeat, metrics)            # compresses repeated data

# Query-Group Compression (KV-Compress):
K_cache_raw = K_cache[:, :H_kv, :]          # [L, 8, d]
for h_kv in 0..H_kv-1:
    metrics[h_kv] = sum_{h in H_kv_group} attention_scores[h]
evict_KVs(K_cache_raw, metrics)              # compresses non-repeated data
# Same max-cache-size C: 4x fewer KVs stored
```

术语一般如何实现？如何使用？

实现方式：(1) 在 prefill 阶段计算完整 attention 后，按 query group 聚合 attention scores；(2) 聚合操作可以累积到 per-KV-head 的 metric tensor 中（而非 repeat 后的 per-query-head tensor）；(3) 后续 eviction 操作基于 M_{h_k, j} 在非 repeat cache 上执行。该方法与 FlashAttention 兼容——attention scores 在 FA 计算中获取（eager mode）或从 observation window queries 单独计算。

适用于所有 GQA 模型（Llama-3, Mistral, Qwen 等）的 KV cache 压缩场景。r 越大（query heads per KV head 越多），query-group compression 相对于 baseline repeat+compress 的优势越显著。在 KV-Compress 中，LongBench C=128 下以 1/4 KVs 达到 state-of-the-art（Mistral-7B: 37.64 vs Ada-SnapKV 36.71; Llama-3.1-8B: 46.26）。

涉及论文标题：
- KV-Compress__Paged_KV-Cache_Compression_with_Variable_Compression_Rates_per_Attention_Head

---
