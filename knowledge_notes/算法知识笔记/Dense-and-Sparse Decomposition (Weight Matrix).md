## Dense-and-Sparse Decomposition (Weight Matrix)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
将权重矩阵分解为 Dense + Sparse 两部分以改善量化性能的方法。SqueezeLLM 提出：W = D + S，S（sparse, CSR FP16）包含 ~0.45% 的异常值（百分位阈值外）和高敏感值（Fisher 排名 top 0.05%），D（dense）包含剩余 99.55% 权重并以非均匀量化存储。动机：(1) LLM 权重中 99.9% 的值集中在 ~10% 的 range 内，少量 outliers 膨胀量化范围 10x → 去除它们大幅缩小 D 的 value range → 提高量化分辨率；(2) 敏感值以 FP16 保留避免扰动最终输出，同时防止 k-means centroids 被它们"拉偏"。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# S = {outliers} ∪ {sensitive}
T_min, T_max = percentile(W, 0.2%), percentile(W, 99.8%)  # outlier 阈值
O = {i | W_i < T_min or W_i > T_max}  # ~0.4%
S_top = topk_indices(F, k=0.05%×|W|)  # Fisher 排名最前的 0.05%
S_indices = O ∪ S_top  # ~0.45% (去重)

# S 存储为 CSR: values (FP16) + col_indices (int16) + row_ptrs (int32)
# D = W - S (S_indices 处清零)
D_indices, LUTs = weighted_kmeans_quantize(D, F, bit)

# 推理: Y = D @ X + S @ X (fused kernel)
```

存储分析：每个 sparse 元素 ~32-33 bits (16b value + 16b index + row_ptr 分摊) → 0.45% sparsity 增加 ~0.24 bit/param 的 overhead。3-bit dense + 0.45% sparsity = 3.24 avg bits。与 grouping 对比：Dense-and-Sparse 是 outlier 问题的直接解决方案（直接移除 outliers），grouping 是间接方案（组内隔离），且 grouping + 非均匀量化组合需 per-group LUT（overhead 更大）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SqueezeLLM 开源：https://github.com/SqueezeAILab/SqueezeLLM。关键参数：sparsity level 0.45%（0.4% outliers + 0.05% sensitive），已通过消融实验验证为 sweet spot——更低的 sparsity 不足以覆盖主要 outliers，更高的 sparsity 引入 diminishing returns（D.2/D.5）。在 GPU 推理中，sparse part 使用 balanced CSR kernel（10 nonzeros/thread）以处理 per-row sparsity skew。SpQR 也采用类似 dense+sparse 策略，但主要区别在于：SpQR 依赖 fine-grained grouping+bi-level quantization 处理 dense part，而 SqueezeLLM 用 sensitivity-based non-uniform quantization，用更少的 sparsity（0.05% vs 1%）和更简单的 quantization scheme 达到更好效果。

涉及论文标题：
- SqueezeLLM Dense-and-Sparse Quantization
