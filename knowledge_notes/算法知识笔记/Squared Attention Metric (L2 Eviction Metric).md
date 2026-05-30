## Squared Attention Metric (L2 Eviction Metric)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Squared Attention Metric 是 KV-Compress 提出的 KV eviction metric 计算方法，使用 attention scores 的平方和 $\sum (A_{h,i,j})^2$ 替代传统的 attention scores 直接求和 $\sum A_{h,i,j}$（L1 aggregation）。前者等价于最小化未来 attention 的 L2 误差，后者等价于最小化 L1 误差。

数学上：对于 key j 和 query i，标准 attention weight $A_{h,i,j} = \text{softmax}(Q_i K_j^T / \sqrt{d})$。L1 metric $M_{h,j}^{(L1)} = \sum_i A_{h,i,j}$。L2 metric $M_{h,j}^{(L2)} = \sum_i (A_{h,i,j})^2$。L2 metric 对高 attention 的 key 更敏感（平方惩罚放大差异），使得 eviction 更倾向于保留高 attention 的 KVs 并更激进地舍弃低 attention 的 KVs。

KV-Compress 实验验证 L2 在所有变体（KVC-w, KVC-full）、所有 max-cache-size（C=128/256/512/1024）和两个模型（Mistral-7B, Llama-3.1-8B）上一致优于 L1。

从算法pipeline角度拆解术语：

```
# L1 vs L2 对比计算
# 假设 attention weight distribution: [0.5, 0.3, 0.15, 0.04, 0.01]

# L1 (standard):
M1 = [0.5, 0.3, 0.15, 0.04, 0.01]  # 直接求和，差异小
# 如果 evict 最后两个: 丢失 0.05 attention mass → L1 error ≤ 0.05

# L2 (squared):
M2 = [0.25, 0.09, 0.0225, 0.0016, 0.0001]  # 平方求和，差异扩大
# 高 attention keys 的 metric 被放大 (0.25 vs 0.09, gap 0.16)
# 低 attention keys 的 metric 被压缩 (0.0016 vs 0.0001, gap 0.0015)
# 排序更确定，eviction 决策更准确

# KVC-w8-L2 computation (Equation with squared attention):
for h in H_k:  # query heads in key's query group
    for i = L-w to L:  # observation window
        for j = 1 to i:  # causal key range
            M_{h_k, j} += (A_{h, i, j})^2  # squared!
# then optional max-pooling
```

术语一般如何实现？如何使用？

实现简单：在 attention 计算的 metric 累积步骤中，将 `M += A` 改为 `M += A*A`（逐元素平方）。不影响其他算法组件（sort, top-k, block eviction selection）。与 GQA query-group aggregation 和 continual compression 正交叠加。

适用场景：所有基于 attention score 聚合的 KV eviction 方法通用。KV-Compress 中默认使用 L2（论文中标为 KVC-w8-L2），除非显式标注 L1（KVC-w8-L1）。

涉及论文标题：
- KV-Compress__Paged_KV-Cache_Compression_with_Variable_Compression_Rates_per_Attention_Head

---
