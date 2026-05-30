## Frobenius Loss-guided Weight Matrix Selection for Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Frobenius Loss-guided Weight Matrix Selection 是 Basis Sharing 中用于确定哪些类型权重矩阵适合跨层共享的分析方法。对每种矩阵类型，评估独立 SVD 压缩 loss 之和 vs 跨层共享 loss，用热力图比较。

分析结论（LLaMA2-7B）：W_K, W_Q, W_V, W_Up, W_Gate 适合共享（共享 loss < 独立 loss 之和）；W_Down 不适合（高维→低维投影拼接后 rank 增大，相同 k 下截断损失更大）；W_O 不适合（注意力输出功能跨层差异大）。具体数值：W_K 9-10 层共享 loss=61817.3 < 独立和=66682.9；W_O 9-10 层共享 loss=10618.3 > 独立和=9250.8。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
热力图生成（32 层 W_K, 20% 压缩比）：

```
for i in range(32):
    X^(i) = collect_activation(W^(i), calib)
    S^(i) = cholesky((X^(i))^T @ X^(i))^{1/2}
    loss_diag[i] = ‖S^(i)·W^(i) - SVD_k(S^(i)·W^(i))‖_F²

for i, j in adjacent_layer_pairs:
    X_cat = concat_vertical(X^(i), X^(j))
    S_ij = cholesky(X_cat^T @ X_cat)^{1/2}
    W_cat = concat_horizontal(W^(i), W^(j))
    heatmap[i][j] = ‖S_ij·W_cat - SVD_k(S_ij·W_cat)‖_F²

suitable = all(heatmap[i][j] < loss_diag[i] + loss_diag[j] for adjacent i,j)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在压缩前离线完成（O(n²)），分析结果决定各类型矩阵的压缩策略。每个 LLM 架构需独立分析，因层间相似性因模型而异。

涉及论文标题：
- Basis Sharing Cross-Layer Parameter Sharing for Large Language Model Compression
