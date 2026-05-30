## Expert Clustering for MoE Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Clustering for MoE Compression 是在 expert merging 之前将功能相似的 expert 分组的技术。MergeMoE 的聚类策略：(1) 选取 top-M 使用频率的 expert 作为聚类中心（确保高频 expert 不被稀释）；(2) 距离度量使用拼接矩阵 [W_U || W_G] 的 L2 距离（而非全部参数），因为 T2/T3 仅作用于 W_G/W_U，在这些矩阵上聚类可直接减少加权平均误差。M-SMoE 则使用路由 logits 余弦相似度、dominant experts 作为中心。消融实验（MergeMoE Table 5）表明聚类质量对最终压缩效果至关重要——即使跳过 T1/T2/T3 优化，仅聚类+直接输出合并性能已接近完整流程。

从算法pipeline角度拆解术语：
```
// MergeMoE 聚类
centers = top-M by frequency f_i
for non-center expert j:
    for center k:
        V_j = concat(W_Uj, W_Gj)  // 仅关注影响 T2/T3 的矩阵
        V_k = concat(W_Uk, W_Gk)
        dist(j,k) = ||V_j - V_k||_2
    分配 j 到最近 center
// 簇内权重: w_j = f_j / Σ_{k∈C_i} f_k
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 替代方案：M-SMoE 用 dominant experts + 路由 logits 余弦相似度；Sub-MoE 用 joint SVD 子空间内聚类；DM-MoE 用混合 drop-then-merge。
- 聚类 vs Pruning 的 tradeoff：聚类保留互补信息但可能引入参数干扰；pruning 消除干扰但信息损失更大；混合方法折中。
- REAP (arXiv 2510.13999) 质疑 merging 方法——实验表明 merging 导致 functional subspace collapse，主张 pruning 可能更优。

涉及论文标题：
- MergeMoE: Efficient Compression of MoE Models via Expert Output Merging

---
