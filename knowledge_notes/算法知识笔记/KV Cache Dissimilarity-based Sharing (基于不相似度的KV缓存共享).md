## KV Cache Dissimilarity-based Sharing (基于不相似度的KV缓存共享)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

KV Cache Dissimilarity-based Sharing 是 KVSharer 论文提出的反直觉层间共享策略。传统方法基于"共享相似的"直觉，但 KVSharer 发现共享不相似的 KV cache 效果更好。具体通过计算任意两层 KV cache 的欧氏距离（分别 flatten keys 和 values 为 1D 向量后取平均作为该层表示），按距离降序排列优先尝试最不相似的层对。消融实验（Figure 6）证明 dissimilarity-based 的 PPL 显著低于 similarity-based（低 2 倍以上）。

从算法pipeline角度拆解术语：

**不相似度计算**：
```
for l in 1..L:
    // 在校准数据集 D 上收集各层 KV cache
    K_avg[l] = mean_{x∈D}(flatten(K_l(x)))
    V_avg[l] = mean_{x∈D}(flatten(V_l(x)))
    KV_repr[l] = concat(mean(K_avg[l]), mean(V_avg[l]))

// 距离矩阵
for i, j in 1..L:
    S[i][j] = ||KV_repr[i] - KV_repr[j]||_2  // Euclidean

R = argsort_descending(S)  // 距离大 = 不相似 = 优先尝试
```

术语一般如何实现？如何使用？

不相似度计算是策略搜索的前置步骤。校准数据集仅需 30 句 64-token 句子（Wikipedia）。距离矩阵 S 规模为 L×L。后续贪心搜索按 R 的顺序尝试替换并验证 hidden-state 相似度。该策略的关键 insight：不相似的 KV cache 捕获了不同的注意力信息，共享后信息多样性得以保留，而相似的层共享可能导致某些注意力模式完全丢失。

涉及论文标题：
- KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing

---
