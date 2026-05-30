## Expert Selection Diversity in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Selection Diversity 指 MoE 模型中被同一 token 选中的 top-k experts 在功能上的互补性——即它们是否学习了不同的、互补的知识而非冗余的、相似的功能。这不同于 load balance（token 分配的数量均衡）。当 expert i 和 expert j 的 gating weight vectors w_{g,i} 和 w_{g,j} 高度相似（cosine similarity → 1）时，它们对相似类型的 token 产生高 logit，倾向于被同时激活（co-activation），从而学习到冗余的功能。GatePro 通过三个指标量化 diversity：Average Cosine Similarity（越低越好）、Average Angle（越大越好）、Spectral Entropy（越高越好，表示 expert 选择的分布更均匀）。GatePro 实验显示其在所有层上都持续维持更优的 diversity metrics，特别是在深层（Layer 16）中 baseline 的 similarity 持续上升而 GatePro 保持稳定低位。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Diversity 计算的三个指标：

- Average Cosine Similarity: $\frac{2}{N(N-1)} \sum_{i<j} |S_{ij}|$，衡量 all expert pairs 的平均 gating weight 对齐程度
- Average Angle: $\frac{2}{N(N-1)} \sum_{i<j} \arccos(S_{ij})$，互补于 cosine similarity，角度越大表示 expert 越正交
- Spectral Entropy: $-\sum_i \tilde{\sigma}_i \log \tilde{\sigma}_i$，其中 $\tilde{\sigma}_i$ 是 similarity matrix S 的标准化奇异值，反映 expert 行为模式的整体分散度

低 diversity 场景的例子（baseline）：
```
Expert 3 和 Expert 17 的 w_{g,3} ≈ w_{g,17} (S_{3,17}=0.92)
Token x: logits[3]=0.8, logits[17]=0.79 → both in top-6
→ 两个几乎等价的 FFN 被同时激活，浪费计算资源
```

高 diversity 场景（GatePro）：
```
Expert 17 被惩罚: logits[17] -= 1e-4 → 0.79-1e-4 < logits[25]=0.08
→ Expert 25 取代 Expert 17 进入 top-6
→ 6 个功能互补的 expert 执行不同计算
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Diversity 提升可通过：GatePro 的 localized competition（最直接的方法），diversity regularization loss（在 training objective 中加入高 S_{ij} 对的惩罚项），正交初始化策略，或动态 expert merging/pruning。GatePro 的优势在于 parameter-free 且不影响 loss landscape。

涉及论文标题：
- GatePro Parameter-Free Expert Selection Optimization for Mixture-of-Experts Models
