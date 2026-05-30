## Multi-Head Mixing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Multi-Head Mixing 是 MHLA 的核心机制，通过一个可学习的系数矩阵 Mc ∈ R^(M×M) 实现 query-conditioned 的 block 级 KV summary 混合。矩阵元素 m_{i,j} 表示 query block i 对 key-value block j 的 affinity（亲和度），第 i 行 m_i 指定 query block i 如何将 M 个局部 summary 线性组合成该 block 专属的全局 summary。

与 standard multi-head attention 沿 channel 维度分头不同，MHLA 的 "multi-head" 指沿 token（spatial）维度的分组。每个 token-level head 独立计算其 local context，再通过可学习的跨 head 混合恢复全局信息。

初始化策略：locality-biased——m_{i,j}^(0) ∝ 1 - dist(i,j)/max_k(dist(i,k))，其中 dist(i,j) 是 block i 和 j 中心在 2D/3D 网格上的欧氏距离。该初始化编码了空间近邻优先的先验，提供更稳定快速的收敛；训练过程中 Mc 完全可学习，并通过 clip 到 (0,1) 保持非负。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Multi-Head Mixing 的 token 级效应**：

```
# block b(t) = token t 所属的 block
# S_j = Σ_{t in block j} K̃_t^T V_t  (local summary)
# Mixed summary for query block i:
S̃_i = Σ_{b=1}^M m_{i,b} S_b = Σ_{t=1}^N m_{i,b(t)} K̃_t^T V_t

# Token-level contribution: query q_i attends to token t via:
# m_{i,b(t)} × (q̃_i^T K̃_t) × V_t^T
#  ^^^^^^^^     ^^^^^^^^^^^^    ^^^^
#  block选择    token内重加权    值
#  (query-      (kernel inner
#   cond.)       product)
```

两阶段权重机制：(1) block 级选择 m_{i,b(t)}——query-conditioned（不同 query block 可获得不同 block 权重）；(2) block 内 token 重加权 q̃_i^T K̃_t——传统 kernel 相似度。两者结合恢复 query-conditioned 的 token 级多样性。

术语一般如何实现？如何使用？

Mc 初始化为 [M, M] 的 float tensor，作为模型参数参与端到端训练。每行 m_i 归一化到和为 1。训练中 Mc 随其他参数一同优化，每步更新后 clip 到 (0,1)。在 chunkwise parallel form 中，Mc 的上三角被 mask 以满足 causality。由于 M 通常较小（≤ sqrt(N)），Mc 的额外存储和计算开销可忽略。

涉及论文标题：
- MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head

---
