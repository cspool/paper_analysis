## Three-Level Attention Classification (Critical/Marginal/Negligible)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SLA的三级注意力权重分类策略，将attention block分为critical（top k_h%, 完整O(N²)计算）、marginal（中间，O(N)线性注意力）和negligible（bottom k_l%, 跳过）。相较于传统稀疏注意力的二级分类（保留/跳过），引入marginal层打破了稀疏度天花板：传统方法跳过中间值会引入显著误差（L1 error从3%跃至33%），保留中间值又严重降低稀疏度（<90%）。SLA对marginal块用几乎免费的线性注意力（cost <0.5% full attention），实现95%有效稀疏度且不损失精度。

从算法pipeline角度拆解：
基于压缩注意力矩阵P_c[i,j]（Q/K mean-pooled后计算dot product），每Q block行独立执行：TopK选前k_h% → critical (M_c=+1)；BottomK选底k_l% → negligible (M_c=-1)；其余 → marginal (M_c=0)。分类参数：k_h=5%源自约8.1%权重大于平均值1/N（保留top 5%已足够），k_l=10%源自约45%权重<1/(100N)（跳过bottom 10%误差<3%）。

术语一般如何实现？如何使用？
在SLA fused GPU kernel内执行。T_m = T_n ≈ 469 for Wan2.1 (N=30K, b=64)，分类开销可忽略。k_h=5%, k_l=10%为推荐默认值；可调整以适应精度-效率tradeoff。

涉及论文标题：
- SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention
