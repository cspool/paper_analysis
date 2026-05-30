## Advanced Knowledge Initialization (AKI)

术语解释
AKI 是 bert2BERT (ACL 2022) 提出的改进版权重初始化方法，通过利用相邻层的权重打破 FPI 的对称性问题，在扩展模型宽度时保持有效参数量不减少。

术语是什么？
AKI 的核心改进：FPI 扩展时从同层权重复制（如 W'_new = W_1/2），而 AKI 从相邻层的权重复制（如 W'_new = W_next_1）。因为相邻层学到不同的特征映射，这样打破对称性。以两层 MLP 为例：y1 = U1^T · W1^T · x, y2 = U2^T · W2^T · y1。FPI 扩展 W1 为 [w1/2; w2/2; w3/2; w1/2]，AKI 扩展 W1 为 [w1/2; w2/2; w3/2; w2_1]（w2_1 是第二层第一个权重），从而打破第一层内部的对称性。

从算法pipeline角度拆解术语：
```
# AKI 宽度扩展（利用相邻层权重）
# 源模型 Layer L: W_L ∈ R^{d_in×d_inter}
# 源模型 Layer L+1: W_{L+1} ∈ R^{d_in×d_inter}
# 扩展 W_L 到 d_inter_new

# FPI (同层复制):
W'_L = concat(W_L[:d_inter], W_L[:d_inter_new-d_inter]复制)
# → 对称性：复制的权重梯度永远相同

# AKI (相邻层构建):
W'_L = concat(W_L[:d_inter], W_{L+1}[:d_inter_new-d_inter])
# → 来自不同层的权重已有不同学习模式 → 打破对称性
```

术语一般如何实现？如何使用？
- 需同时有相邻两层的权重信息（L 和 L+1 层）
- 仅支持 MHA (Multi-Head Attention)，直接不支持 GQA (Group Query Attention)
- 在 AquilaMoE 中作为 baseline：AKI-Stacking validation loss 9.56 at M(32,4096)
- bert2BERT 实验：AKI 在训练一定步数后超过 FPI 的性能
- AquilaMoE 在此基础上改进得到 AKI-Pro

涉及论文标题：
- AquilaMoE Efficient Training for MoE Models with Scale-Up and Scale-Out Strategies

---
