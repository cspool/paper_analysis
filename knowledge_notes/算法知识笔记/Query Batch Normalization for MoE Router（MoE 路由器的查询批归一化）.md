## Query Batch Normalization for MoE Router（MoE 路由器的查询批归一化）

术语是什么？
Query Batch Normalization 是 PEER（继承自 PKM Lample et al. 2019）中应用于 query network 输出上的 Batch Normalization 层，目的是提升 expert 使用的均匀性。由于 product key 的子密钥在训练中可能形成不均匀分布，某些子密钥被频繁选中，导致部分 expert 过载而其他闲置。在 query vector 上添加 BN 层后，query 分布更加均匀（零均值、单位方差），使得 product key 检索到的候选集覆盖更广的 expert 空间。PEER 实验表明：使用 query BN 后，expert 使用率接近 100%（甚至对 N=1M），unevenness（KL 散度，衡量 expert 分布与均匀分布的偏离程度）显著降低，perplexity 也有所改善。

从算法pipeline角度拆解术语：
Query BN 在 PEER 前向传播中的位置：
```
# 无 BN 版本
q = query_proj(x)                           # 可能分布不均
indices, scores = product_key_retrieve(q, ...)

# 有 BN 版本
q = query_proj(x)
q = BatchNorm(q)                            # 标准化 query 分布
indices, scores = product_key_retrieve(q, ...)
```
BN 的作用机制：在训练 batch 上计算 query 的均值和方差，归一化后使各维度 scale 一致。这使得 product key 检索中 q₁^T c_i 和 q₂^T c'_j 的分布更均匀，不同子密钥被选中的概率差异减小。

术语一般如何实现？
标准 BatchNorm1d 应用于 query 向量的 d 维度。PEER 实验中默认启用 query BN。消融实验对比了 16K-1M experts 下使用/不使用 BN 的 expert usage 和 unevenness（表 2）：使用 BN 时 unevenness 约降低 30-50%（如 1M experts: 1.52→1.06），perplexity 从 20.73 降至 20.64（1M experts, C4）。Query BN 在 isoFLOP 最优区域附近改善最明显（Fig. 4）。

涉及论文标题：
- Mixture of A Million Experts
