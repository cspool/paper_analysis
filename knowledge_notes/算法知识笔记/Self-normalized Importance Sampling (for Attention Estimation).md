## Self-normalized Importance Sampling (for Attention Estimation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Self-normalized Importance Sampling (SNIS) 是从提议分布u中采样来估计未知目标分布w期望的统计方法。在attention估计中，目标是对w = Softmax(qK^T/√d)下value的期望o = E_{i~w}[v_i]，但由于计算w需要所有qk_i^T内积（计算量O(nd)），无法直接获得。SNIS允许从提议分布u_i中采样B个索引i_1,...,i_B，然后用$\bar{o} = \sum_{j=1}^B (\tilde{w}_{i_j}/u_{i_j}) v_{i_j} / \sum_{j=1}^B (\tilde{w}_{i_j}/u_{i_j})$ 估计attention output，其中$\tilde{w}_i = \exp(qk_i^T/\sqrt{d})$是未归一化的attention score。该估计器有性质P[lim_{B→∞} X^{IS} = o] = 1。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**SNIS在MagicPIG attention中的实现**：

```
// Self-normalized Importance Sampling for attention
// 提议分布u来自LSH (SimHash碰撞概率)
S = LSH_Sample(q, K, V, HT)  // 采样得到的key索引集合

unormalized_w = []  // 未归一化权重
sample_prob = []    // 采样概率
for idx in S:
  unormalized_w.append(exp(q @ k_idx / sqrt(d)))
  sample_prob.append(u_idx)  // u_i = LSH采样概率

// SNIS估计器（公式9的变体）
weighted_sum_V = 0
weight_sum = 0
for j in range(|S|):
  weight = unormalized_w[j] / sample_prob[j]
  weighted_sum_V += weight * v_S[j]
  weight_sum += weight

o_hat = weighted_sum_V / weight_sum  // Self-normalized
```

术语一般如何实现？如何使用？

在MagicPIG中，SNIS的提议分布u来自LSH SimHash碰撞概率u_i = 1 - (1-p_i^K)^L - L·p_i^K·(1-p_i^K)^{L-1}。由于u_i与qk_i^T/√d（在centering和范数归一化后等价于余弦相似度）单调相关，u近似满足最小方差条件u ∝ w_i|v_i-o|（因log|v_i-o|波动远小于qk_i^T/√d）。"至少2表碰撞"机制（而非标准SimHash的≥1表）极大提升了采样质量——降低了对低相似度key的采样概率。

涉及论文标题：
- MagicPIG: LSH Sampling for Efficient LLM Generation

---
