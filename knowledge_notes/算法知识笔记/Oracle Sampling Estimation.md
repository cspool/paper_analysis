## Oracle Sampling Estimation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Oracle Sampling Estimation是将attention output视为从attention score分布w中独立同分布采样的value期望值o = E_{i~w}[v_i]，然后通过Monte Carlo估计ô = (1/B) Σ_{j=1}^B v_{i_j}。称为"Oracle"是因为它假定了attention score分布w是已知的——在实际稀疏attention中w需要计算所有qk_i^T才能获得，因此oracle sampling不能直接实用化（只能节省wV计算，不能节省qK^T计算，最多2× wall-clock加速）。尽管有重复采样（Theorem 3.3保证实际计算量|S| ≤ 1+B·ε），Oracle采样理论上无偏（Theorem 3.2）且比TopK减少最多4×估计误差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// Oracle Sampling Estimation (预先知道w)
// w = Softmax(qK^T/√d)  // "Oracle"已知
o_hat = 0
S = {}  // 去重后的采样集合
counts = {}  // 重复计数
for j in 1..B:  // B = 采样预算
  i = CategoricalSample(w)  // 从w中采样
  counts[i] = counts.get(i, 0) + 1
  S.add(i)

for i in S:
  o_hat += (counts[i] / B) * v_i  // Equation 5

// Theorem 3.2: E[ô] = o (无偏)
// Theorem 3.3: E[|S|] ≤ 1 + B·(1-max_i w_i)
//   当w峰值明显时，实际计算量远小于B
```

术语一般如何实现？如何使用？

Oracle Sampling在MagicPIG中作为理论motivation而非实用方法。论文通过它证明：(1) 采样估计可以超越TopK的准确率上限；(2) 即使采样预算B很小（如0.002%上下文），oracle sampling仍能保持高准确率。MagicPIG通过LSH近似oracle sampling：用SimHash碰撞概率构造提议分布u，逼近w的分布形状，实现Self-normalized Importance Sampling。

涉及论文标题：
- MagicPIG: LSH Sampling for Efficient LLM Generation

---
