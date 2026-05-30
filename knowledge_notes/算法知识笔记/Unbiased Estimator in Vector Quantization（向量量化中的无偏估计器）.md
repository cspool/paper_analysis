## Unbiased Estimator in Vector Quantization（向量量化中的无偏估计器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
在 RaBitQ 系列中，无偏估计器 ⟨ō,q⟩/⟨ō,o⟩ 是 ⟨o,q⟩ 的无偏估计：E[⟨ō,q⟩/⟨ō,o⟩] = ⟨o,q⟩。前提：(1) 码本由随机旋转的单位向量组成；(2) ō 是 o 在码本中的最近向量。随机性来自随机正交矩阵 P。误差界以高概率成立。论文通过 10⁷ 对估计-真值点对做线性回归（slope=1, intercept=0）验证无偏性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
距离估计公式：
||o_r-q_r||² = ||o_r-c||² + ||q_r-c||² - 2·||o_r-c||·||q_r-c||·⟨o,q⟩
⟨o,q⟩ ≈ ⟨ō,q⟩ / ⟨ō,o⟩（无偏）
其中 ⟨ō,q⟩ = (1/||ȳ||)·(⟨ȳ_u,q'⟩ - (2^B-1)/2·Σq'[i])，q'=P^{-1}q

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
⟨ō,o⟩ 索引时预计算存储；Σq'[i] 查询时一次计算。无偏性意味着无系统性偏差，排序一致性更好。代码: https://github.com/VectorDB-NTU/Extended-RaBitQ

涉及论文标题：
- Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

---
