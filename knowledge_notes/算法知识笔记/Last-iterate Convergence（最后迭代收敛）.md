## Last-iterate Convergence（最后迭代收敛）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Last-iterate Convergence（最后迭代收敛）是随机优化中保证最后一个迭代点 w^T（而非所有迭代的加权平均 w̄^T）收敛到最优解的理论性质。在 QAT 语境中至关重要：即使每个 w^s 都是量化解，其加权平均 w̄^T 通常不再被量化，因此平均迭代收敛结果对 QAT 没有实际意义。最后迭代收敛保证了最终输出的 w^T 本身（量化的）逼近量化约束下的最优解。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
PARQ 的 Theorem 3.2 证明 AProx 的最后迭代收敛：
```
E[F_λ(w^T)] - F_λ(w*) ≤ GR · (2 + 1.5 ln(T)) / √T
```
证明分两步：(1) Lemma A.3 (Orabona 2020) 将 η_T·q_T（最后迭代差距）分解为平均差距 + 尾项修正：η_T q_T ≤ (1/T)Σ η_t q_t + Σ_k (1/k(k+1)) Σ_{t=T-k+1}^T η_t(q_t - q_{T-k})；(2) 平均差距由 Theorem 3.1 的 regret bound 控制，尾项通过 telescoping + (T-k+1..T 区间) 内的单步 bound 控制。最终收敛率与平均迭代同阶 O(1/√T)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
一般 SGD 的最后迭代收敛常需要额外假设（强凸性、平滑性）。PARQ 证明 AProx 在仅凸性下取得最后迭代收敛，归因于 aggregation 机制的隐式正则化效应。这解决了 ProxConnect/Dockhorn et al. 仅证明平均迭代收敛的理论缺口。

涉及论文标题：
- PARQ Piecewise-Affine Regularized Quantization
