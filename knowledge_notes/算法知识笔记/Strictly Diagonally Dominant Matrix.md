## Strictly Diagonally Dominant Matrix

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Strictly Diagonally Dominant Matrix（严格对角占优矩阵）是矩阵的一种性质。定义：对 n×n 矩阵 A，若对每一行 i 满足 |a_ii| > Σ_{j≠i} |a_ij|（对角线元素的绝对值严格大于该行所有非对角线元素绝对值之和），则称 A 为严格对角占优矩阵。两个关键数学推论：(1) Levy-Desplanques 定理：严格对角占优矩阵必然可逆（行列式非零，非奇异）；(2) 存在性：严格对角占优矩阵的逆的无穷范数有界 ||A⁻¹||∞ ≤ 1 / min_i(|a_ii| - Σ_{j≠i}|a_ij|)，即条件数可控。在 AffineQuant 中，该矩阵性质被用作 Gradual Mask 设计的目标和验证标准——只要仿射矩阵 A 保持严格对角占优，其逆矩阵 A⁻¹ 必然存在且数值稳定，使得 XA⁻¹Q(AW) 的计算有效。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 AffineQuant 上下文中，严格对角占优的演化过程：
```
初始状态（epoch=0, GM radius=0）：
A₀ = diag(s₁,...,s_d)       # 纯对角矩阵
行 i: |s_i| > 0 = Σ_{j≠i} 0  → 严格对角占优 ✓

优化中期（epoch=e, GM radius=r）：
Aₑ 的结构: 对角线=1附近、半径r内的非对角线=α倍值、r外=0
行 i: |a_ii| vs Σ_{j≠i,|i-j|≤r} |α·a_ij|
只要 α < min_i(|a_ii| / Σ_{|i-j|≤r}|a_ij|) → 严格对角占优 ✓

优化完成（epoch=f, radius=d）：
所有元素可学习，但 α 持续抑制非对角线幅度
论文实验（Appendix A.6）展示各 block 的 A 矩阵热力图：
所有矩阵均呈现严格对角占优模式（对角线亮，非对角线暗淡）
```
关键：Gradual Mask 通过两阶段机制维持严格对角占优——(1) 前向 Hadamard 积 A* = A ∘ GM 缩小非对角线元素；(2) 反向传播中 GM 调节非对角线梯度更新速率。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在机器学习中，严格对角占优很少作为显式约束，但其导出性质（可逆性、条件数上界）在需要矩阵求逆的场合非常有价值。AffineQuant 的创新在于通过 GM 隐式实现了该约束，无需显式投影或条件检查。在数值线性代数中，严格对角占优矩阵的逆可使用 Jacobi 迭代高效近似，迭代保证收敛。更广泛地，在偏微分方程有限差分法、网络图分析（图 Laplacian）、Markov 链转移矩阵分析中，严格对角占优性质也有重要应用。

涉及论文标题：
- AffineQuant Affine Transformation Quantization for Large Language Models

---
