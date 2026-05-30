## Doubly Stochastic Matrix / Birkhoff Polytope

术语是什么？

双随机矩阵是 $n \times n$ 非负矩阵，满足每行之和 = 1 且每列之和 = 1。所有 $n \times n$ 双随机矩阵的集合构成 Birkhoff polytope $\mathcal{M}^{\text{res}}$。Birkhoff-von Neumann 定理表明其顶点恰好是所有 $n \times n$ 置换矩阵，因此任何双随机矩阵可表示为置换矩阵的凸组合。关键性质：(1) 谱范数 ≤ 1（非膨胀）；(2) 对矩阵乘法封闭（乘积仍为双随机）；(3) 作用于向量时保持均值和范数界限。

从算法pipeline角度拆解：

在 mHC 中 $\mathcal{H}_l^{\text{res}}$ 约束为双随机矩阵 → $\mathcal{H}_l^{\text{res}} \mathbf{x}_l$ 每个输出 stream 是 n 个输入 stream 的凸组合 → 信号均值全局保持。跨 L 层后复合映射 $\prod_{i=1}^{L} \mathcal{H}_{L-i}^{\text{res}}$ 仍为双随机 → 深层信号稳定。n=1 退化为标量 1，完全恢复 identity mapping。

术语一般如何实现？如何使用：

通过 Sinkhorn-Knopp 算法或 Bregman 投影进行约束优化。除 mHC 外，也广泛用于最优传输（耦合矩阵）、图匹配和 ranking 问题。

涉及论文标题：
- mHC Manifold-Constrained Hyper-Connections

---
