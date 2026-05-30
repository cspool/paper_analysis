## α-entmax (Alpha-entmax Sparse Activation Function)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

α-entmax 是 softmax 的稀疏泛化，由 Peters et al. (2019, ACL) 在《Sparse Sequence-to-Sequence Models》中提出。它基于 Tsallis α-entropy 构建，通过参数 α > 1 控制输出概率分布的稀疏程度。核心公式：
$$\alpha\text{-entmax}(\mathbf{s})_i = [(\alpha - 1)s_i - \tau]_{+}^{1/(\alpha - 1)}$$
其中 [x]_+ = max(x, 0) 即 ReLU 截断，τ ∈ R 是归一化常数（阈值），保证输出和为 1。

α 值与稀疏度关系：α → 1：逼近 softmax（稠密，全非零）；α = 1.5：适中稀疏（实践中 ~95% 注意力权重为零）；α = 2.0：sparsemax（高度稀疏 ~99% 零）；α → ∞：逼近 argmax。核心性质：可微分稀疏性——输出天然含有精确零，且整个过程可微支持端到端梯度训练；数据依赖性——稀疏模式由输入 logits 自适应决定，无需预定义固定 mask。

从算法pipeline角度拆解术语。

**α-entmax 替代 softmax 的 Attention Pipeline**：
```
S = QK^T / √d                     // n×n (不物化)
// 对每行 i: 用 Halley-Bisection 求解 τ_i (3 次迭代)
P_i = [(α-1)S_i - τ_i]_+^{1/(α-1)} // S_ij < τ_i/(α-1) → 0
O = PV                              // 零权重不贡献
```
**α 退火训练**（AdaSplash）：从 α=1.0 (dense) 线性增至目标值 (1.5/2.0)，over 1B tokens，避免直接稀疏训练的不稳定性。

术语一般如何实现？如何使用？

核心难点是求 τ：Sorting-based（α=1.5/2.0 精确，O(n log n)，GPU 效率低）；Bisection（线性收敛，~23 次迭代）；Halley-Bisection（三次收敛，~3 次迭代）。GPU 实现：利用 α-entmax Jacobian 的稀疏性加速反向传播；块方式累积 f/f'/f'' 避免物化 S。使用方式：`pip install adasplash`，`output = adasplash(Q, K, V, alpha=1.5)`。

涉及论文标题：
- AdaSplash: Adaptive Sparse Flash Attention
