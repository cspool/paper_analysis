## Newton-Schulz Iteration (牛顿-舒尔茨迭代 / Matrix Orthogonalization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Newton-Schulz 迭代是一种通过矩阵多项式迭代逼近矩阵极分解（polar decomposition）中正交因子的数值方法。给定矩阵 M，其极分解为 M = U P（U 正交，P 半正定），Newton-Schulz 迭代通过纯矩阵乘法（无 SVD/QR 分解）逼近 U = M (M^T M)^(-1/2)。在 Muon 优化器中，Newton-Schulz 用于将梯度动量矩阵 M_t 近似正交化为 O_t ≈ U V^T（即 M_t = U Σ V^T 的奇异向量乘积），用 5 次迭代逼近 (M M^T)^(-1/2) M。使用 5 阶多项式 f(x) = a x + b x³ + c x⁵（a=3.4445, b=-4.7750, c=2.0315），通过对动量矩阵在 Frobenius 归一化后反复应用矩阵乘法实现。该迭代的核心优势：(1) 仅需矩阵乘法——可在 GPU tensor core 上高效执行（bf16），<1% 总训练 FLOPs 开销；(2) 远快于 SVD（O(n³) 且 GPU 不友好）；(3) 5 步迭代足够产生良好正交近似。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Newton-Schulz 迭代在 Muon 中的计算过程：

```
输入: 梯度动量矩阵 M ∈ R^{A×B} (经过 Nesterov 外推)
输出: 近似正交化的更新矩阵 O ≈ (M M^T)^{-1/2} M

# 1. Frobenius 归一化
X = M / ||M||_F                    # 确保 X 的谱范数 ≤ 1，迭代稳定

# 2. 5 阶 Newton-Schulz 迭代 (quintic polynomial)
# 系数: a=3.4445, b=-4.7750, c=2.0315
for step in range(5):
    X_tmp = X @ X^T                 # [A, B] × [B, A] → [A, A]
    B = b * X_tmp + c * (X_tmp @ X_tmp)  # bX² + cX⁴
    X = a * X + B @ X               # aX + bX³ + cX⁵ (等价形式)

return X                            # X ≈ U V^T
```

迭代的数学原理：设 M 的奇异值为 σ₁, σ₂, ..., σ_min(A,B)，则第 k 次迭代后 X 的奇异值变为 f^(k)(σ_i / ||M||_F)，其中 f(x) = ax + bx³ + cx⁵。系数被设计为使 f 在零点导数最大且值域限制在 [0.5, 1.5] 内，这使得 X 的奇异值被推向 1（"半正交化"），而非完全正交化（精确正交化在实际训练中反而性能更差）。N=5 是精度-效率平衡点：N=10 产生更精确的正交化但无性能提升。

与经典 Newton-Schulz（3 阶 X_{k+1} = 1.5 X_k - 0.5 X_k X_k^T X_k）的区别：5 阶多项式通过待定系数法手工调优，在零点附近收敛更快（对小奇异值放大效果更好）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
- Muon 原版实现：`zeropower_via_newtonschulz5` 函数，约 10 行 PyTorch 代码，所有矩阵运算在 bf16 下执行
- 加速变体：CANS (Chebyshev-Accelerated Newton-Schulz, arXiv:2506.10935) 使用 Chebyshev 交替定理和 Remez 算法推导最优系数，改进收敛速度
- 替代方案：AuON (arXiv:2509.24320) 提出 O(n) 的 hyperbolic-cosine RMS 缩放替代 O(n²) 的 Newton-Schulz，在保持性能的同时降低复杂度
- NVIDIA NeMo 的 Scion 优化器实现了可配置系数的 Newton-Schulz（"simple" / "quintic" / "polar_express"）
- 使用时需注意：若 A < B（矩阵更"瘦"），转置后计算可减少计算量（因 X @ X^T 的维度为 min(A,B) × min(A,B)）；N 通常设 5，更多迭代无额外收益

涉及论文标题：
- Muon is Scalable for LLM Training
