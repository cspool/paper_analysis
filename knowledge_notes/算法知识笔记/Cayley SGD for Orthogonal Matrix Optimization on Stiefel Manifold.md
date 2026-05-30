## Cayley SGD for Orthogonal Matrix Optimization on Stiefel Manifold

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
Cayley SGD 是一种在 Stiefel 流形 V_k(R^n) = {X ∈ R^{n×k} : X^T X = I} 上执行随机梯度下降的黎曼优化方法（Li et al., ICLR 2020）。Stiefel 流形是所有 n×k 列正交矩阵构成的黎曼子流形，在此流形上直接用标准 SGD 更新会破坏正交性。Cayley SGD 通过迭代 Cayley 变换实现高效 retraction（将欧几里得空间的更新投影回 Stiefel 流形）：Y = X + α/2 · W(X+Y)，其中 W=GX^T-XG^T 为切线空间中的反对称矩阵。相比闭式 Cayley 变换（需矩阵求逆 O(n³)），迭代形式仅需矩阵乘法，s=2-3 次迭代即可达到正交精度 ~10^{-5}。每次迭代增加约 6n³ 额外计算量（vs 标准 SGD），约 2× 慢于标准 SGD。SpinQuant 和 OSTQuant 使用 Cayley SGD 端到端微调旋转矩阵。

从算法pipeline角度拆解术语，给出具体例子。
```
# Cayley SGD with Momentum (DartQuant Appendix B.2, Algorithm 3)
M_{k+1} = β M_k - G(X_k)                              # 动量更新
W_hat = M_{k+1}@X_k^T - 0.5*X_k@(X_k^T@M_{k+1}@X_k^T) # 反对称: n³+2n³
W_k = W_hat - W_hat^T                                   # 确保反对称性
M_{k+1} = W_k @ X_k                                     # 动量投影: n³
α = min(lr, 2q/(||W_k||+ε))                             # 自适应步长 (q=0.5)
Y_0 = X_k + α M_{k+1}
for i=1 to s:                                           # 迭代 Cayley: 每次 n³
    Y_i = X_k + α/2 * W_k @ (X_k + Y_{i-1})
X_{k+1} = Y_s                                           # 新正交矩阵
总额外计算量: ≈ 6n³（vs 标准 SGD 的 O(n²)）
```

术语一般如何实现？如何使用？
开源实现：SpinQuant（https://github.com/facebookresearch/SpinQuant）的 Cayley SGD 优化器。在 DartQuant 实验中，Cayley SGD 100 步耗时 8.2h（Adam 8.1h）。主要开销：步骤 5（W_hat 计算 ~3n³）、步骤 7（动量投影 ~n³）、步骤 9-11（迭代 Cayley ~2n³）。DartQuant 提出 QR-Orth 替代 Cayley SGD：QR-Orth 100 步耗时 5.7h（1.44× 加速），且 Whip Loss 配合下 6 步即达 Cayley 100 步效果（41× 综合加速比）。

涉及论文标题：
- DartQuant Efficient Rotational Distribution Calibration for LLM Quantization

---
