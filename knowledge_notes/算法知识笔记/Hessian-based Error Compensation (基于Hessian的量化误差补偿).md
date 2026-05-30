## Hessian-based Error Compensation (基于Hessian的量化误差补偿)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Hessian-based Error Compensation 是 GPTQ 的核心机制，源自 Optimal Brain Quantization (OBQ，基于 LeCun 的 Optimal Brain Damage)。基本思想：量化一个权重列 w_j 后产生的 output error Δy = (w_j_hat - w_j) * X[j,:]，该误差可通过对剩余未量化列按 Hessian 逆矩阵 H^{-1} 的比例进行补偿来消除，从而最小化总体输出误差。

Hessian 矩阵 H = X X^T ∈ R^{c×c} 编码了输入激活的二阶统计信息（c 是输入通道数）。H^{-1} 的 (j,k) 元素表示：对第 j 列的量化误差应该以多大比例传导到第 k 列的权值补偿。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# GPTQ Algorithm (simplified)
# W: weight matrix ∈ R^{o×c}, X: calibration input ∈ R^{b×c}
# H = X^T X ∈ R^{c×c}  (Hessian)

H = X.T @ X  # 或 X @ X.T，取决于定义
H_inv = inverse(H)  # 实际使用 Cholesky 分解以提高稳定性

W_q = zeros_like(W)  # 量化后的权重
E = zeros(o, b)      # 累积误差

for j in range(c):  # 逐列量化
    # 量化第 j 列
    for i in range(o):
        W_q[i,j] = clamp(round(W[i,j] / s[i]), q_min, q_max) * s[i]
    
    # 计算量化误差（输出空间）
    err = (W_q[:,j] - W[:,j]).reshape(o, 1)  # [o, 1]
    
    # 用 Hessian 逆补偿剩余列
    for k in range(j+1, c):
        delta = H_inv[j,k] / H_inv[j,j]
        W[:,k] -= err * delta  # 补偿到权重上
    
    # 更新 Hessian 逆
    # 移除第 j 行/列后重新计算逆（实际用 Cholesky 更新更高效）
```

MoEQuant 的 AGQ 改进：传统 Hessian 对所有 token 等权，而 AGQ 将 gating coefficient c_i 引入 Hessian 计算：H = (X ⊙ √c)(X ⊙ √c)^T，使高亲和力 token 对 Hessian 的贡献更大，从而在误差补偿时更准确地保护关键 token 的表达质量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GPTQ (Frantar et al., 2022) 是这一方法的标准实现。其关键优化包括：（1）对 H 做 Cholesky 分解以提高求逆稳定性；（2）对权重列做随机顺序或按 Hessian 对角线排序的贪心量化顺序；（3）对较大矩阵分 block 量化（每 block 128 列）以减少内存。Hessian 的规模为 c×c（c 为 hidden size，如 4096），因此内存开销较大但仍在可接受范围。在 MoE 场景下，每个 expert 的 FFN 矩阵可以独立计算 Hessian。

涉及论文标题：
- MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance
