## EM Algorithm for Codebook Initialization (Hessian-Weighted)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
EM（Expectation-Maximization）Algorithm for Codebook Initialization 是 GPTVQ 中用于初始化 VQ codebook 的聚类算法。与标准 k-means 不同，GPTVQ 的 EM 算法引入 Hessian 加权的距离度量：目标函数为加权平方距离 min_{I,c} Σ_m Σ_{i∈I_m} (x^{(i)} - c^{(m)})^T D^{(i)} (x^{(i)} - c^{(m)})，其中 D^{(i)} = diag(1/H^{-1}_{jj}, ...) 是从校准数据的 Hessian 逆矩阵提取的对角权重矩阵。D^{(i)} 衡量每个维度的"重要性"——H^{-1} 对角元越大（该维度对输出影响越大），该维度的量化误差惩罚越大。E-step 用加权距离（公式 5）分配每个 d 维向量到最优质心；M-step 用 Moore-Penrose 伪逆闭式解 c^{(m)} = (Σ D^{(i)})^{+} Σ D^{(i)} x^{(i)} 更新质心。GPTVQ 默认使用 100 次 EM 迭代（消融显示继续迭代仍有小幅改善）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# GPTVQ EM 初始化伪代码
# 输入: 权重矩阵 W (r × m), 逆 Hessian 对角 D, codebook 大小 k, VQ 维度 d
# 输出: codebook C ∈ R^{d × k}

# 初始化质心（Mahalanobis 或 k-Means++ seeding）
C = initialize_centroids(W_reshaped_to_d_vectors, k)

for iter in range(100):  # 100 次迭代（默认）
    # === E-step: 分配最优质心 ===
    for each d-dim vector x_i in W:
        best_m = 0, best_dist = inf
        for m in range(k):
            diff = x_i - C[:, m]  # [d]
            # Hessian 加权距离: d_j = 1/H^{-1}_{jj}
            dist = Σ_{j=0}^{d-1} diff[j]^2 / H_inv_diag[j]
            if dist < best_dist:
                best_m, best_dist = m, dist
        assignments[i] = best_m
    
    # === M-step: 闭式解更新质心 ===
    for m in range(k):
        # 收集分配给质心 m 的所有向量
        X_m = [x_i for i where assignments[i] == m]
        if len(X_m) == 0: continue
        # 伪逆闭式解: c_m = (Σ D_i)^{+} · (Σ D_i · x_i)
        sum_D = Σ_i D_i              # [d × d] 对角矩阵
        sum_Dx = Σ_i D_i @ x_i       # [d]
        C[:, m] = pinv(sum_D) @ sum_Dx

return C
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GPTVQ 的 EM 实现使用 Mahalanobis initialization 作为 seeding 方法（比 k-Means++ 快 3-5×）：按马氏距离排序所有数据点，均匀采样 k 个质心，确保质心在代表性距离上分布。PyTorch 实现，支持 GPU 加速。D^{(i)} 在实际中简化为仅使用逆 Hessian 对角元（而非完整 d×d 子矩阵），论文发现两者性能无差异。EM 是 GPTVQ 中最耗时的步骤（对高 d 尤为显著），占总运行时间的 50%+。

涉及论文标题：
- GPTVQ: The Blessing of Dimensionality for LLM Quantization

---
