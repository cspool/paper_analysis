## Quantization-Aware SVD Decomposition (QSVD / 量化感知SVD分解)

术语是什么？
Quantization-Aware SVD Decomposition (QSVD) 是 UniQL 提出的一种针对低比特（INT4）量化场景优化的奇异值分解策略。在 LLM 的 value-output 权重对 $(\mathbf{W}_v, \mathbf{W}_o)$ 的联合压缩中，先通过连续两次 SVD 分解排序权重通道：$C^{1/2}\mathbf{W}_v = \mathrm{U}_v \Sigma_v \mathbf{V}_v^{\top}$，然后 $\mathrm{SVD}(\Sigma_v \mathbf{V}_v^{\top} \mathbf{W}_o) = \mathbf{U}\Sigma\mathbf{V}^{\top}$。最终排序权重为 $\mathbf{W}_v = C^{-1/2} \mathbf{U}_v \mathbf{U} \Sigma$，$\mathbf{W}_o = \mathbf{V}^{\top}$。

QSVD 的核心创新在于：将特征值对角阵 $\Sigma$ 融合到 $\mathbf{U}$ 而非 $\mathbf{V}$，使排序后的 $\mathbf{W}_v = (\mathbf{U}\Sigma)$ 而非传统 $\mathbf{W}_v = \mathbf{U}$。其关键洞察是：低比特量化对量化组内的数值分布高度敏感。SVD 的特征值 $\sigma_i$ 呈长尾分布，若将其保留在 $\mathbf{V}$ 侧，$\mathbf{U}$ 列内数值跨度小但 $\mathbf{V}$ 行内跨度大；若融合到 $\mathbf{U}$ 侧，每列 $\mathbf{U}_i$ 乘以对应特征值 $\sigma_i$，则 $\sigma_i$ 自然充当该列的 group-wise 量化缩放因子，避免长尾特征值被 INT4 截断。

从算法pipeline角度拆解：
在 UniQL 的 MHSA value-output 权重排序中（Algorithm 3），QSVD 的流程为：
```
# 输入: W_v ∈ R^{D_h × D_hd}, W_o ∈ R^{D_hd × D_h}, 校准激活 X_h
# Step 1: 计算输入相关性矩阵
C = X_h^T @ X_h                                    # [D_hd, D_hd]
C_half = cholesky(C)                               # 或 sqrtm(C)

# Step 2: 第一次 SVD
U_v, Σ_v, V_v_T = SVD(C_half @ W_v)               # 分解激活加权的 value 投影

# Step 3: 第二次 SVD (joint decomposition)
U, Σ, V_T = SVD(Σ_v @ V_v_T @ W_o)                # 联合分解 value 和 output

# Step 4: QSVD - 融合 Σ 到 U (关键步骤)
W_v_sorted = inv(C_half) @ U_v @ U @ Σ            # U 乘以特征值，每列缩放
W_o_sorted = V_T                                    # V 保留为排序输出矩阵

# 量化时: 每个 group (128列) 共享缩放因子 s = max(|W_v_sorted[:,g]|) / 7
# 由于 Σ 已融合到 U，特征值自然地逐列缩放，避免 INT4 截断失真
```

定量效果：在 W4A16 Llama-3.1-8B 上 25% 剪枝率，不融合 Σ 时精度为 60.2%，融合后（QSVD）提升至 67.7%（+7.5%）；Qwen-2.5-7B 从 61.0% 提升至 64.0%（+3.0%）。

术语一般如何实现？如何使用？
QSVD 的实现依赖于标准数值线性代数库（如 PyTorch 的 `torch.linalg.svd`）。在 LLM 压缩 pipeline 中，QSVD 在权重排序阶段执行（云侧，单次），排序后的 $\mathbf{W}_v$ 列已按重要性降序排列，特征值缩放已内嵌在权重矩阵中。随后进行 GPTQ 量化时（group_size=128, INT4），每组的量化缩放因子 $s_g = \max(|\mathbf{W}_{(i,g)}|) / 7$ 自然受益于 Σ 的预缩放——重要列（大 $\sigma_i$）的权重大，不重要的列（小 $\sigma_i$）权重小，形成天然的数值分布梯度，使 INT4 量化误差最小化。

涉及论文标题：
- UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs
