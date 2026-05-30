## Ridge Leverage Score (Ridge 杠杆分数)

术语是什么？
Ridge Leverage Score 是一种统计度量，用于衡量线性回归中每个数据点对模型拟合的影响力。在 UniQL 的 LLM 压缩语境下，该分数用于评估 MLP 中间层每个通道（$D_{int}$ 维度）对输出重建的重要性。给定校准样本的中间激活相关性矩阵 $C = \mathbf{X}_{int}^{\top} \mathbf{X}_{int} \in \mathbb{R}^{D_{int} \times D_{int}}$，ridge leverage scores 定义为：
$$s = \operatorname{diag}\left(C(C + \lambda I)^{-1}\right), \quad s \in \mathbb{R}^{D_{int}}$$
其中 $\lambda$ 是 ridge 正则化强度（UniQL 设 $\lambda=1$），$I$ 为单位矩阵。该分数对每个通道的"杠杆"进行量化——高分数通道对激活重建更重要，排序矩阵 $\mathbf{S}_m = I[:, \operatorname{argsort}(s)]$ 将权重列按重要性降序排列，使得剪枝时只需丢弃末尾（低杠杆）列。

从算法pipeline角度拆解：
在 UniQL 的结构化权重排序 pipeline 中（Algorithm 1），该分数的工作流程：
1. 从 Alpaca 校准集采样 128 个样本（seq_len=2048），计算每个样本的 MLP 中间激活 $\mathbf{X}_{int}^i = \sigma(\mathbf{X}_h^i \mathbf{W}_g) \odot \mathbf{X}_h^i \mathbf{W}_u$
2. 平均所有样本的相关性矩阵 $C = \frac{1}{N}\sum_{i=1}^N \mathbf{X}_{int}^{i\top} \mathbf{X}_{int}^i$
3. 计算 $s = \operatorname{diag}(C(C + \lambda I)^{-1})$
4. 按 $s$ 降序构造排序矩阵 $\mathbf{S}_m$
5. 重排 MLP 权重：$\mathbf{W}_u' = \mathbf{W}_u \mathbf{S}_m$，$\mathbf{W}_g' = \mathbf{W}_g \mathbf{S}_m$，$\mathbf{W}_d' = \mathbf{S}_m^\top \mathbf{W}_d$

伪代码：
```
# 输入: 校准激活 X_int ∈ R^{N × T × D_int}, λ=1
C = einsum("nti,ntj->ij", X_int, X_int) / N    # [D_int, D_int]
C_reg = C + λ * I                               # ridge 正则化
C_inv = solve(C_reg, I)                         # 矩阵求解 (非伪逆)
s = diag(C @ C_inv)                             # [D_int] ridge leverage scores
idx = argsort(s, descending=True)               # 按重要性排序
S_m = I[:, idx]                                 # 排序矩阵
W_u, W_g = W_u @ S_m, W_g @ S_m                # 重排列
W_d = S_m^T @ W_d                               # 重排行
```

术语一般如何实现？如何使用？
Ridge leverage score 在 statistics 和 numerical linear algebra 中有成熟的理论基础（McCurdy, 2018），UniQL 首次将其应用于 LLM 权重排序。相比基线 MoDeGPT 的伪逆（Moore-Penrose inverse, O(n³), FP64），ridge leverage scores 只需一次矩阵求解（可用 Cholesky 分解或 LU 分解），在 BF16 下运算即可，对于 Llama-3.1-8B 的 MLP 层（$D_{int}=14336$），从 20.58 分钟降至数秒（20× 加速）。且 ridge leverage score 的排序结果对不同剪枝率是等价的（重新排序不需要重新计算），因为排序矩阵 $\mathbf{S}_m$ 基于原始满秩权重计算，而非剪枝后的子矩阵。这支持一次排序 → 多剪枝率的 one-pass 设计。

涉及论文标题：
- UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs
