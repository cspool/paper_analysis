## Model Sparsity Ratio (Ω_MSR) / Effective Sparsity Ratio (Ω_ESR) (模型稀疏率 / 有效稀疏率)

术语是什么？

Ω_MSR 和 Ω_ESR 是 Elastic Attention 提出的形式化稀疏度量。Ω_MSR = 使用 SA 的 head 比例（不考虑 per-head 内部剪枝率），Ω_ESR = 综合 head 比例和 per-head token 剪枝率的实际 attention 覆盖比例。Ω_ESR 用于 fair comparison——不同 SA 方法（SSA vs XA）的 per-head pruning ratio 不同，Ω_ESR 折算到"实际被 attention 覆盖的 token 比例"。

从算法pipeline角度拆解术语。

```
Ω_MSR = (1/(H·L)) · Σ_h Σ_l I[π^{(l,h)} = SA]

Ω_ESR = (1/(H·L)) · Σ_h Σ_l ρ^{(l,h)}
# ρ: FA head=0, SA head=ρ_SA (e.g., SSA prunes 90% → ρ=0.9)

# Example: 50% heads SA with 80% token pruning
# Ω_MSR = 0.5, Ω_ESR = 0.5 × 0.8 = 0.4
```

术语一般如何实现？如何使用？

Ω_MSR 用于训练约束（L_diff = Ω_MSR - t）。Ω_ESR 用于推理效率对比——RULER 实验中 Elastic Attention 在长序列下 Ω_ESR 低于同类方法（更少 token 被 attention 覆盖），证明 adaptive sparsity 比 static assignment 更 effective。Figure 8 使用 Ω_ESR 做 performance-vs-sparsity Pareto frontier 对比。

涉及论文标题：
- Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers（Expectation-Maximization, EM）算法来学习 RoPE-可交换码本。由于可交换码本需满足特定的 $2 \times 2$ 矩阵形式约束（$C = [[x, y], [-y, x]]$），且码本训练本质上是聚类问题，EM 比梯度下降更直接有效。EM 算法的目标：将 N 个 2D 校准向量 $k \in \mathbb{R}^{N \times 2}$ 分配到 $N_{c'}^2$ 个聚类中心（由 $N_{c'}$ 个 $2 \times 2$ 可交换子码本组合而成），最小化 MSE loss。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**CommVQ 码本学习的 EM 算法**：

```
# 输入：校准集 K ∈ R^{N×2}（每个 2D 子空间的 key 向量）
# 参数：码本 C_K^j = {C_K^{j0}, ..., C_K^{j(N_c'-1)}}
#      每个 C_K^{jl} = [[x_l, y_l], [-y_l, x_l]]

# 构建 N_c'^2 个聚类中心
for a in 0..N_c'-1:
    for b in 0..N_c'-1:
        c_ab = [1,0] @ C_K^j[a] + [0,1] @ C_K^j[b]  # [2]

while not converged:
    # E Step: 固定码本，soft assignment + temperature
    D = L2_distance_matrix(K, cluster_centers)  # [N, N_c'^2]
    W = softmax(-D / T)                          # soft assignment 权重
    m = W^T @ K                                  # 加权均值
    N_counts = sum(W, dim=0)                     # 每中心分配数

    # M Step: 固定分配，闭式解更新码本参数
    phi = (T^T @ S @ T)^{-1} @ T^T @ S @ m      # 闭式解

    T = T * decay_rate  # 温度退火
```

**闭式解的矩阵形式**：
$$\phi^* = (T^T S T)^{-1} T^T S m$$

其中 $T \in \{-1,0,1\}^{(2N_{c'}^2) \times (2N_{c'})}$ 是编码聚类中心与码本关系的常数矩阵，$S = \operatorname{diag}(N_{ij})$ 是分配计数的对角矩阵。

术语一般如何实现？如何使用？

为稳定训练，CommVQ 采用两项关键技术：(1) **Soft clustering assignment**：不用 hard assignment，而是根据距离对每个数据点到所有中心赋权重 $W_{ij} = e^{-D_{ij}/T} / \sum_k e^{-D_{ik}/T}$，防止死聚类中心；(2) **Temperature annealing**：温度 T 从高到低指数衰减。对于 1-bit 量化（$N_{c'}=64$）有 4096 个聚类中心，soft assignment 是关键。R 轮迭代式残差量化（每轮在上轮误差上拟合新码本）。训练在 FineWeb-Edu 校准集上进行，每层独立训练。

涉及论文标题：
- CommVQ: Commutative Vector Quantization for KV Cache Compression

---
