## Dyadic Product / Rank-One Adapter (in LoRA)（并矢积 / 秩一适配器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dyadic Product（并矢积，也称 outer product / 外积）是两个向量的矩阵乘积：给定 u ∈ R^{p×1} 和 v ∈ R^{q×1}，其 dyadic product u ⊗ v = u·v^T ∈ R^{p×q}。在 LoRA 中，ΔW = A·B^T 可以按列分解为 r 个 rank-one dyadic product 之和：ΔW = Σ_{j=1}^r (a_j ⊗ b_j)，其中 a_j 是 down-projection 矩阵 A 的第 j 列，b_j 是 up-projection 矩阵 B 的第 j 列。每个 (a_j ⊗ b_j) 构成一个 rank-one adapter，捕获 weight matrix 在特定 rank 维度上的一个方向性变化。MoDE 论文的关键洞察：这种 dyadic decomposition 允许对每个 rank 维度独立进行混合专家（MoE）路由——即每个 rank j 可以有 m 个备选 up-projection 向量 {b_j^1, ..., b_j^m}，router 为每个 rank 独立选择。

从算法pipeline角度拆解术语：
```
# LoRA update 的 dyadic 分解
# A ∈ R^{P×r}, B ∈ R^{Q×r}
# a_j = A[:, j] ∈ R^{P×1}, b_j = B[:, j] ∈ R^{Q×1}

ΔW = A @ B^T                           # r×(P×Q) 低秩矩阵
    = Σ_{j=1}^r (a_j ⊗ b_j)            # r 个 rank-1 矩阵之和
    = Σ_{j=1}^r (a_j @ b_j^T)           # 每个外积贡献一个秩一更新

# 前向计算（对单个 token x ∈ R^{1×P}）
h = x @ A                               # [1×r]  共享 down-projection
dyadic_sum = 0
for j in range(r):
    h_j = h[0, j]                       # 标量
    dyadic_sum += h_j * b_j^T           # h_j ∈ R, b_j^T ∈ R^{1×Q} → [1×Q]
y = x @ W0 + dyadic_sum
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Rank-one adapter 是 MoDE 的核心组件：每个 dyadic term (a_j ⊗ b_j^i) 构成一个独立可路由的 expert。对 rank r 的 up-projection 矩阵，共有 m×r 个 rank-one expert（m 个备选 per rank j）。
- 广义 rank-p adapter（MoDE m×r×p）：将每 p 列 A 和 B 合并为一个 group adapter A_k ∈ R^{P×p}, B_k^i ∈ R^{Q×p}。MoDE 1×r×r = 标准 LoRA rank r，MoDE m×r×r = LoRA-MoE-SD。
- 实现上可通过标准 PyTorch 矩阵操作完成，无需特殊 kernel。

涉及论文标题：
- MoDE: Effective Multi-task Parameter Efficient Fine-Tuning with a Mixture of Dyadic Experts

---
