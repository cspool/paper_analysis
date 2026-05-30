## Expert Decomposition (Low-Rank)

术语解释
专家分解是利用低秩分解技术（如SVD、MPO）将MoE中较大的expert权重矩阵分解为更小的矩阵乘积，从而减少参数量，同时保持计算表达能力。

术语是什么？
低秩分解的直觉：expert权重矩阵通常存在冗余，可以用低秩近似表示：
- **MPOE**：使用矩阵乘积算子（MPO）——一种源自量子多体物理的张量分解技术——将expert权重矩阵分解为中心张量（保留大部分参数和核心信息）+ 若干辅助张量（较小，作为中心张量的补充）。同一层所有expert共享相同的中心张量，大幅减少每层总参数。
- **MC-SMoE**：先合并专家分组，再对合并后专家使用低秩分解（基于合并后专家秩更低的观察）
- **MoE-I²**：识别每个expert的重要性I_{i,j}，为重要expert分配更高秩、不重要expert分配更低秩。秩分配公式：r_{i,j}=⌊(I_{i,j}+ε)^α / Σ(I_{i,j}+ε)^α · R_a · M_i⌋

从算法pipeline角度拆解术语。
```
# Low-Rank Expert Decomposition (SVD-based)
def decompose_expert(W, rank):
    # W: [d_out, d_in]
    U, S, V = SVD(W)
    # 保留前rank个奇异值
    U_r = U[:, :rank]          # [d_out, rank]
    S_r = S[:rank]             # [rank]
    V_r = V[:rank, :]          # [rank, d_in]
    # 分解为两个小矩阵
    A = U_r @ diag(sqrt(S_r))  # [d_out, rank]
    B = diag(sqrt(S_r)) @ V_r  # [rank, d_in]
    # FFN: σ(x @ B^T) @ A^T   (替代 x @ W^T)
    return A, B

# 参数量：d_out*d_in → 2*d_out*rank
# 压缩比 ≈ d_in / (2*rank)
```

术语一般如何实现？如何使用？
- 适用于参数量大的expert进行分解
- 可结合其他压缩技术（量化+分解，剪枝+分解）
- MPO分解在量子物理领域成熟，应用于NN是一种跨学科迁移
- 需要注意分解后的精度恢复（可能需要微调）

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models
- DeRS Towards Extremely Efficient Upcycled Mixture-of-Experts Models

**DeRS-LM (Low-rank Matrix-based DeRS Upcycling)**：DeRS 论文中的 DeRS-LM 采用低秩矩阵 A∈R^{d×r} 和 B∈R^{r×d_h}（A 随机初始化，B 零初始化）表示专家专属增量权重 Δ_i：
- 专家权重合成为 W_i = W_shared + A_i·B_i，其中 W_shared 从原始 FFN 权重初始化
- 训练参数从 N·d·d_h 降至 d·d_h + N·r·(d+d_h)
- rank r=1 时仅增加 ~2.4M 参数（vs Vanilla Upcycling 的 ~2.5B），实现 1041× 参数减少
- 关键设计：B 零初始化确保初始 Δ=0（即初始专家权重等于原始 FFN 权重，保持 upcycling 的 warm-start 特性）
- 与传统的 expert 低秩分解不同：DeRS-LM 分解的是 delta 权重 Δ_i 而非完整的 expert 权重 W_i；base weight W_shared 保持完整 dense
- 当 pretrained dense model 未经过先验微调时，推荐使用 DeRS-LM（低秩矩阵能进行全局修改，即使 rank 很低也能有效调整所有元素）

---
