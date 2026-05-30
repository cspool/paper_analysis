## Centered Kernel Alignment (CKA) for Neural Network Representation Similarity

术语解释
CKA（Centered Kernel Alignment）由 Kornblith et al. (ICML 2019) 提出，是一种用于比较神经网络层间表示相似度的度量方法。CKA 基于 Hilbert-Schmidt Independence Criterion (HSIC)，对正交变换和神经元置换具有不变性——这是比较神经网络表示的关键特性，因为神经元排列不应影响网络功能。DiEP 论文使用 CKA 在两个场景：(1) 可视化不同 MoE 层内和层间的 expert-pair 相似度，作为非均匀剪枝的 motivation；(2) 计算 expert skipping 中 γ₂ 参数（专家输出相似度与平均相似度的比值）。

术语是什么？
Linear CKA 的计算公式：

CKA_linear(X, Y) = ∥X Y^T∥_F² / (∥X X^T∥_F · ∥Y Y^T∥_F)

其中 X ∈ R^(n×p₁)、Y ∈ R^(n×p₂) 为两个层的激活矩阵（n 样本数，p 特征维度）。完整 CKA（含 RBF kernel）流程：
1. 计算 Gram 矩阵 K = XX^T, L = YY^T
2. 使用 centering matrix H = I_n − (1/n)11^T 中心化
3. HSIC(K, L) = tr(K H L H) / (n−1)²
4. CKA(K, L) = HSIC(K, L) / √(HSIC(K,K) · HSIC(L,L))

CKA ∈ [0, 1]，1 表示完全相同的表示结构，0 表示正交。

从算法pipeline角度拆解术语（DiEP 中的使用）：
```
# CKA-based Expert Similarity in DiEP

# 1. Intra-layer CKA (每个 layer 内的 expert-expert 相似度矩阵)
for layer l in 1..L:
    for expert_i, expert_j in pairs(N):
        # 在校准数据上收集 expert 输出
        X_i = collect_expert_outputs(expert_i, D_cal)  # [n_samples, d_model]
        X_j = collect_expert_outputs(expert_j, D_cal)
        # Linear CKA
        CKA[l][i][j] = ∥X_i X_j^T∥_F² / (∥X_i X_i^T∥_F · ∥X_j X_j^T∥_F)

# 2. Inter-layer CKA (相邻层 expert 之间的相似度)
for layer l in 1..L-1:
    for expert_i in layer l, expert_j in layer l+1:
        CKA_inter[l][i][j] = cka_similarity(E_i^(l), E_j^(l+1))

# 3. Adaptive Skipping γ₂ 计算
γ₂ = ρ(y_e0, y_e1) / mean(ρ(y_ei, y_ej))  # 专家输出 CKA 相似度比
# 其中 ρ = linear CKA between expert outputs
# γ₂ > 1 → 两专家特别相似 → 更有可能跳过 e1
```

术语一般如何实现？如何使用？
- PyTorch 实现可参考 github.com/RistoAle97/centered-kernel-alignment
- DiEP 中使用 CKA 做 pre-pruning analysis（可视化动机），不参与 training 计算（仅推理时 skipping 使用）
- CKA 优点：对正交变换不变、可跨维度比较、比 CCA 特异性更好
- CKA 局限：对低方差主成分不敏感（Ding et al. 2021 指出 CKA 主要反映高方差维度的相似性）
- 在 MoE 剪枝上下文中，也可用 cosine similarity 替代（如 Expert Trimming 论文），但 CKA 能捕捉更丰富的结构性相似关系

涉及论文标题：
- DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning
