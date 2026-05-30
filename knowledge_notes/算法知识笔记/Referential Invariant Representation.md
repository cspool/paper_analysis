## Referential Invariant Representation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Referential Invariant Representation（参照不变表示）是 GraphMETRO 的核心概念（Definition 1）：给定图 G、随机变换 τ 和 reference model ξ_0，函数 ξ* 产生的表示被称为 referentially invariant w.r.t. τ，当且仅当 ξ_0(G) ≈ ξ*(τ(G))，∀G ∈ supp(D_s)。本质是通过 reference model ξ_0 的表示空间作为"锚点"来对齐所有 expert 的输出——每个 expert ξ_i 学习对 τ_i(G) 编码以匹配 ξ_0(G)（原图在 reference model 中的表示），而非要求 ξ_i(τ_i(G)) = ξ_i(G)（自不变性）。这解决了多个独立 expert 表示空间不兼容的问题——所有 expert 通过同一 reference space 间接对齐，使 weighted sum aggregation 在数学上有意义。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Referential Invariant Representation 的训练
for G in D_s:
    z_0 = ξ_0(G)              # reference "anchor" 表示（原图）
    G_trans = τ^{(k)}(G)      # 应用组合 shift transforms
    z_i = ξ_i(G_trans) for i=0..K  # 各 expert 对变换后图的编码
    w = ϕ(G_trans)            # gating 预测 shift mixture
    h = Σ Softmax(w)[i] · z_i # 加权聚合
    d = (1/n)·||h - z_0||_F   # Frobenius norm distance
    L_align = λ · d           # λ=1 (所有实验)
    # h(τ_i(G)) = h(G)  (Theorem 1, 单 shift 不变性)
    # h(τ^{(k)}(G)) ≈ h(G) (Theorem 2, 组合 shift 近似不变性)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现关键点：(1) ξ_0 在源数据 D_s 上正常训练，作为"in-distribution expert"；(2) alignment term 权重 λ 至关重要——λ=0 时 WebKB 41.11%→18.79%，验证 alignment 的必需性；(3) Frobenius norm 简单有效：d(z₁,z₂) = (1/n)·||z₁ - z₂||_F = (1/n)·√(Σ(z₁ᵢ - z₂ᵢ)²)；(4) alignment 同时在 τ^{(k)} 组合上执行，保证对复合偏移的不变性；(5) L2 不反向传播到 gating model，避免 gating 和 alignment 目标冲突。

涉及论文标题：
- GraphMETRO Mitigating Complex Graph Distribution Shifts via Mixture of Aligned Experts
