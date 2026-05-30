## Fine-Grained Per-Rank Routing（逐秩细粒度路由）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fine-Grained Per-Rank Routing 是 MoDE 的核心路由创新。传统 LoRA-MoE 的 router 输出 m 维权重，所有 r 个 rank 维度**绑定**在一起路由——即一次 routing decision 选择整个 up-projection 矩阵 B^i。MoDE 的 router 为每个 rank j 独立输出 m 维权重：W_R ∈ R^{r×P×m}，其中 W_{R;j} ∈ R^{P×m} 负责第 j 个 rank。对输入 x，第 j 个 rank 的 routing 权重为 R_j(x) = softmax(x · W_{R;j})。这允许 "B 的第 1 列选 expert 1，第 2 列选 expert 3，第 3 列选 expert 2，第 4 列选 expert 1" 的细粒度组合，可表达 m^r 种不同的 up-projection 矩阵组合（vs 传统 LoRA-MoE 的 m 种），在同等参数量下极大提升模型表达力。

从算法pipeline角度拆解术语：
```
# 输入: x ∈ R^{1×P}
# 共享 A: a_j ∈ R^{P×1} for j=1..r
# Expert up-projection: b_j^i ∈ R^{Q×1} for i=1..m, j=1..r
# Router: W_{R;j} ∈ R^{P×m} for j=1..r

# 传统 LoRA-MoE routing（所有 rank 绑定）
R(x) = softmax(x @ W_R)  ∈ R^{1×m}      # m 种选择
# 对每个 expert i: 贡献 = R_i(x) * x@(Σ_j a_j⊗b_j^{iT})

# MoDE fine-grained routing（每 rank 独立）
dyadic_sum = 0
for j in range(r):                        # 对每个 rank 维度
    R_j = softmax(x @ W_{R;j})            # [1×m] 独立路由权重
    h_j = x @ a_j                         # 标量
    for i in range(m):
        dyadic_sum += R_j[i] * h_j * b_j^{iT}  # 加权 rank-one 贡献
y = x @ W0 + dyadic_sum

# 组合空间: m^r >> m (例如 m=4,r=4: 256 vs 4)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Router 参数 W_R ∈ R^{r×P×m}，推理时需计算 r 次 softmax（每次输出 m 维），计算开销 O(r×P×m)。
- 典型配置：MoDE 4×4（4 experts × rank 4）即可超越 LoRA 64（ROUGE-L 60.18 vs 56.11），参数量仅 1.90%。
- 局限性：router 计算开销随 rank r 增大而线性增长；论文指出在实时应用场景中路由计算可能成为瓶颈。

涉及论文标题：
- MoDE: Effective Multi-task Parameter Efficient Fine-Tuning with a Mixture of Dyadic Experts

---
