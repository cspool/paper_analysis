## Training Success Rate in MoE

术语解释
Training Success Rate 是 ETR 论文为定量比较 TCR 和 ECR 而定义的理论指标：一次训练 step 中，输入样本 x 的 class-discriminative pattern o_i 被正确分发到第 i 个 expert 的概率。

术语是什么？
定义 (Definition 2): 给定 x ∈ R^{s×d} (s tokens, 含 1 个 class-discriminative pattern o_i 和 s-1 个 class-irrelevant pattern r ∼ N)，若 o_i 被正确 dispatch 到第 i 个 expert，则 x 在此 step "训练成功"。Training Success Rate = P(x succeed in training)。

关键理论结果 (Theorem 5 + Corollary 6):
- TCR 成功率: Θ(C·Σp_i/s), p_i = P(δ_{o_i,i} ≥ δ_{x_j,i}) ≥ 1/n
- ECR 成功率: 当 C ≤ (s-1)q_i/2 时 ≤ (1/n)·Σe^{-(s-1)q_i/8}; 当 C ≥ 2s·q_i 时 ≥ 1-e^{-3C/16}
- q_i = P(r 的分数 > o_i 的分数) 衡量 expert 判别能力

术语一般如何实现？如何使用？
纯理论概念，不在代码中直接实现。价值在于提供 TCR→ECR 过渡的理论依据：早期 q_i ≈ Θ(1)→TCR 更优需 C=Θ(s)；后期 q_i << 1→ECR 更优仅需 C=Θ(1)，容量降低 ~40%。

涉及论文标题：
- Expert-Token Resonance Redefining MoE Routing through Affinity-Driven Active Selection
