## Invariant Learning for Graph Out-of-Distribution

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

图分布外泛化中的不变性学习（Invariant Learning for Graph OOD）是一类假设存在跨环境不变的图结构或图表示的方法。代表方法：DIR（因果干预蒸馏 causal subgraph）、EERM（环境划分学习 invariant representations）、GSAT（stochastic attention + information bottleneck）、CIGA（因果不变性表示）。核心范式：将源数据划分为伪环境 {E_1,...,E_K}，学习满足 P(Y|f(G), E=e_i) ≈ P(Y|f(G)), ∀e_i 的 encoder f。GraphMETRO 指出该范式的三方面局限：(1) 环境空间因组合爆炸不可行（环境 = 节点子集 × shift 类型组合）；(2) 忽略 instance-wise heterogeneity（关注 group-level patterns）；(3) 依赖 domain/environment 标签。GraphMETRO 通过 shift component decomposition（K 个 base transforms + continuous weight vector → 无限环境）和 MoE 架构的 instance-adaptive gating 替代传统 env-based 方法。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 标准图不变性学习 (EERM-style) vs GraphMETRO
# Standard:
environments = [E_1, ..., E_K]  # 从源数据构建伪环境
for G in D_s, E_k in environments:
    z = encoder(G in E_k)
    L_k = CE(classifier(z), y)
L_inv = var(best_classifier_params across environments)
L = Σ L_k + β · L_inv

# GraphMETRO (no environment dependency):
w = ϕ(G)  # continuous weight ∈ R^{K+1}, 替代 discrete environments
h = Σ Softmax(w)[i] · ξ_i(G)  # instance-adaptive expert combination
L = CE(μ(h), y) + λ·||h - ξ_0(G)||_F  # task + alignment
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实际使用的主要挑战：(1) 环境构建质量直接影响不变性——划分不当导致 spurious invariance；(2) 需要 domain/environment 标签，但许多场景不可得。GraphMETRO 的替代方案不需要 domain 标签，gating model 从图数据自动推断偏移成分。当限制 gating 输出为 binary 时，GraphMETRO 可退化为传统 finite-environment invariant learning。

涉及论文标题：
- GraphMETRO Mitigating Complex Graph Distribution Shifts via Mixture of Aligned Experts
