## Combinatorial Sparsity in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Combinatorial Sparsity（组合稀疏性）是 MoE 模型中 token 级专家组合多样性的理论度量。由于每个 token 通过 top-K routing 从 N 个 expert 中选择 K 个激活，共有 C(N,K) 种可能的专家组合。这一组合数随 N 和 K 的增长呈指数级（或超指数级）增长，构成了 MoE 模型表达能力的一个重要来源。

LatentMoE 论文首次系统性地将 combinatorial sparsity 纳入 MoE 架构设计的理论框架（Design Principle V）。核心数学关系：

$$\begin{pmatrix} \alpha N \\ \alpha K \end{pmatrix} \ge \left( \begin{pmatrix} N \\ K \end{pmatrix} \right)^{\alpha}$$

即同时将 N 和 K 缩放 α 倍后，组合数以原组合数的 α 次幂增长。例如 C(512,24) >> C(128,6)，不仅仅是线性增加。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
MoE Expressivity = Router Selection Freedom + Expert Specialization + Combinatorial Diversity

Standard MoE (N=128, K=6):
  Expert combinations: C(128,6) = 5.5 × 10^9
  Each token selects from ~5.5 billion possible expert subsets

LatentMoE ℓ-MoE_acc (N=512, K=24, α=4):
  Expert combinations: C(512,24)
  C(512,24) ≥ C(128,6)^4 = (5.5 × 10^9)^4

Pipeline impact:
  - More N → finer-grained expert specialization (each expert covers narrower domain)
  - More K → each token benefits from more expert perspectives
  - C(N,K) diversity → better coverage of input distribution modes
```

Combined with Barron function theory: MoE layer 的有效非线性预算 U_eff ∝ K·m。LatentMoE 的 K'=αK 将 U_eff 提升 α 倍（在 ℓ-MoE_acc 中），同时保持 FLOPs 不变或降低。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：
- 通过增加 expert 数量 N 和 top-K 来实现更大组合空间（LatentMoE 的核心贡献）
- 必须在 iso-inference-cost 约束下进行（通过 latent space projection 补偿增加的 K 带来的 memory/communication cost）
- 实践中的实现：Training 时使用 aux-loss-free load balancing (Wang et al., 2024) + load balancing loss coefficient=10^-4 确保 token 均匀分布
- 组合稀疏性在推理时自动生效：每个 token 选择不同的 K'=24 experts subset，产生动态的专家组合

涉及论文标题：
- LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts
- DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models (Dai et al., 2024)
