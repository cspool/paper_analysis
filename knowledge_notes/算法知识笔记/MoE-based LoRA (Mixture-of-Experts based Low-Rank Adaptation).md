## MoE-based LoRA (Mixture-of-Experts based Low-Rank Adaptation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MoE-based LoRA 将 Mixture-of-Experts 架构集成到 LoRA 框架中。与标准 LoRA 使用单对低秩矩阵 (A, B) 不同，MoE-based LoRA 将适配器分解为 N 个 expert 对 {(A_i, B_i)}_{i=1}^N，每 expert 独立低秩分解，由 router G(x) 动态选择每个 token 激活的部分 expert。正向传播 (Eq. 3)：f_MoE-LoRA(x) = W₀·x + (α/r)·Σ_{i=1}^N G(x)_i·(B_i·A_i·x)，其中 G(x)=top-k(W_g·x)。核心动机：通过 expert 专门化实现 task decoupling，缓解 LoRA 的 intra-task parameter interference。

FlyLoRA 将 Split-LoRA (N×r_i) 作为代表基线。关键洞察 (Sec 2.3)：将 expert 粒度推至 rank-wise (N=r, 每 expert 仅 1 rank) 获最佳 decorrelation (图 1a)，但显式 router W_g ∈ R^{N×n} 随 N 线性增长 (图 1b)，造成参数效率劣化。这驱动 FlyLoRA 提出隐式 router。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// MoE-based LoRA (Split-LoRA(4×8)):
Forward:
  gate_logits = W_g @ x                              // W_g ∈ R^{N×n}
  gate_scores = sigmoid(topk(gate_logits, k_act))    // 选 top-k experts
  delta = Σ_i gate_scores[i] * (B_i @ (A_i @ x))
  output = W₀ @ x + (α/r) * delta

// 激活参数分析 (Table 9, d=hidden_dim, r=total rank, k=activated rank):
// LoRA(r):         param=2dr,   grad=4dr,     optim=24dr
// Split-LoRA(N×r): param=2dk+dN, grad=4dk+2dN, optim=24dk+12dN
// FlyLoRA(k):      param=dk,    grad=2dk,     optim=12dk
//
// Split-LoRA 额外开销来源于 router W_g ∈ R^{N×n}:
// Forward 多 dN 参数, Backward 多 2dN gradient + 12dN optimizer
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Split-LoRA 使用 sigmoid + top-k gate (FlyLoRA Appendix C.3)
- 代表性方法：MoLA (Wu et al. 2024), MixLoRA (Li et al. 2024), HydraLoRA (Tian et al. 2024, 非对称), LoRAMoE (Dou et al. 2023)
- FlyLoRA 改进方向：消除显式 W_g (用冻结 A 替代), 消除 A 的训练开销, 引入跨任务正交性

涉及论文标题：
- FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts
