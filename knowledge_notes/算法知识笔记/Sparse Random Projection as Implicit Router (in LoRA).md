## Sparse Random Projection as Implicit Router (in LoRA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FlyLoRA 的核心创新：将 LoRA 的 A 矩阵替换为**冻结的、稀疏的、随机初始化矩阵**，每行仅 p < n 个非零元素 (p≪n)，采样自 N(0, 1/r²)。A 同时承担两个角色：(1) 下投影 (传统 LoRA A 功能)；(2) 隐式 router——通过 top-k(Ax) 幅值选择应激活的 B 列。灵感来自果蝇嗅觉回路：projection neurons 通过稀疏随机连接投射到 Kenyon cells，然后 winner-take-all 选择性地激活 mushroom body output neurons。

由于 A 的稀疏随机投影近似保持 pairwise 距离 (Theorem 3.1, Johnson-Lindenstrauss 延伸)，语义相似 token 被路由到相似 expert。不同 task 的独立随机 A_i, A_j 天然近似正交 (Theorem 3.4: E[A_i·A_j^T] = 0, P(||A_i·A_j^T||₂ ≥ εr) ≤ p²/(nr²ε²))，实现 inter-task decoupling。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// FlyLoRA 稀疏随机投影 + 隐式路由:
// A ∈ R^{r×n}: 冻结, 每行 p 个非零 ~N(0,1/r²), sparsity ρ=p/n=k/r
// B ∈ R^{m×r}: 可训练, d ∈ R^r: 负载均衡偏置

Forward (Eq. 7-11):
  y = A @ x                               // 稀疏投影: O(r·p) vs LoRA O(r·n)
  y_biased = y + d
  I_topk = argtopk(y_biased, k)           // 隐式路由决定
  delta = (α/r) * sum_{i ∈ I_topk} b_i * y[i]
  output = W₀ @ x + delta

// A 初始化策略对比 (Appendix B.7, MMLU, Llama-3.1-8B):
// Gaussian (默认):      Before 40.88±1.61, Δ after merge -2.02
// Rademacher:           Before 40.42±0.23, Δ -2.35
// FJLT (结构化):         Before 40.57±1.34, Δ -2.50
// Two-Phase (可训练预热): Before 40.76±1.04, Δ -4.86 (破坏正交性!)
//
// 默认配置: total rank r=32, activated rank k=8, sparsity ρ=k/r=0.25
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 与 hash router (Roller et al. 2021) 关系：FlyLoRA 的固定随机投影 A 类似 hash router 的固定映射，通过距离保持性提供更强理论保证
- A 冻结消除：(1) router 参数 W_g；(2) A 的梯度计算和优化器状态；(3) A 相关的激活值存储
- 实现：使用 PyTorch `nn.Linear` weight 冻结 + `requires_grad=False`
- 代码：https://github.com/gfyddha/FlyLoRA

涉及论文标题：
- FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts
