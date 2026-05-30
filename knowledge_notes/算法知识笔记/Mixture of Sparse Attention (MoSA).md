## Mixture of Sparse Attention (MoSA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Mixture of Sparse Attention (MoSA) 是一种受 Mixture of Experts (MoE) 中 Expert-Choice Routing 启发的可学习内容感知稀疏注意力方法。每个 attention head 配有一个可学习的路由权重矩阵 W^r ∈ R^h，通过 sigmoid 函数 σ(x)=1/(1+e^{-x}) 计算每个 token 的选择得分 r=σ(XW^r) ∈ R^T，然后用 TopK 选择得分最高的 k 个 token，仅对这些被选 token 计算 Q、K、V 投影和 attention 矩阵。未被选中的 token 在该 head 的输出中填 0。所有 head 的输出求和，构成 MoSA 层的最终输出。

与标准 dense attention 的 O(T²) 复杂度相比，MoSA 将每 head 的复杂度降至 O(k²+T)：投影成本从 8hh'T 降至 8hh'k，attention 从 4h'T² 降至 4h'k²，额外的 routing overhead 为 2hT + h'k。由于 k << T，节省的 FLOPs 用于增加注意力头数（从 9 增至数百），实现更细粒度的 head 专业化。

核心设计要点：(1) router 输出 r_topk 在 attention 之后通过 diag(r_topk)·A 乘到输出上，使路由决策可通过梯度下降端到端学习；(2) causal mask 基于 token 原始位置索引而非子集位置：M_{a,b}=0 iff I_a≥I_b else -∞；(3) RoPE 旋转角度同样基于原始位置索引，保证位置编码的一致性；(4) Expert-Choice 路由天然保证完美负载均衡——每个 head 恰好处理 k 个 token，无需 auxiliary load-balancing loss；(5) 混合架构：保留 4 个 dense head 提供全局信息流和训练稳定性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# MoSA 单层前向
Input: X ∈ R^{T×h}

for each head i ∈ {1..H}:
  # Step 1: Token Selection via Router
  r = σ(X @ W^r_i)              # r ∈ R^T
  r_topk, I = TopK(r, k)        # I ∈ {0..T-1}^k

  # Step 2: Gather selected tokens
  X^s = X[I]                     # X^s ∈ R^{k×h}

  # Step 3: Q/K/V projections (only on k selected tokens)
  Q = X^s @ W^Q_i               # Q ∈ R^{k×h'}
  K = X^s @ W^K_i               # K ∈ R^{k×h'}
  V = X^s @ W^V_i               # V ∈ R^{k×h'}

  # Step 4: Causal mask based on original positions
  M[a,b] = 0 if I[a] >= I[b] else -∞

  # Step 5: Sparse Attention
  A = softmax(Q @ K^T / √h' + M) @ V  # A ∈ R^{k×h'}

  # Step 6: Router gating + output projection
  X^o = diag(r_topk) @ A @ W^O_i

  # Step 7: Scatter back to full sequence
  Y[j] = X^o[idx] if j == I[idx] else 0

Output: Y = Σ_{i=1..H} Y_i
```

**FLOPs 成本模型**：
- Dense head: FLOP = 8hh'T + 4h'T²
- MoSA head: FLOP = 8hh'k + 4h'k² + 2hT + h'k
- 当 T=1024, k=32, h=1024, h'=64: dense head ≈ 0.805 GFLOPs, MoSA head ≈ 0.019 GFLOPs (42x reduction)

术语一般如何实现？如何使用？

MoSA 使用纯 PyTorch 实现（einsum/scatter/gather），无需专用 CUDA kernel。开源代码：https://github.com/piotrpiekos/MoSA。Router 权重 W^r 与 Q/K/V/O 投影共同通过语言模型目标优化。IsoFLOP 实验中，首个 token 始终被所有 head 选中（attention sink 效应）。下游短序列任务中，自适应调整 k = max(floor(T/ρ), 2)。论文用 C4 数据集训练，T=1024，sparsity ρ=T/k 从 1 到 256。最佳 perplexity 在 ρ≈64 处取得，Small 模型（113M）perplexity 从 16.01 (dense) 降至 12.85 (-19.7%)。KV-cache 在 perplexity-matched 设定下减少 51-70%。

涉及论文标题：
- Mixture of Sparse Attention: Content-Based Learnable Sparse Attention via Expert-Choice Routing
