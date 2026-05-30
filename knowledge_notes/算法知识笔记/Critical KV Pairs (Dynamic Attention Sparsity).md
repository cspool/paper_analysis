## Critical KV Pairs (Dynamic Attention Sparsity)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Critical KV Pairs指在attention计算中对给定query贡献最大的key-value pairs子集。定义：对query q和所有KV pairs S_q = {(k_i, v_i)}，attention score A(q, k_i) = softmax(q·k_i)，critical KV pairs I_q ⊆ S_q为那些A(q, k_i)超过θ-百分位阈值的pairs。DSV论文使用cumulative sum threshold θ=90%，即top KV pairs的attention score之和占总score的90%。整个attention head的sparsity定义为所有queries上非critical KV pairs的平均比例：E_{q~Q}[|S_q\I_q| / |S_q|]。

从算法pipeline角度拆解critical KV pairs的识别和使用：
```
# Critical KV identification (DSV's concept)
# 给定: Q, K, V ∈ R^{S×d_k}

# Step 1: 计算attention scores (conceptual - DSV用低秩近似替代)
scores = Q @ K^T / sqrt(d_k)        # [S, S] - DSV避免物化此矩阵
attn = softmax(scores, dim=-1)       # [S, S]

# Step 2: 识别critical KV pairs per query
for q in range(S):
    sorted_scores, indices = sort(attn[q], descending=True)
    cumsum = cumsum(sorted_scores)
    k = argmin(cumsum >= 0.9)         # θ=90% cumulative sum
    critical_indices[q] = indices[:k]  # top-k KV indices

# Step 3: Sparse attention - 仅对critical KV计算
for q in range(S):
    K_crit = K[critical_indices[q]]   # gather critical KV
    V_crit = V[critical_indices[q]]
    O[q] = softmax(Q[q] @ K_crit^T / sqrt(d_k)) @ V_crit
```

DSV发现Video DiT中的关键特性：(1) attention scores服从power-law分布（少数KV贡献大部分score）；(2) critical KV pairs不具局部性（仅15.1%在5-token半径内），无法用固定窗口模式近似；(3) 稀疏度在attention heads间和training steps间高度异质且动态变化；(4) 相邻token的critical KV pairs高度重叠（2×2×2 3D cube内>92.4%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

识别critical KV的方法分为：(1) 预训练静态方法——LLM推理中用window+token sink（StreamingLLM）、Heavy-Hitter Oracle (H2O)；Minference使用预定义的profile-based attention patterns；(2) 在线预测方法——DSV使用低秩sparsity predictor在线预测（训练阶段）；(3) block-based方法——BLASST的block-level sparsity via softmax thresholding。预定义方法适用于LLM推理（有明确的局部性模式），在线预测方法适用于Video DiT训练（critical KV无局部性模式）。DSV方法的冗余损失极小（>98% attention score被预测的critical KV覆盖）。

涉及论文标题：
- DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training
