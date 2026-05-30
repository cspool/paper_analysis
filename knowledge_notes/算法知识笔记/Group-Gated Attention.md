## Group-Gated Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Group-Gated Attention 是 Focus 论文提出的注意力门控机制：将标准 QK^T 注意力分数乘以一个 group-based gate，使得远距离的 token pair 仅在属于同一 learned group 时才参与注意力。Gate 公式：s_ij = q_i^T k_j · (1_local(i,j) + (1 - 1_local(i,j)) · σ(λ · g_i^T g_j))，其中 1_local 表示 i-j ≤ w（局部窗口），g_i 为 token i 的 group assignment 向量，σ 为 sigmoid。对局部 token，gate=1（全注意力）；对远距离 token，同组 gate≈1（保留注意力），异组 gate≈0（剪枝注意力）。Gate 仅决定信息是否流动（binary routing），q_i^T k_j 决定流动多少（content weighting）。这与 token selection 方法不同——selection 挑选 top-k token 但使用标准 softmax；Focus gate 在 softmax 之前将异组 pair 缩放到零。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 标准 attention: s_ij = q_i^T k_j  (所有 T^2 个 pair)
# Group-gated attention: 
for i in range(T):
    for j in range(i+1):   # causal
        if i - j <= w:
            # 局部窗口内 → 全注意力 (gate = 1)
            s_ij = q_i^T @ k_j
        else:
            # 远距离 → 组门控
            affinity = g[i] @ g[j]        # 同组 ≈ 1.0, 异组 ≈ 0.0
            gate = sigmoid(lambda * affinity)
            s_ij = (q_i^T @ k_j) * gate    # 异组 → s_ij ≈ 0

attn_weights = softmax(s, dim=-1)          # 异组 pair 权重 ≈ 0
output = attn_weights @ V
```

门控机制的关键特性：
- **不重归一化（no re-normalization）**：异组 pair 被 gate 缩放到零，但 softmax 仍在全部 token 上归一化（包括零权重对）——这保留了预训练模型的 softmax 分母统计，是 composability 的来源之一
- **Gate steepness λ**：控制同组/异组的区分锐度。λ 过小 → 门控无力；λ 过大 → 近似 hard assignment
- **Local window 豁免**：局部 token 无论 group 归属均保留全注意力，保证短程依赖不丢失

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Group-gated attention 的推理效率优化：
- 训练时：soft gate 计算全部 O(n²) pair，无训练加速
- 推理时：使用 hard top-k assignment，异组远距离 pair 被完全剪枝（不计算），同组 pair 通过 FlashAttention 分解加速
- K=4, top-k=2: 约 50% 远距离 pair 被保留，约 25% 总 pair 被计算 → 2× 加速
- K=8, top-k=1: 约 12.5% pair 保留 → 理论 8× 加速，实际 8.6×（因 FlashAttention 在短序列上更高效）
- 门控的质量效应：top-k=2 时 PPL 优于 top-k=3/4（更稀疏产生更好质量），验证了"less attention can be more"的核心主张

涉及论文标题：
- Why Attend to Everything? Focus is the Key (Composing Sparse Attention via Learned Grouping)
