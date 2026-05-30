## MoE Gate Network / Expert Routing (门控网络 / 专家路由)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gate Network（亦称 Router）是 MoE 模型中决定每个输入 token 被分配到哪些 expert 的核心组件。通常是一个简单的线性层 W_g [d_model, E]（E 为 expert 总数），将 hidden states 映射到 E 维 logits → softmax 得到每个 expert 的 routing probability → top-K 选择概率最高的 K 个 expert → 输出每个选中 expert 的权重和 index。Gate 网络的设计直接影响 MoE 的负载均衡、模型质量和通信模式。

从算法pipeline角度拆解术语：
Gate Network 在 MoESys 训练中的决策链路：
```
# Input: hidden states H [B, d_model]
# Gate weight: W_g [d_model, E]

# Forward:
logits = H @ W_g                     # [B, E]
probs = softmax(logits)               # [B, E]
weights, indices = top_k(probs, K)    # [B, K], [B, K]

# Auxiliary Loss (load balancing, e.g. Switch Transformer):
# f_e = fraction of tokens routed to expert e
# P_e = mean routing probability for expert e
# Loss_aux = E * sum(f_e * P_e)  # encourages uniform routing

# AlltoAll dispatch:
# Each GPU sends token H[b] to GPU hosting expert indices[b][k]
# → triggers 2× AlltoAll (fwd) + 2× AlltoAll (bwd) per MoE layer
```

MoESys 对 Gate 的利用：Gate 的 expert 选择结果在 AlltoAll 通信中自然可获得 → MoESys 的 2D Prefetch 利用此结果决定 sparse 参数的 prefetch 目标，无需额外通信。hash table 中记录的 hits 频率 = 各 expert 被激活的历史频率 → CPU cache 的 LFU 管理依据。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 常见 routing 变体：top-1 (Switch Transformer, 简化但需更多 expert 或更高 capacity factor)、top-2 (GShard, Mixtral, 平衡负载和计算)、noisy top-K (添加 Gaussian noise 到 logits 鼓励探索)、random routing (ST-MoE, 随机选 expert 减少 bias)。
- Capacity Factor (CF)：限制每个 expert 能处理的最大 token 数 = CF × (tokens_per_batch / num_experts)。超出 capacity 的 token 被"dropped"（不经过该 expert，由 residual connection 绕过）。CF 引入 trade-off：CF 小 → 更多 dropped tokens → 质量下降；CF 大 → 更多 computation 和 memory → 效率下降。
- Auxiliary Loss 的类型：(1) Load balancing loss (Switch Transformer)：L_aux = E·Σ(f_e·P_e)；(2) Z-loss (ST-MoE)：加在 logits 上防止数值溢出。

- 补充 — Noisy Top-K Gating (Shazeer et al. 2017)：在标准 top-K routing 基础上添加两个关键组件。(1) **Tunable Gaussian Noise**：H(x)_i = (x·W_g)_i + StandardNormal()·Softplus((x·W_noise)_i)，噪声幅度由第二个可训练矩阵 W_noise 控制。噪声在训练中提供随机性，防止 Gate 过早收敛到固定 expert；噪声同时使负载均衡损失可微——P(x,i) = Φ((clean_logits_i - kth_excluding(H,k,i)) / noise_std_i)，其中 Φ 是标准正态 CDF，由此构建平滑的 Load(X) 估计器。(2) **KeepTopK**：保留 H(x) 中最大的 k 个值，其余设 -∞，经 Softmax 后对应 gate 值为 0，实现精确稀疏。与后来的 Switch Transformer (top-1) 和 GShard (top-2 + capacity factor) 不同，该论文使用 k=4（LM）或 k=2×2（hierarchical MT），且未使用 capacity factor（因 k 固定且 load balancing loss 已足够均衡）。

涉及论文标题：
- Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
- MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services
- Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without Retraining
- MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance
