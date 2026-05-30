## P-Penalty Loss (Parameter Penalty Loss)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

P-Penalty Loss（Parameter Penalty Loss）是 HMoE 提出的训练辅助损失函数，专为解决异构 MoE 中 router 过度偏好大 expert 的问题设计。传统 MoE 使用 load balancing loss $L_{lb} = N \sum_{i=1}^{N} \mathcal{T}_i \cdot \hat{\mathcal{P}}_i$ 鼓励 expert 被均匀使用，但在异构 MoE 中这一目标不适用——因为 expert 大小不同，"均等使用"不等于"经济使用"。P-Penalty loss 将 expert 的大小（hidden dim $h_{\text{ffn},i}$）直接纳入损失函数：

$$L_{\text{P-Penalty}} = N \sum_{i=1}^{N} \mathcal{M}_i \cdot \hat{\mathcal{P}}_i$$

$$\mathcal{M}_i = \frac{1}{T} \sum_{t=1}^{T} \mathbf{1}\{e_i \in E^t\} \times h_{\text{ffn},i}$$

$$\hat{\mathcal{P}}_i = \frac{1}{T} \sum_{t=1}^{T} P_{i,t}$$

其中 $\mathcal{M}_i$ 是 expert i 的"加权激活计数"——激活次数 × expert 的 hidden dim，$P_{i,t}$ 是 router 分配给 expert i 对 token t 的门控概率。关键特性：(1) 激活大 expert 时 $\mathcal{M}_i$ 更大（因 $h_{\text{ffn},i}$ 更大），P-Penalty 更高，驱动模型优先使用小 expert；(2) 若所有 expert 大小相同（$h_{\text{ffn},i}$ 均等），P-Penalty 退化为标准 load balancing loss。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```python
# P-Penalty Loss 计算（per MoE layer）
def p_penalty_loss(gate_probs, expert_assignments, expert_dims, N):
    """
    gate_probs: [B, S, N] 每个 token-expert 的 softmax 概率
    expert_assignments: [B, S, N] boolean mask, 标记各 expert 是否被激活
    expert_dims: [N] 各 expert 的 hidden dim
    """
    # P_hat_i: 各 expert 的平均门控概率
    P_hat = gate_probs.mean(dim=(0, 1))  # [N]

    # M_i: 各 expert 的加权激活计数
    # 对每个 token，如果 expert i 被激活，累加 h_ffn,i
    is_activated = expert_assignments.float()    # [B, S, N]
    M = is_activated.mean(dim=(0, 1)) * expert_dims  # [N], T 归一化

    # P-Penalty = N * Σ M_i * P_hat_i
    loss = N * (M * P_hat).sum()
    return loss

# 训练: L_total = L_lm + alpha * L_pp (alpha=0.1)
# Top-P 额外: L_total = L_lm + alpha * L_pp + beta * L_entropy (beta=3e-2)
```

P-Penalty vs Load Balancing Loss 的效果对比（Figure 7）：(1) Load balancing loss 无法阻止大 expert 被过度激活——虽然 expert 激活次数趋于均匀，但大 expert 每次激活的计算量更大；(2) P-Penalty loss 成功逆转激活比例——训练后期小 expert 激活率持续上升，大 expert 激活率下降；(3) 最终 HMoE-3B 的激活参数量从 1.23B（homogeneous MoE Top-P）降至 0.68B（HMoE Top-P），同时平均 benchmark 得分从 45.62 提升至 46.53。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

P-Penalty 在每个 MoE 层 forward 时作为辅助 loss 计算，与 language modeling loss 加权求和后反向传播。HMoE 中系数 α=0.1。使用方式：在 PyTorch 训练循环中，每个 MoE 层 forward 后收集 gate_probs 和 expert_assignments，计算各层 P-Penalty 并累加到总 loss。P-Penalty 替换（而非补充）传统 load balancing loss——当所有 expert 大小相同时等价，但对异构 MoE 更有效。局限：仅对 Top-K/Top-P 路由有效，不适用于 expert-choice routing（此时无 token-gate probability 概念）。

涉及论文标题：
- HMoE: Heterogeneous Mixture of Experts for Language Modeling
