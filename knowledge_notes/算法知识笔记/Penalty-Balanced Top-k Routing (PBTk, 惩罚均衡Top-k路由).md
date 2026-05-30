## Penalty-Balanced Top-k Routing (PBTk, 惩罚均衡Top-k路由)

术语解释
Penalty-Balanced Top-k (PBTk) Routing 是 MoE 中最主流的负载均衡路由策略：在语言模型主损失之上添加辅助惩罚项（Auxiliary Loss + Z-Loss），通过梯度反向传播间接约束 router 将 token 均匀分配到各 expert，而非在路由前显式修改路由决策。PBTk 被 DeepSeek-V2/V3、Qwen-MoE、Mixtral 等 SOTA MoE 广泛采用。

术语是什么？
PBTk 路由组合使用两种惩罚损失：
1. **Auxiliary Loss (L_aux)**：L_aux = α · E · Σᵢ fᵢ · Pᵢ，其中 fᵢ = 分配给 expert i 的 token 比例，Pᵢ = router 对 expert i 的平均 softmax 概率。α 典型值 0.01（本文及 ST-MoE）。该损失在 fᵢ = Pᵢ = 1/E 时最小，鼓励均匀分配。
2. **Z-Loss (L_z)**：L_z = λ_z · log²(Σⱼ exp(logitⱼ))，λ_z 典型值 0.001。惩罚 router logit 的过大值，提升训练数值稳定性（Zoph et al., 2022）。

总损失：L_total = L_LM + 0.01 · L_aux + 0.001 · L_z

PBTk 的核心权衡：α 太小 → 路由崩溃（仅少数 expert 被使用）；α 太大 → 梯度干扰主任务，降低模型性能。DeepSeek-V3 后来引入 auxiliary-loss-free 策略（expert-level bias 动态调整）以消除此权衡。

从算法pipeline角度拆解术语：
```python
# PBTk Router forward
def pbtk_router_forward(x, W_router, experts, shared_expert=None):
    # x: [B, S, H] batch of token hidden states
    logits = x @ W_router                    # [B, S, E], E=num experts
    probs = softmax(logits, dim=-1)          # [B, S, E]

    # Top-k selection (no modification to probs)
    topk_probs, topk_indices = topk(probs, k=K)  # K=1 for Switch, K=3 for Granular

    # Expert computation
    output = zeros_like(x)
    if shared_expert is not None:
        output += shared_expert(x)           # shared expert always active

    for i, expert_idx in enumerate(topk_indices):
        expert_out = experts[expert_idx](x)  # GEGLU FFN
        output += topk_probs[i] * expert_out
    output /= sum(topk_probs)                # normalize

    return output, logits, topk_indices

# Loss computation (at training step)
def compute_pbtk_loss(lm_loss, router_logits, topk_indices, batch_size):
    # Auxiliary loss
    f = zeros(E)     # fraction of tokens per expert
    P = zeros(E)     # avg router probability per expert
    for layer_logits, layer_indices in zip(all_router_logits, all_topk_indices):
        probs = softmax(layer_logits, dim=-1)
        for e in range(E):
            f[e] += count(layer_indices == e)
            P[e] += probs[:, :, e].sum()
    f /= total_tokens; P /= total_tokens
    L_aux = 0.01 * E * sum(f[e] * P[e] for e in range(E))

    # Z-loss
    L_z = 0.001 * sum(logsumexp(logits).square().mean() for logits in all_router_logits)

    return lm_loss + L_aux + L_z
```

术语一般如何实现？如何使用？
- **系数选择**：本文使用 α=0.01, λ_z=0.001，与 ST-MoE (Zoph et al., 2022) 和 DeepSeekMoE (Dai et al., 2024) 一致
- **CPT 中的行为**：PBTk 在分布偏移时经历短暂的 MRI spike（路由不均衡激增），但在 ~500 steps 内恢复到比 SBTk 更低的 MRI 水平。说明 PBTk 的路由对分布偏移具有"恢复性鲁棒"（resiliently robust）而非"固有鲁棒"（inherently robust）
- **与 SBTk 的差异**：PBTk 在稳定状态下的 MRI 低于 SBTk（更好的负载均衡），但分布偏移时需要短暂适应期
- **Decayed vs Non-decayed checkpoint**：从衰减 checkpoint 开始 CPT 的 PBTk，在分布偏移后的 MRI spike 稍高（~2-3%），但仍能快速恢复

涉及论文标题：
- Continual Pre-training of MoEs How robust is your router

---
