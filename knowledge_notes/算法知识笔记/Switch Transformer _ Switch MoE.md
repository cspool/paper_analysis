## Switch Transformer / Switch MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Switch Transformer (Fedus et al., 2022) 是 Google 提出的简化 MoE 架构，核心设计：每个 token 仅路由到单个 expert（top-1 routing），而非 GShard 的 top-2。通过减少每 token 的 expert 计算量和通信量，Switch Transformer 以更简单的设计实现更大规模的稀疏模型。关键组件：(1) Top-1 gating (softmax → argmax)，(2) Load Balancing Loss (系数 0.01)，(3) Router Z-Loss，(4) Capacity Factor 控制每 expert 最大 token 数，(5) Truncated Normal Initialization (scale=0.1)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Joint MoE Scaling Laws 使用的 Switch MoE 层前向流程：

```
# 输入: x [B*L, d_model]
# 超参数: E=experts数, CF=capacity_factor (训练), CF=inf (dropless eval)

# 1. Router
router_logits = x @ W_r           # [B*L, E]
router_probs = softmax(router_logits)  # [B*L, E]
expert_idx = argmax(router_probs, dim=-1)  # [B*L]

# 2. Expert FFN (SwiGLU, hidden=3*d_model)
for e in range(E):
    mask = (expert_idx == e)
    tokens_e = x[mask]  # [n_e, d_model]
    # FC1: gate = tokens_e @ W_gate[e], up = tokens_e @ W_up[e]
    # SiLU(gate) * up → h [n_e, 3*d_model]
    # FC2: out_e = h @ W_down[e] [n_e, d_model]
    y[mask] = router_probs[mask, e] * out_e

# 3. 辅助损失
L_aux = 0.01 * LoadBalancingLoss + 0.001 * RouterZLoss
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Switch Transformer 的实现要点：
- Expert FFN 与 dense FFN 同尺寸（expert granularity=1.0），保持简单
- 可用 standard GeMM 或 GroupedGEMM 实现 expert 批量计算
- Joint MoE Scaling Laws 在 280+ 模型中使用 Switch MoE，E∈{1,2,4,8,16,32}（E=1 退化为 dense）
- 评估时设置 capacity factor 为无穷大（dropless），确保所有 token 被处理
- 优势：每 token 仅计算 1 个 expert = 最低计算成本；劣势：load imbalance 比 top-2 更严重

涉及论文标题：
- Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient
