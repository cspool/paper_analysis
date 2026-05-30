## GShard Gate / Top-1 Routing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

GShard Gate（来自论文 GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding, Lepikhin et al., 2020）是 MoE 中最常用的门控路由机制。Gate 是一个可学习的线性变换 W_g，将每个 token 的 d_model 维表示映射到 |E| 维（专家数量），经 softmax 后得到该 token 对各专家的亲和度分数。Top-1 routing 选择分数最高的专家处理该 token。Top-k routing 选择分数最高的 k 个专家。

$$g(x) = softmax(W_g \cdot x), \quad \tau = top\text{-}k(g(x))$$

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在 FOLDMOE 使用的 GPT-MoE 模型中：
- 使用 top-1 GShard gate（k=1，每个 token 只路由到一个专家）
- 每隔一个 Transformer block 替换 FFN 为 MoE 层（alternating pattern）

```
# MoE layer forward with top-1 GShard gate
def moe_layer_forward(x):  # x: [num_tokens, d_model]
    gate_logits = Linear(d_model, num_experts)(x)  # [num_tokens, num_experts]
    gate_probs = softmax(gate_logits, dim=-1)
    expert_idx = argmax(gate_probs, dim=-1)         # top-1

    # Expert Capacity 约束
    for e in 0..num_experts-1:
        tokens_for_e = x[expert_idx == e][:capacity]  # 截断到 capacity
        if len(tokens_for_e) > 0:
            output[expert_idx == e] = expert_e(tokens_for_e) * gate_probs[e]

    return output
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GShard gate 是 MoE 训练的标准选择：
- **Auxiliary loss**: 除主任务 loss 外，通常加辅助 load balancing loss 鼓励 token 均匀分配到各专家，避免某些专家过载或闲置
- **Expert capacity**: 限制每个专家每步最多处理的 token 数（Capacity = CF * B * L / |E|），超出部分被丢弃（token dropping）或用 residual connection 绕过
- **Top-2 routing**: 某些模型（如 Mixtral 8x7B）使用 top-2 gate，每个 token 路由到 2 个专家，增加模型容量但增加计算和通信开销
- FOLDMOE 使用 top-1 + capacity factor=1.0，EP 为每个 GPU 分配 1 个专家

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining
