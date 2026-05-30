## Mixture-of-Experts (MoE / 混合专家模型)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Mixture-of-Experts (MoE) 是一种稀疏激活的神经网络架构，将 FFN 层替换为多个"专家"子网络（experts），通过一个可学习的路由机制（router/gate）为每个输入 token 选择 top-k 个专家进行计算。核心特性：(1) 总参数量巨大（可达万亿级别），但每个 token 仅激活少量参数（稀疏性），推理计算量与激活参数量成正比而非总参数量；(2) 路由函数通常为简单的 softmax 门控：$g(x) = \text{softmax}(W_r x)$，选择 top-k（通常 k=1 或 2）个专家；(3) 需添加负载均衡损失（load balancing loss）防止所有 token 都路由到同一专家。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# MoE Layer Forward Pass
def moe_forward(x, router_W, experts, top_k=2):
    # x: [batch_size, seq_len, d_model]
    # router_W: [d_model, num_experts]
    
    # Step 1: Routing
    router_logits = x @ router_W          # [B, S, E]
    router_probs = softmax(router_logits)
    top_k_weights, top_k_indices = topk(router_probs, top_k)  # [B, S, k]
    
    # Step 2: Dispatch tokens to selected experts
    for expert_id in range(num_experts):
        mask = (top_k_indices == expert_id)  # tokens routed to this expert
        expert_input = x[mask]
        expert_output = experts[expert_id](expert_input)  # FFN per expert
        # Step 3: Combine (weighted sum)
        output[mask] += top_k_weights[mask] * expert_output
    
    # Load balancing loss
    fraction_tokens_routed = mean of router_probs  # [E]
    L_balance = num_experts * sum(fraction_tokens_routed * fraction_tokens_routed)
    return output, L_balance
```

代表性 MoE 模型：Switch Transformer（1.6T 参数，top-1 路由，2048 experts）、GLaM（1.2T 参数，top-2 路由，训练成本仅为 GPT-3 的 1/3）、Mixtral 8×7B（8 experts，每次激活 2 个，总 46.7B 参数，激活 12.9B，性能超 LLaMA2-70B）、DeepSeek-V2（Multi-head Latent Attention + MoE）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

训练框架：DeepSpeed-MoE、Megatron-LM（支持 expert parallelism + tensor parallelism 混合）、Tutel（动态自适应并行和流水线策略）。推理框架：vLLM 支持 expert parallelism、EdgeMoe（端侧通过 expert-wise bit-width adaptation 减少加载时间）、PC-MoE（利用 expert 激活的时间局部性，维护参数委员会减少资源消耗）。MoE 模型可通过 Sparse Upcycling 从 dense checkpoint 初始化（使用约 50% 的原始预训练成本，性能显著超过 dense 对应模型）。

涉及论文标题：
- A Survey of Resource-efficient LLM and Multimodal Foundation Models
