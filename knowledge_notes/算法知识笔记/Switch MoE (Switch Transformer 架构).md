## Switch MoE (Switch Transformer 架构)

术语解释
Switch MoE 是由 Fedus et al. (2022) 提出的简化 MoE 架构：每个 MoE layer 仅激活 1 个 expert (K=1)，相比 K≥2 的方案大幅简化路由和负载均衡。Switch Transformer 证明了 K=1 仍能保证有效的路由梯度（因为 batch 内不同 tokens 选择不同 experts），并成功扩展到 1T+ 参数规模。

术语是什么？
Switch MoE 的关键特征：
- K=1：每 token 仅路由到 1 个 expert
- Full-size experts：每个 expert 的 FFN intermediate size 与 dense model 的 FFN 相同（不使用 fine-grained 拆分）
- 无 shared expert：所有 expert 均为 routed experts
- Token dropping：原始 Switch Transformer 使用 capacity factor 控制 expert 最大容量，超限 token 被 drop（本文不使用 token dropping）

本文的 Switch MoE 配置（570M active / 2B total）：E=8 routed experts, K=1, FFN intermediate=2816 (GEGLU)，无 shared expert。

从算法pipeline角度拆解术语：
```python
# Switch MoE layer forward (K=1)
def switch_moe_forward(x, W_router, experts):
    logits = x @ W_router              # [B, S, 8]
    probs = softmax(logits, dim=-1)    # [B, S, 8]
    top1_idx = argmax(probs, dim=-1)   # [B, S], pick SINGLE expert
    top1_prob = probs[top1_idx]        # scalar per token

    output = zeros_like(x)
    for e in range(8):
        mask = (top1_idx == e)
        if mask.any():
            output[mask] = top1_prob[mask] * experts[e](x[mask])
    return output
```

术语一般如何实现？如何使用？
- **CPT 中的表现**：Switch MoE 的 validation loss 和 benchmark 均弱于 Granular MoE（因为 expert 数量少且无 shared expert）
- **MRI 特征**：Switch MoE 的 early layers（0-6）MRI 显著高于 Granular MoE，且与训练/测试分布无关 → 早期层的路由不稳定可能是 Switch MoE 性能较差的根本原因
- **Code 任务**：Switch MoE 在 HumanEval 上意外优于 Granular MoE（可能因 K=1 的路由更简单，code 数据下 specialization 更清晰）
- **架构比较**：本文的 Granular MoE (E=31, K=3) 在所有主要指标上优于 Switch MoE (E=8, K=1)

涉及论文标题：
- Continual Pre-training of MoEs How robust is your router
