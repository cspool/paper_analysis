## Group-Level Load Balancing Loss

术语解释
Group-Level Load Balancing Loss 是 ARIA 对标准 per-expert load balancing loss 的松弛变体：将 fine-grained MoE 的大量 expert 按固定大小分组，在组级别而非单个 expert 级别施加负载均衡约束，避免过强的负载均衡压制 expert specialization。

术语是什么？
标准 MoE load balancing loss 惩罚每个 expert 的负载不均：$L_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{E} f_i \cdot P_i$。当 expert 数量很大（如 ARIA 的 64 routed experts）时，per-expert 约束过于严格，会阻止 expert 发展出有意义的 specialization。ARIA 将 64 个 routed experts 分为 8 组（每组 8 experts），负载均衡 loss 在组级别计算：

$$L_{balance} = \sum_{g} \alpha \cdot (\text{fraction\_of\_tokens\_routed\_to\_group}_g)^2$$

这样组内 expert 的负载可以自然不均（允许 specialization），但跨组的负载保持相对均衡。

从算法pipeline角度拆解术语：
```
# Group-Level Load Balancing (ARIA style)
num_routed_experts = 64
group_size = 8
num_groups = 8

# Router forward
gate_scores = softmax(W_router @ x)  # [batch, 64]
topk_probs, topk_indices = topk(gate_scores, k=6)

# Group-level load balancing loss
L_balance = 0
for g in range(num_groups):
    group_mask = (topk_indices // group_size == g).any(dim=-1)
    fraction_g = group_mask.sum() / batch_size
    L_balance += fraction_g ** 2
L_balance *= alpha
```

术语一般如何实现？如何使用？
- ARIA 首次在 fine-grained multimodal MoE 中使用 group-level load balancing
- 适用于 expert 数量大（64+）的场景，per-expert balancing loss 过于严格时
- 与 z-loss 配合使用以稳定训练
- 组大小选择是 trade-off：太小组 = 接近 per-expert（约束过强），太大组 = 负载可能严重不均

涉及论文标题：
- Aria An Open Multimodal Native Mixture-of-Experts Model

---
