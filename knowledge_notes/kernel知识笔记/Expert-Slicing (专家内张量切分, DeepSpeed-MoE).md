## Expert-Slicing (专家内张量切分, DeepSpeed-MoE)

术语解释
Expert-Slicing 是 DeepSpeed-MoE 推理系统提出的附加并行维度：将 Tensor-Slicing 应用于 Expert 参数内部，对单个 Expert 的权重矩阵进行行/列切分到多个 GPU，在 Expert Parallelism 的基础上进一步减少每 GPU 的计算量和内存需求。当可用 GPU 数超过 Expert 数时特别有用。

术语是什么？
Expert Parallelism 将不同 Expert 放到不同 GPU，极限是 EP = E（E = Expert 总数，每 GPU 恰好 1 expert）。但当需要更低延迟需要更多 GPU 时（GPU 数 > Expert 数），Expert-Slicing 提供额外的切分维度。

与 Expert Sharding (MoEShard) 的区别：
- **Expert-Slicing (DeepSpeed-MoE)**：Expert Parallelism + Tensor-Slicing within Experts 的组合。每个 GPU 先接收属于其 Expert(s) 的 token（EP 路由），然后 Expert 内部以 tensor-slicing 方式跨多个 GPU 协同计算。本质是 EP 和 TP 的嵌套。
- **Expert Sharding (MoEShard)**：所有 GPU 持有所有 Expert 的部分 shard，所有 GPU 处理所有 token 的 partial computation，完全替代 EP。本质是无 EP 的全 shard 方案。

从kernel调度角度拆解术语：
```
# Expert-Slicing 示例：EP=128, Expert-slicing degree=4 per expert
# Total 512 GPUs, 每 expert 由 4 GPU 协同处理

# 切分方式（per expert）：
W1 [h_in, h_inter] → 列切分为 4 份: W1_g0, W1_g1, W1_g2, W1_g3（每 GPU [h_in, h_inter/4]）
W2 [h_inter, h_out] → 行切分为 4 份: W2_g0, W2_g1, W2_g2, W2_g3（每 GPU [h_inter/4, h_out]）

# Forward（Expert e, on 4 GPUs）:
# Step 1: EP routing - tokens for expert e arrive at GPU group {g0, g1, g2, g3}
# Step 2: Per-GPU computation（所有 4 GPU 并行）:
partial_g0 = tokens @ W1_g0 → GeLU → @ W2_g0
partial_g1 = tokens @ W1_g1 → GeLU → @ W2_g1
partial_g2 = tokens @ W1_g2 → GeLU → @ W2_g2
partial_g3 = tokens @ W1_g3 → GeLU → @ W2_g3
# Step 3: All-Reduce within expert-slicing group
output_expert_e = AllReduce(partial_g0, partial_g1, partial_g2, partial_g3)
```

术语一般如何实现？如何使用？
- 实现于 DeepSpeed-MoE 推理系统（开源：https://github.com/microsoft/DeepSpeed）
- 当 GPU 数超过 Expert 数时自动应用（latency-stringent scenarios）
- 切分策略与 Tensor-Slicing 相同：W1 列切分 + W2 行切分（避免中间同步）
- Expert-Slicing group 内部 All-Reduce 仅限于节点内（NVLink），跨节点使用 Expert Parallelism

涉及论文标题：
- DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale

---
