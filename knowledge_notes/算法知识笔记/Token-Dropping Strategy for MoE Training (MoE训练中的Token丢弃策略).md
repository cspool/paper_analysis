## Token-Dropping Strategy for MoE Training (MoE训练中的Token丢弃策略)

术语解释
Token-Dropping Strategy 是 DeepSeek-V2 在 MoE 训练中提出的负载均衡补充机制：当 balance loss 无法保证严格负载均衡时，在每个设备上按计算预算（capacity factor=1.0）丢弃 affinity score 最低的多余 token，避免因负载不均导致的计算资源浪费。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
尽管 expert-level balance loss、device-level balance loss 和 communication balance loss 三层辅助损失鼓励均衡负载，但它们无法保证严格均衡——某些 expert/device 可能仍收到超过平均值的 token 分配。Token-Dropping Strategy 在每个 device 上强制执行硬性计算预算：(1) 计算每 device 的平均计算预算 = total_tokens / D (capacity factor=1.0)；(2) 若某 device 收到的 token 超过预算，按 affinity score 从低到高丢弃超出的 token；(3) 保留约 10% 的训练序列永远不丢 token（保证训练-推理一致性）。推理时可灵活选择是否启用 token dropping。

为什么需要？MoE 训练中负载不均的两个后果：(1) routing collapse——某些 expert 训练不足；(2) 计算效率下降——expert parallel 下过载 device 成为瓶颈。Balance loss 是软约束，token dropping 提供硬约束兜底。

从算法pipeline角度拆解术语：
```
=== Token-Dropping Strategy (per MoE layer, per training step) ===

Input: tokens assigned to device d: {(t, s_t_expert, g_t)}, capacity C = T/D

// Step 1: Sort tokens by affinity score (descending)
tokens_sorted = sort_by_affinity(tokens_to_device_d, descending=True)

// Step 2: Keep top-C tokens, drop the rest
kept = tokens_sorted[:C]
dropped = tokens_sorted[C:]     // lowest affinity tokens

// Step 3: Guarantee ~10% sequences never drop
// Mark 10% sequences as "protected" before sorting
// Protected tokens always in kept set regardless of affinity

// Step 4: Forward only kept tokens through experts
for (t, s, g) in kept:
    expert_output += g * FFN_expert(h_t)

// Dropped tokens: skip expert computation (output = shared experts only)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
训练时启用加速，评估时不丢 token 以保证结果确定性。推理时：根据效率需求灵活选择——offline batch 推理可启用以提升吞吐，online serving 通常不启用以保证质量。DeepSeek-V2 训练中 capacity factor=1.0，略低于 Riquelme et al. (2021) 的 1.25-1.5（更激进）。与 GShard 的 capacity factor 机制核心区别：GShard 对 expert 做 capacity limit，DeepSeek-V2 对 device 做 capacity limit（因 device-limited routing 后 device 是计算的实际分配单位）。

涉及论文标题：
- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

---
