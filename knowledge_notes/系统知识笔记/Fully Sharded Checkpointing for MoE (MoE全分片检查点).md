## Fully Sharded Checkpointing for MoE (MoE全分片检查点)

术语是什么？
Fully Sharded Checkpointing 是将 checkpoint 保存工作负载均匀分布到所有分布式 ranks 上的策略。不同于 Megatron-DeepSpeed baseline 仅用 EP-Group-0 保存所有 expert 状态、仅用 Rank0 保存非 expert 状态，Fully Sharded 将 (1) Expert Part 以 expert 为最小单位在不同 EP groups 间等分，不同 EP group 的对应 rank 分担同一 expert 的不同参数段；(2) Non-Expert Part 以 layer 为最小单位在所有 DP ranks 间等分。进一步引入 Adaptive Sharding：当 PEC 导致部分 rank 保存更多 experts 时，贪心将非 expert shards 优先分配给负载最轻的 rank。理想 rank 负载：C_rank ≈ (P_ne+P_e)·B_o/D_ep + P_ne·B_w/D_dp + P_e·B_w/D_ep。

从系统架构角度拆解术语：
```
# Equal Sharding for Expert Part
# D_ep=2, Expert(0) → 前半给 EP-Group-0 Rank0, 后半给 EP-Group-1 Rank2
for each expert e:
    shard[i] = e.params[i * len//D_ep : (i+1) * len//D_ep]
    assign shard[i] → EP-Group-i, Rank(i * D_dp/D_ep)

# Equal Sharding for Non-Expert Part
# 以 layer 为最小单位分配到 D_dp 个 ranks
for each layer l:
    target_rank = l % D_dp
    assign layer_params[l] → Rank(target_rank)

# Adaptive Sharding (PEC 不平衡补偿)
# 贪心: 最大非 expert 模块 → 最轻 rank
sorted_layers = sort(non_expert_layers, by_size, descending)
for each layer:
    lightest_rank = argmin(accumulated_workload)
    assign layer → lightest_rank
```

术语一般如何实现？如何使用？
- Megatron-DeepSpeed 框架修改：替换 original checkpoint 的 rank/ep_group 分配逻辑。Sharding pattern 初始计算后固定不变——因 sequential PEC selection pattern 也是确定性的。
- 实测效果：bottleneck rank workload 减少 12%-29%（full saving）和 22%-29%（PEC saving），adaptive sharding 额外减 3.7%-6.1%（K_pec=1 case）。

涉及论文标题：
- Partial Experts Checkpoint: Efficient Fault Tolerance for Sparse Mixture-of-Experts Model Training
