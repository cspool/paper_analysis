## Activated Expert Replicas (激活的专家副本)

术语解释
在 Expert Parallelism MoE 推理中，Activated Expert Replicas 指当前 batch 中实际有 token 需要处理的 expert replicas。与 total expert replicas 不同，activated replicas 仅包括那些收到至少一个 token 的 replica。METRO 论文的关键发现：在 memory-bound decode 阶段，GPU 的 MoE layer runtime 由 activated expert replicas 数量决定，而非由 token 数量决定。

术语是什么？
在 EP 部署中，每个 expert 可以有多个 replicas（副本）分布在不同 GPU 上。当 batch 中的 token 经过 router 选择 top-k experts 后，仅部分 replica 会收到 token——这些收到 token 的 replica 即为 "activated"。例如：expert e 有 3 个 replicas 在 GPU 0/3/5，但当前 batch 仅 2 个 token 选中 e——EPLB routing 可能将 2 个 token 分散到 2 个 replicas（激活 2 个），而 METRO routing 将 2 个 token 集中到 1 个 replica（仅激活 1 个）。在 memory-bound 下，每多激活一个 expert replica 就需要额外加载该 expert 的全部 FFN weight（~200MB）从 HBM 到 Tensor Core——这是延迟的主要来源。

从算法pipeline角度拆解术语：
Activated expert replicas 与 decode latency 的关系：

```
=== 8 GPUs, 128 experts, 1.5x replication, decode batch 256 tokens ===

# EPLB token-balancing routing:
for each expert e with T[e] > 0:
    # 将 T[e] 均匀分配到 e 的所有 replicas
    tokens_per_replica = ceil(T[e] / R[e])
    for g in GPUs hosting e replicas:
        if tokens_assigned_to_g < tokens_per_replica:
            y[e][g] = 1  # 激活该 replica
    
# 结果: 若 expert e 有 3 replicas 和 9 tokens
# → 每个 replica 各 3 tokens → 3 activated replicas
# → 全局: max_g Σ_i y[i][g] ≈ 高 (如 20 activated experts per GPU)

# 延迟构成 (memory-bound):
latency_EPLB = max_g (Σ_{i: y[i][g]=1} load_weight(expert_i, HBM→TC) + compute)
             ≈ max_g (num_activated_experts × weight_load_time_per_expert)
             ≈ 20 × 80μs = 1600μs (weight loading)
             + compute (~100μs, 可忽略)


# METRO expert-minimizing routing:
for each expert e with T[e] > 0:
    G_e = GPUs hosting e replicas (from placement matrix A)
    g* = argmin_{g in G_e} activated_count[g]  # 选 activated experts 最少的 GPU
    y[e][g*] = 1  # 仅激活一个 replica
    x[e][g*] = T[e]  # 所有 token 路由到该 replica

# 结果: 若 expert e 有 3 replicas 和 9 tokens
# → 9 tokens 全部路由到 activated_count 最小的 GPU
# → 仅 1 activated replica
# → 全局: max_g Σ_i y[i][g] ≈ 低 (如 12 activated experts per GPU)

# 延迟构成:
latency_METRO = max_g (num_activated_experts × weight_load_time)
              ≈ 12 × 80μs = 960μs (weight loading, -40% vs EPLB)
              + compute (~100μs) + routing_kernel (~26μs)
              ≈ 1086μs (净节省 ~514μs/layer)
```

术语一般如何实现？如何使用？
- Activated experts 在 decode batch 中通常远少于 total experts（batch size 小，top-k 选择导致各 expert 的 tokens 稀疏分布）
- 减少 activated experts 的方法：(a) **METRO greedy routing**: 将每个 expert 的所有 tokens 集中到单一 replica；(b) **减少 replication factor**: 少 replicas → 少可激活的 replica 数，但会损害 prefill 性能；(c) **Expert pruning**: 跳过不重要的 expert
- METRO 实验：activated experts 在最优解的 10.9% 以内，比 EPLB 降低 up to 42.3%
- 为什么不简单地不复制（1.0x replication）？因为 replication 对 compute-bound prefill 有显著提升（-17% TTFT），但会损害 memory-bound decode。METRO 用 expert-minimizing routing 消除 replication 对 decode 的副作用

涉及论文标题：
- Efficient MoE Serving in the Memory-Bound Regime Balance Activated Experts, Not Tokens

---
