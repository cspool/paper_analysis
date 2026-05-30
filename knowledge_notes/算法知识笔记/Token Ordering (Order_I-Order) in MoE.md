## Token Ordering (Order/I-Order) in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Token Ordering（token 排序/重排）是 MoE 层中在 AlltoAll dispatch 之前对 token 张量进行布局变换的操作。Gate 计算完成后，每个 token 被分配到一个或多个 expert，但此时 token 仍按原始序列顺序存储。Ordering 函数将张量 layout 从 (B, L, M) 变换为 (E, T, M)，其中 T 是 expert 能处理的最大 token 数（T = k×f×B×L/E，f 为 capacity factor）。这个变换使每个 expert 的数据在内存中连续排列，便于后续 Dispatch 和 Expert 计算。

I-Ordering 是 Ordering 的逆操作——在 Expert 计算和 AlltoAll Combine 完成后，将 expert-layout 张量 (E, T, M) 恢复为原始序列 layout (B, L, M)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FSMoE 中 Order/I-Order 的执行流程：

```
# Ordering: (B, L, M) → (E, T, M)
# 输入: hidden_states=[B,L,M], gate_idx=[B,L,k], gate_prob=[B,L,k]

# GShard ordering: 使用 einsum + matmul
# 创建 sparse routing matrix R = [B*L, E] (one-hot per token)
R = scatter_nd(gate_idx, gate_prob, shape=[B*L, E])
# 将 tokens 按 expert 聚合
ordered = einsum("be,blm->etm", R, hidden_states.reshape(B*L, M))
# ordered shape: [E, T, M] where T = capacity * B*L/E

# Tutel ordering: 使用 SIMT-efficient sparse 操作
# 直接按 gate_idx 做 gather/scatter，避免 dense einsum
for expert_id in range(E):
    mask = (gate_idx == expert_id)          # [B, L, k]
    indices = mask.nonzero()                 # N tokens 的索引
    ordered[expert_id, :len(indices), :] = hidden_states[indices]

# I-Ordering: (E, T, M) → (B, L, M)
# 将 expert 计算后的结果 scatter 回原始序列位置
output.zero_()
for expert_id in range(E):
    output[indices[expert_id]] += expert_output[expert_id] * gate_prob
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FSMoE 预实现了两种 Ordering 实现：GShard ordering（einsum+matmul, 适合小规模）和 Tutel ordering（SIMT-efficient sparse ops, 适合大规模和负载不均衡场景）。Order 子模块通过 `OrderBase` 抽象，与 Gate/Dispatch/Expert 解耦，用户可替换而不影响调度器。Capacity factor f 控制每个 expert 能处理的最大 token 数——f=* 表示不丢弃 token（但可能导致显存溢出），f=1.2 表示允许 20% overfill。

**Lancet 的 Gating 约束分区范围分析**（Lancet, MLSys 2024）：

Lancet 发现 gating 方法限制了算子分区的可行范围：(1) **Switch Gate** (Fedus et al., 2022) 和 **Random Gate** (Zuo et al., 2022)：expert assignment 可从部分 batch 决定（每个 token 独立路由），因此可将分区扩展到 MoE layer 之前和之后的 non-MoE 计算（Fig. 4d）；(2) **Batch-Prioritized Routing** (Riquelme et al., 2021)：按整个 batch 内 token 的 importance score 排序后分配 expert（低分 token 先被 drop），沿 batch 维度分区会导致不同 micro-batch 的 token dropping 不同（破坏了数学等价性），因此只能扩展到 MoE layer 之后的 non-MoE 计算（Fig. 4c）。Lancet 的 DP partition range selection 自动感知 gating 类型，对无法分区 before-MoE 的 gating，P(i,n,k) 被设为 ∞。

涉及论文标题：
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
- Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping
