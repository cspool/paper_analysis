## Adaptive Expert Replication (自适应专家复制)

术语解释
Adaptive Expert Replication 是一种 MoE 分布式训练策略，根据每个 expert class 的动态 token popularity 非均匀地调整 expert 的复制份数（replication degree），使热门 expert 获得更多 replica 以处理更多 token、冷门 expert 减少 replica 以避免 GPU 资源闲置。与传统的 uniform static replication（所有 expert 分配相同数量 replica）不同，adaptive replication 直接解决 MoE 训练中的 convergence-latency tradeoff。

术语是什么？
在传统 MoE 训练中，每个 expert class 被复制固定次数 r = sN/E（s=每 rank slots 数, N=rank 数, E=expert class 数）。由于 expert popularity 高度偏斜且快速变化（SYMI 论文 Figure 2 显示 16× fluctuation 在 3 iterations 内），静态复制导致热门 expert 成为 latency bottleneck 并被迫丢弃超出 capacity 的 token，冷门 expert 的 GPU 资源闲置。

Adaptive Expert Replication 的核心公式：
- 每个 expert class e_i 被复制 r_i 次，其中 r_i ∝ popularity_i
- Σ r_i = sN（总 expert instances 数量不变）
- Effective capacity(e_i) = slot_capacity × r_i（而非固定的 capacity_factor × tokens_per_batch / E）

从算法pipeline角度拆解术语：
SYMI 的 Expert Placement Scheduler (Algorithm 1) 实现 adaptive replication：
```
def compute_placement(popularity, E, G, S):
    # popularity: [E] array, per-expert token counts from previous iteration
    # G: world size, S: slots per rank
    goal = (popularity / sum(popularity)) * G * S  # proportional allocation
    exp_counts = maximum(floor(goal), [1] * E)      # at least 1 replica each
    # Rounding correction to match total slots G*S
    while sum(exp_counts) > G * S:
        i = argmax(exp_counts - goal)
        if exp_counts[i] > 1: exp_counts[i] -= 1
    while sum(exp_counts) < G * S:
        i = argmin(exp_counts - goal)
        exp_counts[i] += 1
    # Contiguous assignment (same-class experts grouped together)
    placement = flatten([[exp_id] * count for exp_id, count in enumerate(exp_counts)])
    return placement  # length = G*S, contiguous same-expert blocks
```
流程：前次迭代的 global popularity（通过 all-reduce 聚合）→ 归一化为比例 → floor + rounding correction → contiguous assignment 优先同 rank 内同 expert replica。

术语一般如何实现？如何使用？
- SYMI 基于 DeepSpeed 实现，以 previous iteration popularity 为 proxy（simple yet effective），per-iteration 更新 placement
- 更复杂的策略可使用历史统计、预测模型、或基于数据集特征的 static replication
- 关键前提：需要 Model-Optimizer State Decoupling 来消除 rebalancing 时的 optimizer state 迁移开销
- 与 Top-k gating、Expert Choice routing、auxiliary-loss-free load balancing 等路由策略正交，可组合使用
- LLama 4 和 DeepSeek-V3 使用 shared + routed experts 混合架构，SYMI 可应用于 routed experts 部分

涉及论文标题：
- Accelerating Mixture-of-Experts Training with Adaptive Expert Replication (SYMI)

---
