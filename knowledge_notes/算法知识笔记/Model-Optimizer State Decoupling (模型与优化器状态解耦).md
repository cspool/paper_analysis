## Model-Optimizer State Decoupling (模型与优化器状态解耦)

术语解释
Model-Optimizer State Decoupling 是 SYMI 提出的核心系统设计原则：将 MoE 训练中 expert 的模型参数（weights, 2B/param in fp16）与其优化器状态（Adam optimizer: fp32 param + momentum + variance = 12B/param, 总计 16B/param including gradients）在存储和放置上完全分离。Optimizer state 被均匀静态分片到所有 N 个训练节点的 host memory，永不迁移；而 expert weights 在 GPU HBM 上按 popularity 动态调整放置。

术语是什么？
传统系统（DeepSpeed ZeRO-1, FlexMoE）将 optimizer state 与 expert instance 绑定——optimizer shard 仅分布在持有该 expert 的节点上。当 expert 需要 rebalance 到新 GPU 时，必须同时搬运 optimizer state（8× weight size），导致 rebalancing iteration 延迟为正常的 2.46×-4.10×（FlexMoE 数据），迫使系统只能粗粒度 rebalance（每 50-100 iterations）。

SYMI 的解耦设计：
- **Static optimizer**: optimizer[e_i] 均匀切分为 N 份，分布在所有 N 个节点的 host memory 中，永不迁移
- **Dynamic expert placement**: expert weights 在 GPU slot 上的分配每 iteration 可任意变化
- **No-overhead rebalancing**: Weight Communication Phase 的数据量 = sNW（与 static baseline 完全相同），因为每个 slot 始终接收一个完整的 expert weight，无论 expert class 是否改变

从算法pipeline角度拆解术语：
```
# SYMI Training Iteration (per layer, per rank):
# === State Layout ===
# optimizer_state[e_i] partitioned across ALL N nodes (host memory, static)
# expert_weights[slot_j] on GPU HBM (dynamic, changes per iteration)

# Forward: Router aggregates popularity → all-reduce → store in Layer Metadata Store
popularity[t] = allreduce(count_tokens_per_expert())

# Optimizer Step: Gradient Communication Phase (same data volume as static)
for each optimizer_partition on node_k:
    grad_shard = collect_from_source_expert_instances()  # Algorithm 2
    optimizer.step(grad_shard)  # Adam update → updated weight shard

# Optimizer Step: Expert Placement Scheduling (for iteration t+1)
placement[t+1] = compute_placement(popularity[t])  # Algorithm 1

# Optimizer Step: Weight Communication Phase (key insight - no extra data!)
for each slot_j:
    expert_id = placement[t+1][slot_j]
    # Send updated weights to slot_j - SAME data volume regardless of expert_id!
    send(updated_weights[expert_id], to=slot_j)
    # If expert_id changed: slot_j receives DIFFERENT expert's weights (same size W)
    # If expert_id unchanged: slot_j receives SAME expert's weights (same size W)
    # Data volume per slot = W bytes, ALWAYS.
```

通信量不变性证明：
- Grad Phase: D_G^SYMI = Σ r_i × G/N × N = sNG = D_G^static
- Weight Phase: D_W^SYMI = Σ N × W/N × r_i = sNW = D_W^static
- 仅 locality shift 引入约 1.52% 额外通信时间（N=2048, E=64, s=2）

术语一般如何实现？如何使用？
- SYMI 基于 DeepSpeed 实现，optimizer offload 至 CPU host memory (ZeRO-1 风格)
- 解耦设计不强制 optimizer 必须在 host memory——也可均匀分片在 GPU HBM 中（Appendix A.5），仅 locality 略变
- 与 tensor parallelism、pipeline parallelism、expert-sharding parallelism 正交兼容
- 关键约束：需要高效的跨节点梯度收集和权重分发通信（SYMI 使用 batch point-to-point + pre-registered NCCL groups）

涉及论文标题：
- Accelerating Mixture-of-Experts Training with Adaptive Expert Replication (SYMI)

---
