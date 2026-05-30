## ZeRO-2 Data Parallelism (ZeRO-2 数据并行)

术语是什么？
ZeRO-2 是 Microsoft DeepSpeed 提出的 Zero Redundancy Optimizer 的第二阶段，将 optimizer states 和 gradients 在 data-parallel ranks 之间分片存储，消除冗余，显著降低每 GPU 显存占用。每个 rank 仅持有参数 1/D_dp 的 optimizer states（Adam 所需 momentum + variance，每参数 8 bytes FP32），而 DDP 中每 rank 存储完整 optimizer states。gradients 在 reduce-scatter 后仅保留对应 shard，其余丢弃。Model parameters 仍完全复制——这使 ZeRO-2 在 checkpoint 场景中需特别处理：weights 全复制在每 rank 但 optimizer states 已分片，因此 checkpoint 需聚合所有 ranks 的 shards 才能恢复。

从系统架构角度拆解术语：
```
# ZeRO-2 优化器状态分片
# D_dp=4, params=[w0,w1,w2,w3] → optimizer states=[m0..m3, v0..v3]
# DDP (No ZeRO):  每 rank 存全部 m,v = 8x params
# ZeRO-2:
Rank0: m0,v0 (1/4 optimizer states) + w0..w3 (full params)
Rank1: m1,v1                           + w0..w3
Rank2: m2,v2                           + w0..w3
Rank3: m3,v3                           + w0..w3

# Checkpoint 视角:
# optimizer states: already sharded → natural sharding, save per-rank shards
# model params: replicated → need explicit sharding (Fully Sharded Ckpt)
```

术语一般如何实现？如何使用？
- DeepSpeed 配置文件：`"zero_optimization": {"stage": 2}`，配合 `allgather_bucket_size`、`reduce_bucket_size`、`overlap_comm` 等参数。
- MoE 训练中的特殊考量：ZeRO-2 + EP 混合策略下，expert optimizer states 的分片模式依赖于 EP group 内 DP rank 的划分。MoC-System 的 fully sharded checkpoint 策略针对此场景优化了 weights 的分布（optimizer states 已由 ZeRO-2 分片）。

涉及论文标题：
- Partial Experts Checkpoint: Efficient Fault Tolerance for Sparse Mixture-of-Experts Model Training
