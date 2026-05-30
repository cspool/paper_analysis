## Redundant Expert Deployment (冗余专家部署)

术语解释
Redundant Expert Deployment 是 DeepSeek-V3 针对 MoE 推理负载不均衡提出的部署策略：定期检测高负载 expert，将其复制多份部署到不同 GPU 上，通过增加 expert 副本来分散负载，避免某些 GPU 因承载热门 expert 而过载。该策略是 auxiliary-loss-free load balancing 在推理阶段的补充——训练时 batch-wise balancing 允许 expert 特化，推理时通过冗余部署消除 domain-shift 导致的负载不均衡。

术语是什么？
核心机制：(1) **Hot Expert Detection**：基于 online serving statistics 周期性（每 10 分钟）统计每个 expert 的 token 处理量，识别超过负载阈值的 expert；(2) **Redundant Copy Deployment**：将高负载 expert 的权重复制到多个 GPU 上，prefill 阶段 32 个冗余副本，decode 阶段 64 GPU 承载冗余+共享 expert；(3) **Intra-node Expert Rearrangement**：在 prefill 阶段根据观测负载在节点内重新排列 expert 分布，尽量在 GPU 间均衡负载而不增加跨节点通信开销；(4) **Dynamic Redundancy（探索中）**：每 GPU 承载 16 experts（原 8+冗余 8），推理时仅动态选择激活 9 个，在 all-to-all 前实时计算全局最优路由。

从系统架构角度拆解术语：
```
=== Redundant Expert Deployment 工作流程 ===

// Periodic Update (every 10 minutes)
for each expert i:
    load_i = total_tokens_processed_by_expert_i / time_window
    if load_i > threshold:
        mark_as_hot(expert_i)

// Deployment Adjustment
if prefill_stage:
    redeploy_hot_experts_across_gpus_within_nodes()
    // 32 redundant experts, each GPU: 8 original + 1 redundant
elif decode_stage:
    add_redundant_experts_to_dedicated_gpus()
    // 64 GPUs dedicated to redundant + shared experts
    // Each GPU: 1 expert (no rearrangement needed)

// Online Inference
for each token:
    selected = TopK(s_{i,t} + b_i, K_r=8)
    // Gate selects among original + redundant copies
    // Load distributed across replicas naturally
```

术语一般如何实现？如何使用？
冗余 expert 的权重直接从原始 expert 复制，无需额外训练。prefill 阶段 expert rearrangement 需考虑节点内 GPU 间通信开销（NVLink 160 GB/s，远高于跨节点 IB 50 GB/s）。Dynamic redundancy 需要更高效的全局最优路由算法（all-to-all 前计算）+ 与 dispatch kernel 的融合，以降低 overhead。冗余策略的 trade-off：增加 GPU 显存使用（额外副本）换取负载均衡和吞吐量提升。

涉及论文标题：
- DeepSeek-V3 Technical Report
