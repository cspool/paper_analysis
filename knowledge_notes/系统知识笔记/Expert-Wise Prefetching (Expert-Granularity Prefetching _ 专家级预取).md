## Expert-Wise Prefetching (Expert-Granularity Prefetching / 专家级预取)

术语是什么？
Expert-Wise Prefetching 是 PopFetcher 提出的 MoE 训练优化机制：在 EP 下，各 GPU worker 在非 MoE 计算（Attention 层）期间，利用当前 idle 的 network link 提前从 remote GPU 拉取下一 MoE layer 中预测为热门的 expert 参数到本地 GPU memory。预取后，被预取 expert 对应的 token 无需经过 All-to-All dispatch，直接本地计算。该机制将 expert 调度从与 All-to-All 同阶段执行（reactive）改为提前执行（proactive），消除 expert scheduling 对 token dispatching 的干扰。

从系统架构角度拆解术语：
系统执行流程：
```
// 对当前 Transformer block 的第 l 层 MoE：
// Phase 1: Non-MoE Computation (Attention) 期间
Runtime: attention_forward(local_tokens)
while attention_running:
    for each candidate_expert in prefetch_plan[l+1]:
        async_pull_expert_params(remote_worker, expert_id)  // 独立 CUDA stream
        // 优先通过 NVLink (1800GB/s) 节点内拉取
        // 其次通过 GDR NIC (400Gb/s) 跨节点拉取
        
// Phase 2: Current MoE Layer
Runtime: moe_forward(local_tokens, layer=l)
// 此时 l+1 层热门 expert 已异步到达本地 GPU memory

// Phase 3: Next MoE Layer (l+1)
for expert in prefetched_experts:
    local_compute(tokens_to_expert)   // 零网络传输
for expert in non_prefetched_experts:
    all_to_all_dispatch(tokens)       // 仍需标准 All-to-All
```

核心约束：(1) GPU memory limit：2αH² Σδ_{n,w}^i ≤ Mem_w^{free}（已驻留本地 expert + 中间激活占用了大部分显存）；(2) Transfer time constraint：2αH² Σδ_{n,w}^i / W_{n,w} ≤ Time^{non-MoE}（预取必须在 Attention 计算时间内完成）；(3) Efficiency condition：仅当 compute-to-bandwidth ratio ε = P_w / W_{n,w} > 3αH 时 prefetch 才有效（如 B200 + NVLink 400Gb/s 场景，ε >> 3αH）。

术语一般如何实现？如何使用？
基于 PyTorch 实现：asynchronous scheduling executor 管理独立 CUDA stream（torch.cuda.Stream），在 Attention 层 forward 期间启动 nccl/nvlink P2P 传输。prefetching decision-maker 周期性（中后期训练可降低 replanning 频率）更新各 worker 的预取方案。node-level cache manager 利用 CPU memory 作为中间缓存：同一节点内 GPU 间共享已预取的 remote expert 参数。适用于 bandwidth-constrained 环境（如消费级 GPU + 低速互联），compute-to-bandwidth ratio 越高收益越大。

涉及论文标题：
- PopFetcher Towards Accelerated Mixture-of-Experts Training Via Popularity Based Expert-Wise Prefetch
