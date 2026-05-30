## Load Rebalancing After Expert Dropping

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert dropping 后 router gating 函数被直接修改（W_G^{d×n} → W_G^{d×(n-r)}），删除的 expert 在 router 中的入口消失，但剩余 expert 的 router weights 未重新归一化或重训练，导致 token 分配严重偏斜——某些 retained expert 被过度路由（高 load），另一些则路由不足。Jaiswal et al. (2025) 的 Figure 6 直观展示了 Mixtral-8×7B 在 50% expert sparsity 下 expert load distribution 的偏斜问题（红色虚线），及其通过 task-agnostic finetuning 的显著恢复（绿色实线）。

从系统架构角度拆解：MoE 推理中每个 expert 的计算负载直接决定 GPU memory utilization 和 pipeline 效率。如果某些 expert 被过度路由而另一些闲置，会导致：(1) 热点 expert 处理延迟高 → 整体推理延迟上升；(2) 闲置 expert 的 GPU memory 浪费 → effective speedup 小于理论值；(3) 分布式设置下的负载不均导致部分 GPU 空闲等待。因此 expert dropping 后的 load rebalancing 对实际推理加速至关重要。

```
# Post-Dropping Load Imbalance Measurement
def measure_expert_load_imbalance(M_dropped, X_test):
    """
    M_dropped: expert-dropped SMoE
    X_test: evaluation tokens
    """
    for l in M_dropped.moe_layers:
        load = zeros(n_remaining[l])  # load per expert
        for token in X_test:
            topk = topk(softmax(h @ W_G), k=2)
            for e in topk: load[e] += 1
        
        # Imbalance metric: CV of load distribution
        cv = std(load) / mean(load)  # coefficient of variation
        # 0 = perfectly balanced, higher = more skewed
        
        imbalance_score[l] = cv
    return imbalance_score
```

**Task-agnostic finetuning 对 load balancing 的校正机制**：Finetuning 使用 next-token prediction objective (C4) 更新所有参数（含 router W_G），使 router 自适应学习如何在新 expert 数量下分配 token。Figure 6 显示 finetuning 后的负载分布（绿色实线）显著接近均匀分布，相比 directly dropped（红色虚线）大幅改善。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 在 MoE Lottery Subnetworks 中，每轮 pruning 后的 task-agnostic finetuning 自动实现 load rebalancing
- ~1M training tokens 即可饱和 load rebalancing 效果（Table 5, Mixtral @ 75% sparsity）
- 训练配置：AdamW, lr=1e-6, batch=8, 8×A100, cosine LR scheduler
- 无需专门的 load balance loss（如 Switch Transformer 的 auxiliary loss），next-token prediction 本身即可驱动 router 自适应
- 评估指标：C4 perplexity（language modeling）和 downstream accuracy（zero-shot/k-shot/SFT）
- 与 GRACE-MoE 的 Dynamic Expert Replication 不同——后者处理 placement-based 负载不均（expert placement 在不同 GPU 上），而此处是 routing-based 负载不均（token 分配偏斜）

涉及论文标题：
- Finding Fantastic Experts in MoEs: A Unified Study for Expert Dropping Strategies and Observations
