## SLO Slack-based Batch Size Control (基于SLO松弛的批量大小控制)

术语是什么？

SLO Slack-based Batch Size Control（基于SLO松弛的批量大小控制）是μShare的Batch Manager采用的自适应batching策略：在每个time window结束时计算SLO slack（s→j = Σ 2^(1-i) × (tSLO − ti)，其中ti按时间逆序排列，2^(1-i)实现指数衰减权重，近期请求权重更高），根据slack的符号和大小动态调整下一个window的batch size。Positive slack → 保守线性增加batch（bj+1 = bj + k × s→j），negative slack → 激进指数减少batch（bj+1 = max{bj − e^(λ × s→j), 1}）。这种"conservative increase, aggressive decrease"策略在throughput和SLO guarantee之间提供可控的trade-off。

从系统架构角度拆解术语：

SLO slack-based batch size control的运行时流程：

```
// 参数配置：k=0.05 (线性增加系数), λ=-0.1 (指数减少系数)
// SLO: 200ms (LLaMA-2-7b: 400ms)
// Time window: 论文未明确说明window长度

// 每个window j结束时：
function update_batch_size(window_j):
    // Step 1: 计算SLO slack s→j (指数衰减加权)
    nj = window_j.num_requests
    s_slack = 0
    for i in 1..nj (reverse chronological order):
        s_slack += 2^(1-i) * (t_SLO - t_i)
    
    // Step 2: 调整下一个window的batch size
    if s_slack > 0:
        // Conservative linear increase
        b_{j+1} = b_j + k * s_slack
        // 例如: b_j=10, s_slack=50 → b_{j+1}=10 + 0.05*50 = 12.5 → 12
    else:
        // Aggressive exponential decrease  
        b_{j+1} = max(b_j - e^{λ * |s_slack|}, 1)
        // 例如: b_j=10, s_slack=-20, λ=-0.1
        //   e^{-0.1*20} = e^{-2} ≈ 0.135
        //   b_{j+1} = max(10 - 0.135, 1) = 9.865 → 9
    
    // Step 3: 初始batch size = b_max / n
    //   b_max: 独占GPU时满足SLO的最大batch
    //   n: 当前GPU上co-locating的模型数
    
    return b_{j+1}
```

μShare的dual-level SLO guarantee：
1. **请求侧（Batch Manager）**：SLO slack-based batch size adjustment → 防止input load超过系统容量
2. **内核侧（Block Shaper）**：kernel launch slack-based blocksize adjustment → 通过增减α控制kernel执行速度，加速接近SLO violation的kernel

Hyperparameter sensitivity analysis (μShare v1-v9)：
- k ∈ {0.05, 0.03, 0.01}, λ ∈ {-0.1, -0.15, -0.2}
- μShare v1 (k=0.05, λ=-0.1): max throughput (58.91 normalized), SLO violation 3.35%
- μShare v7 (k=0.05, λ=-0.2): SLO violation 0.84% (< INFless 2.05% and Orion 1.12%), throughput 53.64 (still +19.28%-44.83% over baselines)
- 趋势: throughput 58.91 → 53.64 (↓9%), SLO violation 3.35% → 0.63% (↓81%)

术语一般如何实现？如何使用？

实现要点：
1. **Exponential decay algorithm**：引用自Cormode 2009的forward decay模型，权重衰减使得近期请求对slack的贡献远大于早期请求（最接近的请求权重0.5，2个window前的请求权重0.25，依此类推），避免short-term burst引发震荡
2. **Conservative increase + aggressive decrease**：灵感来自TCP congestion control的AIMD变体，确保SLO violation后迅速削减batch size，而只有当slack持续positive时才缓慢增加
3. **多模型co-location下的初始值**：bmax/n分摊GPU资源，后续各模型独立调整
4. **Integrations**：batch size调整影响PyTorch inference batch的token数，间接影响kernel的blocksize和gridsize → 与block shaper形成两级反馈

涉及论文标题：
- μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs
