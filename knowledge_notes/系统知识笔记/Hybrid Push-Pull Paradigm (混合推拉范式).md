## Hybrid Push-Pull Paradigm (混合推拉范式)

术语是什么？
Hybrid Push-Pull 是 PopFetcher 提出的一种 MoE 训练数据传输范式，结合传统的 expert-centric（push：将 token 发送到 expert 所在 worker）和 data-centric（pull：将 expert 参数拉到 token 所在 worker）两种模式。核心决策依据：当 token 体积超过 expert 参数体积时 pull expert，否则 push token。单个 expert 层参数为 4H²（两个 Linear 层，矩阵维度 H×αH 和 αH×H），两次 All-to-All 中单 token 的跨机传输为 2H → break-even point 为 2048 tokens（H=1024 时 expert 参数约 16MB）。

从系统架构角度拆解术语：
决策流程伪代码：
```
for each worker w:
    for each expert e on remote worker:
        token_volume  = B_{n,w}^i × 2H     // 需发到 remote expert 的 token 总字节数
        expert_volume = 4H² × sizeof(fp32) // expert 参数字节数
        if token_volume > expert_volume:
            strategy = PULL_EXPERT  // 拉 expert 到本地计算
            comm_cost = expert_volume / W_{n,w}
        else:
            strategy = PUSH_TOKEN   // 推 token 到 remote 计算
            comm_cost = token_volume / W_{n,w}
```

与 FasterMoE（push-only，expert shadowing + periodic broadcast）和 Janus（pull-only，pull all experts 易 OOM）的对比：hybrid 根据当前 token distribution 动态选择最优方式，避免单一范式的短板。在 Cluster A (8×RTX 4090, 100Gbps InfiniBand) 上，hybrid 的 throughput 和 per-iteration time 均优于两种 pure 范式。

术语一般如何实现？如何使用？
实现为 EP 训练框架中的 prefetch decision module：在 routing information collector 获取各 expert 的 token 分配后，计算 push/pull 的通信代价，选择低代价策略。适用于 expert 参数和 token batch size 非均匀分布的场景。当 "bad prefetching" 发生时（预取 expert 接收 token 少于预期），PopFetcher 直接回退到传统 push token 模式，无额外开销。

涉及论文标题：
- PopFetcher Towards Accelerated Mixture-of-Experts Training Via Popularity Based Expert-Wise Prefetch
