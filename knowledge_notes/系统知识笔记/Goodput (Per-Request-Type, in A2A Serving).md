## Goodput (Per-Request-Type, in A2A Serving)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Goodput 在 Cornfigurator 中的定义为：对于每种 request type t，在给定 physical plan 下，满足其 per-type latency target L_t 的请求吞吐量。即 goodput_t = throughput_t × Pr(latency_t ≤ L_t)。总体 goodput = Σ_t goodput_t。Goodput 同时捕捉吞吐量和延迟合规性。Latency target L_t 的默认设定：对每种 request type 在 isolation 下 profiling，取最高吞吐配置的 p25 延迟。Appendix A 证明当 L_t ∝ compute cost of type t 时，所有 type 的约束 equally tight。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Per-type goodput 在 Cornfigurator 中的计算流程：

```
Monte Carlo Phase:
  for each request type t:
    采样请求 → 按 routing prob 穿越 plan graph
    累积 per-executor 处理延迟 → empirical latency CDF F_t(l)
    goodput_t = α·R_d·π_t·F_t(L_t)
  Pruning: Pareto-dominance on goodput vector (g1,...,gT)

Simulator Phase:
  以 α·R_d 速率运行 request-level 模拟
  建模 queuing + inter-type contention + CPU-GPU overlap
  输出 per-type latency CDF → goodput_t
  最终选 max Σ_t goodput_t
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Per-type goodput 的关键优势：使用全局单一延迟约束时，仅最重的 request type（如 audio output）约束生效，轻量 type（如 text output）可被无限制降级。Per-type constraint 确保 planner 不能以牺牲轻量 type 延迟为代价提升重量 type 吞吐。

涉及论文标题：
- Cornserve Efficiently Serving Any-to-Any Multimodal Models
