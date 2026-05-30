## Coarse-to-Fine Statistical Evaluation for Serving Plans

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Coarse-to-Fine Statistical Evaluation 是 Cornfigurator 的三阶段评估管道，用于高效搜索 ~500M candidate physical plans。三阶段：(1) Network flow-based throughput estimation——将 plan 建模为流网络，计算 bottleneck node 确定最大吞吐量上界；(2) Monte Carlo latency estimation——随机采样请求并通过 plan 模拟处理延迟；(3) Request-level simulator——完整建模请求队列、inter-type contention 和 runtime 细节。剪枝规则精确——仅丢弃保证冗余或被 Pareto-dominate 的方案。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Qwen 3 Omni 16GPU 的评估流程和时间：

```
Phase 1: Network Flow — 3.48s, 483M→1.95M (0.40% survival)
  for each node v: capacity_v = Σ throughput_k
  找 bottleneck: first node where Σ demand_t ≥ capacity_v
  Pruning: 冗余 GPU 配置的 plans

Phase 2: Monte Carlo Latency — 34.23s, 1.95M→25
  采样 M 请求 → 按 routing prob 穿越 plan → per-type CDF F_t
  goodput_t = α·R_d·π_t·F_t(L_t)
  Pruning: Pareto-dominance on goodput vector

Phase 3: Simulator — 0.83s, 25→5
  以 α·R_d 速率运行，建模 queuing + inter-type contention
  Pruning: Pareto-dominance → select max Σ goodput
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Planning 总时间 < 2 分钟（vs. 全量 simulator 需 4400+ 小时）。F&M 的 mean absolute goodput error 为 10.7%（aggregate），Simulator 降至 4.1%。关键参数 α（吞吐 headroom，默认 0.7）planner 在 ±2% 误差内对 α∈[0.5,0.8] 鲁棒。

涉及论文标题：
- Cornserve Efficiently Serving Any-to-Any Multimodal Models
