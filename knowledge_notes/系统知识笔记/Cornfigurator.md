## Cornfigurator

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Cornfigurator 是第一个面向通用 Any-to-Any（A2A）多模态模型推理 Serving 的自动化部署规划器。它接收 model definition（组件 DAG）、configuration space（executor 类型及配置）、workload（request type 分布及 per-type latency target）、GPU budget 作为输入，自动搜索最优的 colocation/disaggregation 组合、executor 配置（batch size、parallelism degree、实例数）和请求路由策略，输出可被 Cornserve runtime 直接部署执行的 physical plan。优化目标是最大化 per-request-type goodput（满足各自延迟目标的吞吐量）之和。规划器约 5K 行 Rust 实现，使用三阶段粗到细统计评估管道：Network flow（吞吐量上界估计）→ Monte Carlo（延迟采样）→ Request-level simulator（精确队列建模），每阶段后剪枝淘汰劣化方案。开源地址：https://github.com/cornserve-ai/cornfigurator。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Cornfigurator 在 A2A Serving 栈中的位置和运转流程：

```
输入层:
  Model Definition (DAG: E_img→E_vid→L_th→L_ta→G_aud)
  Configuration Space (executor types, batch sizes, TP degrees)
  Workload (8 request types, per-type fractions π_t, latency targets L_t)
  GPU Budget N=16
       │
       ▼
Profiler (§5.1):
  对每个 executor type 在 A100 上 sweep batch size & parallelism
  记录稳态吞吐和延迟（减除 queuing delay）
  输出: per-executor-config throughput & latency profiles
       │
       ▼
Planner - Plan Enumeration (§4.2, Algorithm 1):
  1. Simple subplans: 枚举 model subgraph 的 colocation/disaggregation
  2. Compound subplans: 合并共享节点的 simple subplans (k_c=2)
  3. Logical plans: 组合 subplans 为 supergraph, 覆盖所有 request types (k_s=2)
  4. Physical plans: 注解 GPU 分配 + executor 配置 + routing probabilities
  → 输出 ~483M candidate physical plans
       │
       ▼
Planner - Three-Phase Evaluation (§4.3, Algorithm 2):
  Phase 1: Network Flow (3.48s, 483M→1.95M) - 瓶颈吞吐量上界
  Phase 2: Monte Carlo Latency (34.23s, 1.95M→25) - per-type 延迟 CDF
  Phase 3: Request-Level Simulator (0.83s, 25→5) - queuing + contention
       │
       ▼
输出: Physical Plan (deployed to Cornserve runtime)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Cornfigurator 是 runtime-agnostic 设计，但其 proof-of-concept 实现在 Cornserve 之上。使用流程：(1) 定义 model component graph 和 configuration space；(2) 提供代表性 workload traces；(3) 运行 profiler 收集 per-component 性能数据（平均 4.2 node-hours per pair）；(4) 运行 planner 生成 physical plan（< 2 分钟）；(5) 将 plan 交给 Cornserve runtime 部署执行。支持 workload drift 自适应——当 request type 比例变化时仅需 re-weight profiling samples（无需重新 profiling），re-planning 耗时 single-digit seconds。

涉及论文标题：
- Cornserve Efficiently Serving Any-to-Any Multimodal Models
