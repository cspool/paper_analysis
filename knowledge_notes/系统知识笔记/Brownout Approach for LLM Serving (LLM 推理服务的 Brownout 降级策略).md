## Brownout Approach for LLM Serving (LLM 推理服务的 Brownout 降级策略)

术语解释
Brownout Approach 是借鉴电力系统"降压限电"概念的一种服务降级策略：在计算资源紧张或突发流量时，通过动态降低部分请求的处理质量来保证整体服务的 SLO。BrownoutServe 将其应用于 MoE LLM 推理：用 united expert 替代部分原始 expert 处理 token，以精度换延迟。

术语是什么？
Brownout 在云计算的原始定义（Xu et al. 2016-2021）：在负载高峰期，根据 dimmer 值（0-1）概率性地停用非必需的应用组件，以维持核心服务的可用性和响应时间。BrownoutServe 将此概念迁移到 LLM 推理领域：

- **电力系统 Brownout**: 电压降低 → 非必要设施断电 → 保障关键基础设施
- **LLM 推理 Brownout**: threshold 降低 → 部分 token 走 united expert（精度略降）→ 保障 SLO

核心量化关系：threshold ∈ [0,1] 决定多少比例 token 由原始 experts 处理。threshold=1 为零降级（最高精度），threshold=0 为全降级（最低延迟）。SALC 算法动态调节 threshold。

从系统架构角度拆解术语：
Brownout approach 在 BrownoutServe 系统架构中的运转流程：

```
请求流 → Scheduler (FCFS + ContinuousBatching)
           │
           ├──→ LLM Engine Forward Pass
           │      │
           │      ├── Attention (FlashAttention + PagedAttention)
           │      │
           │      └── BrownoutMoE ← threshold 参数
           │            │
           │            ├── Gate routing (标准)
           │            ├── Token 统计 & Expert 排序
           │            ├── threshold 划分 S1 vs S2
           │            ├── S1 → 原 experts FFN
           │            └── S2 → United experts FFN
           │
           └──→ SLO Analyzer (每 iteration)
                  │
                  ├── get P90 latency (TTFT/TPOT)
                  ├── if latency < warning_line → threshold↑ (提升精度)
                  ├── if latency > SLO → threshold×shrink_ratio (降延迟)
                  └── else → threshold 维持
```

三种 brownout 策略：
1. **Zero-Brownout**: threshold=1，所有 token 走原 experts，等价于标准 MoE 推理
2. **Full-Brownout**: threshold<1 且 use_full_brownout=true，S2 tokens 被直接跳过不处理
3. **Partial-Brownout**: threshold<1 且 use_full_brownout=false，S2 tokens 走 united experts

术语一般如何实现？如何使用？
- 实现依赖 united experts 的预训练（离线知识蒸馏）
- SALC 算法参数：warning_factor（如 0.8）、increment（如 0.1）、shrink_ratio（如 0.8）
- 适用场景：Bursty workloads 下的 chatbot、实时推荐、API 服务等延迟敏感应用
- 与水平扩展的互补性：水平扩展需 1-2 分钟冷启动，Brownout 在秒级内响应突发流量；两者可级联使用

涉及论文标题：
- BrownoutServe: SLO-Aware Inference Serving under Bursty Workloads for MoE-based LLMs
