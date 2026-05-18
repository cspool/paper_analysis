## SLO-aware Dispatcher（SLO感知调度器）

术语是什么？

SLO-aware Dispatcher是MuxWise系统中的调度决策模块，负责根据请求的SLO约束（TTFT和ITL）动态分配GPU SM资源。其核心功能是：为Decode分配恰好足以满足ITL SLO的最小SM数量，将剩余SM全部分配给Prefill，从而在保证服务质量的前提下最大化Goodput。

从系统架构角度拆解术语：

调度流程如下：

1. **ITL SLO解析**：获取当前活跃请求的ITL SLO要求（如P90 < 50ms）。
2. **Decode SM需求计算**：根据当前批处理的Decode请求数量、序列长度分布和KV Cache大小，通过Contention-tolerant Estimator估算满足ITL SLO所需的最少SM数。
3. **SM分区决策**：将`SM_decode`个SM分配给Decode，`SM_total - SM_decode`个SM分配给Prefill。
4. **请求调度**：将新到达请求按TTFT SLO约束排队，插入Prefill执行；持续解码的请求在Decode分区执行。
5. **动态重分配**：当请求完成或新请求到达时，重新计算SM分配，动态调整两个分区的大小。

术语一般如何实现？如何使用？

在MuxWise中实现为控制平面组件，在每次prefill batch完成或decode iteration结束后触发调度决策。具体策略：
- **优先级**：始终优先满足decode SLO（decode使用best-fit最小SM数），prefill尽量早处理。不直接保证prefill SLO——若prefill出现SLO violation说明负载已超系统峰值容量。
- **PL发射数量**：用公式 `NPL = ⌈(Td × NT) / TP⌉` 计算，其中Td为decode估计延时、TP为整个prefill估计延时、NT为模型层数。
- **抢占**：允许短prefill非递归抢占长prefill（长被短抢占后不允许再被其他请求抢占），仅在被抢占的prefill batch仍能满足TTFT SLO时才允许抢占；抢占是可选功能。
- **在线精化**：利用运行时执行数据持续更新contention guard，不断提升估计精度。
通过GreenContext API实时调整SM分区配置（微秒级stream同步开销）。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

---
