## Goodput（有效吞吐量）

术语是什么？

Goodput是LLM推理服务中同时考虑吞吐量和SLO合规性的服务质量指标。定义为**在满足所有SLO约束（如TTFT和ITL/TPOT）的条件下，每秒成功完成的请求数**。区别于原始吞吐量（Throughput），Goodput仅统计满足延迟要求的请求，反映的是"有用"的吞吐量。

从系统架构角度拆解术语：

Goodput的计算与调度流程紧密相关：

1. **SLO定义**：设定两类SLO——TTFT（Time to First Token，首token延迟）和ITL（Inter-Token Latency，token间延迟）。例如"P90 TTFT < 200ms 且 P90 ITL < 50ms"。
2. **请求追踪**：每个请求记录到达时间、首token时间、各token生成时间。
3. **合规判断**：请求的TTFT和ITL均在SLO阈值内，则该请求合规。
4. **Goodput计算**：`Goodput = 合规请求数 / 时间窗口`。
5. **调优目标**：调度器以最大化Goodput为目标，而非最大化原始吞吐量，避免为追求高并发而牺牲服务质量的"假吞吐量"。

系统设计中，Goodput作为调度决策的优化目标，指导SM资源分配、请求接纳控制和批处理策略。

术语一般如何实现？如何使用？

在MuxWise中，SLO-aware Dispatcher以Goodput最大化为目标进行资源分配：先确保Decode有足够SM满足ITL SLO，再将剩余SM全部投入Prefill以提升整体Goodput。实际部署中，Goodput用于比较不同LLM服务系统的综合性能。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing
- Laser: Unlocking Layer-Level Scheduling for Efficient Multi-SLO LLM Serving
- AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding

Laser定义goodput为**维持90% SLO attainment时达到的throughput**（区别于MuxWise的"满足所有SLO约束"的定义）。Laser中SLO attainment衡量decode请求P99 TBT低于SLO target的比例。Laser提升goodput 1.67× vs Sarathi-Serve/DistServe；当goodput attainment target提至99%时，throughput improvement最高达1.85×。

AdaServe定义goodput为**成功满足TPOT SLO的请求产生的tokens/s**（区别于按请求数统计）。AdaServe在multi-SLO混部场景（coding copilot/chatbot/summarization）下goodput最高提升1.9×/1.7× vs best baseline。

TokenFlow提出另一种有效吞吐量指标**Effective Throughput**：不是基于SLO合规（请求级），而是基于token在用户消费buffer中的位置进行加权计数（token级）。buffer中离消费点<10% output_length的token全计、10%-20%线性衰减、>20%不计入。Goodput保证服务质量下限，Effective Throughput反映streaming体验——两者互补但不可互相替代。

---
