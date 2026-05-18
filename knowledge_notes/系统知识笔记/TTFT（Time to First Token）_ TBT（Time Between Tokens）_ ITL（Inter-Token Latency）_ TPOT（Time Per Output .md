## TTFT（Time to First Token）/ TBT（Time Between Tokens）/ ITL（Inter-Token Latency）/ TPOT（Time Per Output Token）

术语是什么？

LLM推理服务的核心延迟SLO指标：
- **TTFT (Time to First Token)**：从请求发送到收到第一个token的时间，主要由Prefill阶段决定。
- **TBT (Time Between Tokens)**：Decode阶段**每个独立token**的生成延迟。与TPOT不同，TBT计量每个token的个体延迟而非平均值，因此能暴露尾延迟问题。MuxWise选择TBT而非TPOT作为SLO指标，因为"TPOT is an average metric that may mask the poor performance of some tokens"。论文设置TBT SLO为Llama-8B 50ms、Llama-70B 100ms。
- **ITL (Inter-Token Latency)**：连续两个输出token之间的平均时间间隔，主要由Decode阶段决定。
- **TPOT (Time Per Output Token)**：所有输出token的**平均**生成时间，等于总decode时间/输出token数。是平均指标，会掩盖单token的差尾延迟。

从系统架构角度拆解术语：

这三个指标是调度系统的SLO约束输入：
1. Scheduler接收带SLO的请求（如"P95 TTFT < 500ms, P95 ITL < 100ms"）。
2. 请求排队时，Scheduler预估能否在TTFT SLO内完成该请求的Prefill，若不能则拒绝或降级。
3. Decode阶段，Scheduler确保批处理大小和SM资源分配使得ITL不超过SLO。
4. 监控系统实时追踪各请求的TTFT/ITL，违反SLO的请求不计入Goodput。

术语一般如何实现？如何使用？

在MuxWise中，TTFT和ITL SLO直接驱动资源分配决策：SLO-aware Dispatcher根据ITL SLO反推所需的最小Decode SM数；TTFT SLO约束Prefill的排队和调度优先级。Bullet进一步使用TTFT和TPOT（而非TBT）作为SLO指标，通过SRM预测不同SM分区下的TTFT/TPOT，在prefill队列堆积时增大prefill SM以降低TTFT，在decode P90 TPOT接近SLO边界时增加decode SM以保障TPOT SLO。

涉及论文标题：
- Towards High-Goodput LLM Serving with Prefill-decode Multiplexing
- Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration
- PASCAL: A Phase-Aware Scheduling Algorithm for Serving Reasoning-based Large Language Models

**PASCAL的重新定义（针对Reasoning-based LLMs）**：在reasoning-based LLMs中，TTFT不再仅由prefill阶段决定。TTFT = prefill stage latency + reasoning phase latency + 从reasoning结束到首个answering token生成的latency。因为reasoning tokens（如CoT的中间推理步骤）对用户不可见，用户要等到answering phase开始后才收到第一个"user-visible"token。这与conventional LLMs中TTFT仅覆盖prefill stage形成根本性区别。

---
