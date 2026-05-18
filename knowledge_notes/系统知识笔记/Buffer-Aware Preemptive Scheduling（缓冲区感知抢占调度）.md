## Buffer-Aware Preemptive Scheduling（缓冲区感知抢占调度）

术语是什么？

Buffer-Aware Preemptive Scheduling是TokenFlow提出的LLM serving抢占策略，相较传统基于固定时间片的抢占（如Round-Robin）或基于QoE的调度（如Andes），其核心创新是**将每个请求的客户端token buffer实时状态作为抢占决策的核心依据**：请求buffer越高（用户有充足已生成未消费token），越容易被安全抢占；buffer越低（用户即将消耗完buffer面临卡顿风险）或尚未获得首token，越优先获得GPU资源。

从系统架构角度拆解术语：

调度执行流程：
1. **Request Tracking**：Request Tracker对每个活跃请求持续追踪：buffer_size（已生成-已消费）、token generation rate（GPU生成速率）、token consumption rate（用户消费速率）、I/O状态（KV cache是否在host同步中）、GPU memory footprint。
2. **Determine Working Set（第一步）**：Scheduler根据GPU显存容量、各请求KV footprint、等待队列长度、I/O队列长度和buffer safety condition，确定可"过量承诺"的请求集合。Safety condition：buffer_size ≥ switching_delay × consumption_rate（即buffer足以覆盖抢占+恢复的时间窗口，不会导致用户端卡顿）。
3. **Buffer Balancing（第二步）**：在working set内按加权排序——权重 = f(buffer_size, generation_rate, consumption_rate, memory_constraint)。优先选出：buffer最低的请求（急需token）、尚未获得首token的请求（降低TTFT）、消费速率最高的请求（token被快速消耗）。Greedy selection + 局部交换优化。
4. **Preemption执行**：当某个请求的buffer满足safety condition且存在更高优先级请求时，scheduler决定preempt。Request Offload Manager将请求移出running set→KV Cache Manager利用后台write-through已同步的KV cache快速offload剩余chunk→GPU资源转向高优先级请求。
5. **Resumption**：当被抢占请求的buffer降至安全阈值以下（逼近耗尽），scheduler将其重新加入running set，KV Cache Manager从host加载KV cache→恢复decode。

术语一般如何实现？如何使用？

TokenFlow在SGLang上约3000行Python代码实现该调度器。调度器周期性reschedule（interval为可配参数）：interval越短，buffer感知越及时但调度开销越高；buffer safety conservativeness越低，抢占越激进响应更快但卡顿风险更高。与Andes（QoE-aware + Token Pacer）的关键区别：Andes仅根据QoE指标调度，缺少与KV cache管理的协同；TokenFlow的scheduler和KV cache manager双向协作，scheduler在决策时考虑I/O overhead，memory manager提前准备减少实际抢占延迟。

涉及论文标题：
- TokenFlow: Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling

---
