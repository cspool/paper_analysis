## Effective Throughput（有效吞吐量，Streaming Token-Weighted）

术语是什么？

Effective Throughput是TokenFlow提出的面向LLM text streaming服务的吞吐量指标，与传统Goodput（SLO合规请求数）不同，Effective Throughput根据**用户token消费进度**对生成的token进行加权计数：buffer中距离用户消费点越近的token权重越高，远离消费点的"超量生成"token不计入有效吞吐。这反映了streaming服务中"用户实际感知到的token输出速率"。

从系统架构角度拆解术语：

Effective Throughput的计算流程：
1. **Buffer状态监控**：Request Tracker持续追踪每个请求的已生成token数（generated）和用户已消费token数（consumed），buffer = generated - consumed。
2. **Token位置权值函数**：对buffer中每个位置的token赋予权值w(i)，其中i为token在buffer中的位置（0=最靠近消费点）：
   - i / output_length < 10%  → w = 1.0（用户即将消费，全额计数）
   - 10% ≤ i / output_length < 20%  → w从1.0线性衰减至0
   - i / output_length ≥ 20%  → w = 0（距消费点太远，不计入有效吞吐）
3. **Effective Throughput计算**：`Effective_Throughput = Σ(每个generated token的w(token)) / 时间窗口`。
4. **调度影响**：Buffer较高的请求（>20% output_length）token权值为0，scheduler将其识别为"低价值占用"，优先抢占这些请求以将GPU切换给buffer较低或尚未获得首token的请求。

与Goodput的关键区别：Goodput从**SLO合规**角度衡量（请求级：TTFT和ITL是否满足阈值），Effective Throughput从**token消费进度**角度衡量（token级：生成的token离用户消费点有多远）。两者互补：Goodput保证服务质量下限，Effective Throughput反映streaming体验优劣。

术语一般如何实现？如何使用？

TokenFlow在scheduler和monitor中实现Effective Throughput的计算：Request Tracker维护每个请求的token生成时间戳和buffer大小，scheduler据此做admission/preemption/resumption决策。实验显示TokenFlow在burst场景下effective throughput最多提升52.9%，Poisson场景下最多提升82.5%。指标设计动机：LLM token生成速率通常远高于用户阅读/听取速率（如30 tokens/s GPU生成 vs 5-10 tokens/s用户消费），超量token堆在buffer中对用户体验无增益，不应计入系统有效产出。

涉及论文标题：
- TokenFlow: Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling

---
