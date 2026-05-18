## Inter-Request Parallelism for Agentic Workloads（Agent负载的请求间并行）

术语是什么？通过联网搜索让回答具体和精准。

Inter-Request Parallelism是针对AI agent workload的系统级并行策略。由于单agent请求内部LLM inference和tool execution存在顺序依赖（LLM输出决定调用哪个tool，tool结果决定下一次LLM输入），intra-request parallelism（请求内并行）机会有限。因此，提高agent serving throughput需要利用inter-request parallelism——让多个并发agent request共享GPU，当一个请求在等待tool返回时，GPU转而服务其他请求的LLM call。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
// Inter-Request Parallelism的时序示意
Time →
请求A: |LLM|====Tool(1.2s)====|LLM|==Tool==|LLM|...
请求B:        |LLM|==Tool==|LLM|====Tool====|...
请求C:              |LLM|====Tool====|LLM|...

GPU执行: |A_LLM|B_LLM|C_LLM|A_LLM|B_LLM|...
         ↑ A等待tool时GPU不idle，被B和C的LLM call填补
```

论文量化结果：
- ReAct sequential (单请求): 0.10 QPS (HotpotQA), 0.19 QPS (WebShop)
- ReAct concurrent (inter-request parallelism): 2.6 QPS (HotpotQA, 25×), 1.2 QPS (WebShop, 6.2×)
- Cost: average latency增加2.1×
- HotpotQA gain > WebShop：因tool latency更长(1.2s vs 20ms)，idle gap更多可填补
- 即使有inter-request parallelism，agent serving仍远低于ShareGPT (2.6/1.2 vs 6.4 QPS)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：(1) vLLM continuous batching自动合并多个concurrent request的decode step；(2) agent server维护多个async worker，各自管理独立request state machine；(3) worker在tool等待期间不占GPU，GPU scheduler从其他worker的pending LLM request中选择执行；(4) FCFS scheduler + Poisson arrival模拟流量。论文AgentBench中多个agent worker异步运行，LLM inference requests汇聚到vLLM backend通过continuous batching合并执行。

涉及论文标题：
- The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective

---
