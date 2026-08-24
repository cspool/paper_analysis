# 4 The Efficiency Paradox: Why Standard Optimizations Fail Agents

While standard optimization techniques—such as post-training quantization, context summarization, and continuous batching—have proven highly effective for stateless, single-turn LLM applications, they often yield counterproductive results when applied to autonomous agents. This section analyzes three critical dimensions where these conventional methods fail to align with the stateful, path-dependent nature of agentic workloads.

#### 4.1 The Quantization Trap

In deep-research agentic workflows, a fundamental conflict exists between inference throughput and reasoning fidelity. While low-bit quantization (e.g., W8A8) is standard for reducing memory bandwidth in chat applications, our analysis reveals it is detrimental to agents requiring strict syntactic precision.

As detailed in Table [1,](#page-3-0) blindly applying quantization creates a *"Latency Trap"*. Although INT8 quantization accelerates atomic forward passes by 45%, it degrades the agent's ability to perform precise operations—such as mathematical derivation and JSON formatting. This precision loss triggers a cascade of failures, forcing the agent into expensive self-correction loops. Consequently, while the *cost per token* decreases, the *total time-to-solution* increases by 70% due to the substantial overhead of retries.

<span id="page-3-0"></span>Table 1: The Quantization Trap. Comparison of FP16 baseline versus INT8 quantization on the HLE benchmark. Although quantization improves throughput (TPS), the loss in precision causes a sharp decline in success rate and triggers a 3.5× increase in retries, ultimately inflating the total time-to-solution.

| Metric                                            | FP16 (Baseline) | INT8 (Quantized) | ∆ Relative       | Impact                          |
|---------------------------------------------------|-----------------|------------------|------------------|---------------------------------|
| Throughput (Tokens/s)<br>Task Success Rate        | 42.5<br>88.2%   | 61.6<br>61.7%    | +45.0%<br>-30.0% | Faster Inference<br>Degradation |
| Breakdown of Latency:                             |                 |                  |                  |                                 |
| Avg. Inference Time (s)<br>Avg. Recovery Time (s) | 45.0<br>5.0     | 30.0<br>55.0     | -33.3%<br>+1000% | Step Speedup<br>Retry Loop      |
| Total Time-to-Solution (s)                        | 50.0            | 85.0             | +70.0%           | Slower End-to-End               |

#### 4.2 Granularity Mismatch

To mitigate the quadratic cost of attention in long-context scenarios, summarization is often employed to compress context. However, we identify a critical *granularity mismatch* for agents. Deep research agents necessitate high-fidelity access to raw details—such as specific variable names in code snippets or numerical values in financial reports—to execute tools correctly.

Table [2](#page-3-1) illustrates this inverse correlation. Summarization acts as a lossy compression layer, stripping away syntactic details deemed "redundant" by the proxy model. This ambiguity forces the agent to issue multiple clarification queries to recover missing information. As a result, the reduction in tokens per step is negated by a 3.5× increase in interaction turns, leading to comparable total token consumption but a significantly higher risk of context drift.

<span id="page-3-1"></span>Table 2: The Summarization Gap. While summarization reduces the context window size per step, it introduces ambiguity that forces the agent to perform more turns to clarify information. This "Scissor Effect" neutralizes the efficiency gains.

| Metric                  | Full Context | Summarized | Observation         |
|-------------------------|--------------|------------|---------------------|
| Avg. Tokens per Step    | 8,500        | 2,100      | 4× Reduction        |
| Avg. Turns to Solve     | 4.0          | 14.0       | 3.5× Increase       |
| Total Token Consumption | ≈ 34k        | ≈ 29.4k    | Marginal Gain       |
| Context Drift Rate      | Low          | High       | Cognitive Ambiguity |

#### 4.3 The Memory Persistence Bottleneck

Modern LLM serving engines typically utilize Shortest Job First (SJF) scheduling, optimizing for stateless, highthroughput chat workloads. This design is fundamentally misaligned with agentic workloads, which are characterized by long-context, multi-turn sessions requiring *memory persistence*.

As shown in Table [3,](#page-4-0) standard schedulers prioritize short incoming requests, aggressively evicting the KV-cache of idle agents waiting for tool execution. Upon resumption, the system must re-compute the entire historical context (Prefill). For agents with contexts exceeding 32K tokens, this leads to a cache hit rate of only 15%, inducing massive latency spikes (up to 3.1s). This demonstrates that agent efficiency relies not merely on compute speed, but on the *temporal locality* of memory management.

<span id="page-4-0"></span>Table 3: Impact of Scheduling Policy on Memory Persistence. Standard SJF scheduling favors short-context agents and causes frequent KV-cache eviction for long-context agents, leading to low cache hit rates and high prefill latency.

| Workload Type       | Standard SJF                                       |                 |  |
|---------------------|----------------------------------------------------|-----------------|--|
|                     | Hit Rate                                           | Prefill Latency |  |
| Short Context (<4K) | 92%                                                | 150 ms          |  |
| Long Context (>32K) | 15%                                                | 3,100 ms        |  |
| Observation         | High eviction and re-computation for long contexts |                 |  |

