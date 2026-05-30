# G. GPU hardware variations

Given the different characteristics of prompt and token generation phases, we measure the performance impact on

|          | Coding   |          |               | Conversation |         |               |
|----------|----------|----------|---------------|--------------|---------|---------------|
|          | A100     | H100     | Ratio         | A100         | H100    | Ratio         |
| TTFT     | 185 ms   | 95 ms    | 0.51×         | 155 ms       | 84 ms   | 0.54×         |
| TBT      | 52 ms    | 31 ms    | $0.70 \times$ | 40 ms        | 28 ms   | $0.70 \times$ |
| E2E      | 856 ms   | 493 ms   | $0.58 \times$ | 4957 ms      | 3387 ms | $0.68 \times$ |
| Cost [5] | \$0.42   | \$0.52   | 1.24×         | \$2.4        | \$3.6   | 1.5×          |
| Energy   | 1.37 Whr | 1.37 Whr | $1 \times$    | 7.9 Whr      | 9.4 Whr | 1.2×          |

TABLE IV: P50 request metrics on A100 vs. H100 without batching on Llama-70B.

the two from running on different hardware. Table I shows the specifications for DGX-A100 [15] and DGX-H100 [16]. The memory-to-compute ratio favors A100 over H100. Table IV shows our findings. We see a lower performance impact on the token generation phase (TBT) as compared to the Prompt phase (TTFT). Since coding requests are dominated by prompt phase, by having very few generated tokens, the E2E latency impact from A100 is worse on coding than conversation. Furthermore, we see that A100 has better or equal inference cost and energy overall compared to H100.

*Insight VII:* Token generation can be run on less compute-capable hardware for better Perf/W and Perf/\$ efficiencies.

#### IV. SPLITWISE

Based on our characterization insights, we propose Splitwise, a technique to split the prompt and generation phases in the LLM inference on to separate machines.

Figure 10 shows the high-level overview of Splitwise. We maintain two separate pools of machines for prompt and token processing. A third machine pool, the mixed pool, expands and contracts as needed by the workload. All machines are preloaded with the model of choice. When a new inference request arrives, the scheduler allocates it to a pair of machines (*i.e.*, prompt and token). The prompt machines are responsible for generating the first token for an input query, by processing all the input prompt tokens in the prompt phase and generating the KV-cache. The prompt machine also sends over the KV-cache to the token machine, which continues the token generation until the response is complete. We use continuous batching at the token machines to maximize their utilization. Machines in mixed pool use mixed continuous batching.

At a lower request rate, we target better latency in Splitwise, while, at a higher request rate, we target avoiding any performance or throughput reduction due to the fragmentation between prompt and token machine pools.

Splitwise uses a hierarchical two-level scheduling as shown in Figure 10. The cluster-level scheduler (CLS) 1 is responsible for machine pool management and for routing incoming inference requests. The machine-level scheduler (MLS) 2 maintains the pending queue and manages batching of requests at each machine.

