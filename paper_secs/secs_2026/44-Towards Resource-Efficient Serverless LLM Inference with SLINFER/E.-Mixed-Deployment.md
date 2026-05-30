# E. Mixed Deployment

To reflect real-world scenarios with mixed model sizes, we evaluate SLINFER under mixed-sized workloads, including CodeLlama-34B deployed with tensor parallelism (2 GPUs/instance). To accommodate the increased workload scale, this experiment runs on 4 CPUs and 6 GPUs. The CPU results are omitted as all systems saturate CPU usage.

Figure 26 shows that SLINFER consistently uses fewer GPUs than both sllm+c and sllm+c+s, but its efficiency varies with model popularity. When small models dominate (4:1:1:1), SLINFER can deploy up to four instances per CPU, while reserving GPUs primarily for large models. In contrast, when large models dominate (1:1:4:1), the deployment density drops due to higher resource demands, reducing sharing efficiency. In the extreme case (0:0:0:1), SLINFER falls back to exclusive GPU allocation, similar to sllm+c and sllm+c+s.

We also observe that sllm+c+s performs worse under large models due to static partitioning that severely limits concurrency under high demands. Overall, since most popular models are relatively small [5], SLINFER can achieve significant resource savings in practice.

### F. Investigate GPU Efficiency

Figure 25 presents an analysis of GPU efficiency when serving mixed models (3B, 7B, and 13B) of 2:2:2 ratio. As discussed in §IX-E, SLINFER's behavior for larger models aligns with baselines and is omitted for brevity.

SLINFER achieves near-optimal memory utilization with close to 1. In contrast, sllm and sllm+c+s both exhibit a three-tier memory utilization pattern corresponding to the three model sizes, with most instances using less than half of their allocated memory. This suggests significant over-provision, since they allocate all available memory in a node (or half of the node) to each instance for KV-cache space.

Despite the sparsity of serverless workloads, SLINFER achieves a 74% higher average batch size than sllm, as instance sharing prolongs execution intervals and accumulates more requests. sllm+c+s suffers from lower peak batch sizes due to fixed resource partitioning that limits concurrency.

TABLE III: Performance under prefill-decode disaggregation. Each cell shows results for aggregated PD / disaggregated PD.

| System   | Load (models) | Avg. GPU Usage | SLO Rate (%) |
|----------|---------------|----------------|--------------|
| sllm+c+s | 32            | 2.0 / 3.0      | 99 / 93      |
|          | 64            | 3.6 / 3.9      | 93 / 70      |
|          | 128           | 4.0 / 4.0      | 65 / 35      |
| SLINFER  | 32            | 0.9 / 1.0      | 99 / 99      |
|          | 64            | 2.5 / 2.9      | 99 / 98      |
|          | 128           | 4.0 / 4.0      | 86 / 69      |

