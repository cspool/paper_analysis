# A Additional Experiments

## <span id="page-15-0"></span>A.1 Throughput analysis

We evaluate inference throughput during scaling transitions using the DeepSeek V2 Lite model in an offline batch processing setting. An offline batch of 10000 requests is drawn from a synthetic workload with 500 prefill tokens and a random range of 250-500 decode tokens. The system is initially provisioned with 6 NPUs and scales up to 8 NPUs at a fixed time for all baselines. Baselines that are not applicable for this transition are excluded. To capture scaling behavior consistently, we divide execution into three windows: before scaling, during scaling, and after scaling. The "during" window is measured as ±5 seconds around the longest scaling transition among all baselines (in this case, Vertical Cold Restart). This setup allows us to examine how different methods perform in steady-state, during the critical transition, and after additional capacity becomes available.

<span id="page-15-3"></span>Table 2. Throughput comparison (requests/sec) for scale-up from DP3TP2 to DP4TP2. We report throughput for three windows: before, during, and after scaling.

| Method                  | Before | During | After |
|-------------------------|--------|--------|-------|
| Vertical (Concurrent)   | 1.338  | 0.467  | 2.268 |
| Vertical (Cold Restart) | 6.002  | 2.064  | 7.818 |
| Elastic (Ours)          | 6.002  | 3.943  | 7.818 |

Table [2](#page-15-3) reports throughput across the three windows. Before scaling, both ElasticMoE and Cold Restart achieve similar throughput, while Concurrent performs poorly because it reserves memory for potential scaling and thus operates at reduced capacity at all times. During scaling, ElasticMoE sustains the highest throughput—nearly double that of Cold Restart—due to lower scaling latency and zero downtime. Although throughput temporarily dips compared to steady state (since the active instance pauses new request intake, lowering effective batch size), service remains uninterrupted, avoiding downtime entirely. After scaling, all methods benefit from the added NPUs and achieve higher throughput than before.

These results demonstrate that ElasticMoE not only scales faster than baselines but also maintains substantially higher throughput during the critical transition period, combining zero downtime with efficient resource utilization.

#### <span id="page-15-1"></span>A.2 Scale-down Latency Analysis

We complement the scale-up evaluation by analyzing scaledown behavior across the same three models. Here, scaling transitions reduce the number of NPUs, with step sizes of 2 for DeepSeek V2 Lite and Qwen 30B-A3B, and progressively larger steps for DeepSeek V3. Figure [12](#page-16-2) reports the results.

As in the scale-up scenario, ElasticMoE achieves substantially lower latency than competing baselines. Across all tested configurations, our method consistently completes scale-down in less than 0.15× the time of the fastest baseline. This translates into latency reductions of 80–90% relative to conventional vertical scaling methods. Vertical (Cold Restart) suffers long downtime since the old instance must terminate before the new instance initializes, while Vertical (Extravagant) and Vertical (Concurrent) temporarily maintain overlapping configurations, inflating latency and memory usage. In contrast, ElasticMoE reclaims resources immediately via live reconfiguration, avoiding redundant weight reloading and memory spikes. The benefits are particularly pronounced in DeepSeek V3, where even aggressive 16→2 NPU reductions complete with ≈ 0.10× baseline latency.

Overall, these results confirm that ElasticMoE not only scales up quickly to meet rising demand but also scales down efficiently to release capacity, ensuring cost-effective elasticity with minimal disruption.

#### <span id="page-15-2"></span>A.3 Ablation Analysis for Scale-Down

Table [3](#page-15-4) reports the progressive ablation study for a scaledown event (DP4→DP3). The trends mirror those in scaleup. Disabling the IPC-safe allocator has negligible effect on latency but increases peak memory. Removing HCCL transfers significantly slows the transition. Eliminating preinitialization and zero-copy reuse further worsens performance: scale-down latency rises above 60s, and without zero-copy, downtime is introduced because weights and KV caches must be duplicated. These results reinforce that all four mechanisms—efficient allocation, P2P transfers, preinitialization, and zero-copy reuse—are jointly essential for low-latency, zero-downtime scaling in both directions.

