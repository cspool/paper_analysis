# *B. End-to-End Performance*

In this section, we evaluate the end-to-end performance of the proposed Node-Link Balance strategy across different hardware configurations and 2D mesh sizes inferred in different batch sizes. The experiments are conducted using three hardware configurations, each with varying compute throughput and communication bandwidth:

- 2.5 TFLOPS compute throughput, 75 GB/s bandwidth
- 5 TFLOPS compute throughput, 50 GB/s bandwidth
- 10 TFLOPS compute throughput, 25 GB/s bandwidth

Additionally, we compare the performance across three different 2D mesh sizes: (4,4), (4,8), and (8,8), with 5 TFLOPS compute throughput and 50 GB/s bandwidth for consistency.

Better TBT latency through different hardware configurations: The results of these experiments, shown in Fig. [8](#page-7-0) and Fig. [9.](#page-7-1) Results in Fig. [8](#page-7-0) reveal how different methods respond to shifts in compute-to-communication ratios. When computation is limited and communication bandwidth is abundant (2.5 TFLOPS, 75 GB/s), EP suffers from severe workload imbalance, resulting in suboptimal TBT latency. In contrast, when computation is sufficient but communication becomes a bottleneck (10 TFLOPS, 25 GB/s), TP incurs heavy all-reduce communication costs, leading to degraded latency performance. Additionally, worth noting is that, for qwen, the expert routing exhibits high imbalance (Fig. [3\(](#page-2-0)a)), causing significant overhead in EP.

The Hybrid TP-EP with Compute-Balanced baseline achieves better performance by distributing expert load more evenly, but ignores communication topology, leading to degraded performance under constrained bandwidth.

In contrast, our Node-Link Balance strategy jointly considers both computation and communication during expert placement. It minimizes per-node compute load, inter-node communication volume, and per-link congestion. As a result, it consistently outperforms all baselines across different system configurations.

On average, our method achieves a speedup ranging from 1.1× to 1.8× compared to TP, 1.1× to 1.5× compared to EP, and 1.0× to 1.4× compared to Hybrid TP-EP with Compute-Balanced.

Better TBT latency through different mesh size: We evaluate the impact of mesh topology on TBT latency under a fixed configuration (5 TFLOPS, 50 GB/s). As shown in Fig. [9,](#page-7-1) our Node-Link Balance strategy consistently achieves low latency across mesh sizes, demonstrating strong adaptability.

An exception occurs in mixtral with an (8,8) mesh, where the Hybrid TP-EP with Compute-Balanced baseline achieves slightly better latency. This is likely due to mixtral's small number of experts, which must be spread across multiple nodes. In large mesh topologies, where communication regularity is more critical, the hybrid baseline benefits from its structured TP communication and moderate message volume.

Overall, our method remains effective across models and mesh sizes, particularly when the number of experts and the topology scale are well matched.

