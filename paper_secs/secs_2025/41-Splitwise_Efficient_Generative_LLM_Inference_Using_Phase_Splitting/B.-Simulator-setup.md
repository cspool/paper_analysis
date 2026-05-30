# B. Simulator setup

We build a simulator to explore cluster designs and evaluate Splitwise at scale. The simulator code is open source [20].

Figure 13 shows the design of our simulator. The simulator is event-driven and faithfully models the Splitwise machine pools, schedulers, machine-level memory and queues, and KV-cache transfer. We first profile the LLM on the target hardware with various input/output sizes (1). Based on the characterization profiles, we build a performance model. The simulator takes as input the request traces, SLOs, the performance model, and the configurations for cluster and scheduler (2). For our

|      | P50        | P90  | P99 |
|------|------------|------|-----|
| TTFT | $2 \times$ | 3×   | 6×  |
| TBT  | 1.25×      | 1.5× | 5×  |
| E2E  | 1.25×      | 1.5× | 5×  |

TABLE VI: SLO expressed as slowdown compared to a request running on DGX-A100 under no contention.

evaluation, we use the prompt and token size distributions from the production traces in Section III. We tune the Poisson arrival rate to increase and decrease the load (requests per second) for cluster sizing. The simulator provides the achieved metrics per request (TTFT, TBT, E2E), and the machine utilization levels 3. We cross-validated the performance model with hardware experiments to ensure accuracy; we also validated the simulator end-to-end using production load with over 50K iterations to ensure fidelity (4).

**Performance model.** We build a piece-wise linear performance model using performance profiles at various batch sizes, input sizes, output sizes, in the required parallelism configuration on A100 and H100 machines from Section III. We validate that our performance model has high accuracy; it incurs a mean absolute percentage error (MAPE) of less than 3% when evaluated with a 80:20 train:test dataset split.

Communication model. In our evaluation, KV-cache transfers cause inter-machine communication, whereas tensor parallelism only causes intra-machine communication. We model intermachine communication overheads by benchmarking our KV-cache transfer implementation over Infiniband in Section VI-A.

**SLOs.** To determine the maximum throughput that can be supported by a given cluster design, we use P50, P90, and P99 SLOs for TTFT, TBT, and E2E latency metrics. Table VI shows our SLO definition using DGX-A100 as a reference. We require all nine SLOs to be met. SLOs on TTFT are slightly looser, since it has a much smaller impact on the E2E latency.

**Baselines.** We compare our Splitwise designs against Baseline-A100 and Baseline-H100. The clusters in these baselines consist of just DGX-A100s and DGX-H100s, respectively. Both baselines use the same mixed continuous batching that Splitwise uses for mixed pool machines (described in Section IV-A).

