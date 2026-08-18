# 2) Real-Time Boundary Adaptation Algorithm

**Runtime Objective.** At each decision epoch, LRP selects a boundary  $b \in \mathcal{K}$  from the pre-profiled portfolio that minimizes predicted end-to-end latency  $\hat{T}_{e2e}(b)$ —comprising queueing, edge execution, network transfer, cloud execution, and switching overhead—under current resource conditions, rather

![](_page_6_Figure_11.jpeg)

Fig. 7: DYNOPIPE mitigates network bandwidth contention by dynamically selecting activation volumes transmitted over the access network, achieving progressively better optimization as request volume increases.

than re-solving the full placement problem online. The Latency-Regulated Placement (LRP) algorithm dynamically reconfigures pipeline boundaries through pre-computed configuration portfolios, addressing the trade-off between adaptation responsiveness and computational overhead.

LRP employs bottleneck-aware adaptive weighting that adjusts placement priorities based on the dominant system bottleneck. When bandwidth is limited,  $\lambda$  increases to favor activation-minimizing boundaries; under heavy computational load,  $\lambda$  decreases for balanced workload distribution; in memory-constrained situations, device memory limits take precedence. This context-sensitive adjustment maintains low latency while ensuring sub-millisecond decision overhead.

Configuration stability uses hysteresis mechanisms preventing oscillatory switching. Boundary transitions occur only when performance improvements exceed threshold  $\delta$  (15-20%), with cooldown periods preventing rapid reconfigurations. This reduces reconfiguration overhead from seconds to milliseconds.

The pre-computed portfolio maintains 3-5 specialized configurations: bandwidth-constrained (boundaries after attention layers), compute-constrained (early cloud placement), memory-constrained (minimal edge footprint), and balanced (proportional distribution). This achieves competitive performance with submillisecond selection latency.

Blended Constraints and Sensitivity. Under simultaneous resource constraints (bandwidth and memory pressure), the SelectBoundary function (Algorithm 1) evaluates active triggers and selects the configuration minimizing worst-case stage latency across all constraints. The hysteresis threshold (15-20%) and cooldown period balance adaptation responsiveness with stability: aggressive switching increases reconfiguration overhead with minimal gains, while conservative thresholds delay beneficial adaptations. The weighting parameter  $\lambda$  uses exponential smoothing to prevent oscillation under fluctuating network conditions.

Assumptions, Rationale, and Limitations. LRP adopts a finite-portfolio heuristic rather than exhaustive online reoptimization or a learned controller: exhaustive search adds O(n) per-decision overhead incompatible with sub-millisecond budgets, while learned policies require training data and are hard to stabilize under shifting edge conditions. The heuristic

assumes (1) a single edge-cloud boundary per request, valid when cross-domain latency dominates intra-domain communication (10–50× in our testbed); (2) monotonic resource-performance relationships, confirmed for uniform transformer architectures (Table III) but potentially weaker for MoE or mixed-modality models; and (3) representative offline profiles. The trigger ordering (bandwidth → compute → memory) reflects empirical bottleneck severity; under blended constraints, SelectBoundary minimizes worst-case stage latency across all active triggers. LRP does not guarantee global optimality—architectures with irregular per-layer costs or highly non-stationary environments may require larger portfolios or learned adaptation.

**Profiling Methodology.**  $T_{\rm comp}$ ,  $T_{\rm mem}$ , and  $T_{\rm comm}$  in Algorithm 1 are obtained from a lightweight offline profiling phase that executes representative prompts (128 tokens, batch=1/4/8) on each device pair. Per-layer execution time and activation sizes are measured and stored in a lookup table (<30 KB per model); runtime adaptation uses these profiles plus live telemetry (bandwidth, GPU utilization, memory pressure sampled every 500 ms) to re-evaluate placement decisions.

**Four-Dimensional Optimization Relationship.** The ablation study in §5.6 reveals that optimal split-point placement is a dynamic function of four interacting variables: request arrival rate  $\lambda$ , network conditions (RTT, bandwidth), edge compute capability  $\alpha_{edge}$ , and model structure. The total request latency decomposes as:

$$T_{\text{total}} = T_{\text{queue}}(\lambda, \mu) + T_{\text{edge}}(SP) + T_{\text{net}}(SP, RTT) + T_{\text{cloud}}(SP)$$
 (4)

where the effective pipeline service rate  $\mu(SP)=1/\max(T_{\text{edge}}(SP),T_{\text{cloud}}(SP))$  determines system capacity. At low load  $(\lambda\ll\mu)$ , minimizing single-request latency favors SP=0 (cloud-only). As load approaches capacity  $(\lambda\to\mu)$ , maximizing throughput through balanced split points that satisfy  $T_{\text{edge}}\approx T_{\text{cloud}}$  becomes critical. Under network contention, the optimal SP shifts to balance transmission volume against pipeline efficiency. This relationship validates DynoPipe's multi-configuration portfolio: different configurations naturally correspond to different operating regions in this four-dimensional space, and the LRP algorithm selects the appropriate configuration based on real-time monitoring of all four dimensions.

