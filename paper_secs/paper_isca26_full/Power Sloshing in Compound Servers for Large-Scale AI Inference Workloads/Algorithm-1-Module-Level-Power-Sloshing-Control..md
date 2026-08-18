# Algorithm 1 Module-Level Power Sloshing Control.

1: **Inputs:** Module power cap  $P_M$ ; GPU utilization targets

 $u_{min} < u_{max}$ ; GPU frequency levels  $\mathcal{F}_G$  with max  $f_{GM}$ ;

```
GPU power model P_G(f_G, u_G); CPU-GPU frequency
    mapping F(\cdot)
 2: while system is running do
      Read current GPU utilization u_G
4:
      if u_G < u_{min} then
5:
         Decrease GPU budget: f_G \leftarrow \text{Lower}(\mathcal{F}_G, f_G)
      else if u_G > u_{max} then
6:
         Increase GPU budget: f_G \leftarrow f_{GM}
7:
      end if
8:
9:
      if power-constrained then
         ▷ Budget reallocation mode
10:
         Estimate GPU power: \hat{P}_G \leftarrow \hat{P}_G(f_G, u_G)
         Compute CPU budget: P_C \leftarrow \max\{0, P_M - \hat{P}_G\}
11:
         Apply CPU power cap: SetCpuPowerLimit(P_C)
12:
13:
      else
         ⊳ Energy optimization mode
14:
         Coordinated CPU scaling: f_C \leftarrow F(f_G)
15:
         Apply CPU frequency: SETCPUFREQ(f_C)
      end if
17: end while=0
```

<span id="page-7-0"></span>If  $u_G > u_{max}$ , the controller immediately reassigns budget to the GPU by setting  $f_G = f_{GM}$  (line 7) to protect latency during bursts.

Depending on the operational regime, the algorithm either optimizes for performance under the fixed aggregate server power budget, or modulates frequency for power savings. In the power-constrained regime (lines 9–12), the selected GPU frequency dictates the respective GPU power budget, and the remaining budget is allocated to the CPU. Because GPU power does not instantaneously track frequency changes and direct per-interval GPU power availability is not explicitly exposed, we estimate the GPU's steady-state power under the chosen  $f_G$  using the power model derived from our characterization. We then compute the residual CPU budget as  $P_C = P_M - \sum_i \hat{P}_{G_i}(f_G, u_G)$  and apply this cap using the CPU power-limit interface. This ensures the module is within  $P_M$  while harvesting and reusing otherwise stranded power.

When the server is not power-constrained—e.g., during periods of low demand—the algorithm instead operates in energy optimization mode (lines 13–16). The CPU frequency is derived from the previously determined GPU frequency (lines 4–8) via a linear mapping  $f_C \leftarrow F(f_G)$ , leveraging the observed correlation between CPU and GPU utilization at power-optimal points and avoids separate CPU monitoring overhead. This simple mapping aligns well with empirical behavior across our workloads; non-linear regions primarily appear only at very high load, but are practically bounded by load balancing and further mitigated by our fast rampup policy (immediately setting  $f_G = f_{GM}$  upon a utilization spike). We use module-level budgeting as the default and deploy coordinated DVFS when global power headroom makes

budget reallocation unnecessary.

Per-Model Target Utilization Selection. The feedback loop in our power management scheme steers GPU utilization (uG) toward a target range, which must be carefully chosen to balance power efficiency and SLO compliance. If the target utilization thresholds (umin, umax) are set too high, the system may lack sufficient headroom to absorb sudden load spikes, increasing the risk of SLO violations. Conversely, thresholds that are too low can miss out on power-saving opportunities.

To address the diversity of models and workloads in largescale datacenters, we select these thresholds on a per-model basis. Specifically, for each model, we analyze historical CPU and GPU utilization traces collected while running at maximum frequency under production workloads. We then set the target utilization range based on a chosen percentile of observed utilization, ensuring that the system maintains enough slack to handle typical workload bursts while still reducing power during periods of lower demand.

We implement two variants of this approach: in a more aggressive *Power-Optimized* configuration, we set the target to the 90th percentile of historical utilization (P90, e.g., 60%– 70% utilization in C1), maximizing power savings. In a more conservative *SLO-Optimized* configuration, we use the 75th percentile (P75, e.g., 40%–50%), providing additional headroom to further reduce the risk of SLO violations. By adapting target utilization thresholds to each model's unique resource profile and workload variability, our approach provides robust SLO compliance and maximizes power savings across heterogeneous deployment scenarios.

Responsiveness and Stability. The proposed dynamic power management scheme must ensure that the control loop can react quickly enough to sudden changes in workload, while avoiding instability or oscillations in frequency settings.

In our system, the main source of control latency is the interval at which GPU utilization (uG) is sampled and reported. We set the interval to 100 ms to balance responsiveness and stability: it is short enough to capture rapid spikes from traffic bursts or batch arrivals, but long enough to filter out transient noise and prevent unnecessary frequency changes. This time granularity also aligns well with the responsiveness needed for production AI inference workloads, with typical p99 SLOs in the tens to hundreds of ms range.

Frequency scaling is highly efficient, with hardware-level transitions typically completing within 100 µs to a few milliseconds. As a result, the end-to-end response time of the control loop is dominated by the utilization sampling interval.

To further improve responsiveness to load surges, we implement an explicit fast-response mechanism. When u<sup>G</sup> exceeds the upper threshold umax, the controller immediately increases the GPU frequency to its maximum value (fGM) in a single step, rather than ramping up gradually (Alg. [1,](#page-7-0) lines 6– 7). This ensures that the system can promptly accommodate sudden increases in query volume, minimizing the risk of SLO violations during bursts. In contrast, when reducing frequency in response to lower utilization, the controller steps down one frequency level at a time (Alg. [1,](#page-7-0) lines 4–5). The conservative approach prevents over-correction and oscillatory behavior, which could otherwise degrade both performance and power efficiency. This *asymmetric adjustment policy*: rapid upscaling and gradual downscaling, provides robust stability across a wide range of production workloads.

Implementation Simplicity and Generality. The proposed mechanism is intentionally simple, requiring only access to standard GPU utilization counters and frequency control interfaces (e.g., via NVIDIA's NVML or similar APIs). It does not depend on application-specific instrumentation, detailed workload characterization, or offline profiling. These characteristics make it practical, broadly applicable across diverse server types and workloads, and easy to deploy at scale.

By prioritizing GPU control and deriving CPU power budget accordingly, we further reduce monitoring overhead and complexity. This design decision is justified by the observation that, in AI inference servers, the GPU is the primary power consumer and performance bottleneck, while the CPU's role is secondary and closely coupled to GPU activity.

Summary. In summary, our mechanism provides a practical, low-overhead template for real-time server power management in datacenters. By leveraging GPU utilization as a proxy for load and employing a simple, robust control loop, it achieves power savings with minimal risk to performance or SLOs. We treat DVFS as an actuation mechanism for *dynamic server-level power budgeting*: under a fixed module power cap, we continuously harvest slack power from underutilized components and reassign it to the currently performancecritical components. This approach establishes a new baseline for server-level power management, upon which more sophisticated or workload-specific strategies can be built.

