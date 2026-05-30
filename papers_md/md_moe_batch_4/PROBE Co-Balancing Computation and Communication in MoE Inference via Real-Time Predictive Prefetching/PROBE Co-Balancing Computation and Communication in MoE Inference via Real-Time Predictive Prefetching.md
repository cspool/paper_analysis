# PROBE: CO-BALANCING COMPUTATION AND COMMUNICATION IN MOE INFERENCE VIA REAL-TIME PREDICTIVE PREFETCHING

Qianchao Zhu <sup>1</sup> Xucheng Ye <sup>1</sup> Yuliang Liu <sup>1</sup> Haodong Ouyang <sup>1</sup> Chengru Song <sup>1</sup>

## ABSTRACT

Mixture-of-Experts models have become a dominant architecture for scaling Large Language Models by activating only a sparse subset of experts per token. However, latency-critical MoE inference faces a fundamental tension: while expert parallelism improves memory efficiency, it also amplifies execution stragglers. In real-world serving, continuous batching and diverse concurrent requests induce rapid semantic shifts, causing expert hotspots to migrate abruptly across GPUs and triggering the "double penalty" of coupled computational skew and network congestion.

We propose PROBE, an inference system that co-balances computation and communication in real time. PROBE introduces Continuous Lookahead Pipelining, which proactively predicts, plans, and prefetches for upcoming layers while keeping all control overheads off the critical path. PROBE consists of: (1) a *Gate-Initialized Lookahead Predictor* that distills the target router to forecast next-layer expert activation with high fidelity; (2) a *Hardware-Aware Balance Planning* solver that jointly optimizes dynamic expert replication and token assignment under strict hiding-window constraints; and (3) a *Phase-Locked Co-Scheduling* policy that uses split-phase transmission to hide bandwidth-intensive expert transfers behind computation without contending with All-to-All collectives. Experiments show that PROBE reduces prefill latency by up to 1.32× and improves decoding throughput by up to 1.26× over state-of-the-art baselines, especially under extreme workload volatility.

## 1 INTRODUCTION

The insatiable demand for model intelligence has driven Large Language Models (LLMs) [\(Vaswani et al.,](#page-12-0) [2017;](#page-12-0) [Co](#page-11-0)[manici et al.,](#page-11-0) [2025;](#page-11-0) [Achiam et al.,](#page-11-0) [2023;](#page-11-0) [Liu et al.,](#page-12-0) [2025\)](#page-12-0) toward trillion-parameter scales [\(Kaplan et al.,](#page-11-0) [2020\)](#page-11-0). To sustain this scaling without incurring prohibitive computational costs, the Mixture-of-Experts (MoE) architecture has emerged as the de facto standard [\(Fedus et al.,](#page-11-0) [2022;](#page-11-0) [Jiang et al.,](#page-11-0) [2024;](#page-11-0) [Du et al.,](#page-11-0) [2022\)](#page-11-0). By decoupling parameter count from active computation—activating only a sparse subset of experts per token—MoE enables models like GPT-OSS [\(Agarwal et al.,](#page-11-0) [2025\)](#page-11-0), DeepSeek-V3 [\(Liu et al.,](#page-12-0) [2024\)](#page-12-0), Qwen3-MoE [\(Yang et al.,](#page-12-0) [2025\)](#page-12-0) to achieve massive capacity with manageable FLOPs.

While expert load balancing has been extensively explored in the training regime—often relying on auxiliary losses [\(Lepikhin et al.,](#page-12-0) [2021;](#page-12-0) [Fedus et al.,](#page-11-0) [2022\)](#page-11-0) or capacity constraints [\(He et al.,](#page-11-0) [2022;](#page-11-0) [Zhai et al.,](#page-12-0) [2023\)](#page-12-0), recent stateof-the-art models have increasingly shifted toward finergrained sparsity [\(Team et al.,](#page-12-0) [2025\)](#page-12-0) and deep expert specialization [\(Liu et al.,](#page-12-0) [2025;](#page-12-0) [Yang et al.,](#page-12-0) [2025;](#page-12-0) [Wang et al.,](#page-12-0) [2024\)](#page-12-0) to enhance model capabilities. This paradigm shift

relaxes balancing constraints and significantly intensifies workload skewness. Consequently, serving these models for latency-critical inference creates a fundamental tension between maintaining memory efficiency, mitigating severe stragglers, and handling dynamic imbalance.

As illustrated in [Figure 1,](#page-1-0) while Expert Parallelism (EP) enables massive models to fit within GPU memory, the resulting workload imbalance becomes a critical bottleneck for inference efficiency. This performance degradation is driven by a complex interplay of spatial and temporal dimensions. Spatially, unlike the uniform workload of dense models, MoE routing and execution is dictated by input semantics; popular experts create "hotspots" that inflict a *double penalty*, where the overloaded rank is simultaneously throttled by computational skew and network congestion during All-to-All collectives. Temporally, this instability is exacerbated by the stochastic nature of continuous batching [\(Kwon et al.,](#page-11-0) [2023\)](#page-11-0), where the global batch composition churns rapidly as requests join and depart at arbitrary intervals. Consequently, expert hotspots migrate abruptly, especially during the prefill phase with limited steps, rendering static expert placement obsolete.

To mitigate the straggler effect, existing solutions [\(Li et al.,](#page-12-0) [2025b;](#page-12-0) [Han et al.,](#page-11-0) [2025;](#page-11-0) [Yun et al.,](#page-12-0) [2024;](#page-12-0) [He et al.,](#page-11-0) [2022;](#page-11-0) [Dai](#page-11-0) [et al.,](#page-11-0) [2024;](#page-11-0) [Doucet et al.,](#page-11-0) [2025;](#page-11-0) [Zeng et al.,](#page-12-0) [2025\)](#page-12-0) largely

<sup>1</sup>Kling Infra, Kuaishou Technology.

<span id="page-1-0"></span>![](_page_1_Figure_1.jpeg)

Figure 1. Spatial and Temporal Challenges in MoE Inference. Under Expert Parallelism, efficiency is constrained by: (1) Spatial imbalance, where skewed token-to-expert routing creates computational stragglers and communication bottlenecks; and (2) Temporal volatility, where expert hotspots shift rapidly over time under continuous batching. This motivates a system that can handle both instantaneous skew and continuous distribution shifts.

converge on selective expert replication, trading memory for improved load balance. However, this paradigm fails to adequately address the unique challenges of latency-critical inference. First, strategies that permanently replicate popular experts inflate memory usage, competing with the KV cache for limited HBM capacity. Second, expert hotspots exhibit high-frequency shifts, a volatility that is particularly acute during the condensed prefill phase. Consequently, reactive approaches relying on historical statistics inherently lag behind abrupt variations, producing obsolete placement decisions. Furthermore, given the strict requirement to avoid stalling the critical path, any exposed overhead (e.g., offloading-based approaches (Hu et al., 2026; Yu et al., 2026), host-based balancing solvers, or blocking expert transfers) can neutralize the balancing gains. Finally, techniques relying on pre-gated routing (Hwang et al., 2024) require training-time adaptation, violating strict correctness requirements.

In this paper, we propose **PROBE**, an inference system designed to co-balance computation and communication in real-time. PROBE fundamentally shifts the paradigm from *reactive adjustment* to *proactive preparation*. While token arrival is stochastic, the semantic routing of deep models is predictable. PROBE implements a **Continuous Lookahead Pipelining** mechanism. Instead of blocking the critical path, PROBE overlaps the *Predict*, *Plan*, and *Prefetch* phases for the subsequent layer with the main stream, effectively hiding these control overheads.

Specifically, we make the following contributions:

- Gate-Initialized Lookahead Predictor: We introduce a lightweight predictor that distills the routing logic of the target layer. By freezing the target layer's router as a prior and leveraging the previous layer's hidden states as input, it forecasts next-layer expert hotspots with  $\approx 90\%$  accuracy while incurring negligible overhead.
- Hardware-Aware Balance Planning: We formulate straggler mitigation as a resource assignment problem. Unlike solvers that ignore transfer costs, our planner strictly bounds expert replication decisions within the device-specific "hiding window", dynamically replicates experts onto underutilized ranks, and ensures these overheads do not stall the pipeline.
- Phase-Locked Co-Scheduling: PROBE operates on a dual-track architecture. Within this framework, we design a split-phase transmission scheduling that orchestrates prediction, planning, and prefetching to execute orthogonally to the main stream. This ensures that bandwidth-heavy prefetching never contends with All-to-All collectives, guaranteeing zero contention on hardware resources.
- Evaluation: Experiments show that PROBE effectively neutralizes stragglers, reducing prefill latency by up to  $1.32 \times$  and improving decoding throughput by up to  $1.26 \times$  over state-of-the-art baselines under extreme workload volatility.

## 2 BACKGROUND

The MoE architecture has emerged as a dominant paradigm for scaling LLMs by decoupling parameter size from computational cost (Jacobs et al., 1991; Shazeer et al., 2017; Fedus et al., 2022). While historical approaches addressed expert load balancing during training via auxiliary losses (Lepikhin et al., 2021; Xue et al., 2024), recent state-of-the-art models increasingly prioritize expert specialization and adopt finer-grained expert granularity with higher sparsity to maximize performance (Liu et al., 2025; Guo et al., 2025a; Qiu et al., 2025; Team et al., 2025), often relaxing balancing constraints. This shift significantly intensifies workload skewness during inference: under expert parallelism, the synchronous execution of MoE layers transforms skew into a severe straggler effect, bounding layer latency by the most heavily loaded device. In this section, we (§ 2.1) characterize the manifestations of load imbalance in real-time inference, (§ 2.2) analyze the trade-offs between parallelism strategies, and (§ 2.3) identify the system constraints imposed by low-latency serving.

## 2.1 Characterizing Expert Load Imbalance

To systematically analyze the straggler effect in EP, we quantify load skewness using the *Imbalance Ratio* ( $\mathcal{IR}$ ). Defined at the rank granularity,  $\mathcal{IR}$  measures the disparity

<span id="page-2-0"></span>![](_page_2_Figure_1.jpeg)

Figure 2. Expert activation patterns across prefill and decoding. Measurements use ep=8 with a standard sharded expert placement policy. Subfigures (a) and (b) show the concentrated, bursty skew during prefill ( $\approx$ 32K tokens). Subfigures (c) and (d) show rapid load shifts during decoding ( $\approx$ 8K tokens), where expert popularity changes with semantic transitions. Comparing GPT-OSS-120B (Top-4 out of 128 experts) and Qwen3-235B (Top-8 out of 128 experts) illustrates that model sparsity patterns further modulate imbalance severity.

between the maximum and average workload across an EP group of size ep. For a rank r hosting experts  $\mathcal{E}_r$ , the local load is  $\mathcal{L}_r = \sum_{e \in \mathcal{E}_r} n_e$ , yielding:

$$\mathcal{IR} = \frac{\max_{r \in \{0,\dots,ep-1\}} \mathcal{L}_r}{\frac{1}{ep} \sum_{r=0}^{ep-1} \mathcal{L}_r}$$
(1)

An  $\mathcal{IR}$  of 1.0 represents ideal balance. However, as ep scales MoE models with higher sparsity, the probability of "hot" experts colliding on a single device increases. Consequently, a substantially elevated  $\mathcal{IR}$  implies that the cluster's aggregate throughput is throttled by the straggler device, forcing underutilized resources to idle at synchronization barriers.

**Prefill: Burstiness from Semantic Clustering.** During prefill, the parallel processing of massive prompt sequences triggers severe workload concentration. Unlike the uniform distribution assumed by statistical multiplexing, the semantic locality inherent in input contexts causes specific experts to be disproportionately activated. As shown in Figure 2(a-b), injecting new datasets transforms prompt semantics into instantaneous traffic bursts, manifesting as frequent spikes in the  $\mathcal{IR}$  above **2.6** even with large batches ( $\approx$ 32K tokens). Consequently, the overloaded rank dictates global tail latency, significantly inflating Time-To-First-Token (TTFT).

**Decoding: Volatility under Continuous Batching.** In contrast to the concentrated skewness observed in prefill, the decoding phase is characterized by rapid load volatility. While token aggregation from diverse semantics results in a lower peak  $\mathcal{IR}$ , the workload distribution is destabilized by the mechanics of continuous batching (Kwon et al., 2023). The constant churn of arriving and departing requests, coupled with the semantic evolution of generated sequences, results in an unstable expert distribution. As shown in Figure 2(c-d), expert popularity shifts unpredictably during workload transitions, causing  $\mathcal{IR}$  to fluctuate between 1.43 and 2.28. Crucially, this volatility creates a significant bottleneck, forcing approximately 50% of global compute capacity to idle at synchronization barriers.

### 2.2 The Dilemma of MoE Parallelism

To scale the inference of massive MoE models, existing frameworks have largely converged on a hybrid paradigm: applying Data Parallelism (DP) to attention modules and EP to MoE modules (Zheng et al., 2024; Kwon et al., 2023; Dai et al., 2024; NVIDIA, 2023). While EP is essential for managing the massive parameters of modern MoE models, it introduces a fundamental tension between memory efficiency and straggler effect. EP maximizes batch size per expert but suffers from the straggler effect. Conversely, DP eliminates imbalance but necessitates full weight replication, which is often infeasible. We deconstruct these trade-offs across three critical dimensions:

Memory Efficiency: Capacity vs. Bandwidth. As MoE models scale toward trillions of parameters, the full weight replication required by DP becomes prohibitively expensive, making EP a necessity for memory capacity. Beyond mere storage, EP optimizes memory bandwidth through global token consolidation. By aggregating tokens across all ranks, EP ensures large effective batch sizes for each expert. In contrast, DP fragments the global batch across independent replicas. This fragmentation forces the loading of full expert weights for a small number of local tokens, drastically inflating redundant memory accesses and computation. Consequently, DP degrades arithmetic intensity, pushing the computation into a memory-bound bottleneck, particularly for "cold" experts with negligible utilization.

## Computational Intensity: Straggler & Fragmentation.

MoE efficiency hinges on the arithmetic intensity of Grouped GEMMs. EP maximizes this by aggregating tokens to saturate Tensor Cores, yet incurs severe load imbalance. As quantified in Figure 3, the significant gap between maximum and average rank latency confirms that system throughput is bound by the single slowest straggler. In contrast, DP eliminates imbalance but suffers a severe **fragmentation penalty**. Processing fragmented local batches dilutes arith-

<span id="page-3-0"></span>![](_page_3_Figure_1.jpeg)

Figure 3. **MoE compute latency.** Profiling via SGLang on GPT-OSS-120B (128 experts, Top-4). We compare EP (Max/Avg/Min) with DP and EP + 4 extra experts. DP is bottlenecked by fragmentation (low arithmetic intensity and padding), while modest EP redundancy mitigates stragglers with minimal memory overhead.

metic intensity, pushing computation into a memory-bound regime. Additionally, rigid kernel tiling necessitates extensive padding for irregular token counts; the resulting waste in FLOPs becomes increasingly severe with fine-grained sparsity (Guo et al., 2025b). Crucially, the "EP + Extra Experts" profile reveals that the bottleneck is skewness, not aggregate workload. Selectively replicating experts reduces the tail latency with minimal memory overhead, effectively balancing load without incurring the performance penalties of full DP.

Communication: Coupled Skew and the Double Penalty. EP relies on bandwidth-bound All-to-All collectives, where workload skew creates a **double penalty**: computational hotspots are inevitably coupled with network congestion. Even with topology-aware and token deduplication optimizations (Zhao et al., 2025), ranks hosting popular experts attract excessive unique tokens. Figure 4 confirms that this skew drastically inflates the maximum receive volume on specific ranks. Since collective operations are synchronized by the slowest device, this local congestion collapses effective cluster-wide bandwidth. Consequently, the overloaded rank suffers as a compounded bottleneck, sequentially throttled by high network ingress, computation, and egress.

#### 2.3 Challenges in Real-Time MoE Inference

To navigate the complex trade-offs between memory, computation, and communication, recent research has explored various hybrid strategies. Approaches like Grace-MoE (Han et al., 2025) and Libra employ expert replication to trade memory for better load balance, while systems such as FasterMoE (He et al., 2022) and FlexMoE (Yun et al., 2024) leverage dynamic offloading to smooth out peaks. However, transposing the paradigm that predominantly designed for throughput-oriented settings to latency-critical inference is challenging. Unlike training, where complex solver costs

![](_page_3_Figure_7.jpeg)

Figure 4. Skew hurts All-to-All efficiency. Benchmarked on  $8 \times H800$ , GPT-OSS-120B with DeepEP (Zhao et al., 2025). Top: effective All-to-All Dispatch bandwidth. Bottom: max per-rank traffic volume. Compared to a manually balanced top-K baseline, real workloads create receiver hotspots and reduce effective bandwidth; Combine phase shows similar behavior.

can be amortized over long backward passes, online inference operates under strict Service Level Objectives. In this regime, even modest scheduling overheads or reactive data movement can nullify the performance gains. Consequently, adapting dynamic balancing to online inference requires overcoming three distinct hurdles:

Expert Hotspot Shifts in Inference The first challenge stems from the inherently dynamic routing behavior (Zhang et al., 2025). Unlike the static batches in training, inference engines handle requests that join and depart at arbitrary intervals. This stochasticity introduces high variance in token distribution per step, causing rapid shifts in expert popularity that render historical heuristic methods ineffective. Therefore, the system requires a high-fidelity predictor capable of anticipating router decisions ahead of time. Crucially, this prediction must feed into a lightweight but effective solver that constructs a distribution strategy in real time, accounting for the constantly fluctuating batch composition rather than relying on stale statistics.

Enforcing Zero-Overhead Balancing. The stringent Time-Per-Output-Token (TPOT) constraints mandate that auxiliary load-balancing operations remain strictly hidden behind the critical path (Zhao et al., 2024; Chen et al., 2024). This creates a strong dependence on hardware characteristics. On configurations with high compute capability but limited interconnect bandwidth, fast compute kernels (e.g., Attention or Grouped GEMM) shrink the overlap window available for expert transfers, forcing the system to cap the transfer volume. Conversely, limited bandwidth can prolong the All-to-All communication phase, inadvertently widen-

<span id="page-4-0"></span>ing the window available for solver execution and tolerating more planning steps. This imposes a rigid constraint: the load balancer must be hardware-aware, dynamically trading off solver complexity against transfer volume to maximize balance gains within device-specific execution budgets without stalling the critical pipeline.

Mandating CUDA Graph Compatibility. Modern serving engines often rely on CUDA Graph (NVIDIA Corporation, 2024a) to accelerate the decoding phase (Zheng et al., 2024; Kwon et al., 2023). Without CUDA Graph, CPU-side dispatch overhead for long sequences of kernels prevents back-to-back execution, leaving large idle bubbles on the device timeline. However, dynamic load balancing creates a fundamental conflict: variable control flow in solvers and dynamic P2P expert transfers can preclude static graph capture, forcing a regression to eager execution. To preserve graphbased speedups, the balancing mechanism must eliminate host-device synchronization (e.g., host-based ILP solvers). This motivates implementing solvers natively on the GPU, leveraging kernel fusion, and developing custom communication kernels to absorb dynamic logic while adhering to CUDA Graph requirements.

### 3 PERFORMANCE MODELING

Building on the system challenges identified, we formulate an analytical performance model for the hybrid parallel setting (DP for Attention, EP for MoE). The model quantifies end-to-end layer latency as a function of token distribution, capturing the interaction among three components: (1) **computation latency**, inflated by load skew and Grouped GEMM fragmentation; (2) **communication latency**, which degrades under traffic congestion during both dispatch and combine; and (3) **prefetch overhead**, bounded by the available hardware overlap window.

## 3.1 System Setup and Notation

Table 1 summarizes the notation. The router produces a global token count  $n_e$  for each expert e. We denote by  $n_{e,r_t}^{r_s}$  the number of tokens from source rank  $r_s$  routed to expert e on rank  $r_t$  (so  $\sum_{r_s,r_t} n_{e,r_t}^{r_s} = n_e$ ). When the source rank is irrelevant, we write  $n_{e,r_t} = \sum_{r_s} n_{e,r_t}^{r_s}$ .

In a standard EP system with sharded placement (no replication), each expert is hosted on a unique rank, so all tokens of expert e are processed on its home rank. In the presence of expert replication, the workload  $n_e$  can be partitioned across multiple ranks that host a copy of expert e; we denote by  $n_{e,r}$  the number of tokens of expert e assigned to (and processed on) rank r, subject to the conservation constraint  $\sum_r n_{e,r} = n_e$ .

| Symbol                    | Definition                                                          |
|---------------------------|---------------------------------------------------------------------|
| System & Model Parameters |                                                                     |
| ep                        | Expert Parallelism size (number of ranks)                           |
| B                         | Global batch size (tokens per step)                                 |
| H                         | Hidden dimension size                                               |
| ${\mathcal W}$            | Parameter size per expert                                           |
| $\bar{F}$                 | Per-token FLOPs per expert                                          |
| $\eta_g(\cdot)$           | GEMM efficiency function (w.r.t. tokens/expert)                     |
| Workload & Distribution   |                                                                     |
| $\mathcal{E}_r$           | Set of experts <i>physically hosted</i> on rank r                   |
| $\Delta_r$                | Set of <i>redundant</i> experts replicated on rank $r$              |
| $n_e$                     | Global tokens routed to expert $e\left(\sum n_e = B \cdot k\right)$ |
| $n_{e,r_t}^{r_s}$         | Tokens on source $r_s$ routed to $e$ hosted on $r_t$                |
| $\lambda_r^{in/out}$      | Token ingress/egress deduplication ratio on rank $r$                |
| $\mathcal{V}_r^{in/out}$  | Network traffic volume for rank $r$                                 |
| $\mathcal{IR}$            | Imbalance Ratio ( $\max_r \text{Load}_r/\text{Avg Load}$ )          |

Table 1. Key notation for the MoE performance model.

## 3.2 Computation: Skew and Fragmentation

**Rank-Level Latency.** After dispatch, rank r executes its assigned experts using Grouped GEMM. The effective compute time depends not only on FLOPs but also on kernel efficiency  $\eta_g(\cdot)$ , which degrades for small token counts due to padding and reduced arithmetic intensity. The processing time for each expert e on rank r is modeled as:

$$T_{e,r}(n_{e,r}) = \frac{n_{e,r} \cdot \bar{F}}{\eta_g(n_{e,r}) \cdot F_{peak}},\tag{2}$$

where F denotes the per-token FLOPs. The total compute latency for rank r is the summation over all locally hosted experts, comprising both native and replicated:  $T^r_{comp} = \sum_{e \in \mathcal{E}_r \cup \Delta_r} T_{e,r}$ .

**Straggler Effect.** Since EP inference is synchronous, the layer latency is dictated by the slowest rank. We relate the tail latency to the cluster average via the  $\mathcal{IR}$ :

$$T_{comp} = \max_{r} T_{comp}^{r} \approx \mathcal{IR} \cdot \left(\frac{1}{ep} \sum_{r} T_{comp}^{r}\right).$$
 (3)

This formulation highlights a compounding degradation: high skew ( $\mathcal{IR} \gg 1$ ) creates a straggler rank with large  $T^r_{comp}$ , while fragmentation of splitting  $n_e$  across ranks reduces  $n_{e,r}$ , pushing computations into the low-efficiency regime of  $\eta_q(\cdot)$ .

## 3.3 Communication: The Double Penalty

**Traffic Volume and Send/Recv Congestion.** The All-to-All dispatch and combine phases are bandwidth-intensive. Their latency is dictated by the bottleneck rank handling the maximum data volume (either send or receive). Crucially, token deduplication dynamics differ for ingress and egress traffic. Let  $\lambda_r^{in}$  denote the deduplication factor for traffic

received by rank r (i.e., how many experts on r are hit by a unique remote token), and  $\lambda_r^{out}$  the corresponding factor for traffic sent out of rank r. The ingress  $(\mathcal{V}_r^{in})$  and egress  $(\mathcal{V}_r^{out})$  volumes are formulated as:

$$\mathcal{V}_r^{in} = \frac{H}{\lambda_r^{in}} \sum_{r' \neq r} \sum_{e \in \mathcal{E}_r \cup \Delta_r} n_{e,r}^{r'}, \quad \mathcal{V}_r^{out} = \frac{H}{\lambda_r^{out}} \sum_{e \notin \mathcal{E}_r \cup \Delta_r} n_e^r.$$
(4)

The critical communication volume for rank r is determined by the maximum congestion across both directions:  $V_r = \max(V_r^{in}, V_r^{out})$ .

Coupled Latency and The Double Penalty. Unlike the uniform traffic in DP, EP exhibits a structural correlation between traffic and computation, imposing a double penalty on the straggler rank. Specifically, the rank  $r^*$  hosting global hotspots (maximizing  $T^r_{comp}$ ) inherently attracts the highest volume of unique tokens during the dispatch phase (i.e.,  $\mathcal{V}^{in}_{r^*} \approx \max_r \mathcal{V}^{in}_{r}$ ). Symmetrically, during the Combine phase, this same rank must redistribute the largest volume of results, creating an egress bottleneck. Consequently, the end-to-end MoE latency is dictated by this single overloaded device, sequentially throttled by network ingress, computation, and network egress:

$$T_{MoE} \approx \underbrace{\max_{r} T_{comp}^{r}}_{\text{Compute Skew}} + \underbrace{2 \cdot \max_{r} \left(\frac{\mathcal{V}_{r}}{BW_{net}}\right)}_{\text{Network Skew}}.$$
 (5)

This formulation highlights that the load imbalance  $\mathcal{IR}$  proxies the total system slowdown, magnifying the latency penalty beyond computational delays.

## 3.4 Constrained Expert Prefetching

**Expert Transfer Latency.** Dynamic redundancy introduces overheads for moving expert weights. For a rank r prefetching a set of experts  $\Delta_r^{in}$  and evicting  $\Delta_r^{out}$ , the transfer latency is dictated by the maximum of read/write volumes:

$$T_{trans}^{r} = \frac{\max(|\Delta_{r}^{in}|, |\Delta_{r}^{out}|) \cdot \mathcal{W}}{BW_{net}}.$$
 (6)

**Hiding Window.** To ensure non-blocking execution on the critical path, expert transfers must be confined within the computation window  $T^r_{window}$  of noncommunication kernels. The exposed overhead is modeled as  $\max{(0, \max_r T^r_{trans} - T^r_{window})}$ . This imposes a hardware-aware constraint: to maintain zero overhead, the system must bound the replica volume  $|\Delta_r|$  according to the device's compute-to-bandwidth ratio.

## 4 SYSTEM DESIGN

Building on the performance modeling in §3, we present PROBE, a runtime scheduling system designed to neutral-

![](_page_5_Picture_13.jpeg)

Figure 5. Overview of PROBE. The system implements a dual-track execution that overlaps control-plane operations with the main stream on complementary resources: predictor/planner run during network-bound All-to-All, while P2P prefetching is overlapped with compute-bound GEMM and attention to hidden management overhead.

ize the straggler effect in latency-critical MoE inference. PROBE introduces a **Continuous Lookahead Pipelining** paradigm: rather than blocking the critical path to deliberate on load balancing, PROBE exploits the execution time of the current layer to predict, plan, and prefetch resources for the next. By decoupling control plane decisions from the main execution flow, PROBE achieves just-in-time expert reconfiguration with nearly zero-overhead.

## 4.1 Architecture Overview

As illustrated in Figure 5, PROBE establishes a dual-track execution model to isolate management overheads from the critical path:

- The Deterministic Track (Main Stream): Executes the standard sequence of MoE operators: Attention, All-to-All Dispatch, MoE Computation, and All-to-All Combine. While this track remains semantically strictly equivalent to standard execution, its alternating resource demands (compute-bound vs. bandwidth-bound) expose valuable "execution slacks" for the auxiliary track.
- The Auxiliary Track (Control Plane): Operates asynchronously alongside the main stream via three pipelined stages: (1) Lookahead Prediction forecasts expert activation for the upcoming layer; (2) Balance-Optimal Planning derives the optimal replication strategy; and (3) Preemptive Expert Transfer materializes the plan via P2P communication.

We denote by P' the baseline expert placement (e.g., the static EP sharding placement), and by P the updated place-

ment after applying the current planning decision. For expert transfers,  $\Delta_r^{in}$  and  $\Delta_r^{out}$  represent the sets of experts to be prefetched into and evicted from rank r, respectively.

#### 4.2 Predictive Lookahead Mechanism

To enable proactive balancing, PROBE must break the synchronous dependency of standard MoE architectures, where expert activation is known only after the gating network executes. We therefore introduce *Lookahead Gating* to anticipate the next-layer expert distribution one layer ahead.

**Gate-Initialized Lookahead Predictor.** Leveraging the empirical continuity of hidden states across adjacent transformer layers, we construct a lookahead predictor by reusing the target layer's pre-trained router as a strong prior. Concretely, for each MoE layer L, we clone its router parameters  $(\mathbf{W}_L, \mathbf{b}_L)$  and freeze them as a base. To compensate for cross-layer feature drift while avoiding cold-start instability, we attach a lightweight, trainable residual MLP:

$$\hat{\mathbf{l}}_{L} = \underbrace{\mathbf{W}_{L}\mathbf{h}_{L-1} + \mathbf{b}_{L}}_{\text{Frozen Prior}} + \underbrace{\hat{\mathbf{W}}_{L}^{2} \sigma(\hat{\mathbf{W}}_{L}^{1}\mathbf{h}_{L-1})}_{\text{Trainable Residual}}, \tag{7}$$

where  $\mathbf{h}_{L-1}$  is the hidden state from the previous layer and  $\sigma(\cdot)$  is the SiLU activation. We zero-initialize the residual component to match the cloned router initially, ensuring stable starting performance while allowing gradual, data-driven corrections. Crucially, local predictions derived from  $\hat{\mathbf{l}}_L$  are aggregated via a lightweight All-Gather to enable global prefetch planning; the actual token dispatch strictly follows the ground-truth router outputs during execution, preserving semantic equivalence.

Scale-Driven Online Distillation. To ensure robustness against dynamic semantic shifts in real-world traffic, we adopt a scale-driven distillation strategy inspired by Eagle-3 (Li et al., 2025a). We treat the continuous stream of inference requests spanning diverse domains as a mixed dataset. By minimizing the Cross-Entropy loss between the predictor's output and the ground-truth router's probability distribution, we force the lightweight MLP to align its trajectory with the actual gating logic. This massive exposure to online data enables the compact predictor to generalize to complex routing patterns, achieving ≈90% Top-K accuracy while incurring negligible overhead.

## 4.3 Balance-Optimal Planning

Given the predicted per-expert workload  $\hat{\mathbf{n}}$ , the solver jointly optimizes expert placement  $\mathbf{P}$  and per-expert token assignment  $\mathbf{A}$  after planning. This formulation minimizes critical path latency while ensuring that the cost of dynamic expert replication is strictly bounded by the computation phase of the concurrent pipeline.

**Optimization Formulation.** We formulate straggler mitigation as a resource allocation problem subject to routing conservation and latency hiding constraints. The objective is to minimize the bottleneck rank's latency:

$$\begin{aligned} & \underset{\mathbf{P}, \mathbf{A}}{\min} & & \max_{r} \left( T_{comp}^{r}(\mathbf{A}) + T_{comm}^{r}(\mathbf{A}) \right) \\ & \text{s.t.} & & n_{e,r} > 0 \implies \mathbf{P}_{r,e} = 1, \quad \sum_{r} n_{e,r} = n_{e}, \ \forall e, r \\ & \underbrace{T_{trans}^{r}(\mathbf{P})}_{\text{Prefetch Latency}} \leq \underbrace{T_{window}^{r}}_{\text{Hiding Window}}, \quad \forall r \end{aligned}$$

$$(8)$$

 $T^r_{window}$  here denotes the rank-local overlap window available for expert transfers, i.e., the executation of attention or Grouped GEMM. The first constraint ensures routing validity. The second constraint enforces a zero-penalty, bounding the prefetching latency within the available computation window. This guarantees that transfers are fully overlapped, preventing bandwidth contention with critical communications.

**Greedy Rebalancing Strategy.** Since finding the global optimum for joint placement and routing is computationally expensive, we employ an iterative heuristic detailed in Algorithm 1. The process repeatedly identifies the global bottleneck rank  $r_{src}$  and pairs it with the least loaded rank  $r_{dst}$  to offload the hottest expert  $e^*$ . Crucially, every replication move is gated by a dual-side budget check (Line 9) to ensure the transfer latency strictly fits within the hiding window. Upon validation, we apply a locality-aware water-filling rebalance strategy to determine the optimal token distribution. Adhering to a locality-first principle, tokens generated on  $r_{src}$  remain pinned to the local replica to eliminate network overhead. In contrast, remote traffic comprising requests for  $e^*$  originating from ranks without local replicas is dynamically partitioned among all available replicas. Rather than enforcing strict peer equality, this redistribution follows a water-filling logic at rank-level granularity, greedily redirecting remote tokens to  $r_{dst}$  until the load on  $r_{src}$  aligns with the cluster-wide average or the transferable pool is exhausted. The loop persists until convergence or the iteration budget is consumed, ensuring the planning phase completes within the strict lookahead timeframe to yield the optimized expert placement P and routing assignment A.

## 4.4 Phase-Locked Co-Scheduling

To materialize the plan without stalling the critical path, PROBE implements a Phase-Locked Co-Scheduling policy. This mechanism maps each stage of the auxiliary track to a complementary, orthogonal phase in the main track, ensuring zero contention on hardware resources.

#### <span id="page-7-0"></span>Algorithm 1 Greedy Balance-Optimal Planning

```
Require: Predicted Workload \hat{\bf n}, Baseline Placement {\bf P}'
Ensure: Final Placement P, Routing Assignment A
 1: Initialize sets \Delta_r^{in}, \Delta_r^{out} \leftarrow \emptyset for all r; k \leftarrow 0
 2: Initialize A using \hat{\mathbf{n}} and \mathbf{P}' (Locality-First)
 3: \mathbf{L} \leftarrow \text{ComputeLatencies}(\mathbf{A})
 4: loop
 5:
            r_{src} \leftarrow \arg\max \mathbf{L}

    ▷ Identify bottleneck rank

            r_{dst} \leftarrow \arg\min \mathbf{L}
 6:

    ▶ Identify helper rank

 7:
            e^* \leftarrow \text{SelectHeavyExpert}(r_{src}, \hat{\mathbf{n}})
            if not CheckDualBudget(r_{src}, r_{dst}, e^*, \Delta^{in/out}, \mathbf{L}) then
 8:
 9.
                 Mark pair (r_{src}, r_{dst}) invalid; continue
10:
            end if
            (\mathbf{A}', \text{gain}) \leftarrow \text{WaterFillingRebalance}(e^*, r_{src}, r_{dst}, \mathbf{A})
11:
12:
            if gain \leq \epsilon or k \geq k_{max} then

13:
14:
            end if
            \Delta_{r_{src}}^{out} \leftarrow \Delta_{r_{src}}^{out} \cup \{e^*\}; \quad \Delta_{r_{dst}}^{in} \leftarrow \Delta_{r_{dst}}^{in} \cup \{e^*\}
15:
            A \leftarrow A'; L \leftarrow ComputeLatencies(A)
16:
17:
18: end loop
19: \mathbf{P} \leftarrow \text{UpdatePlacement}(\Delta^{in/out}, \mathbf{P}')
20: return P, A
```

## Orthogonal Pipelining with Split-Phase Transmission.

To neutralize overheads, PROBE employs a resource-aware scheduling policy that maps auxiliary-track tasks to complementary main-track phases. On the compute side, the lightweight MLP-based predictor and the single-SM optimized planning solver initiate concurrently with the bandwidth-bound All-to-All dispatch. While the predictor is hidden by the dispatch latency, the solver's minimal footprint permits non-intrusive overlap with the subsequent MoE computation. On the network side, the bandwidth-intensive expert transfer is hidden behind compute-heavy windows via a split-phase transmission mechanism. To prevent contention, transfer initiates during layer L's MoE computation but is preemptively suspended to yield bandwidth to the critical All-to-All Combine phase; it resumes only after the combine completes, finalizing during layer L+1's attention. This orchestration ensures that management overheads are strictly masked by orthogonal hardware resources.

### 5 IMPLEMENTATION

We implement PROBE atop the SGLang (Zheng et al., 2024) framework, integrating DeepEP (Zhao et al., 2025) (normal mode) as the communication backend while maintaining CUDA Graph capture compatibility. We leverage symmetric memory provided by NVSHMEM (NVIDIA Corporation, 2024b) to manage a dedicated replicated-expert buffer region. For prediction, we implement a lightweight global All-Gather with NVSHMEM primitives to synchronize perrank estimates. For planning, the solver is realized as a single-SM CUDA kernel that performs serial iterative updates, with a hard cap of  $k_{max}=16$  iterations to bound

overhead. For prefetching, we use a custom Triton (Tillet et al., 2019) kernel to issue remote put operations with controlled SM occupancy. To support at most three redundant experts per rank, we adopt double buffering for the replica region, limiting memory overhead to six expert slots per device and enabling asynchronous writes of next-layer weights while the current layer computes.

### **6** EXPERIMENTS

We conduct a comprehensive evaluation to demonstrate the efficiency of PROBE. Specifically, our experiments are designed to answer the following key questions:

<u>End-to-End Performance</u>: How much acceleration does PROBE achieve in both prefill and decoding phases compared to state-of-the-art baselines? (§6.2)

**Robustness:** Can PROBE maintain stability under dynamic workloads with abrupt semantic shifts? (§6.3)

Predictor Fidelity: Does the lookahead predictor capture expert activation patterns with sufficient accuracy? (§6.4)

<u>Latency Breakdown:</u> How effectively does our dual-track pipeline overlap communication with computation to hide system overheads at the micro-operation level? (§6.5)

## 6.1 Experimental Setup

Environments. We evaluate PROBE on an 8×NVIDIA Hopper-141GB node interconnected via 900 GB/s NVSwitch. The software stack includes PyTorch 2.9, CUDA 12.9, NCCL 2.27.3 (NVIDIA, 2022) and NVSHMEM 3.3.20 (NVIDIA Corporation, 2024b).

**Models and Datasets.** We benchmark on two models representing distinct sparsity configurations: **Qwen3-MoE-235B** (128 experts, Top-8, 93 layers, BF16) and the more sparse **GPT-OSS-120B** (128 experts, Top-4, 36 layers, BF16). For evaluation, we construct three datasets: *Chinese* and *Code*, aggregated from multiple open-source corpora (e.g., Alpaca-zh (Peng et al., 2023), CodeAlpaca-20k (Chaudhary, 2023), OpenAI-humaneval (Chen et al., 2021)), and a synthetic *Repeat* dataset. The latter is constructed by duplicating a narrow set of prompts to simulate extreme expert skew within the *ep*=8 environment.

**Baselines.** We compare PROBE against two representative systems: **SGLang** (Zheng et al., 2024), the standard EP baseline employing static sharded placement; and **DeepSeek-EPLB** (Zhao et al., 2025), a statistic-based load balancing strategy. For EPLB, we configure 2 redundant expert slots per layer per rank, constraining the global rebalancing transfer to complete within 2 decoding steps.

<span id="page-8-0"></span>![](_page_8_Figure_1.jpeg)

Figure 6. Prefill latency scaling. Evaluated on an node with ep=8. We use chunked prefill of 8K (GPT-OSS) / 16K (Qwen3-MoE) tokens per rank; the x-axis shows total input tokens across ranks. We omit DeepSeek-EPLB because extra replicas can trigger OOM under prefill memory pressure, and reactive transfers are too costly given the limited prefill steps.

#### 6.2 End-to-End Performance

Prefill Latency (TTFT). We first evaluate the performance during the compute-intensive prefill phase, as detailed in Figure 6. Standard EP implementation suffers significantly from the bursty nature of prompt processing. PROBE effectively neutralizes these stragglers. By dynamically balancing the expert load, PROBE achieves consistent acceleration across both models and total input tokens, peaking at a 1.32× speedup compared to SGLang. Although the hybrid parallelism (DP for attention module) introduces potential attention workload skew, its impact is mitigated in our experiments by the usage of chunked prefill and dominance of short-context prompts, leaving MoE stragglers as the predominant bottleneck. Furthermore, these gains are more pronounced on the sparser GPT-OSS-120B model, as its inherently higher IR exacerbates straggler bottlenecks, thereby offering a larger optimization margin compared to Qwen3-MoE. Notably, we exclude DeepSeek-EPLB from this evaluation due to its fundamental incompatibility with the prefill regime. First, its reliance on historical statistics proves ineffective against the rapid, instantaneous semantic shift characteristic of the condensed prefill stage. Second, unlike PROBE's hidden scheduling, EPLB's rebalancing incurs transfer overheads that significantly outweigh the potential gains given the limited number of prefill steps. Finally, the alternative of static expert replication is infeasible, as the additional memory footprint triggers OOM errors under the high memory pressure of large-batch processing.

**Decoding Throughput-Latency Trade-off.** Figure 7 illustrates the system performance during the decoding phase. We report the average throughput over the initial 500 decoding steps by sweeping batch sizes. PROBE consistently pushes the Pareto frontier towards the optimal bottomright corner across all datasets, demonstrating a superior trade-off between throughput and latency. Compared to DeepSeek-EPLB configured with one-shot rebalancing, PROBE achieves up to **1.26**× higher throughput at the same

![](_page_8_Figure_6.jpeg)

Figure 7. Throughput-latency Pareto frontier of decoding stage. GPT-OSS with ep=8. We sweep per-rank batch size from 512 to 1536 on *Chinese*, *Code*, and *Repeat*. PROBE consistently dominates the frontier; on *Repeat*, it avoids the large latency spikes caused by extreme expert skewness in baselines.

batch size. The performance gap stems from the temporal dynamics of the workload: EPLB's static placement, derived from a single snapshot, progressively degrades as the semantic distribution drifts over the 500-step window. In contrast, PROBE's continuous lookahead prediction ensures optimal expert locality for every layer of every step, thereby providing robustness against such volatility, which is especially evident on the high-skew Repeat dataset. Furthermore, PROBE optimizes memory efficiency by cyclically reusing expert slots, avoiding EPLB's requirement for static per-layer placeholders that that compete with KV cache for memory. Our dynamic approach strictly limits the redundancy footprint, preserving maximum capacity for long-context inference.

#### 6.3 Robustness to Semantic Shifts

![](_page_8_Figure_10.jpeg)

Figure 8. Throughput under abrupt semantic shifts. GPT-OSS-120B with ep=8: the workload switches from *Code* to *Chinese* at step  $\approx$ 200. DeepSeek-EPLB incurs warm-up and degrades after the shift due to stale placement, while PROBE remains stable via real-time predictive planning.

To validate system robustness against workload volatility, we designed a "stress test" to simulate abrupt semantic transitions. As shown in Figure 8, we initiate the decoding

<span id="page-9-0"></span>process with the Code dataset, followed by an instantaneous switch to the *Chinese* dataset with higher  $\mathcal{IR}$  at step  $\approx$ 200. DeepSeek-EPLB exposes the limitation of historical statistic-based approaches. Initially, EPLB operates with a default placement with no redundant expert, suffering from suboptimal throughput until step  $\approx$ 110. At this point, sufficient historical activation data is collected to trigger a rebalancing event, resulting in a visible performance jump. However, this gain is temporary, when the workload shifts at step≈205, EPLB's performance drastically degrades. This occurs because the system retains the expert placement optimized for the previous distribution, which is now mismatched with the new high-skew workload. In sharp contrast, PROBE demonstrates robustness to volatility. By leveraging the lookahead predictor, PROBE anticipates the expert hotspots of the next layer in real time, rather than relying on the statistics of past steps. Consequently, PROBE requires no warm-up period and maintains a stable, high-throughput trajectory across the shift boundary. The system instantly adapts to the new dataset without the lag observed in EPLB, proving its suitability for environments where request variability is the norm.

![](_page_9_Figure_2.jpeg)

Figure 9. Predictor fidelity across layers. The untrained baseline (frozen router only) suffers from feature drift, while online distillation improves Top-K accuracy to  $\approx 90\%$ . The Top-Half-K hit rate and  $2\times \text{Top-}K$  recall (within a  $2\times$  prediction window) both approach 100%.

### 6.4 Analysis of Predictor Fidelity

The efficacy of PROBE's pipelining depends on the predictor's ability to anticipate expert activation with high fidelity. We first validate the necessity of the trainable residual component: as shown in the Figure 9, while the *Untrained* baseline suffers from feature drift with only around 70%–80% accuracy, our online distillation strategy significantly corrects this, elevating the *Top-K Accuracy* to 87%–94% across layers and ensuring bandwidth is consumed only for valid transfers. Beyond standard accuracy, the predictor

![](_page_9_Figure_6.jpeg)

Figure 10. Timeline breakdown of an single decoding step. GPT-OSS with ep=8 and per-rank b=768, averaged over Layers 1–35 (excluding Layer 0). **Top:** baseline Rank-0 timeline, where *Combine* is inflated by synchronization with stragglers. **Bottom:** PROBE's dual-track timeline; *Prefetch* visualizes transferring up to three experts (0–3 selected per layer) to demonstrate hiding.

exhibits near-deterministic reliability for critical workload components; the *Top-Half-K Hit-Rate* and the 2×*Top-K Recall* consistently approach 100%, serving as a virtually perfect predictor against false negatives. Crucially, this certainty implies that the potential of predictive execution extends far beyond expert prefetching, especially in disaggregation frameworks (Zhu et al., 2025; Wang et al., 2025): since the destinations of the majority of tokens are known effectively before routing computation completes, future optimizations could leverage this signal to pre-dispatch hidden states to high-confidence experts, potentially overlapping the entire All-to-All communication latency with routing itself.

## 6.5 Latency Breakdown

Figure 10 validates PROBE's dual-track scheduling via the averaged micro-operation timeline across Layers 1-35 of GPT-OSS. The breakdown confirms the effective concealment of all control overheads: the **Predict** phase (comprising MLP inference and global Gather) and the single-SM **Planner** execute concurrently with *Dispatch*, with the planner's tail latency naturally overlapped by the MoE Compute. Furthermore, the **Prefetch** latency with 3 expert budget is masked via split-phase transmission, overlapping sequentially with the current MoE Compute and the subsequent layer's Attention, while the **Update** phase prepares expert and assignment masks without stalling execution. Crucially, PROBE effectively neutralizes stragglers: the average IRacross 35 layers sees a modest reduction from 2.13 to 1.09, accompanied by a substantial drop in computation latency skew (Max/Avg) from 2.27 to 1.18. This alignment eliminates synchronization idle time, which otherwise manifests as inflated Combine latency. Notably, given the abundant bandwidth, the visible reduction in the *Combine* phase is driven primarily by this elimination of wait times rather than accelerated data transfer.

## 7 CONCLUSION

In this paper, we identify that MoE inference efficiency is fundamentally limited by the interplay of spatial stragglers and temporal workload volatility. To address these challenges, we propose PROBE, an inference system that shifts balancing paradigm from reactive adjustment to proactive preparation. PROBE leverages Continuous Lookahead Pipelining that combining a high-fidelity predictor, a hardware-aware planner, and phase-locked co-scheduling, to co-balance computation and communication in real time without stalling execution. Extensive evaluations show that PROBE improves both prefill latency and decoding throughput over state-of-the-art baselines. Collectively, our results validate predictive lookahead execution as a promising approach for efficient trillion-parameter MoE deployment.

## <span id="page-11-0"></span>REFERENCES

- Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023.
- Agarwal, S., Ahmad, L., Ai, J., Altman, S., Applebaum, A., Arbus, E., Arora, R. K., Bai, Y., Baker, B., Bao, H., et al. gpt-oss-120b & gpt-oss-20b model card. *arXiv preprint arXiv:2508.10925*, 2025.
- Chaudhary, S. Code alpaca: An instruction-following llama model for code generation, 2023.
- Chen, C., Li, X., Zhu, Q., Duan, J., Sun, P., Zhang, X., and Yang, C. Centauri: Enabling efficient scheduling for communication-computation overlap in large model training via communication partitioning. In *Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3*, pp. 178–191, 2024.
- Chen, M., Tworek, J., Jun, H., Yuan, Q., de Oliveira Pinto, H. P., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., Ryder, N., Pavlov, M., Power, A., Kaiser, L., Bavarian, M., Winter, C., Tillet, P., Such, F. P., Cummings, D., Plappert, M., Chantzis, F., Barnes, E., Herbert-Voss, A., Guss, W. H., Nichol, A., Paino, A., Tezak, N., Tang, J., Babuschkin, I., Balaji, S., Jain, S., Saunders, W., Hesse, C., Carr, A. N., Leike, J., Achiam, J., Misra, V., Morikawa, E., Radford, A., Knight, M., Brundage, M., Murati, M., Mayer, K., Welinder, P., McGrew, B., Amodei, D., McCandlish, S., Sutskever, I., and Zaremba, W. Evaluating large language models trained on code, 2021.
- Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. *arXiv preprint arXiv:2507.06261*, 2025.
- Dai, D., Deng, C., Zhao, C., Xu, R., Gao, H., Chen, D., Li, J., Zeng, W., Yu, X., Wu, Y., et al. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. *arXiv preprint arXiv:2401.06066*, 2024.
- Doucet, Z., Sharma, R., de Vos, M., Pires, R., Kermarrec, A.-M., and Balmau, O. Harmoeny: Efficient multi-gpu inference of moe models. *arXiv preprint arXiv:2506.12417*, 2025.
- Du, N., Huang, Y., Dai, A. M., Tong, S., Lepikhin, D., Xu, Y., Krikun, M., Zhou, Y., Yu, A. W., Firat, O., et al. Glam:

- Efficient scaling of language models with mixture-ofexperts. In *International conference on machine learning*, pp. 5547–5569. PMLR, 2022.
- Fedus, W., Zoph, B., and Shazeer, N. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *The Journal of Machine Learning Research*, 2022.
- Guo, H., Lu, H., Nan, G., Chu, B., Zhuang, J., Yang, Y., Che, W., Leng, S., Cui, Q., and Jiang, X. Advancing expert specialization for better moe. *arXiv preprint arXiv:2505.22323*, 2025a.
- Guo, W., Mishra, M., Cheng, X., Stoica, I., and Dao, T. Sonicmoe: Accelerating moe with io and tile-aware optimizations. *arXiv preprint arXiv:2512.14080*, 2025b.
- Han, Y., Pan, L., Peng, J., Tao, Z., Zhang, W., and Zhang, Y. Grace-moe: Grouping and replication with locality-aware routing for efficient distributed moe inference. *arXiv preprint arXiv:2509.25041*, 2025.
- He, J., Zhai, J., Antunes, T., Wang, H., Luo, F., Shi, S., and Li, Q. Fastermoe: modeling and optimizing training of large-scale dynamic pre-trained models. In *Proceedings of the 27th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming*, pp. 120–134, 2022.
- Hu, J., Xu, M., Ye, K., and Xu, C. Brownoutserve: Sloaware inference serving under bursty workloads for moebased llms. *IEEE Transactions on Computers*, 2026.
- Hwang, R., Wei, J., Cao, S., Hwang, C., Tang, X., Cao, T., and Yang, M. Pre-gated moe: An algorithm-system codesign for fast and scalable mixture-of-expert inference. In *2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)*, pp. 1018–1031. IEEE, 2024.
- Jacobs, R. A., Jordan, M. I., Nowlan, S. J., and Hinton, G. E. Adaptive mixtures of local experts. *Neural computation*, 3(1):79–87, 1991.
- Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D. S., Casas, D. d. l., Hanna, E. B., Bressand, F., et al. Mixtral of experts. *arXiv preprint arXiv:2401.04088*, 2024.
- Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*, 2020.
- Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., and Stoica, I. Efficient memory management for large language model serving

- <span id="page-12-0"></span>with pagedattention. In *Proceedings of the 29th symposium on operating systems principles*, pp. 611–626, 2023.
- Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N., and Chen, Z. Gshard: Scaling giant models with conditional computation and automatic sharding. In *International Conference on Learning Representations*, 2021.
- Li, Y., Wei, F., Zhang, C., and Zhang, H. Eagle-3: Scaling up inference acceleration of large language models via training-time test. *arXiv preprint arXiv:2503.01840*, 2025a.
- Li, Y., Zheng, P., Chen, S., Xu, Z., Lai, Y., Du, Y., and Wang, Z. Speculative moe: Communication efficient parallel moe inference with speculative token and expert pre-scheduling. *arXiv preprint arXiv:2503.04398*, 2025b.
- Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., et al. Deepseek-v3 technical report. *arXiv preprint arXiv:2412.19437*, 2024.
- Liu, A., Mei, A., Lin, B., Xue, B., Wang, B., Xu, B., Wu, B., Zhang, B., Lin, C., Dong, C., et al. Deepseek-v3. 2: Pushing the frontier of open large language models. *arXiv preprint arXiv:2512.02556*, 2025.
- NVIDIA. NVIDIA Collective Communication Library (NCCL) Documentation. [https:](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html) [//docs.nvidia.com/deeplearning/nccl/](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html) [user-guide/docs/index.html](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html), 2022.
- NVIDIA. Tensorrt-llm: A tensorrt toolbox for optimized large language model inference. [https://github.](https://github.com/NVIDIA/TensorRT-LLM) [com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), 2023.
- NVIDIA Corporation. *NVIDIA CUDA C++ Programming Guide*, 2024a. URL [https://docs.nvidia.com/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs) [cuda/cuda-c-programming-guide/index.](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs) [html#cuda-graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs). Version 12.6.
- NVIDIA Corporation. NVSHMEM: NVLink-accelerated communication library. [https://developer.](https://developer.nvidia.com/nvshmem) [nvidia.com/nvshmem](https://developer.nvidia.com/nvshmem), 2024b.
- Peng, B., Li, C., He, P., Galley, M., and Gao, J. Instruction tuning with gpt-4. *arXiv preprint arXiv:2304.03277*, 2023.
- Qiu, Z., Huang, Z., Zheng, B., Wen, K., Wang, Z., Men, R., Titov, I., Liu, D., Zhou, J., and Lin, J. Demons in the detail: On implementing load balancing loss for training specialized mixture-of-expert models. *arXiv preprint arXiv:2501.11873*, 2025.

- Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., and Dean, J. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In *International Conference on Learning Representations*, 2017.
- Team, K., Bai, Y., Bao, Y., Chen, G., Chen, J., Chen, N., Chen, R., Chen, Y., Chen, Y., Chen, Y., et al. Kimi k2: Open agentic intelligence. *arXiv preprint arXiv:2507.20534*, 2025.
- Tillet, P., Kung, H.-T., and Cox, D. Triton: an intermediate language and compiler for tiled neural network computations. In *Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages*, pp. 10–19, 2019.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. *Advances in neural information processing systems*, 30, 2017.
- Wang, B., Wang, B., Wan, C., Huang, G., Hu, H., Jia, H., Nie, H., Li, M., Chen, N., Chen, S., et al. Step-3 is large yet affordable: Model-system co-design for cost-effective decoding. *arXiv preprint arXiv:2507.19427*, 2025.
- Wang, L., Gao, H., Zhao, C., Sun, X., and Dai, D. Auxiliaryloss-free load balancing strategy for mixture-of-experts. *arXiv preprint arXiv:2408.15664*, 2024.
- Xue, F., Zheng, Z., Fu, Y., Ni, J., Zheng, Z., Zhou, W., and You, Y. Openmoe: An early effort on open mixture-of-experts language models. *arXiv preprint arXiv:2402.01739*, 2024.
- Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. *arXiv preprint arXiv:2505.09388*, 2025.
- Yu, H., Cui, X., Zhang, H., and Wang, H. Taming latencymemory trade-off in moe-based llm serving via finegrained expert offloading. 2026.
- Yun, S., Choi, I., Peng, J., Wu, Y., Bao, J., Zhang, Q., Xin, J., Long, Q., and Chen, T. Flex-moe: Modeling arbitrary modality combination via the flexible mixture-of-experts. *Advances in Neural Information Processing Systems*, 37: 98782–98805, 2024.
- Zeng, Y., Huang, C., Mei, Y., Zhang, L., Su, T., Ye, W., Shi, W., and Wang, S. Efficientmoe: Optimizing mixture-ofexperts model training with adaptive load balance. *IEEE Transactions on Parallel and Distributed Systems*, 2025.
- Zhai, M., He, J., Ma, Z., Zong, Z., Zhang, R., and Zhai, J. {SmartMoE}: Efficiently training {Sparsely-Activated}

- <span id="page-13-0"></span>models through combining offline and online parallelization. In *2023 USENIX Annual Technical Conference (USENIX ATC 23)*, pp. 961–975, 2023.
- Zhang, M., Li, P., Peng, J., Qiu, M., and Chen, T. Advancing moe efficiency: A collaboration-constrained routing (c2r) strategy for better expert parallelism design. *arXiv preprint arXiv:2504.01337*, 2025.
- Zhao, C., Zhou, S., Zhang, L., Deng, C., Xu, Z., Liu, Y., Yu, K., Li, J., and Zhao, L. Deepep: an efficient expert-parallel communication library. *https://github. com/deepseek-ai/DeepEP*, 2025.
- Zhao, Y., Yang, S., Zhu, K., Zheng, L., Kasikci, B., Zhou, Y., Xing, J., and Stoica, I. Blendserve: Optimizing offline inference for auto-regressive large models with resourceaware batching. *arXiv preprint arXiv:2411.16102*, 2024.
- Zheng, L., Yin, L., Xie, Z., Sun, C. L., Huang, J., Yu, C. H., Cao, S., Kozyrakis, C., Stoica, I., Gonzalez, J. E., et al. Sglang: Efficient execution of structured language model programs. *Advances in neural information processing systems*, 37:62557–62583, 2024.
- Zhu, R., Jiang, Z., Jin, C., Wu, P., Stuardo, C. A., Wang, D., Zhang, X., Zhou, H., Wei, H., Cheng, Y., et al. Megascale-infer: Serving mixture-of-experts at scale with disaggregated expert parallelism. *arXiv preprint arXiv:2504.02263*, 2025.