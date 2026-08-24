# 5 Preliminary Results

To assess the effectiveness of our system, we conducted an evaluation using a representative conversational voice agent composed of modular components: speech-to-text, text-to-speech, web search, and a central LLM node, as depicted in Figure [2.](#page-3-1) These findings are preliminary, and comprehensive system validation is currently underway. Table [5](#page-16-0) summarizes the accelerator hardware included in our current evaluation.

Our optimization framework places the non-LLM components of the voice agent on CPUs given the task characteristic (relatively computationally light) and the relative cost of a CPU, hence the dominant factor impacting overall TCO is the LLM component which is the most computationally demanding part. As a result, the following focuses on exploring optimizations on the LLM component. For the LLM, we evaluated four configurations of the LLaMA 3 model: 8B and 70B parameter sizes, each in FP16 and FP8 precisions (see Table [4\)](#page-16-1). Computational and memory demands were profiled based on model size, sequence lengths, and architectural details as an input to the optimization framework. Device-specific performance metrics, such as latency and throughput, incorporate empirical measurements when available and are augmented by theoretical roofline modeling [\[38\]](#page-23-21) to represent realistic performance boundaries. All reported FLOP values assume dense computation, without accounting for sparsity.

To precisely isolate scheduling and hardware allocation benefits, we simulated a continuous workload scenario with unconstrained hardware availability. We evaluate which heterogeneous configuration leads to the maximum throughput (tokens/sec by maximizing batch size) under two different scenarios with SLAs that correspond to interactive and offline usage scenarios:

- Latency SLA (Interactive workloads): Time-to-First-Token (TTFT) 250 ms, Token-to-Token (TBT) 20 ms.
- Throughput SLA (Offline workloads): Maximize tokens/s/\$.

| Model                | Parameters (B) | Precision | Source       |
|----------------------|----------------|-----------|--------------|
| LLaMA 3 - 8B - FP16  | 8              | FP16      | Meta AI [39] |
| LLaMA 3 - 8B - FP8   | 8              | FP8       | Meta AI [39] |
| LLaMA 3 - 70B - FP16 | 70             | FP16      | Meta AI [39] |
| LLaMA 3 - 70B - FP8  | 70             | FP8       | Meta AI [39] |

<span id="page-16-1"></span>Table 4: Model configurations used in evaluation.

| Device | Manufacturer | Cost (\$) | Memory (GB) | Bandwidth (GB/s) | TFLOPs (FP16) | Operating Cost (\$/hr) |
|--------|--------------|-----------|-------------|------------------|---------------|------------------------|
| A40    | NVIDIA       | \$3,000   | 48          | 696              | 75            | \$0.15                 |
| A100   | NVIDIA       | \$8,000   | 80          | 2039             | 322           | \$0.25                 |
| Gaudi3 | Intel        | \$12,500  | 128         | 3700             | 1678          | \$0.49                 |
| MI300x | AMD          | \$20,000  | 192         | 5300             | 1307          | \$0.52                 |
| H100   | NVIDIA       | \$25,000  | 80          | 3350             | 1979          | \$0.60                 |
| B200   | NVIDIA       | \$40,000  | 192         | 8000             | 2250          | \$0.83                 |

<span id="page-16-0"></span>Table 5: Specifications of accelerator hardware used in the optimizer. Costs averaged across a representative sample of hardware resellers available in public listings as of June 2025.

> **[图片提取文字 (无描述)]:**
> Latency SLA Throughput SLA 4.0 3.5 3.0 2.5 2.0 2.0 1.5 H100 H100 1.0 0.5 0.0 8200:9audi3 8200:H100 H100:9audi3 8200:8200 Device Pair Device Pair Model Llama 3 - 8B - FP8 Llama 3 - 70B - FP8 Llama 3 - 8B - FP16 Llama 3 - 70B - FP16
![](_page_16_Figure_9.jpeg)

TCO Benefit for Heterogeneous Configs (input=512, output=4096)

<span id="page-16-2"></span>Figure 8: TCO Benefit for Heterogeneous Configs (input=512, output=4096). Comparison of cost efficiency across different Llama 3 models and device pairings. Dashed line at 1.0 indicates baseline TCO for H100::H100. Bars show top configurations that meet SLA constraints: Latency SLA (TTFT  $\leq$  250ms, TBT  $\leq$  20ms) and Throughput SLA (Maximize tokens/s/\$). Results are based on a performance model fit to real measurements and explore heterogeneous configurations that leverage both tensor parallelism and pipeline parallelism with disaggregated inference.

#### 5.1 TCO of Heterogeneous Systems

The evaluated accelerator hardware specifications are detailed in Table 5, covering GPUs and ASIC accelerators from multiple vendors to demonstrate the broad applicability of our framework. The operating cost model assumes that hardware is financed over a fixed amortization period of 4 years with an interest rate of 8%. For utility costs, we assume

Latency SLA

Throughput SLA

Throughput SLA

Throughput SLA

Throughput SLA

Throughput SLA

Throughput SLA

Latency SLA

Throughput SLA

Latency SLA

Throughput SLA

Latency SLA

Throughput SLA

Latency SLA

Throughput SLA

Latency SLA

Throughput SLA

Latency SLA

Throughput SLA

Latency SLA

Throughput SLA

Latency SLA

Throughput SLA

Latency SLA

Latency SLA

Throughput SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

Latency SLA

TCO Benefit for Heterogeneous Configs (input=4096, output=512)

<span id="page-17-0"></span>Figure 9: **TCO Benefit for Heterogeneous Configs (input=4096, output=512).** Comparison of cost efficiency across different Llama 3 models and device pairings. Dashed line at 1.0 indicates baseline TCO for H100::H100. Bars reflect top-performing configurations that satisfy SLA constraints: Latency SLA (TTFT  $\leq$  250ms, TBT  $\leq$  20ms) and Throughput SLA (Maximize tokens/s/\$.). These results are derived from a performance model calibrated to hardware measurements, incorporating both tensor parallelism and pipeline parallelism under disaggregated inference.

each node operates at its maximum rated TDP, with a cost of \$0.40/kWh. Other operational expenses, such as datacenter or colocation fees and nonrecurring engineering (NRE) costs, are excluded from the operating cost. Additionally, to describe the heterogeneous configurations, we leverage the operator "::" as a notation to denote disaggregated inference. The left and right operands correspond to the hardware configurations used during the prefill and decode stages, respectively.

Figures 8, and 9 demonstrate the TCO improvements achievable through heterogeneous hardware configurations compared against the homogeneous baseline configuration (H100::H100). We focus on 6 possible combinations of hardware pairs, as they best illustrate the variations between performance and cost. We evaluated two scenarios for input-output sequence lengths, corresponding to reasoning tasks (long intermediate/output token sizes) and summarization tasks (short output sequence length). For each configuration, the system automatically explores options and selects the best combination of tensor and pipeline parallelism based on the available network bandwidth (both scale up and scale out) for that configuration and the latency SLA. Initial increases in tensor parallelism substantially reduced latency; however, further increases introduced significant device-to-device communication overhead, negating the computational efficiency gains. Additionally, our framework automatically incorporates optimizations such as paged attention [12], further enhancing the efficiency of execution.

There are two interesting observations from the results.

- **B200::Gaudi 3** has the best overall TCO benefit, especially for FP8 model configurations, for both interactive as well as batch workloads. The benefits are present (albeit smaller) even compared to a B200::B200 baseline which is the latest generation system.
- H100::Gaudi 3 configuration is often comparable or slightly better than a B200::B200 configuration, implying that the Gaudi 3 can effectively complement the H100 and overall the heterogeneous configuration can deliver compelling performance, reducing the need to upgrade to Blackwell. The benefits are likely even higher if we incorporate the depreciation of the Hopper GPUs that have already been partially amortized, which is outside the scope of this paper.

#### 5.2 Deployment requirements and considerations

One of the central challenges in deploying workloads across distributed systems lies in managing the bandwidth and latency constraints imposed by the interconnect fabric linking accelerators. These interconnects are typically categorized as *scale-up* or *scale-out* fabrics. *Scale-up fabrics* aim to deliver high-bandwidth, low-latency connections with shared memory semantics across multiple accelerators within a single system, as exemplified by NVLink-based designs such as NVL72 [20]. In contrast, *scale-out fabrics* rely on commodity networking technologies such as Ethernet

and InfiniBand [\[22\]](#page-23-5), enabling the interconnection of large-scale clusters without shared memory, thereby requiring explicit software coordination for data movement.

In our system design, we assume that scale-up fabrics are confined to a single chassis, typically supporting up to 8 accelerators. Beyond this, we rely on high-speed *RDMA over Converged Ethernet (RoCE)* [\[40\]](#page-24-0), which is commonly deployed in modern large-scale AI datacenters [\[41\]](#page-24-1).

We utilize the underlying fabric for two primary purposes:

- 1. Inter-node parallelism: Distributing computation across multiple machines (for example tensor parallelism)
- 2. State transfer across pipeline stages: Moving shared runtime state between nodes (for example key-value (KV) caches, during prefill/decode disaggregation.)

Both inter-node parallelism and state transfer are incorporated into our total cost of ownership (TCO) model. The scalability of inter-node parallelism is constrained by the efficiency of data movement between accelerators, while state transfer primarily affects the end-to-end latency of the deployed agent.

Importantly, state transfer latency can often be partially amortized by overlapping communication with computation. For example, in prefill/decode disaggregation, key-value (KV) cache transfers contribute to the latency of the *second token*, as the cache must be transmitted from the prefill stage to the decode stage. Fortunately, the bandwidth demands of this transfer are typically well-supported by modern AI datacenter networks [\[42\]](#page-24-2). For completeness, we present the high-level bandwidth model that can be used to model the minimum bandwidth required to allow non-blocking pipelining of disaggregated inference:

$$BW_{\text{PeakEgress}} = \frac{\text{KV Cache Size}}{\text{TTFT} \cdot N_{PrefillGPU}} \tag{1}$$

$$BW_{\text{PeakIngress}} = \frac{\text{KV Cache Size}}{\text{TBT} \cdot N_{DeocodeGPU}}$$
 (2)

It is important to note that the above equations represent the *peak* bandwidth required to transfer a single KV cache instance. In practice, inference systems often operate on batched inputs, which linearly scales the effective KV cache size and, correspondingly, the peak bandwidth requirement.

However, if the primary concern is overall task completion time—as is common in batch-oriented workloads—then it is more appropriate to consider *amortized* bandwidth.

For practical workloads, we can estimate the peak bandwidth required based on the KV cache size and compute time. We compute the size of the key-value (KV) cache required for transformer-based models such as LLaMA using the following expression:

$$\text{KVCacheSize}_{\text{peak}} = 2 \cdot N_{\text{layers}} \cdot d_{\text{model}} \cdot \left(\frac{N_{\text{kv}}}{N_{\text{heads}}}\right) \cdot \text{ISL} \cdot BS \cdot \text{BPE} \tag{3}$$

### Legend:

• Nlayers: Number of transformer layers

• dmodel: Hidden dimension of the model

• Nkv: Number of key/value heads

• Nheads: Total number of attention heads

• ISL: Input sequence length (tokens)

• BS: Batch size

• BPE: Bytes per element (e.g., 2 for FP16)

Using the derived expressions, we observe that a 200–400 Gbps link is sufficient to meet the SLA requirements for transferring KV caches for input sequence lengths up to 32K tokens, depending on the specific LLaMA model variant employed. Such high-bandwidth interconnects are commonly available in modern high-performance AI datacenters.

While our TCO model incorporates a detailed treatment of networking latency and cost, we find that practical provisioning of interconnect bandwidth is generally sufficient to mitigate performance bottlenecks. Moreover, as noted by [\[42\]](#page-24-2), increases in model and context size can actually reduce bandwidth requirements in practice. For instance, total time for first token (TTFT) tends to grow superlinearly with input sequence length (ISL), whereas the KV cache size grows only linearly. Similarly, while decode latency depends on the number of decoding GPUs, the corresponding ingress bandwidth requirement decreases inversely. Additionally, recent models with more efficient attention mechanisms—such as Multi-Linear Attention (MLA)—require smaller KV cache sizes [\[43\]](#page-24-3), further reducing pressure on interconnect bandwidth.

#### 5.3 Analysis

To understand the above results, we explored how our optimization framework is making decisions on which parts of the voice agent workload are placed on which hardware. For example, the hardware allocations of different LLM inference stages (prefill and decode) are quite distinct given their different computational needs (prefill is computationally intensive whereas decode is more memory capacity intensive). Our framework inherently accommodates such optimizations by decomposing the LLM workload into granular components, enabling hardware resources to be matched precisely with operational demands.

Optimal hardware configurations varied significantly depending on input sequence length and decode tokens. For longer input sequences (Figure [9\)](#page-17-0), Intel Gaudi 3 accelerators emerged as the most cost-effective choice for prefill tasks due to their superior cost-performance ratio relative to NVIDIA B200. Conversely, when latency or FP8 performance is the primary concern, the higher computational power of the B200 justified its selection despite higher associated costs.

In decode-intensive scenarios (Figure [8\)](#page-16-2), Gaudi3 accelerators were selected for decode tasks due to their lowest marginal cost, as indicated in Figure [4,](#page-9-0) assuming the workload can accommodate slightly longer token-to-token latency. Conversely, the B200 provides the best overall performance at an increased cost but remains relatively efficient compared to previous-generation systems such as the H100.

In conclusion, our optimization framework effectively leverages the diverse performance characteristics of heterogeneous hardware resources, dynamically allocating workloads based on specific SLA requirements. This adaptability enables optimal utilization of hardware capabilities, ensuring both cost efficiency and performance responsiveness tailored to individual requests.

