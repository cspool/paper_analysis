# Understanding Inference Scaling for LLMs: Bottlenecks, Trade-offs, and Performance Principles

Moiz Arif\*, Avinash Maurya<sup>†</sup>, Sudharshan Vazhkudai\*, Bogdan Nicolae<sup>†</sup>

\*Micron Technology Inc., Austin, TX, USA

<sup>†</sup>Argonne National Laboratory, Lemont, IL, USA

Email: \*{marifa, svazhkudai}@micron.com; <sup>†</sup>{amaurya, bnicolae}@anl.gov

Abstract—The transition from standard generative AI to reasoning-centric architectures, exemplified by models capable of extensive Chain-of-Thought (CoT) processing, marks a fundamental paradigm shift in system requirements. Unlike traditional workloads dominated by compute-bound prefill, reasoning workloads generate long chains of reasoning tokens that shift inference into a Capacity-Bound regime. This paper presents a comprehensive system characterization, evaluating models ranging from 8B to 671B parameters on GPUs clusters. By systematically exploring the interplay between Data, Tensor, and Pipeline parallelism, we identify critical bottlenecks that defy standard scaling heuristics. Our analysis reveals that data parallelism is throughput efficient for small models but hits a capacity trap on reasoning workloads as KV-cache fragmentation forces early throttling resulting in sub-optimal compute utilization. Tensor parallelism unlocks stranded memory and delivers sublinear gains near the 32B crossover. At frontier scale, dense models (e.g., Llama-405B) are interconnect and memory-bandwidth bound and favor high-degree TP, while sparse Mixture-of-Experts (MoE) models (e.g., DeepSeek-R1) are limited by routing and synchronization latency and benefit from hybrid strategies. These insights provide a rigorous decision framework for navigating the reasoning cliff, establishing new architectural imperatives for the next generation of inference infrastructure.

Index Terms—LLM inference, parallelism and scheduling Analysis, performance characterization, memory capacity wall

#### I. Introduction

#### A. Motivation

The deployment of Large Language Models (LLMs) has transitioned from a capability demonstration to a cornerstone of industrial computing. However, the emergence of reasoning-centric architectures, exemplified by DeepSeek-R1 [16] and OpenAI's o1 [28] marks a fundamental paradigm shift in system requirements. Unlike standard generative models that prioritize fluent text generation, reasoning models employ CoT [35] like processing to generate extensive intermediate logic traces before producing a final answer.

This shift fundamentally alters the resource signature of AI inference. While traditional LLM serving is often dominated by compute-bound prefill or bandwidth-bound decoding of short responses, reasoning workloads typically demonstrate a *Capacity-Bound* inference regime. The generation of thousands of intermediate "thinking" tokens [34] creates massive, persistent Key-Value (KV) cache footprints that saturate High Bandwidth Memory (HBM) long before compute utilization peaks [37]. Consequently, standard scaling heuristics such as

maximizing batch size for throughput or relying solely on Data Parallelism (DP) become counter-productive, triggering severe memory thrashing, scheduler preemption, and non-linear latency spikes [1], [23].

Deploying these frontier-scale models (often exceeding 400B parameters [18], [25]) requires navigating a complex, high-dimensional design space [21]. System architects must balance the low-latency requirements of interactive reasoning against the strict capacity limits of modern accelerators. This necessitates a move beyond monolithic scaling toward nuanced combinations of Data, Tensor [33], and Pipeline parallelism [20], tailored specifically to the sparsity and sequence length characteristics of the model [39].

#### B. System Challenges in Reasoning Inference

Deploying reasoning-centric models at scale introduces distinct system-level bottlenecks that are structurally different from training [6] or standard chat inference:

- The Capacity Wall: Reasoning traces with Output Sequence Length  $(OSL)\gg 10k$  cause the KV cache to grow linearly, rapidly exhausting per-GPU HBM [24]. In DP with engines such as vLLM [22], this leads to "stranded capacity" where fragmented memory across replicas forces premature request throttling despite low compute utilization [37].
- The Parallelism Efficiency Gap: While Tensor Parallelism (TP) alleviates memory pressure by using collective HBM of multiple GPUs, it incurs communication overheads that degrade performance beyond intra-node scaling [10]. Conversely, Pipeline Parallelism (PP) offers memory relief but suffers from "pipeline bubbles" that are difficult to hide in serial, auto-regressive generation tasks [3].
- Architectural Divergence: The optimal strategy is no longer universal as we show in our analysis. As we demonstrate, dense models (e.g., Llama-3.1-405B) behave fundamentally differently from sparse MoE models (e.g., DeepSeek-R1-671B), requiring bespoke parallelization hierarchies to align with their respective compute-to-communication ratios [8].

While recent work has improved kernel efficiency and KV management, the dominant scaling limits for reasoning workloads increasingly arise from system-level capacity and scheduling dynamics rather than operator inefficiencies. This work isolates and quantifies these limits under realistic inference conditions.

## *C. Contributions*

This paper presents a systematic characterization of GPUbased inference for reasoning-centric LLMs. We evaluate models ranging from 8B to 671B parameters on a cluster of NVIDIA H200 GPUs, identifying the breakpoints where standard scaling laws fall short. Our specific contributions are:

- 1) Characterization of the "Parallelism Transition Point": We empirically determine the model size and sequence length thresholds where DP collapses due to KV saturation, necessitating a transition to hybrid or TP approaches.
- 2) Quantification of the "Reasoning Gap": We demonstrate how the extended decoding phases of reasoning models shift the critical bottleneck from prefill compute (TTFT) to decode memory capacity (TPOT).
- 3) Evaluation of Frontier Architectures: We provide a comparative analysis of large dense versus sparse scaling, revealing that MoE architectures favor hybrid strategies (PP+TP) to mitigate synchronization costs, whereas dense models demand high-bandwidth TP configurations.
- 4) Operational Guidance: Based on these insights, we provide guidance for selecting optimal parallelism strategies that maximize fleet-level throughput while meeting strict latency SLAs for reasoning tasks.

#### II. BACKGROUND AND RELATED WORK

## *A. Reasoning Inference Breakdown*

Reasoning-centric inference is distinct from standard chat workloads in its execution flow and resource demands. It proceeds in two phases with orthogonal hardware requirements:

Prefill (The Compute Phase): The model processes the user prompt (input sequence) in parallel and dominated by large matrix-matrix multiplications (GEMMs) and is strictly compute-bound. Modern GPUs effectively utilize their tensor cores, achieving high Streaming Multiprocessor (SM) occupancy. The latency of this phase determines the TTFT.

Decode (The Memory Phase): Following prefill, the model enters the autoregressive generation loop. Unlike standard chat (OSL ≈ 500), reasoning traces can exceed 10000 tokens [36]. Each token generation requires reading the entire model weight set and the active KV cache from HBM [37]. This phase is strictly *bandwidth-bound*; the arithmetic intensity collapses, and the GPU spends the majority of cycles performing memory reads. Our telemetry indicates that for reasoning workloads, the system spends > 99% of its wall-clock time in this inefficient regime [2].

## *B. Model Architectures and Attention Mechanisms*

The impact of the GPU memory exhaustion is modulated by the specific architecture of the model. We characterize two distinct paradigms found in state-of-the-art reasoning engines:

Dense Architectures (Grouped-Query Attention): Models like *Llama-3.1-405B* [18] utilize a dense transformer architecture where every parameter is active for every token. To mitigate memory pressure, these models employ *Grouped-Query Attention (GQA)* (typically 8 KV heads) [4], which reduces the KV footprint by 3×–8× compared to standard Multi-Head Attention (MHA) [12]. However, the memory cost remains linear with layer count [5]. For the 405B model (≈126 layers), the KV cache consumes ≈1.05 MB per token in FP16. Serving a batch of 128 requests with 10k reasoning tokens each consumes over 1.3 TB of memory solely for the cache, far outstripping the capacity of a single H200 GPU.

Sparse Architectures (Multi-Head Latent Attention): The *DeepSeek-R1-671B* [16] model utilizes a MoE architecture, activating only ≈37B parameters per token. Crucially, it employs *Multi-Head Latent Attention (MLA)*, which compresses the KV cache into a low-rank latent vector. This architectural choice effectively decouples the KV cache size from the number of attention heads, allowing R1 to sustain long reasoning contexts with a relatively lower memory footprint than dense models of same scale [26].

#### *C. Memory Hierarchy and Scheduling*

To manage these massive footprints, we utilize *vLLM* [22], an advanced, widely-used inference engine with *PagedAttention*, which partitions the KV cache into non-contiguous blocks to eliminate internal fragmentation. The scheduler plays a critical role in this architecture:

- Chunked Prefill: To maximize GPU throughput, the scheduler splits long input sequences into smaller chunks [3]. These chunks are processed iteratively and added greedily to batches, allowing the system to interleave prefill computation with ongoing decode requests. This smooths SM occupancy and mitigates head-of-line blocking, though it can fragment memory traffic.
- Preemption: When the "Reasoning Cliff" is reached (HBM saturation), the scheduler must preempt active requests [37], moving them to a "Waiting" queue or swapping them to CPU host memory. This is a defensive mechanism to prevent OOM but triggers a re-computation or swapping penalty that degrades tail latency.

We focus on HBM-resident KV caches to isolate fundamental capacity and bandwidth limits of reasoning-centric inference without introducing latency trade-offs from disaggregated memory tiers. Techniques such as KV offloading [27], prefetching [17], quantization [11], and compression expand the optimization space but are ultimately constrained along the decode path by latency, bandwidth, and scheduling overheads. We therefore treat these techniques as complementary and orthogonal to our analysis, which characterizes when KVcache pressure dominates even under HBM-only conditions.

#### *D. Parallelism Taxonomy*

Scaling these models requires distributing computation across several GPUs. We evaluate four strategies:

Data Parallelism (DP): The model is replicated on each GPU. While efficient for throughput, DP suffers from *KV Fragmentation* as each replica holds a full copy of weights (e.g., ≈800GB for 405B which exceeds single GPU memory capacity and even for smaller models, weight replication reduces available KV space).

Tensor Parallelism (TP): Individual layers are sharded across GPUs. TP uses the aggregated HBM capacity without unnecessary model replication. However, it introduces highfrequency all-Reduce communication at every layer [1]. For the 32B model, we observe a transition where the benefit of aggregate memory outperforms the cost of communication.

Pipeline Parallelism (PP): Layers are partitioned sequentially across GPUs. PP reduces memory footprint without the high communication frequency of TP. However, it introduces "Pipeline Bubbles" (idle time) [20] due to lack of sufficient concurrent requests to fill these bubbles without incurring unacceptable queueing latency.

Hybrid Parallelism: A hierarchical combination (e.g., TP within a node, PP across nodes). Our analysis suggests this is critical for frontier sparse models to balance memory pooling with synchronization overheads.

## III. EXPERIMENTAL METHODOLOGY

#### *A. Testbed and Inference Engine*

All experiments are performed on a high-performance inference node equipped with 8× NVIDIA H200 Tensor Core GPUs (SXM5 form factor). Each H200 GPU features 141 GB of HBM3e memory providing a peak memory bandwidth of 4.8 TB/s, and delivers a theoretical peak of 1,979 TFLOPS for FP16/BF16 tensor operations. The GPUs are fully interconnected via fourth-generation NVLink and NVSwitch, providing a bidirectional GPU-to-GPU bandwidth of 900 GB/s per GPU, facilitating efficient all-reduce operations essential for TP. The host system is driven by dual Intel Xeon Platinum 8558P processors with 2TB DDR5 system memory.

We employ the vLLM v1 inference engine, leveraging its PagedAttention mechanism to mitigate memory fragmentation. The engine is configured with a block size of B = 16 to balance quantization granularity with memory access efficiency. We utilize the default First-Come-First-Served (FCFS) scheduling policy but tune the max\_num\_batched\_tokens and max\_num\_seqs parameters to characterize the concurrency limits.

Our evaluation focuses on system-level behavior within a single NVLink-connected 8-GPU node, which serves as the fundamental scaling unit for modern LLM inference. While kernel-level optimizations and fine-grained overlap of communication and computation can improve absolute efficiency, the per-replica constraints imposed by KV-cache capacity, memory bandwidth, and scheduler dynamics under long-context reasoning workloads remain the same. Larger deployments primarily scale throughput via DP replication of this node-level behavior, with TP confined to the NVLink domain. Accordingly, we abstract away micro-architectural kernel specialization to quantify when system-level effects dominate and common scaling heuristics break down.

## *B. Dataset Characterization*

To evaluate the "Reasoning Shift", we utilize Meta's Natural Reasoning dataset [38], comprised of 1.15 million multi-hop reasoning and commonsense inference samples. Unlike standard chat workloads dominated by input processing (prefill), this dataset forces models into a generation-heavy regime. Figure 1 illustrates the divergent token length distributions that characterize "reasoning" versus traditional generative LLM workloads. Specifically, it compares input sequence lengths (ISL) versus output sequence lengths (OSL), highlighting that the majority of prompts remain relatively short while their outputs become exceptionally long. A thorough analysis of 100k random samples uncovers a distinct workload profile:

- Input Sequence Length (ISL): 77% of prompts are 50– 150 tokens, and very few exceed 300 tokens (Figure 1a).
- Output Sequence Length (OSL): 45% of responses exceed 5000 tokens, indicating not only larger outputs but also complex reasoning chains (Figure 1b).
- Reasoning Density: Similar to OSL, reasoning steps are highly verbose, with 43.04% of responses contain over 5000 "reasoning tokens" reflecting detailed multi-step reasoning token generation (Figure 1c).

Architecturally, these distributions underscore a key stress point: even modest numbers of reasoning requests can saturate GPU memory capacity due to massive KV cache footprints from long outputs, well before reaching full compute utilization. This shifts the paradigm from prefill compute density to decode memory bandwidth/capacity and interconnect latency, providing critical motivation for treating memory capacity as a first-class design parameter– since conventional focus on FLOPs/bandwidth alone overlooks this token explosion effect inherent to reasoning workloads.

