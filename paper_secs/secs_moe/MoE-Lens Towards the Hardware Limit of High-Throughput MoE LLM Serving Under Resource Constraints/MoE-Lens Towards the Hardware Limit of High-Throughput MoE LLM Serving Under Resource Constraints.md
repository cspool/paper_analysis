## **MoE-Lens**: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints

Yichao Yuan University of Michigan Ann Arbor, Michigan, USA yichaoy@umich.edu

Lin Ma University of Michigan Ann Arbor, Michigan, USA linmacse@umich.edu

Nishil Talati University of Michigan Ann Arbor, Michigan, USA talatin@umich.edu

## Abstract

Mixture of Experts (MoE) LLMs, characterized by their sparse activation patterns, offer a promising approach to scaling language models while avoiding proportionally increasing the inference cost. However, their large parameter sizes present deployment challenges in resource-constrained environments with limited GPU memory capacity, as GPU memory is often insufficient to accommodate the full set of model weights. Consequently, typical deployments rely on CPU-GPU hybrid execution: the GPU handles compute-intensive GEMM operations, while the CPU processes the relatively lightweight attention mechanism. This setup introduces a key challenge: how to effectively optimize resource utilization across CPU and GPU? Prior work has designed system optimizations based on performance models with limited scope. Specifically, such models do not capture the complex interactions between hardware properties and system execution mechanisms. Therefore, previous approaches neither identify nor achieve the hardware limit.

This paper presents MoE-Lens, a high-throughput MoE LLM inference system designed through holistic performance modeling for resource-constrained environments. Our performance model thoroughly analyzes various fundamental system components, including CPU memory capacity, GPU compute power, and workload characteristics, to understand the theoretical performance upper bound of MoE inference. Furthermore, it captures the system execution mechanisms, including workload scheduling and the effects of paged KV cache, to identify the key hardware bottlenecks and accurately predict the achievable throughput. Informed by our performance model, MoE-Lens introduces an inference system featuring a resource-aware scheduler for prefill and decode phases, an execution engine that overlaps their computation, a data transfer mechanism for model weights, and an optimized CPU-based attention implementation. Evaluated on diverse MoE models and datasets, MoE-Lens outperforms the state-of-the-art MoE-Lightening by 4.6× on average (up to 25.5×), with our theoretical model predicting performance with an average 94% accuracy.

## Keywords

LLM, MoE, resource-constraint env, high-throughput serving

## 1 Introduction

The emergence of Mixture-of-Experts (MoE) models [\[11,](#page-11-0) [16,](#page-11-1) [37\]](#page-13-0) has marked a significant evolution in the design of Large Language Models (LLMs). In contrast to dense models [\[5,](#page-11-2) [20\]](#page-12-0) that activate the full set of model parameters for every input, MoE models introduce sparsity by routing each input through a small subset of expert networks. This design enables MoE models to scale up the total parameter count substantially without proportionally increasing the number of floating-point operations (FLOPs) per inference step. As a result, MoE-based LLMs have demonstrated strong empirical performance across a wide range of tasks [\[11,](#page-11-0) [15\]](#page-11-3), while maintaining manageable compute requirements for inference.

However, the practical deployment of MoE models is challenging due to their high memory capacity demand to store the model weights. Although only a few experts are activated per token, all expert parameters must reside in memory to allow flexible routing decisions at runtime. This leads to substantial memory pressure that often exceeds the capacity of a GPU. For instance, recent models such as DeepSeek-V3/R1 [\[15,](#page-11-3) [16\]](#page-11-1) and Mixtral-8x22B [\[2\]](#page-11-4) size hundreds of gigabytes, significantly outpacing the requirements of dense models with similar FLOPs. These memory capacity demands hinder the use of MoE models in resource-constrained environments, such as low-cost servers, where the available GPU memory capacity is far less than the model size.

A key technique for enabling MoE inference in resource constrained environments is CPU offloading [\[9,](#page-11-5) [38\]](#page-13-1). In this approach, the model weights and Key-Value (KV) cache are stored in CPU memory and transferred to the GPU on demand during inference. As a result, CPU-GPU IO becomes a critical bottleneck. Prior work has sought to mitigate this bottleneck through improved scheduling strategies, including pipelining [\[38\]](#page-13-1), attention offloading [\[9,](#page-11-5) [24\]](#page-12-1), and model-aware prefetching [\[19\]](#page-11-6). While these techniques have led to notable, hardware utilization remains low even in state-ofthe-art systems, leaving room for significant performance improvements. For example, we find only 16.5% of GPU utilization for MoE-Lightning [\[9\]](#page-11-5) during the generation stage. This raises important questions: what is the upper bound on achievable performance, and how can the system achieve such upper bound?

In this paper, we present MoE-Lens, a high-throughput MoE inference framework designed for resource-constrained environments, achieving up to 25.5× and an average of 4.6× speedup over state-of-the-art MoE-Lightening [\[9\]](#page-11-5). MoE-Lens's design contains three stages. First, unlike prior work that relies on limited-scope performance models for system optimizations, MoE-Lens employs a two-stage holistic performance model that considers factors beyond CPU-GPU I/O bandwidth. It not only identifies theoretical performance upper bounds but also accurately predicts the execution time. In the third stage, we propose a system design guided by this model and jointly optimize the execution pipeline and sequence-level scheduling to bring the system closer to hardware limits. Similar to prior works [\[9,](#page-11-5) [19,](#page-11-6) [38,](#page-13-1) [45\]](#page-13-2), our focus is offline, batching processing inference tasks, such as model evaluation [\[27\]](#page-12-2), data wrangling [\[31\]](#page-13-3), form processing [\[10\]](#page-11-7), LLM for relational analytics [\[30\]](#page-13-4), and synthetic data generation [\[20\]](#page-12-0), where maximizing inference throughput directly reduces total job completion time.

MoE-Lens's two-stage performance modeling accounts for critical factors that represent missed opportunities in prior works [\[9,](#page-11-5) [19,](#page-11-6) [38\]](#page-13-1) to show how they influence throughput. In the first stage, the model analyzes the theoretical performance upper bound of MoE inference based on the fundamental system components. It identifies CPU memory capacity, an element overlooked by prior work, as a primary limiting factor and quantifies how prompt and generation lengths impact the memory utilization. The second stage captures how system execution mechanisms, including workload scheduling and paged KV cache, affect memory/compute utilization and overall system performance. By integrating all these dimensions, our model accurately predicts end-to-end wall-clock inference time for systems operating near hardware limits.

In the third stage, MoE-Lens introduces a high-throughput MoE LLM inference system design informed by our holistic performance model that significantly outperforms state-of-the-art solutions. The system maximizes hardware utilization by addressing key inefficiencies in CPU-side resource usage and balancing compute across the prefill and decode stages. To this end, we introduce resourceaware scheduling that enables effective prefill/decode overlapping, reducing idle time and smoothing workload distribution. We also propose a novel pipeline design, VSLpipe, which includes a contiguous data mover to maximize CPU-GPU bandwidth utilization during weight transfers. Our hand-optimized CPU attention kernel using instrinsics fully leverages the vector units of modern CPUs, preventing the CPU compute throughput from becoming a bottleneck and improving GPU utilization.

Evaluated on diverse models and datasets, MoE-Lens achieves on average 4.6×, up to 25.5×, speedup over the state-of-the-art solution MoE-Lightning [\[9\]](#page-11-5). The results show the importance of holistic performance modeling and architecture-aware design decisions for high-throughput MoE inference in resource-constrained environments. In summary, MoE-Lens makes the following contributions.

- A holistic performance model for MoE inference in resourceconstrained environments capturing complex interactions between hardware properties and system execution mechanisms.
- An accurate throughput predictor for MoE LLM inference under hardware constraints.
- A system design informed by the model, featuring resource-aware phase scheduling, CPU-side attention execution, and efficient weight/KV cache transfer.
- MoE-Lens: an architecture-aware, end-to-end CPU-GPU hybrid system with theoretical underpinnings that achieves an average 4.6× throughput improvement over the state-of-the-art.

## 2 Background

Mixture-of-Expert (MoE) LLMs. MoE LLMs achieve strong benchmark performance while reducing compute needs compared to dense LLMs with similar parameter counts, primarily composed of attention and MoE layers. For architectural details, see [\[9,](#page-11-5) [19\]](#page-11-6). A defining trait of modern MoE models is their large size: hundreds of GBs [\[14–](#page-11-8)[16\]](#page-11-1)—which exceeds standard GPU memory [\[9,](#page-11-5) [19,](#page-11-6) [38\]](#page-13-1).

Concepts in LLM Model Inference. One key module in LLM Models is the attention module, where key and value vectors are calculated and cached in KV Cache. Group Query Attention (GQA) is a commonly used attention variant in MoE models, which allows a group of query vectors to share a single pair of key and value vectors, thereby reducing the size of the KV cache. The LLM inference consists of two stages: the prefill stage, typically compute-bound, where the initial prompt is processed in parallel, and the decode stage, typically memory-bound [\[33\]](#page-13-5), where tokens are generated sequentially in an auto-regressive manner.

Resource-Constrained LLM Inference. LLM inference in resourceconstrained environments prioritizes high-throughput batch processing on systems where the GPU memory is significantly smaller than the model's total parameter size, while the CPU has sufficient memory or disk capacity to store model weights. This setting differs from traditional LLM serving systems [\[26,](#page-12-3) [33,](#page-13-5) [36\]](#page-13-6) optimized for latency-sensitive applications like chatbots and code completion, where low response time is critical. Instead, resource-constrained inference systems, typically equipped with GPUs like T4 (16GB) or L4 (24GB) [\[9,](#page-11-5) [19,](#page-11-6) [38\]](#page-13-1), prioritize overall throughput and can afford to trade off latency. Since these GPUs lack sufficient memory to store the full model, weights must be streamed from CPU memory over PCIe, introducing substantial overhead. MoE-Lightning [\[9\]](#page-11-5) addresses this by offloading attention computation to the CPU, forming a CPU–GPU hybrid system. This design avoids transferring the large KV cache to GPU, which is essential as growing GPU parallelism leads to KV sizes exceeding memory capacity. Moreover, because attention has low arithmetic intensity [\[9,](#page-11-5) [13\]](#page-11-9), the CPU can execute it efficiently while the GPU focuses on compute-intensive MoE layers.

## 3 Motivation

While MoE-Lightning [\[9\]](#page-11-5), the state-of-the-art MoE inference system for resource-constrained environments, leverages a performance model to guide system design and achieves substantial speedups, an important question remains: does the state-of-the-art fully harness the capabilities of the underlying hardware?

## 3.1 Limited-Scope of Performance Modeling in Prior Work

MoE-Lightning introduced the Hierarchical Roofline Model (HRM) to address the CPU-GPU IO bottleneck, achieving notable gains by offloading decode-stage attention to the CPU and avoiding frequent KV cache transfers. While effective, HRM's focus is limited to arithmetic intensity and IO bandwidth, overlooking two crucial factors that influence performance ceilings: CPU memory capacity and the characteristics of input requests, such as prompt and generation lengths. These factors directly affect how much weight transfer overhead can be amortized and how well pipelining can hide data movement latency. As shown in Table [1,](#page-2-0) typical execution plans generated by MoE-Lightning result in underutilized CPU memory, revealing inefficiencies in resource allocation. Moreover, sustaining high concurrency requires not just fast IO, but also ample memory bandwidth and compute throughput on the CPU side: elements

<span id="page-2-0"></span>

| Prefill Length | Generation Length | CPU Memory (GB) | CPU Memory Utilization |
|----------------|-------------------|-----------------|------------------------|
| 98             | 32                | 265             | 52.0%                  |
| 98             | 64                | 265             | 56.2%                  |
| 926            | 128               | 265             | 35.0%                  |

Table 1: CPU memory utilization for execution plans generated by MoE-Lightening [9], showcasing under-utilization.

<span id="page-2-1"></span>![](_page_2_Figure_4.jpeg)

Figure 1: Sample of an execution timeline of GPU computation and CPU-GPU IO during the prefill and decode stages of MoE-Lightning.

HRM does not model. This presents a *significant opportunity* to rethink scheduling and architecture-aware execution strategies that better align with hardware constraints.

**Motivated Approach.** These limitations motivate the development of the *Stage 1 Model* in MoE-Lens's three-step approach, which extends beyond operator-level IO analysis to capture a more complete picture of system performance. This model establishes a theoretical upper bound on throughput for processing requests with varying prompt and generation lengths, incorporating the often-overlooked impact of CPU memory capacity.

## <span id="page-2-4"></span>3.2 Resource Utilization Imbalance between Prefill and Decode Stages.

To approach the throughput upper bound, it is essential to account for workload heterogeneity in LLM inference, which causes imbalanced resource utilization between the prefill and decode stages. As shown in Figure 1, the prefill stage fully utilizes GPU compute but leaves CPU-GPU IO bandwidth underutilized, while the decode stage suffers from low GPU utilization due to on-demand weight transfers from CPU to GPU, constrained by limited GPU memory. Existing MoE inference systems in resource-constrained settings handle these stages separately, simplifying scheduling but exacerbating the imbalance. Profiling MoE-Lightning [9] with a 98-token prompt and 32-token generation reveals this inefficiency: CPU-GPU IO is active only 23.9% of the time during prefill, while GPU utilization drops to 16.5% during decode.

Motivated Approach. This observation motivates the need to overlap prefill and decode execution to approach the hardware limit. MoE-Lens formalizes this in the Stage 2 Model, which incorporates key execution factors to make the model both resource- and workload-aware while still reflecting hardware constraints. By accounting for overlapped scheduling and system-level interactions, the Stage 2 Model aligns closely with real system behavior, enabling practical guidance for system design and accurate estimation of end-to-end execution time in MoE inference.

### <span id="page-2-5"></span>4 Overview of MoE-Lens

In this work, we aim to answer the following critical questions:

- What is the throughput upper bound for a machine?
- How to systematically approach that performance upper bound?

<span id="page-2-2"></span>![](_page_2_Figure_15.jpeg)

Figure 2: Overview of MoE-Lens that combines a theoretical performance upper bound, resource and workload aware performance model, and an informed system design to reach hardware limits.

In MoE-Lens, we take a three-step approach as shown in Figure 2, building up an increasingly detailed understanding of MoE inference in a resource-constrained environment, and eventually reach a concrete design that reaches the hardware limit. In Stage 1, we develop a model to understand the theoretical performance upper bound of MoE inference based on the fundamental system's architectural components (§5.1, §5.2). We further analyze the requirements on CPU memory bandwidth and compute throughput, which support the performance upper bound (§5.3) and the benefits of prefill/decode overlapped scheduling (§5.4). Holistically considering all factors above, and request batch size and paged KV cache, we derive our resource and workload aware performance model in Stage 2 from an implementation viewpoint (§5.5), providing realistic insights for concert design and predict system performance (with 94% accuracy). One important property of our Stage 2 model is that it converges to the performance upper bound with an increasing batch size, thus still modeling the hardware limits but pricing in the physical execution factors. Finally in Stage 3, we provide a system design based on the insights from modeling and the guidance of the Stage 2 model, which adapts to the variance of the real execution environments while achieving the high hardware utilization outlined by the Stage 2 model (§6).

## <span id="page-2-6"></span>5 MoE-Lens Performance Model

This section describes the details of our *Stage 1* and 2 models.

#### <span id="page-2-3"></span>5.1 CPU Memory Capacity as a Limiting Factor

As the CPU memory stores the KV cache, its capacity directly determines the number of sequences that can be processed in parallel. A large number of active sequences allows a large number of tokens to reuse the weights loaded from the CPU for computation, amortizing the cost of moving the weights from the CPU to the GPU. An important question is how much CPU memory is necessary to fully utilize the GPU?

Let  $N_e$  denote the number of experts,  $N_k$  the number of top-k experts selected per token token, h the model dimension,  $h_i$  the intermediate dimension of the expert networks (typically,  $h_i = mh$ , where m > 1), s the GQA group size and n as the number of tokens processed in parallel. The GEMM arithmetic-to-IO intensity, which is the amount of GEMM computation divided by the amount of

<span id="page-3-2"></span>

| -                                   | Sequence Length 256 |       | Sequence Length 512 |       |       |       |
|-------------------------------------|---------------------|-------|---------------------|-------|-------|-------|
| NVIDIA GPU                          | A40                 | L40   | A100                | A40   | L40   | A100  |
| BF16 Throughput (in T FLOPS)        | 150                 | 181   | 312                 | 150   | 181   | 312   |
| # of Tokens to Saturate GPU compute | 19.2k               | 23.2k | 40.0k               | 19.2k | 23.2k | 40.0k |
| KVCache Size to Saturate (in GB)    | 614                 | 741   | 1277                | 1228  | 1482  | 2554  |

Table 2: KV Cache Size Needed to Saturate GPU compute.

weight data accessed, of a MoE model is

$$I = n \frac{6N_k h h_i + 4h^2 + 4\frac{h^2}{s}}{6N_e h h_i + 4h^2 + 4\frac{h^2}{s}} = n \frac{6mN_k + 2 + \frac{2}{s}}{6mN_e + 2 + \frac{2}{s}} \approx n \frac{N_k}{N_e}$$
(1)

Here,  $\frac{N_k}{N_e}$  reflects the sparsity of the MoE layer. Let  $C_{GPU}$  be the GPU's GEMM throughput and  $B_{IO}$  the GPU-CPU IO bandwidth. To saturate the GPU computation power, we require

<span id="page-3-4"></span>
$$I \ge \frac{C_{GPU}}{B} \Leftrightarrow n \ge \frac{C_{GPU}}{B} \frac{N_e}{N_k} \tag{2}$$

For example, saturating the compute of a single NVIDIA A40 GPU ( $C_{\rm GPU}=150$  TFLOPS, B=32,  $N_e=8$ ,  $N_k=2$ ) when running Mixtral-8x7B requires processing 19,200 tokens in parallel. Table 2 quantifies the number of tokens and corresponding KV cache size, assuming a 512-token sequence length (sum of prompt and generation length), needed to saturate different GPUs. Even under a throughput-optimized system design, where abundant pending sequences are assumed, achieving this level of parallelism is challenging due to the **large CPU memory capacity** required to hold the KV cache. Each token contributes to the cumulative KV cache footprint, which can quickly exceed the available CPU memory in resource-constrained settings. For instance, supporting 512-token sequences would require 1.2TB of CPU memory for KV cache: disproportionate for a single GPU. This memory demand grows with increasing GPU compute capabilities, as illustrated in Table 2.

**Takeaway:** CPU memory capacity for KV cache storage is a limiting factor to fully utilize GPU compute resources.

