# Henry Hoffmann

University of Chicago Chicago, IL, United States hankhoffmann@cs.uchicago.edu

speedup over the best open-source baseline for 95th percentile requests. Code for SwiftSpec will be available at https://github.com/ByteDance-Seed/SwiftSpec

CCS Concepts: • Computer systems organization → Real-time system architecture; Heterogeneous (hybrid) systems; • Computing methodologies → Machine learning.

**Keywords:** large language model serving, speculative decoding, kernel optimization

#### **ACM Reference Format:**

Ziyi Zhang, Ziheng Jiang, Chengquan Jiang, Menghan Yu, Size Zheng, Haibin Lin, Xin Liu, and Henry Hoffmann. 2026. SwiftSpec: Disaggregated Speculative Decoding and Fused Kernels for Low-Latency LLM Inference. In *Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '26), March 21–26, 2026, Pittsburgh, PA, USA.* ACM, New York, NY, USA, 15 pages. https://doi.org/10.1145/3779212.3790246

#### 1 Introduction

Interactive coding assistants [5, 13, 17], robotics [47, 52], and AI-augmented search [3, 44] all need low-latency responses from large language models (LLMs). This increasingly means dedicating an entire 8-GPU node to a *single* request. For example, GitHub Copilot meets SLOs by pinning each query to multiple GPUs [10], NVIDIA's Medusa benchmark drives an 8×H200 node at 268 tokens/s for one Llama-3-70B request [31], and ServerlessLLM provisions 8–24 GPUs per request to keep tail latency below 100 ms/token [12]. Robots that use an LLM to reason and plan [36] also benefit from lower pertoken latency. In all these scenarios, it is impractical to batch the current request with other requests as in a centralized, cloud-based LLM server. Therefore, the system processes

<span id="page-1-0"></span>![](_page_1_Figure_2.jpeg)

**Figure 1.** Speculative decoding overview. (a) Conventional speculative decoding with sequential draft/verify steps. (b) Tensorparallel speculative decoding with reduced latency but communication overhead and GPU underutilization. (c) SwiftSpec: Our approach, combining disaggregated tree generation (§3.1), evolving tree cache (§3.2), and latency-optimized kernels (§3.3).

<span id="page-1-1"></span>**Table 1.** Time per inference vs. tensor parallelism when serving Llama3 models using vLLM [19].

|            | # gpus=1 | # gpus=2 | # gpus=4 | # gpus=8 |
|------------|----------|----------|----------|----------|
| Llama3-1B  | 1.58ms   | 1.58ms   | 1.73ms   | 1.72ms   |
| Llama3-3B  | 2.77ms   | 2.61ms   | 2.80ms   | 3.27ms   |
| Llama3-8B  | 4.30ms   | 3.46ms   | 3.50ms   | 3.78ms   |
| Llama3-70B | 24.78ms  | 15.90ms  | 11.86ms  | 11.22ms  |

one request at a time. Given that single-request, full-node deployment is already practiced, the scientific question becomes: what architectural and system bottlenecks limit per-request performance, and how can we redesign multi-GPU runtime and kernels to push single-request latency even lower?

In LLM-serving, there is an inherent trade-off between throughput and latency. This paper studies how to utilize all on-server GPUs to achieve ultra-low latency decoding in single-request scenarios, where existing serving frameworks—designed to maximize throughput—often fall short. From a system architecture viewpoint, understanding the performance bottlenecks of such a single-request regime reveals design principles for future AI acceleration. Specifically, it is challenging to efficiently combine speculative decoding with tensor parallelism.

**Speculative decoding** [2, 4, 20, 28] accelerates single-request LLM inference by splitting the process into two distinct phases: draft and verification. During the draft phase, a small *draft* model rapidly generates a sequence of candidate tokens (and, in some variants, a tree-structured set of candidates). During the subsequent verification phase, a much larger *target* model validates all candidates through batch inference. This process emits multiple tokens at once, reducing decoding latency. Prior work typically treats the draft and verification phases as strictly sequential operations because of their data dependencies [2, 21, 28], as shown in Figure 1(a). This design places the draft phase on the critical path as an additional overhead, preventing speculative decoding from fully realizing its latency-reduction potential.

**Tensor parallelism** [35] reduces decoding latency by scaling computation resources. Tensor parallelism partitions the model weights across multiple GPUs and then aggregates partial results through all-reduce operations. However, a straightforward combination of tensor parallelism with speculative decoding, as shown in Figure 1(b), is ineffective. In speculative decoding, the draft and target models are co-located on the same devices. Because the two models differ greatly in size, applying the same degree of tensorparallelism to both does not produce minimal system latency. The smaller draft model reaches the point of diminishing returns sooner: once its weights are finely sharded, further increasing the tensor-parallelism no longer reduces latency, because other overheads—most notably inter-GPU communication—dominate. Crucially, Table 1 shows that draft and target models have very different GPU-scaling curves, motivating our decision to disaggregate them.

To effectively combine speculative decoding with tensor parallelism, we redesign the speculative decoding process in an asynchronous, disaggregated manner (Figure 1(c)). Rather than co-locate on the same hardware, we dedicate separate GPUs for draft and target models. The draft and target models work in parallel: while the target verifies iteration n-1, the draft produces candidates for iteration n. When a verification iteration is complete, the target synchronizes the validated tokens with the draft and obtains the next set of candidate tokens. Under this design, (1) the draft and target models can be flexibly scaled to different degrees of parallelism, and (2) the dependencies between the two models are decoupled, removing the draft phase from the critical path.

Realizing this design poses three challenges. First, while the target model is validating the current iteration, the draft model must generate candidates for the next iteration. Second, maintaining key-value cache consistency between complex draft models (e.g., tree-structured draft models) and the target model is non-trivial. When tree-based draft generation runs in parallel, newly accepted tokens may force the draft model to discard invalid branches, but the draft and target KV caches should be consistent, and the valid branches

should be preserved to avoid recomputation. Third, hiding communication latency during decoding is challenging under tensor parallelism because it is hard to overlap all-reduce operations with compute operations since they usually remain on the critical path. Furthermore, the GPU kernels are usually optimized for higher throughput and have suboptimal performance under low batch sizes, spending significant time on data movement and kernel launch.

We address these challenges with SwiftSpec, a novel system providing ultra-low latency LLM decoding for singlerequest scenarios (Figure 1(c)). Specifically, SwiftSpec introduces: (a) disaggregated tree generation, which runs the draft and target models on disjoint GPUs, allowing each model to scale according to its own compute requirements. While the target model verifies one batch, the draft model produces future candidate tokens, ensuring high GPU utilization. This requires us to implement (b) Evolving tree cache with synchronization and maximized reuse. After each verification step, SwiftSpec reorganizes both the draft and target models' KV Caches for consistency. For the draft model, we keep the accepted and future tokens consistent with the draft tree, even when some guesses are incorrect. This approach also maximizes reuse of the previously computed KV cache values. (c) Latency-optimized kernels: we reduce synchronization and data transfers, accelerating inference in low-batch scenarios. Using the NVIDIA Collective Communication Library's Low Latency (NCCL LL) protocol, we develop a fused GEMM with all-reduce and an attention operator without any explicit synchronization barriers. We further decrease latency by fusing operations in the Switched Gated Linear Unit (SwiGLU) [34].

We evaluate SwiftSpec using five different model families and six different datasets. SwiftSpec consistently outperforms the baselines, achieving an average of 1.75× decoding speed over the best baseline using 8 H800 GPUs. As a highlight, SwiftSpec serves Llama3-70B with an average decoding speed of 347 output tokens/s, higher than the 268 token/s NVIDIA reports using 8 H200s [31]. SwiftSpec not only improves average speed, but is faster than baselines across the *entire* range of requests when serving Llama-3-70B (thus reducing tail latency and demonstrating performance stability across queries). In summary, our contributions are:

- Identifying scalability challenges of speculative decoding under tensor parallelism.
- Presenting SwiftSpec, which integrates disaggregated tree generation, an evolving tree cache, and latencyoptimized kernels to support speculative decoding in an asynchronous, disaggregated manner.
- Demonstrating, to our knowledge, the first LLM speculative decoder to achieve 300+ tokens/s on a full 8-GPU Nvidia Hopper node for serving single user requests using Llama3-70B model, validating the practical feasibility of such deployment.

