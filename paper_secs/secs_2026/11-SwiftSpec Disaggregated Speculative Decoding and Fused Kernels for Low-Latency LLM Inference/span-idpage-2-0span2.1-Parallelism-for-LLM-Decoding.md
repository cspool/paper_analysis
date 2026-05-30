# <span id="page-2-0"></span>2.1 Parallelism for LLM Decoding

LLM inference (e.g., [32, 40, 41]) is typically accelerated through model parallelism, which distributes computation across multiple GPUs. The two most common strategies are intra-operator parallelism (such as tensor parallelism, which splits matrix multiplications across GPUs) [35] and inter-operator parallelism (such as pipeline parallelism, which distributes layers across devices) [11, 16].

In single-request serving, tensor parallelism (TP) dominates. Unfortunately, applying TP to speculative execution is challenging because small draft models gain relatively little from TP, while large target models benefit more from increased TP (see Table 1). We synthesise these small-batch constraints and the resulting design opportunities in §2.4.

#### <span id="page-2-1"></span>2.2 GPU Constraints for Low Batch Size

Modern GPUs are often optimized for large-batch throughput workloads. In small-batch inference, e.g., interactive LLM serving, performance suffers due to kernel under-utilization and communication overhead. For transformers, three operators dominate: GEMM, attention [42], and all-reduce.

Unfortunately, small batch size performance is a requirement for single-request serving. While the details are omitted for space, we find that increasing the draft or target model batch size beyond 16 results in minor performance gains (< 5%) because the draft model no longer generates useful tokens and the time spent verifying them is wasted. Under such small batch sizes ( $\le 16$ ), the GEMM, attention, and all-reduce operators exhibit poor efficiency. Some targeted optimizations, such as low-bit quantization [23, 43], reduce communication costs but are not a holistic solution.

Similarly, state-of-the-art LLM serving frameworks have reduced communication costs for small sizes by using the NVIDIA collective communication library (NCCL) all-reduce operator [19, 50]. NCCL-LL offers fine-grain collectives, but prior systems do not fuse them with computation. SwiftSpec instead introduces latency-optimized kernels (§3.3) that (1) combine GEMM and all-reduce operators and (2) combine attention computation and communication to greatly reduce overhead and enable fine-grained communication.

#### 2.3 Tree-based Speculative Decoding

Speculative decoding [20, 27, 28] accelerates large language model inference by employing a smaller, faster draft model to rapidly propose candidate tokens ahead of the main model. The larger target model then verifies these candidates in parallel using a batch inference, accepting those that align with its own probability distribution while discarding incorrect ones. This process allows the system to generate multiple tokens for every single invocation of the computationally expensive target model, significantly reducing overall latency. Prior works denote *average acceptance length* [2, 20, 21] as

<span id="page-3-1"></span>

|                   |                                      |               |            | SOTA LLM engines |           |
|-------------------|--------------------------------------|---------------|------------|------------------|-----------|
|                   | Feature                              | PipeInfer [1] | PEARL [24] | [30, 31, 37, 50] | SwiftSpec |
| (1) Independent   | Tree-based speculation               | ✓             | Х          | ✓                | ✓         |
| scalability &     | Draft/target on disjoint compute     | ✓             | ✓          | X                | ✓         |
| parallelism       | Flexible draft/target GPU allocation | ×             | ×          | X                | ✓         |
| (2) Consistent    | Fine-grained KV reuse (zero waste)   | ×             | ✓          | ✓                | ✓         |
| KV-cache reuse    | Robust to misprediction (re-rooting) | ×             | ✓          | ✓                | ✓         |
| (3) Small-batch   | GEMM & Atten. fused w/ NCCL-LL       | Х             | Х          | Х                | ✓         |
| kernel efficiency | Fused SwiGLU                         | ×             | ×          | limited          | ✓         |

Table 2. Comparison of SwiftSpec (last column) and prior speculative decoding techniques.

the average number of tokens verified per target inference. Average acceptance length directly impacts the end-to-end latency, as higher lengths reduce the number of target model invocations.

Conventional speculative decoding approaches impose dependencies between the draft and target models. Sequence-based speculative decoding uses a pipeline: running the draft model to generate the next guess while the target model verifies the preceding guess [24, 27?]. z Sequence-based methods, however, typically exhibit lower-quality guesses compared to tree-based approaches [28] where the draft model proposes a tree of candidate tokens and the target model verifies a path through this tree. This approach tends to achieve higher *average acceptance length* since it considers more possible future tokens per output position.

However, tree-based methods impose more system complexity. First, while sequential approaches generate the single most probable next token, it is challenging to concurrently generate tree nodes with high verification probability. Second, KV cache management requires consistency between target-accepted tokens and potentially useful but unverified draft tokens. SwiftSpec addresses these timitations through disaggregated tree generation and an evolving tree cache with maximized reuse (§3.1–3.2).

