# Rethinking LLM Inference Bottlenecks: Insights from Latent Attention and Mixture-of-Experts

Sungmin Yun<sup>†</sup>, Seonyong Park<sup>†</sup>, Hwayong Nam<sup>†</sup>, Younjoo Lee<sup>†</sup>, Gunjun Lee<sup>†</sup>, Kwanhee Kyung<sup>†</sup>, Sangpyo Kim<sup>§</sup>, Nam Sung Kim<sup>‡</sup>, Jongmin Kim<sup>†</sup>, Hyungyo Kim<sup>‡</sup>, Juhwan Cho<sup>†</sup>, Seungmin Baek<sup>†</sup>, Jung Ho Ahn<sup>†</sup>

\*\*Seoul National University, Seoul, South Korea,

\*\*University of Illinois at Urbana-Champaign, Champaign, Illinois, USA

\*\*CryptoLab Inc., Seoul, South Korea

{sungmin.yun, seonyong.park, nhy4916, younjoo0614, kevin970401, kwanhee5,

Abstract—Computational workloads composing traditional transformer models are starkly bifurcated. Multi-Head Attention (MHA) and Grouped-Query Attention are memory-bound due to low arithmetic intensity, while FeedForward Networks are compute-bound. This dichotomy has long motivated research into specialized hardware to mitigate the attention bottleneck.

This paper argues that recent architectural advances in transformer models—Multi-head Latent Attention (MLA) and Mixture of Experts (MoE)—introduce new dominant bottlenecks, shifting the challenge away from memory-intensive attention. We make two key observations. First, the arithmetic intensity of MLA is over two orders of magnitude higher than that of MHA, moving it toward a compute-bound regime well-matched to modern accelerators such as GPUs. Second, distributing MoE experts across a pool of accelerators allows batching to tune their arithmetic intensity to that of dense layers, producing a more balanced computational profile. Consequently, the focus of hardware and system optimization should shift from attention acceleration to high-bandwidth interconnects and balancing expert workloads across accelerators.

#### I. INTRODUCTION

Transformer-based large language models (LLMs) [58] have achieved remarkable accuracy across various natural language processing tasks [10]. An LLM summarizes (or prefills) an input sequence of tokens; then, it generates output tokens one-by-one through decode steps. A conventional LLM consists of a sequence of decoder blocks, each comprising a Multi-Head Attention (MHA) or Grouped-Query Attention (GQA) sub-block and a FeedForward Network (FFN) sub-block.

To improve scalability and efficiency, recent LLM architectures have adopted optimizations such as Multi-head Latent Attention (MLA) [10], first appeared in DeepSeek-V2 [32], which reduces the memory footprint of attention. They also incorporate Mixture of Experts (MoE) [7], [15], [26], [33], [60] to increase the model capacity with multiple *experts* for the FFN sub-block, where only a subset of experts are activated per token to mitigate the compute cost [53].

Maximizing accelerator utilization when serving these models is critical for improving throughput and end-to-end latency [1]. Serving LLMs in production requires deploying systems that integrate hundreds or even thousands of accelerators (e.g., GPUs [38] and TPUs [56]) to handle high query volumes and large models [4], [48]. Inefficient utilization results in compute resources remaining idle, higher infrastructure costs, and failure to meet service-level objectives (SLOs) [68].

A key factor in serving (inferencing) LLMs efficiently is arithmetic intensity (ArI), the ratio of arithmetic operations to memory access, measured in operations per byte (Op/B). The ridge point of an accelerator, a concept from the roofline model [59], defines the ArI at which performance transitions from being *memory-bound* to *compute-bound*. To fully exploit an accelerator's capabilities, the ArI of each layer in the model should be configured to approach this ridge point.

In this paper, we argue that the introduction of MLA and MoE fundamentally reshapes the computational landscape of LLM inference. Crucially, both techniques significantly reduce per-request memory capacity and computation demands, especially in the decode stage. By introducing a latent space to attention, MLA significantly reduces the KV cache (KV\$) size, a major bottleneck in conventional LLMs. The reduced KV\$ size enables the use of much larger batch sizes. Furthermore, while MoE significantly increases the model size with multiple experts, associated computational costs do not increase proportionally due to the sparse activation of the experts.

We demonstrate how these architectural shifts in LLMs synergistically increase throughput while reducing latency compared to conventional models by operating near the accelerator's ridge point. With layer reordering applied in the decode stage of an MLA sub-block, the ArI of its core-attention layer approaches the accelerator's ridge point regardless of the batch size, by reducing the number of memory accesses. This enables significantly large batch sizes while still satisfying SLOs. Along with the reduced per-expert computation, the large batch sizes also drive the ArI of each expert's FC layers closer to the accelerator's ridge point. This, in turn, enables serving systems to support MoE layers more efficiently.

Building on these observations, we present a serving-system methodology that holistically integrates MLA and MoE within

multi-accelerator environments. We distill our findings into three design principles for large-batch inferences, aiming to maximize throughput while satisfying SLO constraints. First, attention-specialized hardware such as processing-in-memory (PIM) is no longer necessary as MLA and MoE shift attention and expert computation toward the compute-bound regime. Second, high-bandwidth interconnects (e.g., NVLinks) are indispensable for reducing latency in token dispatch and aggregation within MoE layers. Third, balancing expert workloads is critical; mitigating load imbalance across accelerators caused by skewed expert distributions [24] preserves throughput scalability. These principles guide the design of balanced, high-throughput LLM serving systems cooptimized across model, hardware, and interconnect levels.

The key contributions of this paper are as follows:

- We discover that layer reordering in MLA increases the arithmetic intensity of core-attention layer, making it approach the accelerator's ridge point.
- We demonstrate that the reduced KV-cache size in MLA and the decreased computation from sparsely activated MoE experts synergistically enable efficient serving of large batches.
- We highlight that the inference bottlenecks lie in interconnect bandwidth and load imbalance across tokens.

#### II. BACKGROUND

#### A. Standard LLM architecture and its layers

Despite rapid advancement in LLM algorithms [21], [34], transformer-based architectures [58], especially those only with decoders [6], remain the standard backbone for modern LLMs (Figure 1). A transformer consists of a series of  $n_{\rm decoder}$  decoder blocks. Given an input sequence of  $\ell$  tokens (e.g., words) for an inference (serving) request, each token is embedded into a  $d_{\rm emb}$ -dimensional hidden state for  $d_{\rm emb}$  typically on the order of thousands. This forms a hidden state matrix  $\mathbf{H}_{\ell} \in \mathbb{R}^{\ell \times d_{\rm emb}}$  that is provided as input to the decoder blocks. This matrix passes through the decoder blocks, each with its own trained weights, and is transformed into an output vector.

LLM inference consists of a prefill (summarization) stage and a decode (generation) stage. The prefill stage processes the entire input hidden state matrix ( $\ell=L_{\rm in}$ ) to generate the first output token. This stage mainly performs matrix-matrix multiplications, implemented as GEMM operations. Subsequently, the decode stage generates the remaining tokens auto-regressively, where each step takes the previously generated single token ( $\ell=1$ ) as input to produce the next one, continuing until an end-of-sequence token is generated or the output sequence reaches the maximum output token. Also, this stage is dominated by GEMV operations, as each step multiplies a single-token vector ( $\ell=1$ ) with weight matrices for a single-batch inference.

A decoder block in a conventional LLM consists of two sub-blocks: a **Multi-Head Attention** (MHA) sub-block and a **FeedForward Network** (FFN) sub-block. In the rest of the paper, we refer to a sub-block simply as a block. In MHA, a single hidden state matrix **H** (hence *self*-attention) is first

![](_page_1_Figure_10.jpeg)

<span id="page-1-0"></span>Fig. 1. Conventional transformer-decoder-based LLM architecture.

linearly transformed (projected) into  $\underline{\mathbf{Q}}$ uery (Q),  $\underline{\mathbf{K}}$ ey (K), and  $\underline{\mathbf{V}}$ alue (V) matrices, each dimensioned by a decompression dimension  $d_{\mathrm{dec}}$ , by passing through fully-connected (FC) layers with pre-trained weights. These matrices are split into  $n_{\mathrm{hd}}$  heads, each dimensioned by  $d_{\mathrm{hd}}$  (i.e.,  $d_{\mathrm{dec}} = n_{\mathrm{hd}} \cdot d_{\mathrm{hd}}$ ). Eq. 1 shows how  $\mathbf{Q}$ ,  $\mathbf{K}$ , and  $\mathbf{V}$  for each head are computed where L denotes the current sequence length, defined as the sum of the input sequence length  $L_{in}$  and the number of tokens generated so far. During auto-regressive decoding, KV cache (KV\$) stores past K and V values to maintain context without costly recomputation. Thus, only the new K and V vectors for the current input token are computed and appended to KV\$.

<span id="page-1-1"></span>
$$\underbrace{\mathbf{Q}_{i}}_{\mathbb{R}^{\ell \times \frac{d_{\text{dec}}}{n_{\text{hd}}}}} = \underbrace{\mathbf{H}_{\ell}}_{\mathbb{R}^{\ell \times d_{\text{emb}}}} \cdot \underbrace{\mathbf{W}_{\mathbf{Q}_{i}}}_{\mathbb{R}^{d_{\text{emb}} \times \frac{d_{\text{dec}}}{n_{\text{hd}}}}}$$

$$\underbrace{\mathbf{X}_{i}}_{\mathbb{R}^{L \times \frac{d_{\text{dec}}}{n_{\text{hd}}}}} = \underbrace{\mathbf{H}_{L}}_{\mathbb{R}^{L \times d_{\text{emb}}}} \cdot \underbrace{\mathbf{W}_{\mathbf{X}_{i}}}_{\mathbb{R}^{d_{\text{emb}} \times \frac{d_{\text{dec}}}{n_{\text{hd}}}}}$$
for  $\mathbf{X} \in \{\mathbf{K}, \mathbf{V}\}$  and  $i \in [1, n_{\text{hd}}]$ 

Each head independently performs a sequence of operations referred to as a core-attention layer, which computes score (Eq. 2), softmax, and context (Eq. 3) operations.

<span id="page-1-2"></span>
$$\underbrace{\mathbf{S}_{i}}_{\mathbb{R}^{\ell \times L}} = \underbrace{\mathbf{Q}_{i}}_{\mathbb{P}^{\ell \times \frac{d_{\text{dec.}}}{n_{\text{hd}}}}} \cdot \underbrace{\mathbf{K}_{i}^{T}}_{\mathbb{P}^{n_{\text{hd}} \times L}} \tag{2}$$

<span id="page-1-3"></span>
$$\underbrace{\mathbf{O}_{i}}_{\mathbb{R}^{\ell \times \frac{d_{\mathrm{dec}}}{n_{\mathrm{hd}}}}} = \underbrace{\mathrm{Softmax}(\frac{\mathbf{S}_{i}}{\sqrt{d_{\mathrm{dec}}/n_{\mathrm{hd}}}})}_{\mathbb{R}^{\ell \times \frac{d_{\mathrm{dec}}}{n_{\mathrm{hd}}}}} \underbrace{\mathbf{V}_{i}}_{\mathbf{R}^{L \times \frac{d_{\mathrm{dec}}}{n_{\mathrm{hd}}}}} \tag{3}$$

Finally, another FC layer called attention output projection follows, generating the MHA block's output U.

**Grouped-Query Attention (GQA)** has been widely adopted in modern LLMs [19], [26], [33], [60]. GQA groups multiple Q heads ( $deg_{grp}$ ) to share a single KV head within each group. GQA reduces the KV\$ size at the cost of possible reduction in accuracy [2]. While Multi-Query Attention (MQA) [51] further reduces KV\$ size by sharing a single KV head across all Q heads, this significantly degrades accuracy [2]. Hence, we exclude MOA from our analysis.

An FFN block in recent LLMs consists of three FC layers and one non-linear activation. Until GPT-3 [25], most models

used an FFN block with two FC layers and one activation. More recent LLMs employ three FC layers, which improves response quality at the cost of additional computation by introducing gated activation functions [52].

Modern LLMs commonly apply Rotary Positional Embedding (RoPE) [55] to inject positional information into token generations. RoPE encodes token positions by applying a position-dependent rotational transformation to Q and K before the core-attention layer while leaving V unchanged.

In LLM inference, multiple requests in a batch share the same model weights, allowing weight reuse and reducing memory access for FC layers. In the decode stage, batching converts FC-layer operations from GEMV to GEMM, as activations from multiple requests are multiplied jointly with the model weights. However, as each request accompanies its own KV\$, increasing the batch size *B* results in higher memory usage. As a result, the maximum feasible batch size is constrained by both memory capacity and SLOs [68].

#### B. Hardware efficiency & arithmetic intensity (ArI)

When serving LLMs, three key factors determine service latency and throughput: *arithmetic throughput, memory bandwidth*, and *memory capacity* of an accelerator. High arithmetic throughput enables fast execution of compute-intensive layers, such as GEMMs on large tensors during the prefill stage (*e.g.*, Eq. 1). To fully leverage this arithmetic capability, sufficient memory bandwidth is essential to quickly supply the data required for these computations. Such high bandwidth can only be fully utilized when the memory capacity is large enough to accommodate the entire working set; otherwise, data must be fetched from lower-bandwidth sources (*e.g.*, via a PCIe link), severely limiting performance.

ArI is useful for evaluating the expected throughput of an algorithm with ample parallelism on a given accelerator (e.g., GPU or TPU). It is the ratio of operations performed to the amount of data accessed from memory in bytes (Op/B). The ridge point of an accelerator (RPacc) [59] refers to the ArI at which performance shifts from being memorybound to compute-bound. It is calculated as the ratio of peak arithmetic throughput to peak memory bandwidth of the accelerator. If the ArI of a computation is below the ridge point, its performance is limited by memory bandwidth; its execution time is largely determined by # of memory accesses memory bandwidth. If it is above RP<sub>acc</sub> such that a sufficient number of operations are performed per unit of data to make the computation computebound, the achieved arithmetic throughput will saturate. Then, # of operations the execution time would be bounded by  $\frac{\text{\# or operations}}{\text{arithmetic throughput}}$  [59].

#### C. Model parallelism

To distribute model weights and computations across multiple accelerators [5], [14], modern LLM serving systems employ tensor parallelism (TP) and data parallelism (DP) [54]. With model size now exceeding the capacity of a single accelerator device, multi-accelerator serving has become essential. TP partitions activation, weights, or both across accelerators, allowing each to compute partial results of a single operation

![](_page_2_Figure_8.jpeg)

<span id="page-2-0"></span>Fig. 2. Time per output token (TPOT) and per-GPU throughput of GPT-3, Llama4-Maverick, and DeepSeek-R1 across various sequence lengths and batch sizes. The blurred area indicates configurations where out-of-memory errors occur. See §VI for experimental details.

in parallel; however, TP introduces inter-accelerator communication. DP replicates weights and processes independent subbatches on each accelerator to avoid the communication but requires larger memory capacity. We use  $deg_{\rm TP}$  and  $deg_{\rm DP}$  to denote the degrees of TP and DP, respectively.

