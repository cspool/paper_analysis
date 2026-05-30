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

# III. EMERGING LLM OPTIMIZATIONS: IMPACT ON PERFORMANCE AND MEMORY CAPACITY DEMAND

For efficient LLM serving, a myriad of optimizations have been proposed, spanning both algorithmic strategies and hardware-aware techniques, in response to ever-growing demand for LLM services. Contrary to conventional LLMs, DeepSeek-R1 [10] leverages **Multi-head Latent Attention** (**MLA**) and **Mixture of Experts** (**MoE**). Figure 2 shows that DeepSeek-R1 delivers up to 41× and 2× higher perdevice throughput, and 30% and 34% lower time per output token (TPOT) than GPT-3 and Llama4-Maverick, respectively, even with 3.8× and 1.7× larger model capacity (number of parameters). We provide a high-level explanation of two key **R**easons why DeepSeek-R1 offers higher throughput and lower latency compared to conventional LLMs.

(R1) Reduced core-attention layer latency in MLA: Figure 3 shows that the core-attention layer of GPT-3, based on MHA, presents an ArI of approximately 1 even with batching, making it the primary bottleneck preventing throughput and latency improvements during the decode stages. Unlike FC layers, which amortize the cost of accessing their weights by reusing them across multiple batched requests, the coreattention layer operates on KV\$ and attention score values, which are unique to each request. Thus, these values cannot be shared across batched requests, limiting ArI to 1 regardless of B in MHA. Because of this extremely low ArI, performance is bounded by memory bandwidth, leading to severe underutilization of compute resources. GQA reduces the number of memory accesses by sharing KV\$ across multiple queries, thereby slightly improving the ArI; in Llama4-Maverick, each KV\$ is shared among five queries. Still, the operations remain largely bounded by memory bandwidth.

By contrast, the core-attention layer of DeepSeek-R1, based on MLA, significantly reduces the number of memory accesses

<span id="page-3-3"></span>

|                           |        |        |        | V100 SXM2 [35] A100 SXM4 [36] H200 SXM5 [39] B200 SXM6 [41] TPU V5P [57] TPU V7 [56] MI325X [3] |      |        |        |
|---------------------------|--------|--------|--------|-------------------------------------------------------------------------------------------------|------|--------|--------|
| BF16 throughput (TFLOPS)  | 125    | 312    | 989.5  | 2250                                                                                            | 459  | 2307   | 1307.4 |
| Memory bandwidth (GB/s)   | 900    | 2039   | 4800   | 8000                                                                                            | 2765 | 7400   | 6000   |
| HBM capacity per GPU (GB) | 32     | 80     | 141    | 192                                                                                             | 95   | 192    | 256    |
| Ridge point (BF16)        | 138.89 | 153.02 | 206.15 | 281.25                                                                                          | 166  | 320.42 | 217.9  |

![](_page_3_Figure_2.jpeg)

<span id="page-3-0"></span>Fig. 3. Roofline plot of layers in decoder block at sequence length L = 4096, obtained from real-machine measurements on an NVIDIA H100 GPU.

and brings the ArI close to the ridge point of modern accelerators (RPacc) through layer reordering (detailed in [§IV\)](#page-3-1) as shown in Figure [3.](#page-3-0) This not only reduces the core-attention layer latency but also increases compute utilization during the decode stages.

(R2) Larger *B* and less computation for MoE: Larger batch size (B) improves the ArI of memory-bound FC layers, enhancing compute utilization of accelerators and maximizing throughput with negligible latency increase. However, B is limited by the KV\$ size, which grows with sequence length. In GPT-3, the KV\$ size of a request reaches 9 GB for a sequence length of 2048 (detailed in [§VII\)](#page-9-1). The large KV\$ size limits B, leaving FC layers memory-bound, reducing throughput and underutilizing compute resources [\[23\]](#page-13-9), [\[62\]](#page-14-18).

In contrast, DeepSeek-R1 requires more memory to store all model parameters, but it uses 67× smaller KV\$ per token compared to GPT-3, as shown in Figure [4.](#page-3-2) This reduction enables larger B and higher throughput under the same memory budget. Also, while GPT-3 performs computations with all 175B parameters for every token, DeepSeek-R1, leveraging MoE, activates only 8 out of 256 experts per token. As a result, it utilizes just 37B out of 671B total parameters. This sparsity significantly reduces the per-token computation cost, making DeepSeek-R1 more suitable for large-batch inference.

Llama4-Maverick uses 2.84× larger KV\$ per token than Deepseek-R1. Given that the hidden dimension of Llama4- Maverick is 5120, whereas that of DeepSeek-R1 is 7168, we observe that MLA effectively reduces the KV\$ size compared to GQA—by approximately 4× for the same hidden dimension. Consequently, for a sequence length of 8192, Llama4- Maverick cannot support a batch size of 128 per GPU, whereas DeepSeek-R1 can, despite its larger model size (see Fig. [2\)](#page-2-0).

![](_page_3_Figure_8.jpeg)

<span id="page-3-2"></span>Fig. 4. Comparison of GPT-3, Llama4-Maverick, and DeepSeek-R1 for 8M tokens with BF16 precision: the amount of parameters accessed and computed per token (Activated) and the total memory capacity required for storing attention weights, FFN/MoE weights, and KV\$ (Total).

<span id="page-3-4"></span>TABLE II SYMBOLS USED THROUGHOUT THIS PAPER, THEIR DESCRIPTIONS, AND THE EXEMPLAR PARAMETERS USED IN DEEPSEEK-R1 [\[10\]](#page-12-0)

| Term  | Description                                    | DeepSeek-R1 |
|-------|------------------------------------------------|-------------|
| demb  | Embedding dimension                            | 7168        |
| nhd   | Number of heads                                | 128         |
| dhd   | Head dimension                                 | 128         |
| ddec  | Decompressed Q/K/V dimensions, nhd · dhd 16384 |             |
|       | dQco, dKVco Compressed Q, KV dimensions        | 1536, 512   |
| dRoPE | Rotary Positional Embedding dimension          | 64          |
| dFFN  | FFN intermediate dimension                     | 18432       |
| dMoE  | MoE intermediate dimension                     | 2048        |
| ne    | Number of routed experts                       | 256         |
| nk    | Number of routed experts per token             | 8           |

Lastly, further increasing B drives the ArI beyond the RPacc and makes the FC layers compute-bound, hence increasing latency without a corresponding throughput improvement. Thus, any changes to an algorithm and/or its operating parameters, such as B, must consider the RPacc of the target serving system. The latest B200 GPU, for instance, provides approximately 18× higher arithmetic throughput than a V100 GPU, representing the most significant improvement among the accelerators listed in Table [I.](#page-3-3) Nevertheless, since both arithmetic throughput and memory bandwidth have scaled with the technology, the B200's RPacc increases by only a modest factor of 2 compared to the V100's; contemporary accelerators exhibit RPaccs within a narrow range of 200–400 Op/B.

## <span id="page-3-1"></span>IV. INSIGHTS ON MULTI-HEAD LATENT ATTENTION

We analyze the computational characteristics of an MLA block, which comprises multiple FC layers and a core-attention layer, using the symbolic definitions in Table [II.](#page-3-4)

#### *A. Introducing a latent space to attention*

MLA employs a low-rank joint compression for the attention block, primarily reducing KV\$ capacity requirements while also lowering projection weight size and the associated

![](_page_4_Figure_0.jpeg)

<span id="page-4-0"></span>Fig. 5. Computation flow of multi-head latent attention (MLA) with/without layer reordering. ⓐ to ⓑ refer to the layers of MLA (e.g., ⓐ: QKV compression, ⓑ: Q RoPE, ⓒ: Q decompression, ⓓ: K decompression, ⓔ: V decompression, ⓓ: Score, ②: K RoPE, ⓑ: Context).

FC layer computations. It first compresses a hidden state matrix, mapping it to a lower-dimensional latent space (a) in Figure 5) to form a compressed Q ( $\mathbf{C}_{\mathrm{Q}}$ ) and a compressed KV ( $\mathbf{C}_{\mathrm{KV}}$ ) through projection using  $\mathbf{W}_{\mathrm{CQ}}$  and  $\mathbf{W}_{\mathrm{CKV}}$  (Eq. 4). The resulting  $\mathbf{C}_{\mathrm{Q}}$  and  $\mathbf{C}_{\mathrm{KV}}$  are then decompressed (©, d), and (e) through projections using the corresponding decompression weights  $\mathbf{W}_{\mathrm{DX}_i}$ , where  $\mathbf{X} \in \{\mathbf{Q}, \mathbf{K}, \mathbf{V}\}$ , to reconstruct the higher-dimensional full  $\mathbf{Q}_i$ ,  $\mathbf{K}_i$ , and  $\mathbf{V}_i$ . These are then used to perform the same core-attention layer as in a standard MHA block (Eq. 5 and Eq. 6).

<span id="page-4-1"></span>
$$\underbrace{\mathbf{C}_{\mathbf{Q}}}_{\mathbb{R}^{\ell \times d_{\mathbf{Qco}}}} = \mathbf{H}_{\ell} \cdot \underbrace{\mathbf{W}_{\mathbf{CQ}}}_{\mathbb{R}^{d_{\mathbf{emb}} \times d_{\mathbf{Qco}}}}, \underbrace{\mathbf{C}_{\mathbf{KV}}}_{\mathbb{R}^{L \times d_{\mathbf{KVco}}}} = \mathbf{H}_{L} \cdot \underbrace{\mathbf{W}_{\mathbf{CKV}}}_{\mathbb{R}^{d_{\mathbf{emb}} \times d_{\mathbf{KVco}}}}$$
(4

<span id="page-4-2"></span>
$$\mathbf{S}_{i} = \mathbf{Q}_{i} \cdot (\mathbf{K}_{i})^{\mathrm{T}} = (\mathbf{C}_{\mathrm{Q}} \cdot \underbrace{\mathbf{W}_{\mathrm{DQ}_{i}}}_{\mathbb{R}^{d_{\mathrm{Qco}} \times \frac{d_{\mathrm{dec}}}{n_{\mathrm{hd}}}}} \cdot (\mathbf{C}_{\mathrm{KV}} \cdot \underbrace{\mathbf{W}_{\mathrm{DK}_{i}}}_{\mathbb{R}^{d_{\mathrm{KVco}} \times \frac{d_{\mathrm{dec}}}{n_{\mathrm{hd}}}}})^{\mathrm{T}}$$
(5

<span id="page-4-3"></span>
$$\mathbf{O}_{i} = \operatorname{Softmax}(\frac{\mathbf{S}_{i}}{\sqrt{d_{\operatorname{dec}}/n_{\operatorname{hd}}}}) \cdot (\mathbf{C}_{\operatorname{KV}} \cdot \underbrace{\mathbf{W}_{\operatorname{DV}_{i}}}_{\mathbb{R}^{d_{\operatorname{KVco}} \times \frac{d_{\operatorname{dec}}}{n_{\operatorname{hd}}}}})$$

The attention block's parameter footprint shrinks substantially both in absolute size and as a fraction of total model parameters. Employing the low-rank joint compression reduces the weight used to generate **K** and **V** from  $d_{\rm emb} \times d_{\rm dec} = 7K \times 16K = 112M$  to  $d_{\rm emb} \times d_{\rm KVco} + d_{\rm KVco} \times d_{\rm dec} = 7k \times 0.5K + 0.5K \times 16K = 11.5M$  (see Table II for notations). With this reduced weight burden, replicating its FC layer parameters across devices is far more tractable, making DP for the attention block a compelling choice.

For conventional LLMs, a key limiting factor of the batch size is KV\$ size. MLA introduces a latent space on attention, drastically reducing the KV\$ ( $C_{\rm KV}$ ) storage (Figure 4). For conventional LLMs,  $d_{\rm dec}(=d_{\rm emb})$ -dimensioned K and V are cached per layer. In DeepSeek-R1, K and V share the same compressed cache with a dimension of  $d_{\rm KVco}+d_{\rm RoPE}$ , which is usually an order of magnitude smaller than both  $d_{\rm emb}$  and  $d_{\rm dec}$ . In GPT-3, KV\$ consumes 4.5MB (=  $d_{\rm dec} \times n_{\rm decoder} \times$  (K & V)×FP16 = 12288×96×2×2B) per token, whereas  $C_{\rm KV}$  in DeepSeek-R1 requires only 68.6KB (=  $(d_{\rm KVco}+d_{\rm RoPE}) \times n_{\rm decoder} \times$  BF16 = 576×61×2B) per token.

<span id="page-5-1"></span>TABLE III COMPARISON OF MLA WITH AND WITHOUT THE REORDERING IN TERMS OF FLOPS, MEMORY ACCESS, AND ARI FOR PREFILL AND DECODE STAGES ASSUMING  $B,L\gg n_{\rm hd},d_{\rm hd}$ .

| Layer    | Phase        | Reordering | FLOPs                                 | Asymptotic Memory Access                              | ArI                                                  | ArI in DeepSeek-R1 |
|----------|--------------|------------|---------------------------------------|-------------------------------------------------------|------------------------------------------------------|--------------------|
| Prefill  | K decompress | without    | $B2Ld_{\rm KVco}n_{\rm hd}d_{\rm hd}$ | $2BLn_{\rm hd}d_{\rm hd}$                             | $\approx d_{\rm KVco}$                               | $\approx 512$      |
|          |              | with       | $B2Ld_{\rm KVco}n_{\rm hd}d_{\rm hd}$ | $2B(Ln_{\rm hd}d_{\rm hd} + Ln_{\rm hd}d_{\rm KVco})$ | $\approx (d_{\rm hd}^{-1} + d_{\rm KVco}^{-1})^{-1}$ | ≈ 100              |
| 1 ICIIII | Score        | without    | $B2n_{\mathrm{hd}}L^2d_{\mathrm{hd}}$ | $2Bn_{\rm hd}L^2$                                     | $\approx d_{\rm hd}$                                 | ≈ 128              |
|          |              | with       | $B2n_{\rm hd}L^2d_{\rm KVco}$         | $2Bn_{\rm hd}L^2$                                     | $\approx d_{\rm KVco}$                               | $\approx 512$      |
| Decode   | K decompress | without    | $B2d_{\rm KVco}Ln_{\rm hd}d_{\rm hd}$ | $2BLd_{\rm dec}$                                      | $\approx d_{\rm KVco}$                               | $\approx 512$      |
|          |              | with       | $B2d_{\rm KVco}n_{\rm hd}d_{\rm hd}$  | $2Bd_{\rm KVco}n_{\rm hd}$                            | $\approx d_{\rm hd}$                                 | ≈ 128              |
|          | Score        | without    | $B2n_{\rm hd}Ld_{\rm hd}$             | $2Bn_{\rm hd}d_{\rm hd}L$                             | $\approx 1$                                          | ≈ 1                |
|          | Score        | with       | $B2n_{\rm hd}Ld_{\rm KVco}$           | $2B(d_{\mathrm{KVco}}L + n_{\mathrm{hd}}L)$           | $\approx (n_{\rm hd}^{-1} + d_{\rm KVco}^{-1})^{-1}$ | ≈ 100              |

Smaller KV\$ size allows larger batch sizes for FC layers, which improve compute utilization. However, MLA shares the same core-attention layer structure as conventional MHA, suffering from extremely low ArI ( $\approx$  1). Also,  $C_{\rm KV}$  decompression needs to be computed on demand during runtime.

Nevertheless, MLA enables *layer reordering* (or simply *reordering*) [32] to improve data reuse by rearranging the layers in the attention block. One of MLA's key features is decoupled RoPE: Instead of applying RoPE directly to Q and K, RoPE is computed separately, the result of which is added element-wise in the score layer. By removing the nonlinearity between the QKV generation layer and the coreattention layer, Eq. 5 can be algebraically rewritten:

<span id="page-5-2"></span>
$$\mathbf{S}_{i} = \mathbf{Q}_{i} \cdot (\mathbf{C}_{KV} \cdot \mathbf{W}_{DK_{i}})^{T}$$

$$= \mathbf{Q}_{i} \cdot (\mathbf{W}_{DK_{i}}^{T} \cdot \mathbf{C}_{KV}^{T})$$

$$= (\mathbf{Q}_{i} \cdot \mathbf{W}_{DK_{i}}^{T}) \cdot \mathbf{C}_{KV}^{T}$$

$$\mathbf{S} = \begin{bmatrix} \mathbf{Q}_{1} \cdot \mathbf{W}_{DK_{1}}^{T} \\ \mathbf{Q}_{2} \cdot \mathbf{W}_{DK_{2}}^{T} \\ \vdots \\ \mathbf{Q}_{n_{hd}} \cdot \mathbf{W}_{DK_{n_{hd}}}^{T} \end{bmatrix} \cdot \mathbf{C}_{KV}^{T}$$
(7)

A similar reordering can also be applied to the context layer:

$$\begin{aligned} \mathbf{O}_{i} &= \operatorname{Softmax}(\frac{\mathbf{S}_{i}}{\sqrt{d_{\operatorname{dec}}/n_{\operatorname{hd}}}}) \cdot (\mathbf{C}_{\operatorname{KV}} \cdot \mathbf{W}_{\operatorname{DV}_{i}}) \\ &= (\operatorname{Softmax}(\frac{\mathbf{S}_{i}}{\sqrt{d_{\operatorname{dec}}/n_{\operatorname{hd}}}}) \cdot \mathbf{C}_{\operatorname{KV}}) \cdot \mathbf{W}_{\operatorname{DV}_{i}} \end{aligned} \tag{8}$$

#### B. Impact of reordering MLA

The *reordering*<sup>1</sup> optimization improves hardware utilization and drastically reduces the latency of attention during the decode stage. However, it rather increases the latency during the prefill stage. This is because reordering in MLA significantly changes the computational characteristics, such as FLOPs, the number of memory accesses, and ArI.

Table III compares the FLOPs and memory requirements in K decompression and score layers across both prefill and decode stages when a single accelerator is used, with and without reordering. V decompression and context layers exhibit similar trends regardless of reordering because  $\mathbf{K}$  and  $\mathbf{W}_{\mathrm{DK}}$  share the same structure with V and  $\mathbf{W}_{\mathrm{DV}}$ , respectively. This analysis leads to a number of notable observations.

Without reordering, the core-attention layer in MLA preserves the same computational flow as MHA, except for the runtime compression and decompression layers, which generate Q, K, and V. The required amounts of computations (FLOPs) and memory accesses for the K decompression and score layers are proportional to B and L during the decode stage (Table III), whereas those of the other FC layers do not scale with L. As a result, KV decompression and coreattention layers dominate the execution time of an attention block as L increases (Figure 6(b)). At B = 128 and L = 4096, KV decompression and core-attention layers account for 59% and 40% of the attention block latency, respectively.

*Observation-1:* In the decode stages of MLA without reordering, KV decompression and core-attention layers dominate the runtime of MLA's attention blocks.

After applying layer reordering, MLA multiplies  $\mathbf{W}_{\mathrm{DK}}$  with  $\mathbf{Q}$ . Because the size of  $\mathbf{Q}$  is independent of L in the decode stage, layer reordering eliminates the need to decompress the entire  $\mathbf{C}_{\mathrm{KV}}$ , reducing the cost of  $\mathbf{K}$  decompression by a factor of L (Table III). The portion of  $\mathbf{K}$  decompression—previously the dominant component—has been significantly reduced due to reordering (see Figure 6(c)). By contrast, this benefit disappears in the prefill stage as the size of  $\mathbf{Q}$  is proportional to  $L = L_{\mathrm{in}}$ , leaving the computational cost of  $\mathbf{K}$  decompression unchanged.

Layer reordering increases computations required for the score layer, which is a part of the core-attention layer, by  $d_{\rm KV^{co}}/d_{\rm hd}$  (4 for DeepSeek-R1) times in both prefill and decode stages. The original score layer is replaced by a multiplication between  $\mathbf{Q}_i \cdot \mathbf{W}_{\rm DK_i}^{\rm T}$  and  $\mathbf{C}_{\rm KV}^{\rm T}$  (Eq. 7), where one of the matrix dimensions becomes  $d_{\rm KVco}$  instead of  $d_{\rm hd}$ .

<span id="page-5-0"></span><sup>&</sup>lt;sup>1</sup>Although the DeepSeek papers [11], [32] refer to this technique as "absorption," we use this term to distinguish it from weight fusion, which merges multiple weight matrix multiplications into a single computational step.

![](_page_6_Figure_0.jpeg)

![](_page_6_Figure_1.jpeg)

<span id="page-6-0"></span>Fig. 6. (a) Normalized latency of the attention block in the decode stage without reordering compared to a reordered attention block. (b) and (c) show the execution time ratio of each layer in the attention block in the decode stage with and without reordering, across varying sequence length and batch size. All experiments assumed a 32 B200 GPU system. See §VI for our experimental setup in detail.

**Observation-2:** While layer reordering maintains or reduces the FLOPs of KV decompression, it increases the FLOPs of core-attention layers in both prefill and decode stages.

With layer reordering in the decode stage, the core-attention layer reads  $C_{\rm KV}$  instead of decompressed KV\$, reducing memory access by  $d_{\rm dec}/d_{\rm KVco}$ . Because  $C_{\rm KV}$  can be shared among the heads, the ArI of both score and context layers reaches approximately  $n_{\rm hd}d_{\rm KVco}/(n_{\rm hd}+d_{\rm KVco})$  ( $\approx 100$  for DeepSeek-R1) in the decode stages.

FlashMLA [27], a GPU-optimized implementation, further doubles this Op/B by reusing  $C_{\rm KV}$  loaded during the score layer in the subsequent context layer. The resulting doubled ArI (e.g.,  $\approx$  200 Op/B in DeepSeek-R1) closely approaches the ridge point RP $_{\rm acc}$  of modern accelerators, exhibiting a balance between computation and memory bandwidth for modern accelerators.

Despite using a latent space, explicitly generating decompressed  $\mathbf{K}$  and  $\mathbf{V}$  requires a significant amount of memory for activation, limiting B as shown in Figure 6. For DeepSeek-R1, decompressing the  $\mathbf{K}$  tensor with a per-accelerator batch size of 256 and L=4096 inflates the activation footprint to  $\approx 50 \mathrm{GB}$ . As layer reordering shrinks the size of this on-the-fly activation, the maximum feasible B increases. The increased B delivers proportionally higher throughput on FC layers.

**Observation-3:** Layer reordering improves hardware utilization by increasing the ArI of the core-attention layer to approach  $\mathrm{RP}_{\mathrm{acc}}$  of modern accelerators in the decode stage and by enabling sufficient batching for FC layers.

As the ArI of the core-attention layer, which dominates the runtime of attention blocks, approaches  $RP_{\rm acc}$ , the time spent on computations remains approximately equal to the time spent on accessing  $C_{\rm KV}$ . Thus, the latency of the core-attention layer is reduced approximately by  $2d_{\rm dec}/d_{\rm KVco}$  (=64 for DeepSeek-R1) after layer reordering.

In contrast, reordering increases the latency of attention blocks during the prefill stage. Layer reordering expands the

![](_page_6_Figure_10.jpeg)

<span id="page-6-1"></span>Fig. 7. Computation flow of an MLA attention block with  $deg_{\mathrm{TP}}=2$  and  $n_{\mathrm{hd}}=4$  on our experimental setup.

activation's dimension per head from  $d_{\rm hd}$  to  $d_{\rm KVco}$ . For the KV decompression layer, while the FLOPs remains unchanged, the number of memory accesses increases, leading to longer execution times. Also, the FLOPs increases for the coreattention layer (Obs.2).

Putting it all together, layer reordering significantly reduces the latency of attention blocks in the decode stage by up to  $103.12\times$  (see Figure 6). By contract, the attention block latency in the prefill stage slows down by up to  $2.21\times$  with layer reordering. Hereafter, the prefill stage uses MLA without reordering and the decode stage uses MLA with reordering.

**Observation-4:** Layer reordering substantially reduces attention block latency by reducing memory access and drastically mitigating the impact of KV decompression overhead.

#### C. Parallelism on MLA

TP offers little latency benefit once layer reordering is applied to the attention block. Although reordering drastically lowers the core-attention layer latency, the latency still scales

![](_page_7_Figure_0.jpeg)

<span id="page-7-0"></span>Fig. 8. Latency of the attention block with and without reordering as batch sizes and degTP vary when L= 4096.

with B and L in the decode stage and becomes dominant at large B and L values (Figure [6\(](#page-6-0)c)). When heads are independent, as in MHA, head-wise TP can distribute the KV\$ across multiple accelerators, reducing latency. Also, the ArI is preserved within a head as it is not affected by TP.

In the reordered MLA, however, all heads share the same CKV; thus, all accelerators need to store and read the whole CKV, nullifying the performance and capacity gains from scaling out to degTP accelerators. Also, TP reduces the number of heads per accelerator, thereby reducing the ArI by degTP. As depicted in Figure [7,](#page-6-1) TP reduces the number of heads batched on each accelerator, thereby reducing the ArI by degTP.

Figure [8](#page-7-0) compares the latency impact of TP on reordered and non-reordered MLA attention blocks. Although the FC layers in an attention block benefit from TP, the core-attention layer dominates the runtime and limits improvements in the non-reordered case. Thus, using DP alone is preferable to combining TP and DP in the attention block.

*Observation-5:* In the decode stage with reordering, tensor parallelism fails to provide a meaningful latency reduction.

# V. INSIGHTS ON MIXTURE OF EXPERTS

#### <span id="page-7-2"></span>*A. Mixture of Experts (MoE)*

Although it is a common belief that larger models with more parameters produce higher-quality responses [\[28\]](#page-13-11), the substantial computational overhead associated with scaling LLMs hinders further growth. MoE [\[15\]](#page-13-1) addresses this issue by employing sparse activation in FFN blocks; MoE introduces a pool of *experts* and activates only a small subset of experts for each input. Recent LLMs [\[10\]](#page-12-0), [\[11\]](#page-12-8), [\[33\]](#page-13-3) adopt a hybrid architecture consisting of two types of experts: a *shared expert* and *routed experts*. The former is activated for every token during inference, whereas the latter are selectively activated based on a routing mechanism that dynamically assigns n<sup>k</sup> experts out of n<sup>e</sup> experts to each token. The computational procedure of an MoE block can be described as

$$\mathsf{MoE}(\mathbf{u}) = \left(\sum_{e \in \{1, \dots, n_e\}} \mathsf{Expert}_e(\mathbf{u})\right) + \mathsf{Expert}_{\mathsf{shared}}(\mathbf{u}) \quad (9)$$

By utilizing only n<sup>k</sup> experts (n<sup>k</sup> < ne), along with one shared expert per token at runtime, MoE effectively scales the model with low computational overhead. n<sup>k</sup> and n<sup>e</sup> vary across models: DeepSeek-R1 [\[10\]](#page-12-0) employs eight routed experts selected from a pool of 256 while Llama4-Maverick [\[33\]](#page-13-3) uses a single routed expert selected from a pool of 128.

The execution time of an MoE block is dominated by expert computations and by the communication required to dispatch/combine tokens to/from the selected experts [\[24\]](#page-13-4), [\[63\]](#page-14-19). A common strategy to scale MoE inference is exploiting expert parallelism (EP), which distributes experts across accelerators. Under EP, tokens are transferred to the devices holding their routed experts; after the computation, the results are combined with additional communication overhead. As different types of parallelism can be selectively applied at the block level, we use degEP to denote the degrees of EP.

Moreover, Using EP for MoE in a multi-accelerator system introduces complications regarding batching, where we need to maximize the usage of each accelerator's arithmetic throughput while also satisfying the memory capacity limitations and SLO constraints. Meanwhile, communication overhead between the accelerators in a serving system is highly dependent on the system-wide interconnect (*e.g.*, NVLink [\[17\]](#page-13-12), [\[41\]](#page-14-16), InfiniBand [\[20\]](#page-13-13), [\[43\]](#page-14-20), and optical links [\[49\]](#page-14-21)) specification. In this section, we analyze the impact of both factors on the performance of MoE blocks.

#### *B. Maximize compute utilization in MoE blocks*

In a multi-accelerator system, an efficient expert computation requires careful design to make the best use of each accelerator's arithmetic throughput by batching tokens for each expert. Although attention and MoE blocks both contain FC layers, their effective B's differ due to the MoE sparsity and distinct parallelization strategies. For attention, requests are split across degDP DP groups and each group processes B/degDP requests using TP; the ArI of the FC layers in an attention block scales with B/degDP.

In contrast, as we utilize EP for MoE, each expert handles B · nk/n<sup>e</sup> tokens on average. Because tokens are dynamically routed to experts at runtime, the number of tokens assigned to each expert can vary across experts. Hereafter, we denote by Γimb *the load imbalance ratio*, defined as the ratio between the actual number of tokens processed by an expert and the number for an ideal uniform distribution. Considering the load imbalance, the ArI of each expert would be Γimb · B · nk/ne.

For analytical simplicity, we conduct the following analysis based on the average behavior of all experts. The effects of load imbalance will be deeply discussed in [§VII-C.](#page-10-0) To reach the ridge point RPacc for FC layers in each block, the batch size B must satisfy:

<span id="page-7-1"></span>
$$B \ge B_{\text{attn}} = RP_{\text{acc}} \cdot \deg_{\text{DP}}$$

$$B \ge B_{\text{MoE}} = RP_{\text{acc}} \cdot \frac{n_e}{n_k}$$
(10)

Battn and BMoE are the batch sizes that reaches RPacc for the FC layers of an attention and an MoE block, respectively. We denote the minimum B that satisfies Eq. 10 as  $B_{RP} = \max(B_{attn}, B_{MoE})$ .

While  $B_{\rm attn}$  is influenced by  $\deg_{\rm DP}$ ,  $B_{\rm MoE}$  only depends on the model and the target accelerator and is independent of the number of accelerators. Since  $n_e$  and  $n_k$  are model parameters, the ArI of the FC layers in an MoE block is determined once  $RP_{\rm acc}$  and the model are fixed.

**Observation-6:**  $B_{\mathrm{MoE}}$  is determined once the model and the target accelerator are fixed.

#### C. Two primary factors limiting batch size

While batching  $B_{\rm RP}$  requests is desirable, the feasible batch size is limited by two factors: memory capacity and SLO.

**Memory capacity**: To fully utilize the accelerator's computational resources, data must be served at high bandwidth. To achieve that, the entire working set should reside in the main memory (e.g., HBM). This working set includes the weights for attention and MoE blocks, as well as KV\$. As the weight size is predetermined, serving systems typically use the remaining memory for activation and KV\$, whose sizes are proportional to B. Thus, the memory space requirements for model weights determine the maximum feasible batch size ( $B_{\text{cap}}$ ) as follows:

<span id="page-8-0"></span>
$$B_{\text{cap}} = \frac{M_{\text{cap}} \cdot n_{\text{acc}} - n_{decoder} \cdot (M_{\text{attn}} \cdot \deg_{\text{DP}} + M_{\text{MoE}})}{n_{decoder} \cdot M_{\text{KV}} \cdot L + M_{\text{act}}(L)}$$
(11)

where  ${\rm M_{cap}} \cdot n_{\rm acc}$  denotes the memory capacity of a system composed of  $n_{acc}$  accelerators, each having a  ${\rm M_{cap}}$  capacity.  ${\rm M_{attn}}$  and  ${\rm M_{MoE}}$  represent the model weight sizes of a single decoder block's attention and MoE, respectively. We denote the KV\$ size per token for each decoder block as  ${\rm M_{KV}}$ . The activation memory space required by a decoder block per token on each accelerator,  ${\rm M_{act}}$ , depends on the sequence length L. As this memory space is reused across multiple decoder blocks, the  ${\rm M_{act}}(L)$  term in Eq. 11 does not scale with  $n_{\rm decoder}$ . To batch  ${\rm B_{RP}}$  requests (from the previous section),  ${\rm B_{cap}}$  should be greater than  $B_{\rm RP}$ .

**SLO**: As excessive batching would incur latency overheads, SLO becomes another limiting factor for feasible batch sizes. In a disaggregated system, the time per output token (TPOT), a key latency metric in LLM serving, is determined by the latency of each decode stage and expressed as follows:

<span id="page-8-1"></span>
$$\text{TPOT(B, L)} = n_{\text{decoder}} \cdot \left( \underbrace{\frac{\mathbf{M}_{\text{attn}} \cdot \text{deg}_{\text{DP}} + \mathbf{M}_{\text{MoE}}}{n_{\text{acc}} \cdot \mathbf{BW}_{\text{Mem}}}}_{\text{model load lat.}} + \underbrace{\delta(\mathbf{B, L})}_{\text{additional lat.}} \right)$$

where both the first and the second terms in the parentheses represent latencies for each decoder block: the first accounts for the latency to read model weights and the second,  $\delta(B,L)$ , includes additional latency such as memory access time for the KV\$ and activations, communication overhead, and any remaining computation time. The additional latency term is a function of B and L.

<span id="page-8-3"></span>TABLE IV

MODEL CONFIGURATION USED IN EVALUATION. BOTH
LLAMA4-MAVERICK AND DEEPSEEK-R1 HAVE 1 SHARED EXPERT.

| Model       | # of par. | $d_{\mathrm{emb}}$ | $deg_{\mathrm{grp}}$ | $ d_{\mathrm{FFN}} $ | $d_{\rm MoE}$ | $n_{\rm e}$ | $n_{\mathrm{k}}$ | $deg_{\mathrm{TP}}$ | $deg_{\mathrm{DP}}$ | $deg_{\mathrm{EP}}$ |
|-------------|-----------|--------------------|----------------------|----------------------|---------------|-------------|------------------|---------------------|---------------------|---------------------|
| GPT-3       | 175B      | 12K                | 1                    | 48K                  | -             | -           | -                | 8                   | 4                   | -                   |
| Llama4      | 400B      | 5K                 | 5                    | 16K                  | 8K            | 1           | 128              | 8                   | 4                   | 32                  |
| DeepSeek-R1 | 671B      | 7K                 | 1                    | 18K                  | 2K            | 8           | 256              | 1                   | 32                  | 32                  |

As the memory access time for the KV\$ and activations, along with communication time, is unavoidable when processing each decoder block, the minimum bound of this additional latency,  $\delta_{\min}(B,L)$ , is given by

$$\delta_{\min}(\mathbf{B}, \mathbf{L}) \ge \mathbf{B} \cdot \left(\frac{\mathbf{M}_{\mathrm{KV}} \cdot \mathbf{L} + \mathbf{M}_{\mathrm{act}}(L)}{n_{\mathrm{acc}} \cdot \mathbf{BW}_{\mathrm{mem}}}\right) + \mathrm{Comm}(\mathbf{B}, \mathbf{L})$$
 (12)

where  $\operatorname{Comm}(B,L)$  denotes the communication overhead between the accelerators. Increasing B leads to larger KV\$ and activation sizes, thus increasing the minimum bound of TPOT. The theoretical maximum batch size,  $B_{SLO}$ , that satisfies the SLO time limit (TPOT $_{SLO}$ ) can be achieved under the minimum latency. A batch size B greater than  $B_{SLO}$  can never satisfy the TPOT $_{SLO}$  time limit, establishing an upper bound on the feasible batch size.

*Observation-7:* While MoE tightens the batch size limit due to weight overheads, MLA complements this through its small KV\$ size.

MoE weights  $(M_{\rm MoE})$  are typically larger than the FFN weights in standard LLMs, increasing memory requirements for the model weights. Then,  $B_{\rm cap}$  decreases as less memory space remains for KV\$ (see Eq. 11). It also increases the model load latency, which shortens the time available for  $\delta_{\rm min}(B_{\rm SLO},L)$ , thereby reducing  $B_{\rm SLO}$ . In contrast, the reduction of  $M_{\rm KV}$  and  $M_{\rm attn}$  by MLA enables storing the KV\$ for more requests in the main memory, thereby increasing  $B_{\rm cap}$ . It also reduces the load time for  $M_{\rm KV}$  and  $M_{\rm attn}$ , allowing higher  $B_{\rm SLO}$  (see Eq. 12). Thus, as for the batch size limits, MLA and MoE impose complementary effects.

#### D. Communication cost

To reduce MoE execution time, both interconnect bandwidth and expert-distribution skew must be addressed. Besides expert computations, communication is a dominant contributor to the overall MoE execution time [24]. As experts are distributed across multiple accelerators using EP, the system must dispatch tokens to, and combine tokens from, the selected experts. As tokens are transferred over the interconnect between the accelerators, communication time is determined by the size of the transferred tokens and the available interconnect bandwidth. In the MoE blocks, the communication time varies across the accelerators because each accelerator sends or receives a different amount of data due to the imbalance in expert distributions.

<span id="page-8-2"></span>
$$Comm_{MoE}(B) = 2 \cdot \max_{a \in Acc.} (\Gamma_{imb}^{acc}(a)) \cdot \frac{M_{token} \cdot n_k \cdot B}{BW_{Int} \cdot n_{acc}} + \alpha$$
 (13)

![](_page_9_Figure_0.jpeg)

<span id="page-9-3"></span>Fig. 9. Throughput-latency graph for the decode stages of GPT-3, Llama4-Maverick, and DeepSeek-R1. We assume a 32 B200 GPU system.

Eq.  $13^2$  provides a simplified model of the MoE communication time.  $\Gamma^{acc}_{imb}$  denotes the load imbalance ratio at the accelerator level, computed from the total number of tokens processed by all the experts assigned to an accelerator. When the batch size increases, the interconnect bandwidth and  $\Gamma^{acc}_{imb}$  become the most critical factors. While a larger batch size can improve throughput by increasing compute utilization, it also incurs significant communication overheads. Thus, a high-bandwidth interconnect is required to fully exploit the benefits of batching in LLM inference [65]. Moreover, as the expert distribution becomes more skewed (larger  $\Gamma^{acc}_{imb}$ ), tokens concentrate on a small subset of accelerators, leading to longer communication times.

In summary, reducing the communication cost in MoE requires both high-bandwidth interconnects and an effective mitigation of load imbalance. The gating operation computes expert scores through a lightweight FC layer, and its computational cost is negligible compared to the expert computation.

*Observation-8:* The interconnect bandwidth and expert load imbalance are the dominant factors that determine the communication time of MoE blocks at large batch sizes.

#### VI. EXPERIMENTAL SETUP

<span id="page-9-0"></span>To evaluate LLM serving performance in various configurations, we conducted real-system experiments on DGX H100 [42] and developed an in-house simulator based on LLMSimulator [50], [62]. In our simulator, we modeled modern kernel- and system-level optimizations (e.g., FlashAttention [9], FlashMLA [27], fused kernels, and optimized communication) to ensure fair and realistic execution-time estimation. We verified the computational characteristics at the node level using a real system. For inter-node communication time (e.g., dispatch and combine communication in the MoE block), we validated our simulation results against the timing data reported in DeepEP [66].

We configured the accelerator as a modern NVIDIA B200 GPU, whose key parameters are listed in Table I. By default, we assumed all GPUs in a group are fully connected via

<span id="page-9-2"></span> $^2We$  assume a fully connected, switch-based interconnect topology that offers uniform bidirectional bandwidth (BW $_{\rm Int}$ ) among all accelerators.  $M_{\rm token}$  represents the size of token by a single decoder.  $\alpha$  denotes the additional latency in the network.

NVLink fifth generation, providing 1.8TB/s of bidirectional bandwidth following the NVL72 system topology [38]. For each experiment, we specify the number of GPUs per group and note when InfiniBand XDR (100 GB/s) is used for inter-group communication. We used DeepSeek-R1, Llama4-Maverick, and GPT-3 (key parameters specified in Table II and Table IV); all experiments were performed with BF16 precision for all parameters, KV\$, and activations. We used BF16 as the baseline, but our observations also hold for lower precisions (*e.g.*, FP8), as further discussed in \$VIII.

To accurately model real-world serving scenarios, we assumed a Zipfian distribution for token routing [29]. We varied the degree of skewness (s) to thoroughly study its impact on system performance. For better interpretability, we annotated each distribution with the corresponding load imbalance metrics (e.g,  $\Gamma_{imb}$ ) defined in §V.

Following common practices [16], [40], [46], [68], we assume a *disaggregated* system where the prefill and decode stages are executed on separate machines. We focus on the decode phase as the prefill phase is generally compute-bound and already achieves high utilization without batching [1], [13], [68]. Moreover, the insights gained from analyzing the communication time of MoE blocks also apply to the prefill phase as large-batch decode scenarios exhibit similar interconnect traffic patterns to prefill. For model deployment for the decode system, we set  $deg_{TP}$  to 8 for both GPT-3 and Llama 4-Maverick, chosen to maximize performance of the model, while aligning with the typical 8-GPU topology of NVIDIA DGX systems [37]. For DeepSeek-R1, we set the  $deg_{TP}$  to 1, in accordance with our observation (Obs. 5).

#### <span id="page-9-1"></span>VII. END-TO-END MODEL EXECUTION ANALYSIS

#### A. The Synergistic impact of MLA and MoE

LLMs adopting MLA and MoE achieve significantly higher throughput than conventional models. This is because MLA and MoE have a powerful synergistic relationship. MLA's highly-compressed KV\$ dramatically increases the memory capacity available for batching (B<sub>cap</sub>). This, in turn, allows the system to form the large batches required to fully utilize the compute resources of the sparsely activated experts in MoE blocks, which would otherwise be constrained by memory.

Figure 9 illustrates this by comparing DeepSeek-R1 with Llama4-Maverick and GPT-3. For a sequence length of 8192,

![](_page_10_Figure_0.jpeg)

<span id="page-10-1"></span>Fig. 10. System throughput and execution time ratio of the decode stage of DeepSeek-R1 when using InfiniBand XDR (100GB/s) among a group of GPUs (DGX), varying sequence lengths and batch sizes. We assume 32 B200 GPU system.

DeepSeek-R1's  $B_{\rm cap}$  (7360) is nearly  $60\times$  larger than GPT-3's (124) and  $2.21\times$  larger than Llama4-Maverick's (3328), which has an even larger model size. Consequently, DeepSeek-R1 can be configured with a batch size large enough to approach its ridge point  $B_{\rm RP}(=B_{\rm attn})$ , whereas Llama4-Maverick and GPT-3 become memory-capacity-limited long before their compute resources can be saturated.

#### B. The critical role of interconnect

The performance of a scaled-out MoE-based system is highly sensitive to interconnect bandwidth [8]. The all-to-all communication pattern, required to dispatch every token to its designated experts and then combine the results, creates dense network traffic that can easily become a bottleneck. As shown in Figure 10, moving from a high-bandwidth fabric, such as NVLink, to a lower-bandwidth one, such as InfiniBand, dramatically increases this communication overhead. For a per-accelerator batch size of 128, our measurements show that a single all-to-all communication task (e.g., dispatch/combine) takes 151.8 µs on a lower-bandwidth fabric, compared with 17.65 µs on the high-bandwidth fabric. Higher communication latency consumes a larger portion of the per-token time budget, which directly reduces the achievable batch size under a given SLO (B<sub>SLO</sub>) and leads to underutilization. Thus, for efficient system deployment, it is critical to have interconnects with high bisection bandwidth.

This sensitivity forces a critical deployment decision: using multiple small, tightly-coupled instances (e.g., 32 GPU×8) versus one large, monolithic instance (e.g., 256 GPU). Since it is difficult to scale the number of accelerators while maintaining high bisection bandwidth, we vary the interconnect bandwidth of the 256 GPU configuration to 900 GB/s, 300 GB/s, and 100 GB/s, which are equal to or lower than that of each 32 GPU instance.

As Figure 11 shows, the optimal choice depends on the workload. For shorter sequences, multiple small instances are more cost-effective because communication is contained within high-bandwidth domains, and the memory overhead of replicating MoE weights is manageable. When L=2048 (Figure 11(a)), at a batch size of  $B_{\rm RP}$  to maximize throughput,

![](_page_10_Figure_7.jpeg)

<span id="page-10-2"></span>Fig. 11. Throughput comparison of **256 GPU** and **32 GPU** $\times$ **8** systems of the decode stage of DeepSeek-R1 when L=2048 and L=16384. 900 GB/s denotes NVLink, while 100 GB/s corresponds to InfiniBand.

**32 GPU**×**8** achieves equivalent throughput as **256 GPU** with 900 GB/s interconnect bandwidth. At this point, in **256 GPU**, each GPU is responsible for executing only one expert, but each expert processes 8 times more tokens than in **32 GPU**×**8**. As the Op/B of experts in **256 GPU** belongs to the compute-bound region, it results in higher latency. Thus, the latency of MoE blocks becomes similar across the systems.

For very long sequences (e.g., L = 16384 in Figure 11(b)), however, a single large instance is superior. The memory savings from storing the massive MoE weights only once by **256 GPU** frees up system-wide capacity for a larger  $B_{\rm cap}$ , which is essential for handling the large KV\$, over **32 GPU** $\times$ **8**. This leads to higher overall throughput, even if the large-scale interconnect has higher latency. For example, even with a reduced interconnect bandwidth of 300 GB/s, **256 GPU** delivers better throughput by reducing MoE execution latency.

#### <span id="page-10-0"></span>C. Skewed expert distribution

Mitigating skewness in expert distribution is essential for achieving high-throughput and low-latency MoE execution. Figure 12 presents the throughput-latency trade-off for the decode stages of DeepSeek-R1 under varying degrees of expert routing skewness (s). As s increases (from 0.2 to 0.8), the overall system throughput gradually decreases due to the load imbalance among the accelerators. Also, the latency increases as more tokens are concentrated on a smaller subset of experts. As the distribution get more skewed, the rate of increase in both communication latency and MoE latency with respect to the batch size also grows, indicating more severe performance degradation under skewed conditions. These results indicate that skewed expert routing reduces the effectiveness of batching, as increasing skewness leads to higher latency and diminishing throughput gains.

Our observations remain valid even with a skewed distribution of experts; however, the preferred deployment configurations will be affected by this skewness. Under a uniform random distribution, the batch size that saturates the throughput is close to  $B_{\rm MoE}$ . However, with skewness, hot experts become saturated before the total batch size reaches  $B_{\rm MoE}$ , while cold experts process fewer tokens, resulting in lower ArI and reduced throughput. When the total batch size increases

![](_page_11_Figure_0.jpeg)

<span id="page-11-1"></span>Fig. 12. Throughput-latency graph for the decode stages of DeepSeek-R1 with skewed expert routing and 2048 sequence length in 32 GPU system.

![](_page_11_Figure_2.jpeg)

<span id="page-11-2"></span>Fig. 13. Throughput and load imbalance ratio comparison between **256 GPU** and **32 GPU** $\times$ **8** systems for the decode stage of DeepSeek-R1 under varying skewness of expert distribution when L=2048. Both systems use a 900 GB/s interconnect among GPUs.

beyond  $B_{\mathrm{MoE}}$ , the ArI of cold experts can eventually reach  $RP_{\mathrm{acc}}$ , leading to a larger batch size required for throughput saturation. Nevertheless, saturating all experts increases latency, as the hot experts have already reached their maximum throughput and only contribute additional latency without improving overall throughput. Therefore, service providers must select an appropriate batch size that balances the tradeoff between throughput and latency, considering the skewness.

Smaller deployment units such as 32 GPU×8 can more effectively mitigate the load imbalance compared to a monolithic 256 GPU. Figure 13 compares the system throughput when serving DeepSeek-R1 with different deployment granularities, either using a single deployment of 256 GPU or eight deployments of 32 GPU×8 each, under varying levels of expert routing skewness. When there is no skew (s = 0.0), the 256 GPU configuration achieves higher throughput due to larger aggregate compute and communication bandwidth. However, as skewness increases, throughput degrades more severely in 256 GPU, while 32 GPU×8 maintains higher throughput. When the skewness is 0.8,  $\Gamma_{imb}^{acc}$  of **256 GPU** is  $6.13 \times$  higher than that of 32 GPU $\times$ 8. In 256 GPU, each GPU handles only one expert; thus, the token imbalance among the experts directly translates to a load imbalance across GPUs. In contrast, in 32 GPU×8, each GPU handles 8 experts, which naturally balances the token distribution and mitigates the load imbalance. Both systems assume a 900 GB/s interconnect;

![](_page_11_Figure_6.jpeg)

<span id="page-11-3"></span>Fig. 14. Normalized throughput of Duplex [62], which PIM devices process only MoE execution, compared to the baseline GPU system. We used PIM devices with  $RP_{\rm acc}$ =8, utilizing 4 times of HBM memory bandwidth of GPU.

considering the significantly higher networking cost required to fully connect **256 GPU**, **32 GPU**×**8** offers a more balanced and cost-efficient deployment unit for large-scale MoE serving.

#### D. Effectiveness of Processing-In-Memory architectures

At low-batch inference scenarios, where latency-sensitive workloads or on-device inference are required, executing MoE layers on Processing-in-Memory (PIM) architectures [22], [23], [44], [45], [62] provides better efficiency by exploiting higher memory bandwidth compared to GPUs. We modeled Duplex [62], a state-of-the-art HBM-based PIM architecture designed to accelerate MoE layers, and compared its throughput with that of GPUs. Figure 14 shows normalized throughput improvements when using PIM for MoE execution.

When the batch size per GPU is smaller than 32, PIM devices can effectively reduce latency and increase throughput by processing expert computations faster through their high memory bandwidth. However, as the batch size increases, PIM devices struggle to sustain performance because the ArI of the experts increases, making computation rather than memory bandwidth the dominant bottleneck. We conclude that, when MLA and MoE are employed, PIM devices are more suitable for low-batch, low-sequence-length inference scenarios and, in particular, decode stages in such scenarios.

#### VIII. DISCUSSION

<span id="page-11-0"></span>Low weight precision: Recent LLMs support low-precision weights such as FP8 to alleviate memory capacity constraints, while accepting a modest loss in accuracy [10]–[12], [18], [30], [31], [47], [61], [64], [67]. The peak FLOPS of accelerators increase when low-precision weights are used. For

example, latest GPUs can achieve up to two times higher peak FLOPS when executing FP8 operations compared to FP16 or BF16. Thus, RPacc doubles; however, BRP remains unchanged because memory access also decreases by half, due to the reduced data size. In contrast, Bcap increases because low-precision weights reduce the memory footprint of the model, thereby expanding the available memory capacity for KV\$. In Figure [9\(](#page-9-3)b), when L = 8192, the system is unable to reach BRP due to the Bcap constraint. By adopting FP8 for model weights, Bcap increases sufficiently to match BRP, enabling the system to maximize throughput.

#### IX. CONCLUSION

Advances in large language models (LLMs) have reshaped the computational landscape of inference. Multi-head Latent Attention (MLA) and Mixture-of-Experts (MoE) move the performance bottleneck away from memory bandwidth. With layer reordering, MLA becomes mostly compute-bound, which is well-suited for contemporary accelerators, diminishing the need for dedicated hardware. MoE achieves scalability through sparse expert activation but demands large batches to sustain utilization; MLA complements this by reducing the KV\$, enabling large-batch inference efficiently even for long sequences. Finally, we highlight that interconnect bandwidth and expert skewness become the primary factors determining end-to-end performance. Future serving systems must emphasize high-bandwidth interconnects and balanced workloads to achieve scalable, low-latency LLM serving.

# REFERENCES

- <span id="page-12-2"></span>[1] A. Agrawal, N. Kedia, A. Panwar, J. Mohan, N. Kwatra, B. S. Gulavani, A. Tumanov, and R. Ramjee, "Taming Throughput-latency Tradeoff in LLM Inference with Sarathi-serve," in *Proceedings of the 18th USENIX Conference on Operating Systems Design and Implementation*, 2024. [Online]. Available:<https://dl.acm.org/doi/10.5555/3691938.3691945>
- <span id="page-12-5"></span>[2] J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebron, and S. Sanghai, "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," in *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, Singapore, 2023, pp. 4895–4901. [Online]. Available: <https://aclanthology.org/2023.emnlp-main.298>
- <span id="page-12-7"></span>[3] AMD, "AMD INSTINCT™ MI325X ACCELERATOR Leading-Edge, industry-standard accelerator module for generative AI, inference, training, and high performance computing," 2025. [Online]. Available: [https://www.amd.com/content/dam/amd/en/documents/](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/instinct-mi325x-datasheet.pdf) [instinct-tech-docs/product-briefs/instinct-mi325x-datasheet.pdf](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/instinct-mi325x-datasheet.pdf)
- <span id="page-12-3"></span>[4] R. Y. Aminabadi, S. Rajbhandari, A. A. Awan, C. Li, D. Li, E. Zheng, O. Ruwase, S. Smith, M. Zhang, J. Rasley, and Y. He, "DeepSpeed-inference: enabling efficient inference of transformer models at unprecedented scale," in *Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis*, 2022, pp. 1–15. [Online]. Available: [https://dl.acm.org/doi/](https://dl.acm.org/doi/abs/10.5555/3571885.3571946) [abs/10.5555/3571885.3571946](https://dl.acm.org/doi/abs/10.5555/3571885.3571946)
- <span id="page-12-6"></span>[5] A. Bambhaniya, R. Raj, G. Jeong, S. Kundu, S. Srinivasan, S. Subramanian, M. Elavazhagan, M. Kumar, and T. Krishna, "Demystifying AI Platform Design for Distributed Inference of Next-Generation LLM models," 2025. [Online]. Available: [https:](https://arxiv.org/abs/2406.01698) [//arxiv.org/abs/2406.01698](https://arxiv.org/abs/2406.01698)
- <span id="page-12-4"></span>[6] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei, "Language

- Models are Few-Shot Learners," in *Proceedings of the 34th International Conference on Neural Information Processing Systems*, 2020. [Online]. Available: [https://proceedings.neurips.cc/paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) files/ [paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
- <span id="page-12-1"></span>[7] D. Dai, C. Deng, C. Zhao, R. Xu, H. Gao, D. Chen, J. Li, W. Zeng, X. Yu, Y. Wu, Z. Xie, Y. Li, P. Huang, F. Luo, C. Ruan, Z. Sui, and W. Liang, "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models," in *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 2024, pp. 1280–1297. [Online]. Available: <https://aclanthology.org/2024.acl-long.70/>
- <span id="page-12-10"></span>[8] W. Dally and B. Towles, *Principles and Practices of Interconnection Networks*. Morgan Kaufmann, 2004.
- <span id="page-12-9"></span>[9] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Re, "FlashAttention: ´ Fast and Memory-Efficient Exact Attention with IO-Awareness," in *Proceedings of the 36th International Conference on Neural Information Processing Systems*, 2022, pp. 16 344 – 16 359. [Online]. Available: <https://dl.acm.org/doi/10.5555/3600270.3601459>
- <span id="page-12-0"></span>[10] DeepSeek-AI, D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi, X. Zhang, X. Yu, Y. Wu, Z. F. Wu, Z. Gou, Z. Shao, Z. Li, Z. Gao, A. Liu, B. Xue, B. Wang, B. Wu, B. Feng, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, D. Dai, D. Chen, D. Ji, E. Li, F. Lin, F. Dai, F. Luo, G. Hao, G. Chen, G. Li, H. Zhang, H. Bao, H. Xu, H. Wang, H. Ding, H. Xin, H. Gao, H. Qu, H. Li, J. Guo, J. Li, J. Wang, J. Chen, J. Yuan, J. Qiu, J. Li, J. L. Cai, J. Ni, J. Liang, J. Chen, K. Dong, K. Hu, K. Gao, K. Guan, K. Huang, K. Yu, L. Wang, L. Zhang, L. Zhao, L. Wang, L. Zhang, L. Xu, L. Xia, M. Zhang, M. Zhang, M. Tang, M. Li, M. Wang, M. Li, N. Tian, P. Huang, P. Zhang, Q. Wang, Q. Chen, Q. Du, R. Ge, R. Zhang, R. Pan, R. Wang, R. J. Chen, R. L. Jin, R. Chen, S. Lu, S. Zhou, S. Chen, S. Ye, S. Wang, S. Yu, S. Zhou, S. Pan, S. S. Li, S. Zhou, S. Wu, S. Ye, T. Yun, T. Pei, T. Sun, T. Wang, W. Zeng, W. Zhao, W. Liu, W. Liang, W. Gao, W. Yu, W. Zhang, W. L. Xiao, W. An, X. Liu, X. Wang, X. Chen, X. Nie, X. Cheng, X. Liu, X. Xie, X. Liu, X. Yang, X. Li, X. Su, X. Lin, X. Q. Li, X. Jin, X. Shen, X. Chen, X. Sun, X. Wang, X. Song, X. Zhou, X. Wang, X. Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. Zhang, Y. Xu, Y. Li, Y. Zhao, Y. Sun, Y. Wang, Y. Yu, Y. Zhang, Y. Shi, Y. Xiong, Y. He, Y. Piao, Y. Wang, Y. Tan, Y. Ma, Y. Liu, Y. Guo, Y. Ou, Y. Wang, Y. Gong, Y. Zou, Y. He, Y. Xiong, Y. Luo, Y. You, Y. Liu, Y. Zhou, Y. X. Zhu, Y. Xu, Y. Huang, Y. Li, Y. Zheng, Y. Zhu, Y. Ma, Y. Tang, Y. Zha, Y. Yan, Z. Z. Ren, Z. Ren, Z. Sha, Z. Fu, Z. Xu, Z. Xie, Z. Zhang, Z. Hao, Z. Ma, Z. Yan, Z. Wu, Z. Gu, Z. Zhu, Z. Liu, Z. Li, Z. Xie, Z. Song, Z. Pan, Z. Huang, Z. Xu, Z. Zhang, and Z. Zhang, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning," 2025. [Online]. Available:<https://arxiv.org/abs/2501.12948>
- <span id="page-12-8"></span>[11] DeepSeek-AI, A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, D. Dai, D. Guo, D. Yang, D. Chen, D. Ji, E. Li, F. Lin, F. Dai, F. Luo, G. Hao, G. Chen, G. Li, H. Zhang, H. Bao, H. Xu, H. Wang, H. Zhang, H. Ding, H. Xin, H. Gao, H. Li, H. Qu, J. Cai, J. Liang, J. Guo, J. Ni, J. Li, J. Wang, J. Chen, J. Chen, J. Yuan, J. Qiu, J. Li, J. Song, K. Dong, K. Hu, K. Gao, K. Guan, K. Huang, K. Yu, L. Wang, L. Zhang, L. Xu, L. Xia, L. Zhao, L. Wang, L. Zhang, M. Li, M. Wang, M. Zhang, M. Zhang, M. Tang, M. Li, N. Tian, P. Huang, P. Wang, P. Zhang, Q. Wang, Q. Zhu, Q. Chen, Q. Du, R. Chen, R. Jin, R. Ge, R. Zhang, R. Pan, R. Wang, R. Xu, R. Zhang, R. Chen, S. Li, S. Lu, S. Zhou, S. Chen, S. Wu, S. Ye, S. Ye, S. Ma, S. Wang, S. Zhou, S. Yu, S. Zhou, S. Pan, T. Wang, T. Yun, T. Pei, T. Sun, W. Xiao, W. Zeng, W. Zhao, W. An, W. Liu, W. Liang, W. Gao, W. Yu, W. Zhang, X. Li, X. Jin, X. Wang, X. Bi, X. Liu, X. Wang, X. Shen, X. Chen, X. Zhang, X. Chen, X. Nie, X. Sun, X. Wang, X. Cheng, X. Liu, X. Xie, X. Liu, X. Yu, X. Song, X. Shan, X. Zhou, X. Yang, X. Li, X. Su, X. Lin, Y. Li, Y. Wang, Y. Wei, Y. Zhu, Y. Zhang, Y. Xu, Y. Xu, Y. Huang, Y. Li, Y. Zhao, Y. Sun, Y. Li, Y. Wang, Y. Yu, Y. Zheng, Y. Zhang, Y. Shi, Y. Xiong, Y. He, Y. Tang, Y. Piao, Y. Wang, Y. Tan, Y. Ma, Y. Liu, Y. Guo, Y. Wu, Y. Ou, Y. Zhu, Y. Wang, Y. Gong, Y. Zou, Y. He, Y. Zha, Y. Xiong, Y. Ma, Y. Yan, Y. Luo, Y. You, Y. Liu, Y. Zhou, Z. Wu, Z. Ren, Z. Ren, Z. Sha, Z. Fu, Z. Xu, Z. Huang, Z. Zhang, Z. Xie, Z. Zhang, Z. Hao, Z. Gou, Z. Ma, Z. Yan, Z. Shao, Z. Xu, Z. Wu, Z. Zhang, Z. Li, Z. Gu, Z. Zhu, Z. Liu, Z. Li, Z. Xie, Z. Song, Z. Gao, and Z. Pan, "DeepSeek-V3 Technical Report," 2024. [Online]. Available:<https://arxiv.org/abs/2412.19437>

- <span id="page-13-18"></span>[12] T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer, "LLM.int8(): 8-bit matrix multiplication for transformers at scale," in *Proceedings of the 36th International Conference on Neural Information Processing Systems*, 2022. [Online]. Available: [https://dl.acm.org/doi/10.5555/](https://dl.acm.org/doi/10.5555/3600270.3602468) [3600270.3602468](https://dl.acm.org/doi/10.5555/3600270.3602468)
- <span id="page-13-16"></span>[13] K. Du, B. Wang, C. Zhang, Y. Cheng, Q. Lan, H. Sang, Y. Cheng, J. Yao, X. Liu, Y. Qiao, I. Stoica, and J. Jiang, "PrefillOnly: An Inference Engine for Prefill-only Workloads in Large Language Model Applications," 2025. [Online]. Available: <https://arxiv.org/abs/2505.07203>
- <span id="page-13-8"></span>[14] A. Elmeleegy, S. Raj, B. Slechta, and V. Mehta, "Demystifying AI Inference Deployments for Trillion Parameter Large Language Models." [Online]. Available: [https://developer.nvidia.com/blog/demystifying-ai](https://developer.nvidia.com/blog/demystifying-ai-inference-deployments-for-trillion-parameter-large-language-models/)[inference-deployments-for-trillion-parameter-large-language-models/](https://developer.nvidia.com/blog/demystifying-ai-inference-deployments-for-trillion-parameter-large-language-models/)
- <span id="page-13-1"></span>[15] W. Fedus, B. Zoph, and N. Shazeer, "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity," *Journal of Machine Learning Research*, vol. 23, no. 120, pp. 1–39, 2022. [Online]. Available:<http://jmlr.org/papers/v23/21-0998.html>
- <span id="page-13-15"></span>[16] J. Feng, Y. Huang, R. Zhang, S. Liang, M. Yan, and J. Wu, "WindServe: Efficient Phase-Disaggregated LLM Serving with Streambased Dynamic Scheduling," in *ISCA*, 2025. [Online]. Available: <https://dl.acm.org/doi/10.1145/3695053.3730999>
- <span id="page-13-12"></span>[17] D. Foley and J. Danskin, "Ultra-Performance Pascal GPU and NVLink Interconnect," *IEEE Micro*, vol. 37, no. 2, pp. 7–17, 2017. [Online]. Available:<https://dl.acm.org/doi/abs/10.1109/MM.2017.37>
- <span id="page-13-19"></span>[18] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, "OPTQ: Accurate Quantization for Generative Pre-trained Transformers," in *The Eleventh International Conference on Learning Representations*, 2023. [Online]. Available:<https://openreview.net/forum?id=tcbBPnfwxS>
- <span id="page-13-6"></span>[19] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, A. Yang, A. Fan, A. Goyal, A. Hartshorn, A. Yang, A. Mitra, A. Sravankumar, A. Korenev, A. Hinsvark, A. Rao, A. Zhang, A. Rodriguez, A. Gregerson, A. Spataru, B. Roziere, B. Biron, B. Tang, B. Chern, C. Caucheteux, C. Nayak, C. Bi, C. Marra, C. McConnell, C. Keller, C. Touret, C. Wu, C. Wong, C. C. Ferrer, C. Nikolaidis, D. Allonsius, D. Song, D. Pintz, D. Livshits, D. Wyatt, D. Esiobu, D. Choudhary, D. Mahajan, D. Garcia-Olano, D. Perino, D. Hupkes, E. Lakomkin, E. AlBadawy, E. Lobanova, E. Dinan, E. M. Smith, F. Radenovic, F. Guzman, F. Zhang, G. Synnaeve, G. Lee, G. L. Anderson, ´ G. Thattai, G. Nail, G. Mialon, G. Pang, G. Cucurell, H. Nguyen, H. Korevaar, H. Xu, H. Touvron, I. Zarov, I. A. Ibarra, I. Kloumann, I. Misra, I. Evtimov, J. Zhang, J. Copet, J. Lee, J. Geffert, J. Vranes, J. Park, J. Mahadeokar, J. Shah, J. v. d. Linde, J. Billock, J. Hong, J. Lee, J. Fu, J. Chi, J. Huang, J. Liu, J. Wang, J. Yu, J. Bitton, J. Spisak, J. Park, J. Rocca, J. Johnstun, J. Saxe, J. Jia, K. V. Alwala, K. Prasad, K. Upasani, K. Plawiak, K. Li, K. Heafield, K. Stone, K. El-Arini, K. Iyer, K. Malik, K. Chiu, K. Bhalla, K. Lakhotia, L. Rantala-Yeary, L. v. d. Maaten, L. Chen, L. Tan, L. Jenkins, L. Martin, L. Madaan, L. Malo, L. Blecher, L. Landzaat, L. d. Oliveira, M. Muzzi, M. Pasupuleti, M. Singh, M. Paluri, M. Kardas, M. Tsimpoukelli, M. Oldham, M. Rita, M. Pavlova, M. Kambadur, M. Lewis, M. Si, M. K. Singh, M. Hassan, N. Goyal, N. Torabi, N. Bashlykov, N. Bogoychev, N. Chatterji, N. Zhang, O. Duchenne, O. C¸ elebi, P. Alrassy, P. Zhang, P. Li, P. Vasic, P. Weng, P. Bhargava, P. Dubal, P. Krishnan, P. S. Koura, P. Xu, Q. He, Q. Dong, R. Srinivasan, R. Ganapathy, R. Calderer, R. S. Cabral, R. Stojnic, R. Raileanu, R. Maheswari, R. Girdhar, R. Patel, R. Sauvestre, R. Polidoro, R. Sumbaly, R. Taylor, R. Silva, R. Hou, R. Wang, S. Hosseini, S. Chennabasappa, S. Singh, S. Bell, S. S. Kim, S. Edunov, S. Nie, S. Narang, S. Raparthy, S. Shen, S. Wan, S. Bhosale, S. Zhang, S. Vandenhende, S. Batra, S. Whitman, S. Sootla, S. Collot, S. Gururangan, S. Borodinsky, T. Herman, T. Fowler, T. Sheasha, T. Georgiou, T. Scialom, T. Speckbacher, T. Mihaylov, T. Xiao, U. Karn, V. Goswami, V. Gupta, V. Ramanathan, V. Kerkez, V. Gonguet, V. Do, V. Vogeti, V. Albiero, V. Petrovic, W. Chu, W. Xiong, W. Fu, W. Meers, X. Martinet, X. Wang, X. Wang, X. E. Tan, X. Xia, X. Xie, X. Jia, X. Wang, Y. Goldschlag, Y. Gaur, Y. Babaei, Y. Wen, Y. Song, Y. Zhang, Y. Li, Y. Mao, Z. D. Coudert, Z. Yan, Z. Chen, Z. Papakipos, A. Singh, A. Srivastava, A. Jain, A. Kelsey, and A. Shajnfeld, "The Llama 3 Herd of Models," 2024. [Online]. Available:<https://arxiv.org/abs/2407.21783>
- <span id="page-13-13"></span>[20] P. Grun, "Introduction to Infiniband for End Users," *White paper, InfiniBand Trade Association*, vol. 55, 2010. [Online]. Available: [https:](https://network.nvidia.com/pdf/whitepapers/Intro_to_IB_for_End_Users.pdf) [//network.nvidia.com/pdf/whitepapers/Intro](https://network.nvidia.com/pdf/whitepapers/Intro_to_IB_for_End_Users.pdf) to IB for End Users.pdf

- <span id="page-13-5"></span>[21] A. Gu and T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," 2024. [Online]. Available: [https:](https://arxiv.org/abs/2312.00752) [//arxiv.org/abs/2312.00752](https://arxiv.org/abs/2312.00752)
- <span id="page-13-17"></span>[22] Y. Gu, A. Khadem, S. Umesh, N. Liang, X. Servot, O. Mutlu, R. Iyer, and R. Das, "PIM Is All You Need: A CXL-Enabled GPU-Free System for Large Language Model Inference," in *ASPLOS*, 2025. [Online]. Available:<https://doi.org/10.1145/3676641.3716267>
- <span id="page-13-9"></span>[23] G. Heo, S. Lee, J. Cho, H. Choi, S. Lee, H. Ham, G. Kim, D. Mahajan, and J. Park, "NeuPIMs: NPU-PIM Heterogeneous Acceleration for Batched LLM Inferencing," in *ASPLOS*, 2024, p. 722–737. [Online]. Available:<https://doi.org/10.1145/3620666.3651380>
- <span id="page-13-4"></span>[24] H. Huang, N. Ardalani, A. Sun, L. Ke, H.-H. S. Lee, S. Bhosale, C.-J. Wu, and B. Lee, "Toward efficient inference for mixture of experts," in *Advances in Neural Information Processing Systems*, 2024, pp. 84 033– 84 059.
- <span id="page-13-7"></span>[25] Y. Huang, Y. Cheng, A. Bapna, O. Firat, D. Chen, M. Chen, H. Lee, J. Ngiam, Q. V. Le, Y. Wu, and Z. Chen, "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism," in *Advances in Neural Information Processing Systems 32*, 2019, pp. 103–112. [Online]. Available: [https://proceedings.neurips.cc/paper/](https://proceedings.neurips.cc/paper/2019/hash/093f65e080a295f8076b1c5722a46aa2-Abstract.html) [2019/hash/093f65e080a295f8076b1c5722a46aa2-Abstract.html](https://proceedings.neurips.cc/paper/2019/hash/093f65e080a295f8076b1c5722a46aa2-Abstract.html)
- <span id="page-13-2"></span>[26] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. de las Casas, E. B. Hanna, F. Bressand, G. Lengyel, G. Bour, G. Lample, L. R. Lavaud, L. Saulnier, M.-A. Lachaux, P. Stock, S. Subramanian, S. Yang, S. Antoniak, T. L. Scao, T. Gervet, T. Lavril, T. Wang, T. Lacroix, and W. E. Sayed, "Mixtral of Experts," 2024. [Online]. Available:<https://arxiv.org/abs/2401.04088>
- <span id="page-13-10"></span>[27] S. L. Jiashi Li, "FlashMLA: Efficient MLA decoding kernels," 2025. [Online]. Available:<https://github.com/deepseek-ai/FlashMLA>
- <span id="page-13-11"></span>[28] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei, "Scaling Laws for Neural Language Models," 2020. [Online]. Available: <https://arxiv.org/abs/2001.08361>
- <span id="page-13-14"></span>[29] Y. Lei, D. Lee, L. Zhao, D. Kurniawan, C. Kim, H. Jeong, C. Kim, H. Choi, L. Yu, A. Krishnamurthy, J. Sherry, and E. Nurvitadhi, "FAST: An Efficient Scheduler for All-to-All GPU Communication," 2025. [Online]. Available:<https://arxiv.org/abs/2505.09764>
- <span id="page-13-20"></span>[30] J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao, X. Dang, C. Gan, and S. Han, "AWQ: Activationaware Weight Quantization for On-Device LLM Compression and Acceleration," in *Proceedings of Machine Learning and Systems*, 2024. [Online]. Available: [https://proceedings.mlsys.org/paper](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf) files/paper/ [2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf)
- <span id="page-13-21"></span>[31] Y. Lin, H. Tang, S. Yang, Z. Zhang, G. Xiao, C. Gan, and S. Han, "QServe:W4A8KV4 Quantization and System Co-design for Efficient LLM Serving," in *Eighth Conference on Machine Learning and Systems*, 2025. [Online]. Available:<https://openreview.net/forum?id=1FfmStySS1>
- <span id="page-13-0"></span>[32] A. Liu, B. Feng, B. Wang, B. Wang, B. Liu, C. Zhao, C. Dengr, C. Ruan, D. Dai, D. Guo, D. Yang, D. Chen, D. Ji, E. Li, F. Lin, F. Luo, G. Hao, G. Chen, G. Li, H. Zhang, H. Xu, H. Yang, H. Zhang, H. Ding, H. Xin, H. Gao, H. Li, H. Qu, J. Cai, J. Liang, J. Guo, J. Ni, J. Li, J. Chen, J. Yuan, J. Qiu, J. Song, K. Dong, K. Gao, K. Guan, L. Wang, L. Zhang, L. Xu, L. Xia, L. Zhao, L. Zhang, M. Li, M. Wang, M. Zhang, M. Zhang, M. Tang, M. Li, N. Tian, P. Huang, P. Wang, P. Zhang, Q. Zhu, Q. Chen, Q. Du, R. Chen, R. Jin, R. Ge, R. Pan, R. Xu, R. Chen, S. Li, S. Lu, S. Zhou, S. Chen, S. Wu, S. Ye, S. Ma, S. Wang, S. Zhou, S. Yu, S. Zhou, S. Zheng, T. Wang, T. Pei, T. Yuan, T. Sun, W. Xiao, W. Zeng, W. An, W. Liu, W. Liang, W. Gao, W. Zhang, X. Li, X. Jin, X. Wang, X. Bi, X. Liu, X. Wang, X. Shen, X. Chen, X. Chen, X. Nie, X. Sun, X. Wang, X. Liu, X. Xie, X. Yu, X. Song, X. Zhou, X. Yang, X. Lu, X. Su, Y. Wu, Y. Li, Y. Wei, Y. Zhu, Y. Xu, Y. Huang, Y. Li, Y. Zhao, Y. Sun, Y. Li, Y. Wang, Y. Zheng, Y. Zhang, Y. Xiong, Y. Zhao, Y. He, Y. Tang, Y. Piao, Y. Dong, Y. Tan, Y. Liu, Y. Wang, Y. Guo, Y. Zhu, Y. Wang, Y. Zou, Y. Zha, Y. Ma, Y. Yan, Y. You, Y. Liu, Z. Ren, Z. Ren, Z. Sha, Z. Fu, Z. Huang, Z. Zhang, Z. Xie, Z. Hao, Z. Shao, Z. Wen, Z. Xu, Z. Zhang, Z. Li, Z. Wang, Z. Gu, Z. Li, and Z. Xie, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model," 2024. [Online]. Available:<https://arxiv.org/abs/2405.04434>
- <span id="page-13-3"></span>[33] A. Meta, "The llama 4 herd: The beginning of a new era of natively multimodal ai innovation," 2025. [Online]. Available: <https://ai.meta.com/blog/llama-4-multimodal-intelligence/>

- <span id="page-14-8"></span>[34] S. Nie, F. Zhu, Z. You, X. Zhang, J. Ou, J. Hu, J. Zhou, Y. Lin, J.-R. Wen, and C. Li, "Large Language Diffusion Models," 2025. [Online]. Available:<https://arxiv.org/abs/2502.09992>
- <span id="page-14-13"></span>[35] NVIDIA, "NVIDIA V100 GPU," 2017. [Online]. Available: [https://images.nvidia.com/content/volta-architecture/pdf/volta](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)[architecture-whitepaper.pdf](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)
- <span id="page-14-14"></span>[36] NVIDIA, "NVIDIA A100 GPU," 2020. [Online]. Available: [https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf) [a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf)
- <span id="page-14-28"></span>[37] NVIDIA, "NVIDIA DGX-B200," 2024. [Online]. Available: [https:](https://www.nvidia.com/en-us/data-center/dgx-b200/) [//www.nvidia.com/en-us/data-center/dgx-b200/](https://www.nvidia.com/en-us/data-center/dgx-b200/)
- <span id="page-14-3"></span>[38] NVIDIA, "NVIDIA GB200 NVL72," 2024. [Online]. Available: <https://www.nvidia.com/en-us/data-center/gb200-nvl72/>
- <span id="page-14-15"></span>[39] NVIDIA, "NVIDIA H100 GPU," 2024. [Online]. Available: [https://resources.nvidia.com/en-us-hopper-architecture/nvidia](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-tensor-core-gpu-datasheet)[tensor-core-gpu-datasheet](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-tensor-core-gpu-datasheet)
- <span id="page-14-26"></span>[40] Nvidia, "Dynamo," 2025. [Online]. Available: [https://github.com/ai](https://github.com/ai-dynamo/dynamo?tab=readme-ov-file)[dynamo/dynamo?tab=readme-ov-file](https://github.com/ai-dynamo/dynamo?tab=readme-ov-file)
- <span id="page-14-16"></span>[41] NVIDIA, "NVIDIA Blackwell Architecture Technical Brief," 2025. [Online]. Available: [https://resources.nvidia.com/en-us-blackwell](https://resources.nvidia.com/en-us-blackwell-architecture)[architecture](https://resources.nvidia.com/en-us-blackwell-architecture)
- <span id="page-14-23"></span>[42] NVIDIA, "NVIDIA DGX H100," 2025. [Online]. Available: [https://lambda.ai/hubfs/4.%20Resources/Datasheets/NVIDIA%](https://lambda.ai/hubfs/4.%20Resources/Datasheets/NVIDIA%20DGX/2024-04-nvidia-dgx-h100-datasheet-nvidia-us-web.pdf) [20DGX/2024-04-nvidia-dgx-h100-datasheet-nvidia-us-web.pdf](https://lambda.ai/hubfs/4.%20Resources/Datasheets/NVIDIA%20DGX/2024-04-nvidia-dgx-h100-datasheet-nvidia-us-web.pdf)
- <span id="page-14-20"></span>[43] NVIDIA, "NVIDIA Quantum-X800 InfiniBand Switches," 2025. [Online]. Available: [https://nvdam.widen.net/s/nfdzskhmnc/infiniband](https://nvdam.widen.net/s/nfdzskhmnc/infiniband-datasheet-quantum-family-3231555)[datasheet-quantum-family-3231555](https://nvdam.widen.net/s/nfdzskhmnc/infiniband-datasheet-quantum-family-3231555)
- <span id="page-14-29"></span>[44] Y. Pan, Z. Xia, P.-K. Hsu, L. Hu, H. Kim, J. Sharda, M. Zhou, N. S. Kim, S. Yu, T. Rosing, and M. Kang, "Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving," in *Proceedings of the 58th IEEE/ACM International Symposium on Microarchitecture*, ser. MICRO '25. New York, NY, USA: Association for Computing Machinery, 2025, p. 1–17. [Online]. Available:<https://doi.org/10.1145/3725843.3756043>
- <span id="page-14-30"></span>[45] J. Park, J. Choi, K. Kyung, M. J. Kim, Y. Kwon, N. S. Kim, and J. Ahn, "AttAcc! Unleashing the Power of PIM for Batched Transformer-based Generative Model Inference," in *ASPLOS, Volume 2*, 2024, p. 103–119. [Online]. Available:<https://doi.org/10.1145/3620665.3640422>
- <span id="page-14-27"></span>[46] P. Patel, E. Choukse, C. Zhang, A. Shah, ´I. Goiri, S. Maleki, and R. Bianchini, "Splitwise: Efficient Generative LLM Inference Using Phase Splitting," in *ISCA*, 2024. [Online]. Available: [https:](https://doi.org/10.1109/ISCA59077.2024.00019) [//doi.org/10.1109/ISCA59077.2024.00019](https://doi.org/10.1109/ISCA59077.2024.00019)
- <span id="page-14-31"></span>[47] H. Peng, K. Wu, Y. Wei, G. Zhao, Y. Yang, Z. Liu, Y. Xiong, Z. Yang, B. Ni, J. Hu, R. Li, M. Zhang, C. Li, J. Ning, R. Wang, Z. Zhang, S. Liu, J. Chau, H. Hu, and P. Cheng, "FP8-LM: Training FP8 Large Language Models," 2023. [Online]. Available: <https://arxiv.org/abs/2310.18313>
- <span id="page-14-5"></span>[48] R. Pope, S. Douglas, A. Chowdhery, J. Devlin, J. Bradbury, J. Heek, K. Xiao, S. Agrawal, and J. Dean, "Efficiently Scaling Transformer Inference," in *Efficiently Scaling Transformer Inferenc*, 2023. [Online]. Available: [https://proceedings.mlsys.org/paper](https://proceedings.mlsys.org/paper_files/paper/2023/hash/c4be71ab8d24cdfb45e3d06dbfca2780-Abstract-mlsys2023.html) files/paper/2023/ [hash/c4be71ab8d24cdfb45e3d06dbfca2780-Abstract-mlsys2023.html](https://proceedings.mlsys.org/paper_files/paper/2023/hash/c4be71ab8d24cdfb45e3d06dbfca2780-Abstract-mlsys2023.html)
- <span id="page-14-21"></span>[49] L. Poutievski, O. Mashayekhi, J. Ong, A. Singh, M. Tariq, R. Wang, J. Zhang, V. Beauregard, P. Conner, S. Gribble, R. Kapoor, S. Kratzer, N. Li, H. Liu, K. Nagaraj, J. Ornstein, S. Sawhney, R. Urata, L. Vicisano, K. Yasumura, S. Zhang, J. Zhou, and A. Vahdat, "Jupiter Evolving: Transforming Google's Datacenter Network via Optical Circuit Switches and Software-Defined Networking," in *Proceedings of ACM SIGCOMM 2022*, 2022, p. 66–85. [Online]. Available: <https://doi.org/10.1145/3544216.3544265>
- <span id="page-14-24"></span>[50] SCALE-SNU, "LLMSimulator — GitHub Repository," 2025. [Online]. Available:<https://github.com/scale-snu/LLMSimulator>
- <span id="page-14-9"></span>[51] N. Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need," 2019. [Online]. Available:<https://arxiv.org/abs/1911.02150>
- <span id="page-14-10"></span>[52] N. Shazeer, "GLU Variants Improve Transformer," 2020. [Online]. Available:<https://arxiv.org/abs/2002.05202>
- <span id="page-14-2"></span>[53] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, "Outrageously Large Neural Networks: The Sparsely-

- Gated Mixture-of-Experts Layer," 2017. [Online]. Available: [https:](https://arxiv.org/abs/1701.06538) [//arxiv.org/abs/1701.06538](https://arxiv.org/abs/1701.06538)
- <span id="page-14-12"></span>[54] M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, and B. Catanzaro, "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism," 2020. [Online]. Available: <https://arxiv.org/abs/1909.08053>
- <span id="page-14-11"></span>[55] J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu, "RoFormer: Enhanced Transformer with Rotary Position Embedding," 2023. [Online]. Available:<https://arxiv.org/abs/2104.09864>
- <span id="page-14-4"></span>[56] A. Vahdat, "Ironwood: The First Google TPU for the Age of Inference," 2025. [Online]. Available: [https://blog.google/products/](https://blog.google/products/google-cloud/ironwood-tpu-age-of-inference/) [google-cloud/ironwood-tpu-age-of-inference/](https://blog.google/products/google-cloud/ironwood-tpu-age-of-inference/)
- <span id="page-14-17"></span>[57] A. Vahdat and M. Lohmeyer, "Enabling next-generation AI workloads: Announcing TPU v5p and AI Hypercomputer," 2023. [Online]. Available: [https://cloud.google.com/blog/products/ai-machine-learning/](https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-tpu-v5p-and-ai-hypercomputer?hl=en) [introducing-cloud-tpu-v5p-and-ai-hypercomputer?hl=en](https://cloud.google.com/blog/products/ai-machine-learning/introducing-cloud-tpu-v5p-and-ai-hypercomputer?hl=en)
- <span id="page-14-0"></span>[58] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. u. Kaiser, and I. Polosukhin, "Attention is All you Need," in *Proceedings of the 31st International Conference on Neural Information Processing Systems*, 2017. [Online]. Available: <https://dl.acm.org/doi/10.5555/3295222.3295349>
- <span id="page-14-7"></span>[59] S. Williams, A. Waterman, and D. Patterson, "Roofline: an insightful visual performance model for multicore architectures," *Commun. ACM*, vol. 52, p. 65–76, 2009. [Online]. Available: [https://doi.org/10.1145/](https://doi.org/10.1145/1498765.1498785) [1498765.1498785](https://doi.org/10.1145/1498765.1498785)
- <span id="page-14-1"></span>[60] xAI, "grok1," 2024. [Online]. Available: [https://github.com/xai-org/](https://github.com/xai-org/grok-1) [grok-1](https://github.com/xai-org/grok-1)
- <span id="page-14-32"></span>[61] G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han, "SmoothQuant: accurate and efficient post-training quantization for large language models," in *Proceedings of the 40th International Conference on Machine Learning*, 2023. [Online]. Available: [https:](https://dl.acm.org/doi/10.5555/3618408.3619993) [//dl.acm.org/doi/10.5555/3618408.3619993](https://dl.acm.org/doi/10.5555/3618408.3619993)
- <span id="page-14-18"></span>[62] S. Yun, K. Kyung, J. Cho, J. Choi, J. Kim, B. Kim, S. Lee, K. Sohn, and J. Ahn, "Duplex: A Device for Large Language Models with Mixture of Experts, Grouped Query Attention, and Continuous Batching," in *MICRO*, 2024, pp. 1429–1443. [Online]. Available: <https://ieeexplore.ieee.org/abstract/document/10764531>
- <span id="page-14-19"></span>[63] S. Zhang, N. Zheng, H. Lin, Z. Jiang, W. Bao, C. Jiang, Q. Hou, W. Cui, S. Zheng, L.-W. Chang, Q. Chen, and X. Liu, "COMET: Fine-grained Computation-communication Overlapping for Mixture-of-Experts," in *Proceedings of Machine Learning and Systems*, 2025. [Online]. Available:<https://openreview.net/forum?id=fGgQS5VW09>
- <span id="page-14-33"></span>[64] Y. Zhang, P. Zhang, M. Huang, J. Xiang, Y. Wang, C. Wang, Y. Zhang, L. Yu, C. Liu, and W. Lin, "QQQ: Quality Quattuor-Bit Quantization for Large Language Models," 2024. [Online]. Available: <https://arxiv.org/abs/2406.09904>
- <span id="page-14-22"></span>[65] C. Zhao, C. Deng, C. Ruan, D. Dai, H. Gao, J. Li, L. Zhang, P. Huang, S. Zhou, S. Ma, W. Liang, Y. He, Y. Wang, Y. Liu, and Y. Wei, "Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures," in *Proceedings of the 52nd Annual International Symposium on Computer Architecture*, ser. ISCA '25. New York, NY, USA: Association for Computing Machinery, 2025, p. 1731–1745. [Online]. Available:<https://doi.org/10.1145/3695053.3731412>
- <span id="page-14-25"></span>[66] C. Zhao, S. Zhou, L. Zhang, C. Deng, Z. Xu, Y. Liu, K. Yu, J. Li, and L. Zhao, "DeepEP: an efficient expert-parallel communication library," [https://github.com/deepseek-ai/DeepEP,](https://github.com/deepseek-ai/DeepEP) 2025.
- <span id="page-14-34"></span>[67] Y. Zhao, C.-Y. Lin, K. Zhu, Z. Ye, L. Chen, S. Zheng, L. Ceze, A. Krishnamurthy, T. Chen, and B. Kasikci, "Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving," in *Proceedings of Machine Learning and Systems*, 2024. [Online]. Available: [https://proceedings.mlsys.org/paper](https://proceedings.mlsys.org/paper_files/paper/2024/hash/5edb57c05c81d04beb716ef1d542fe9e-Abstract-Conference.html) files/paper/2024/ [hash/5edb57c05c81d04beb716ef1d542fe9e-Abstract-Conference.html](https://proceedings.mlsys.org/paper_files/paper/2024/hash/5edb57c05c81d04beb716ef1d542fe9e-Abstract-Conference.html)
- <span id="page-14-6"></span>[68] Y. Zhong, S. Liu, J. Chen, J. Hu, Y. Zhu, X. Liu, X. Jin, and H. Zhang, "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving," in *18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)*. USENIX Association, 2024, pp. 193–210. [Online]. Available: <https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin>

# APPENDIX

TABLE V Symbols used throughout this paper, their descriptions, and the exemplar parameters used in DeepSeek-R1 [10]

| Term                                       | Description                        | DeepSeek-R1 | Term                                                                                                                                | Description                        | DeepSeek-R1                                    |
|--------------------------------------------|------------------------------------|-------------|-------------------------------------------------------------------------------------------------------------------------------------|------------------------------------|------------------------------------------------|
| TP/DP/EP                                   | Tensor / Data / Expert Parallelism | -           | $\mathbf{O}_t$                                                                                                                      | Context output                     | -                                              |
| $\overline{\mathrm{deg_{TP}/_{DP}/_{EP}}}$ | TP / DP / EP degree                | -           | $\mathbf{U}_t$                                                                                                                      | Final attention output / FFN input | -                                              |
| В                                          | Batch size                         | -           | $\mathbf{H}_t$                                                                                                                      | FFN output / Decoder block input   | -                                              |
| $\overline{L}$                             | Sequence length                    | -           | $RP_{device}$                                                                                                                       | Ridge point of device              | -                                              |
| $\overline{n_{\mathrm{dec}}}$              | Decoder blocks                     | 61          | Q, K, V Query, Key, Value                                                                                                           |                                    | -                                              |
| $d_{\rm emb}$                              | Embedding dimension                | 7168        | $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ Weight for Q, K, V generation                                                            |                                    | -                                              |
| $n_{\rm head}$                             | Number of heads                    | 128         | $\mathbf{W}_{\mathrm{attn\_out}}$                                                                                                   | Out projection weight in attention | (16384, 7168)                                  |
| $d_{\text{head}}$                          | Head dimension                     | 128         | $\mathbf{W}_{\mathrm{gate}},\mathbf{W}_{\mathrm{up}}$                                                                               | Weight for gate/up in FFN          | (7168, 18432)                                  |
| $d_{\text{dec}}$                           | Decompressed Q, KV dimension       | 16384       | $\mathbf{W}_{\text{down}}$                                                                                                          | Weight for down in FFN             | (18432, 7168)                                  |
| $d_{\mathrm{Qco}}, d_{\mathrm{KVco}}$      | Compressed Q, KV dimension         | 1536, 512   | $\mathbf{W}_{\mathrm{route}}$                                                                                                       | MoE route weight                   | (7168, 256)                                    |
| $d_{\text{RoPE}}$                          | Rotary PE dimension                | 64          | $\mathbf{W}_{\mathrm{exp}_{n},\mathrm{gate}},\mathbf{W}_{\mathrm{exp}_{n},\mathrm{up}},\mathbf{W}_{\mathrm{exp}_{n},\mathrm{down}}$ | MoE up/down projection weights     | (7168, 2048),<br>(7168, 2048),<br>(2048, 7168) |
| $\overline{d_{\text{MoE}}}$                | MoE dimension                      | 2048        | $\mathbf{W}_{\text{CQ}},\mathbf{W}_{\text{CKV}}$                                                                                    | Q comp / KV compression            | (7168, 1536),<br>(7168, 512)                   |
| $\overline{n_{\mathrm{k}}}$                | Top-k experts                      | 8           | $\mathbf{W}_{\mathrm{DQ}}$                                                                                                          | Q decompression weight             | (1536, 16384)                                  |
| $\overline{n_{\mathrm{e}}}$                | Number of experts                  | 256         | $\mathbf{W}_{\mathrm{DK}},\mathbf{W}_{\mathrm{DV}}$                                                                                 | K, V decompression weights         | (512, 16384)                                   |
| $\overline{\mathbf{Q}_{\text{NoPE}}}$      | Query vector (No RoPE)             | (1, 16384)  | $\mathbf{W}_{\mathrm{RQ}}$                                                                                                          | RoPE Q weight                      | (1536, 8192)                                   |
| $\overline{\mathbf{Q}_{\mathrm{RoPE}}}$    | Query after RoPE                   | (1, 8192)   | $\mathbf{W}_{\mathrm{RK}}$                                                                                                          | RoPE K weight                      | (7168, 64)                                     |
| $\mathbf{K}_{\mathrm{RoPE}}$               | Key vector for RoPE                | (1, 64)     | $\mathbf{C}_Q$                                                                                                                      | Latent Q (compressed)              | -                                              |
| $\overline{\mathbf{S}_{\mathrm{RoPE}}}$    | Score output with RoPE             | -           | $\mathbf{C}_{\mathrm{KV}}$                                                                                                          | Latent KV (compressed)             | -                                              |
| $\overline{\mathbf{S}_{\text{NoPE}}}$      | Score output without RoPE          | -           | $n_{\rm acc}$                                                                                                                       | Number of accelerators             | -                                              |
| $\overline{\mathbf{S}_t}$                  | Final score output                 | -           | -                                                                                                                                   | -                                  |                                                |