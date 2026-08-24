# Training Long-Context LLMs Efficiently via Chunk-wise Optimization

#### Wenhao Li1,2, Yuxin Zhang<sup>1</sup> , Gen Luo<sup>2</sup> , Daohai Yu<sup>1</sup> , Rongrong Ji<sup>1</sup>

<sup>1</sup>Key Laboratory of Multimedia Trusted Perception and Efficient Computing, Ministry of Education of China, Xiamen University, 361005, P.R. China <sup>2</sup>OpenGVLab, Shanghai AI Laboratory

Correspondence: [rrji@xmu.edu.cn](mailto:rrji@xmu.edu.cn)

### Abstract

While long-context large language models (LLMs) exhibit remarkable document processing capabilities, their prohibitively high training costs often hinder customized applications. To mitigate this issue, we propose *Sequential Chunk-wise Optimization* (SeCO), a memory-efficient training paradigm that partitions lengthy inputs into manageable chunks. Each chunk independently constructs its computational graph and performs localized backpropagation, ensuring that only one chunk's forward activations are stored in memory. Building on SeCO, we further introduce *Sparse Chunk-wise Optimization* (SpaCO), which reduces computational overhead by selectively propagating gradients to specific chunks and incorporates a carefully designed compensation factor to ensure unbiased gradient estimation. SpaCO decouples the computational cost of backpropagation from the context length, enabling training time to gradually converge to inference time as sequences become longer. Implemented as lightweight training wrappers, both SeCO and SpaCO offer substantial practical benefits. For example, when fine-tuning an 8B model with LoRA on a single RTX 3090 GPU, SeCO expands maximum sequence length from 1K to 16K tokens, while SpaCO demonstrates accelerated training speed—achieving up to 3× faster than SeCO under the same experimental setup. These innovations provide new insights into optimizing long-context models, making them more accessible for practical applications. We have open-sourced the code at [here.](https://github.com/wenhaoli-xmu/seco)

### 1 Introduction

Recent advancements in long-context LLMs [\(Chen](#page-8-0) [et al.,](#page-8-0) [2024;](#page-8-0) [An et al.,](#page-8-1) [2024;](#page-8-1) [Peng et al.,](#page-8-2) [2024;](#page-8-2) [Zhao](#page-9-0) [et al.,](#page-9-0) [2024\)](#page-9-0) have demonstrated unprecedented capabilities in processing lengthy documents, offering superior retrieval quality compared to retrievalaugmented generation (RAG) approaches [\(Liu,](#page-8-3) [2022\)](#page-8-3), making them particularly valuable for commercial applications requiring nuanced document understanding.

However, fine-tuning these models faces significant resource challenges: *(i) Time Overhead:* The quadratic scaling of attention mechanisms leads to prohibitive training time [\(de Vries,](#page-8-4) [2023\)](#page-8-4). *(ii) Memory Constraints:* Despite optimizations like FlashAttention [\(Dao,](#page-8-5) [2024\)](#page-8-5), the storage requirements for forward activations still increases linearly with sequence length, quickly depleting GPU memory. As a result, fine-tuning 8B models [\(Meta-](#page-8-6)[AI,](#page-8-6) [2024\)](#page-8-6) with LoRA [\(Hu et al.,](#page-8-7) [2022\)](#page-8-7) on a single RTX 3090 GPU is limited to sequences of only 1K tokens.

Existing architectural modifications, exemplified by LongLoRA's S 2 -attention [\(Chen et al.,](#page-8-0) [2024\)](#page-8-0), aim to alleviate these issues by reducing computational overhead to sub-quadratic through attention approximation. However, these methods incur gradient accuracy compromises while offering limited resource savings,[1](#page-0-0) motivating our exploration of alternative efficiency improvements strategies.

We introduce *Sequential Chunk-wise Optimization* (SeCO), a novel training method that preserves exact gradients while dramatically reducing memory consumption. The key innovation of SeCO is the application of gradient checkpointing [\(Bulatov,](#page-8-8) [2018;](#page-8-8) [Chen et al.,](#page-8-9) [2016\)](#page-8-9) along the sequence dimension using chunk-level checkpoints. This approach represents a fundamental departure from traditional gradient checkpointing, which typically employs a fixed number of checkpoints for static layer-wise or block-wise partitioning of the computational graph. Unlike these conventional techniques, where memory requirements for forward activations scale linearly with sequence length, SeCO maintains a con-

<span id="page-0-0"></span><sup>1</sup>As shown in Figure 1 (*Mid*) of the LongLoRA paper, the proposed method exhibits memory scaling patterns similar to those of full fine-tuning baseline, achieving only about a 2-fold extension in sequence length.

<span id="page-1-0"></span>> **[图片提取文字 (无描述)]:**
> SeCO step 2 SeCO step 1 error (chunk 4) KV (chunk 4)  $m_4$  $m_3$ Θ SeCO step 4 SeCO step 3 error (chunk 2)  $m_2$ KV cache (chunk 1) language modeling error (chunk 1)
![](_page_1_Figure_0.jpeg)

Figure 1: (*Left*) Computational graph for chunk-wise optimization with k = 4 chunks. The dense connections among KV caches (green arrows) complicate memory management, leading popular training frameworks [\(MicroSoft,](#page-8-10) [2021;](#page-8-10) [Gugger et al.,](#page-8-11) [2022\)](#page-8-11) to rely on end-to-end parallel training. (*Right*) By analyzing the topology of this graph, we propose SeCO, a bootstrapping method leveraging gradient checkpointing along the sequence dimension. SeCO ensures that only the computational graph of a single chunk is stored at any time.

stant overhead for forward activations, regardless of sequence length. This innovation achieves orderof-magnitude memory reduction while maintaining manageable training time overhead, establishing it as an efficient solution for fine-tuning long-context LLMs under resource-constrained conditions.

While SeCO successfully mitigates memory constraints in long-context LLM fine-tuning, it maintains computational overhead comparable to naive parallel training. This computational burden significantly undermines its applicability for processing extended sequences, thereby limiting its practical utility. To address this limitation, we propose *Sparse Chunk-wise Optimization* (SpaCO), an enhanced variant of SeCO that achieves substantial computational savings. SpaCO preserves the integrity of forward propagation while implementing selective backpropagation through a fixed subset of chunks. This modification decouples computational cost from sequence length during gradient computation. Our theoretical framework reveals a crucial architectural insight: The gradient chain length between key-value (KV) cache chunks exhibits inherent boundedness determined by model depth (as also noted in [Dai et al.,](#page-8-12) [2019\)](#page-8-12). This fundamental property enables SpaCO to employ randomized chunk sampling while preserving unbiased gradient estimation, achieving significant computational reduction without compromising theoretical guarantees (for more detailed explanation, please refer to Section [5\)](#page-4-0).

Empirical evaluations highlight the substantial practical advantages of SeCO and SpaCO:

- Scalability: The memory overhead for SeCO and SpaCO scales minimally with increasing sequence length, as the only contributing factor is the storage of the KV cache. Moreover, SpaCO's training time converges to inference time as the sequence length expands, demonstrating efficient computational scaling.
- Performance: Although SpaCO does not compute exact gradients like SeCO, it incurs only a small performance gap. Specifically, at a sparsity ratio of 1/8, the language modeling error increases by less than 0.1 compared to exact gradient training.

### Our contributions are threefold:

- A memory efficient training paradigm (SeCO) that enables long-context fine-tuning through sequence dimensional gradient checkpointing.
- A computation-efficient extension (SpaCO) leveraging sparsification, with theoretical guarantees of unbiased gradient estimation.
- Open-source implementations that achieve up to an order of magnitude training sequence length improvements on consumer hardware.

### 2 Related Works

Long-Context LLMs. Efforts to extend the context window of LLMs primarily rely on augmenting positional embeddings and applying limited posttraining to adapt models pre-trained on shorter contexts [\(Chen et al.,](#page-8-0) [2024;](#page-8-0) [Peng et al.,](#page-8-2) [2024\)](#page-8-2). While

> **[图片提取文字 (无描述)]:**
> **High** Memory Efficiency SeCO SpaCO Low Time Efficiency High Time Efficiency Gradient Checkpointing ZeRO3 Naive Parallel Training Low Memory Efficiency
![](_page_2_Figure_0.jpeg)

Figure 2: Qualitative comparison of different methods. *(i)* SeCO achieves significant memory reduction while maintaining time efficiency comparable to layerlevel gradient checkpointing. *(ii)* Building upon SeCO, SpaCO significantly reduces computational overhead, making the training time converges to inference time as the sequence length expands.

these methods are effective, the inherent quadratic computational complexity of attention mechanisms renders long-context training prohibitively expensive [\(de Vries,](#page-8-4) [2023;](#page-8-4) [Hu et al.,](#page-8-13) [2024\)](#page-8-13). Recent works, such as LongLoRA [\(Chen et al.,](#page-8-0) [2024\)](#page-8-0), address this issue using S 2 -attention, which achieves linear computation scalability. However, its architectural modification introduces biased gradient computation.

Gradient Checkpointing. Gradient checkpointing techniques [\(Chen et al.,](#page-8-9) [2016;](#page-8-9) [Bulatov,](#page-8-8) [2018\)](#page-8-8) optimize memory consumption by recomputing activations during backpropagation rather of storing them. In Transformer [\(Vaswani et al.,](#page-8-14) [2017\)](#page-8-14) architectures, conventional layer-level checkpointing offers limited benefits for long sequences due to the static partitioning of the computational graph. Gradient checkpointing applied along the sequence dimension enables maintaining a constant memory footprint for storing forward activations, presenting substantial advantages. Nevertheless, current implementations in mainstream deep learning frameworks [\(Chintala et al.,](#page-8-15) [2016;](#page-8-15) [Google,](#page-8-16) [2015\)](#page-8-16) are primarily designed for checkpointing within individual forward pass, lacking the capability to handle concatenated computational graphs that emerge across multiple iterative processes.

Gradient Estimation. The concept of approximate gradient predates modern deep learning, exemplified by stochastic gradient descent's use of mini-batch [\(Bottou and Bousquet,](#page-8-17) [2007\)](#page-8-17). SpaCO introduces a novel paradigm that aligns with the philosophy of SGD: by utilizing a limited number

of gradient propagation pathways to estimate the underlying true gradient, it significantly reduces computational overhead while maintaining acceptable performance.

### 3 Preliminary

When processing large amounts of data all at once, GPU threads can become saturated, causing parallel processing time to scale linearly with data size, offering no advantage over sequential processing while increasing memory usage. To address this, efficient LLM serving frameworks such as vLLM [\(Kwon et al.,](#page-8-18) [2023\)](#page-8-18) and FlashInfer [\(Ye et al.,](#page-8-19) [2025\)](#page-8-19) adopt a chunk pre-filling strategy, splitting long contexts into smaller chunks and processing them sequentially. We extend this idea from LLM inference to LLM training.

Computational Graph. For an input sequence X partitioned into k chunks {xj} k <sup>j</sup>=1, let m<sup>j</sup> denote the KV cache and J<sup>j</sup> the error component for chunk j. The model f with parameter Θ processes chunks sequentially:

<span id="page-2-1"></span>
$$(J_j, m_j) = f(x_j; m_1, m_2, ..., m_{j-1}; \Theta).$$
 (1)

The parameter gradient combines direct and indirect contributions through KV cache:

$$\nabla_{\Theta} J_{j} = \underbrace{\frac{\partial J_{j}}{\partial \Theta}}_{\text{Direct term}} + \sum_{i=1}^{j} \underbrace{\frac{\mathrm{d} J_{j}}{\mathrm{d} m_{i}} \frac{\partial m_{i}}{\partial \Theta}}_{\text{Indirect contributions}}. \quad (2)$$

Due to the iterative nature of f(·), the computation of dJj/dm<sup>i</sup> involves nested dependencies:

<span id="page-2-0"></span>
$$\frac{\mathrm{d}J_{j}}{\mathrm{d}m_{i}} \frac{\partial m_{i}}{\partial \Theta} = \frac{\partial J_{j}}{\partial m_{i}} \frac{\partial m_{i}}{\partial \Theta} + \sum_{i < t_{1} \leq j} \frac{\partial J_{j}}{\partial m_{t_{1}}} \frac{\partial m_{t_{1}}}{\partial m_{i}} \frac{\partial m_{i}}{\partial \Theta} + \sum_{i < t_{1} < t_{2} \leq j} \frac{\partial J_{j}}{\partial m_{t_{2}}} \frac{\partial m_{t_{2}}}{\partial m_{t_{2}}} \frac{\partial m_{t_{1}}}{\partial m_{i}} \frac{\partial m_{i}}{\partial \Theta} + \dots$$

$$(3)$$

Although Eq. [\(3\)](#page-2-0) appears complex, it fundamentally demonstrates that gradients propagate through all possible multi-hop paths among {mt} j t=i . To visualize this process, we present partial derivatives ∂ △/∂ ⃝ as directed edges from △ to ⃝, forming the complete computational graph of gradient propagation from {Jj} k <sup>j</sup>=1 to Θ as illustrated in Figure [1.](#page-1-0) This graph explicitly captures the computational dependencies for chunk-wise optimization.

<span id="page-3-1"></span>> **[图片提取文字 (无描述)]:**
> ) forward prop (no grad) stage 1 chunk 2 forward prop backprop memory allocated stage 2 memory released chunk 1
![](_page_3_Figure_0.jpeg)

Figure 3: A visualized illustration of Algorithm 1. Stage 1 corresponds to the first for-loop, generating KV caches for all data chunks through inference-mode, serving as checkpoints. Stage 2 corresponds to the second forloop, where the computational graph is constructed and localized backpropagation is performed.

**Gradient Checkpointing.** Gradient checkpointing trades computational time for reduced memory usage. The fundamental principle guiding checkpoint placement requires that the complete subsequent computational graph must be reconstructible using only these designed checkpoints and existing leaf nodes. As formalized in Eq. (1), the output of chunk-j is uniquely determined by two components: (i) the preceding KV cache sequence  $\{m_i\}_{i=1}^{j-1}$  and (ii) the model parameters  $\Theta$  (leaf nodes). By storing these KV caches, we enable the complete reconstruction of any subsequent chunk's output during backpropagation, making them ideal checkpoint candidates. During the forward propagation, only the checkpointed KV caches need to be computed and stored, eliminating the need to retain intermediate activations.

### **Sequential Chunk-wise Optimization**

SeCO is a plain version of this chunk-wise optimization that does not save any computation and obtains exact gradients (verified in Appendix D).

#### 4.1 Methodology

During forward propagation, we compute all chunks sequentially in inference mode to generate corresponding KV caches  $\{m'_1, m'_2, \dots, m'_k\}$ , where prime notation distinguishes inferencegenerated caches from training-phase counterparts.

For backpropagation, the computational graph topology in Figure 1 dictates a sequential reverseorder reconstruction strategy. For chunk j:

- 1. Reconstruct computational graph using Eq. (1) to compute error  $J_i$  and KV cache  $m_i$ .
- 2. Transfer gradients from  $m'_i$  to  $m_j$ .
- 3. Backpropagate  $J_i$  and  $m_i$  to accumulate gradients for model parameters and preceding checkpoints  $m'_1, \ldots, m'_{i-1}$ .

After processing all chunks, accumulated parameter gradients match those from naive parallel training modulo numerical precision. Implementation details follow Algorithm 1, and the corresponding visualization is presented in Figure 3.

### 4.2 Efficiency

We analyze the theoretical efficiency of SeCO, in terms of computation and storage.

Memory Savings. By reconstructing at most one chunk's computational graph at any given time, SeCO effectively prevents forward activations from scaling linearly with sequence length. This design reduces the memory requirements for storing forward activations by a factor of k. However, it is important to note that SeCO does not optimize fixed memory components such as optimizer states and model parameters, nor does it alleviate the memory overhead of the KV cache.

Computational Overhead. SeCO introduces two primary sources of computational overhead: (i) additional recomputation during backpropagation, and (ii) frequent kernel launches for small-scale tensor operations.

For the first component, since backpropagation typically requires approximately twice the FLOPs of forward propagation (DeepSpeed, 2021; Kaplan, 2019), the subgraph reconstruction introduces an estimated 33% computational overhead.

Regarding the second component, modern GPUs like the RTX 3090 contain fixed computational resources (82 streaming multiprocessors with 128 cores each). When using sufficiently large chunk sizes, these resources can achieve near-saturation utilization. Experimental results demonstrate that increasing chunk sizes beyond 128 yields diminishing returns, with only marginal reductions in computational time observed.

<span id="page-3-0"></span>Algorithm 1 Sequential Chunk-wise Optimization

**Require:** Model f, data  $X = \{x_1, x_2, \dots, x_k\}$ , parameters  $\Theta$ 

Ensure:  $\nabla_{\Theta}$ 

```
1: for i = 1 to k do
```

2: 
$$m'_i \leftarrow f(x_i; \{m'_i\}_{i=1}^{i-1}; \Theta)$$

3: end for

4: **for** i = k to 1 **do** 

5: 
$$J_i, m_i \leftarrow f(x_i; \{m_j'\}_{j=1}^{i-1}; \Theta)$$
  
6:  $m_i. \operatorname{grad} \leftarrow m_i'. \operatorname{grad}$ 

 $backprop(J_i)$ 

8: end for

<span id="page-4-1"></span>> **[图片提取文字 (无描述)]:**
> longest chain = 2 layer 3 layer 2 layer 1 chunk 1 chunk 2 chunk 3 chunk 4
![](_page_4_Picture_0.jpeg)

Figure 4: In the Transformer architecture, the gradient flow traverses through at most a number of KV caches equal to the layer depth. This observation was also highlighted in Transformer-XL [\(Dai et al.,](#page-8-12) [2019\)](#page-8-12).

### <span id="page-4-0"></span>5 Sparse Chunk-wise Optimization

While SeCO reduces memory consumption, it introduces additional computational overhead, further prolonging the already time-consuming training process. This limitation hinders its ability to handle ultra-long sequences efficiently. To alleviate this issue, we propose SpaCO, an improvement over SeCO. By introducing sparsification in backpropagation, SpaCO significantly accelerates training while maintaining memory efficiency.

The key insight stems from the observation that the checkpoints {m′ 1 , . . . , m′ k } enable independent computational graph construction for individual chunk through Eq. [\(1\)](#page-2-1). Capitalizing on this, SpaCO implements a stochastic backpropagation scheme that randomly selects a subset of chunks for gradient computation during each training iteration.

This sparsification, however, poses the risk of biased gradient estimation. In extreme cases where only one chunk is selected, the gradient flow between chunks is disrupted—akin to nonoverlapping chunked attention mechanisms, which constrain the model to local dependencies.

One might hypothesize that dense gradient propagation is essential for learning global patterns, given that the longest gradient chain spans all KV cache chunks. However, our theoretical analysis reveals that this is not necessary.

The Longest Gradient Chain. In the computational graph shown in Figure [1,](#page-1-0) the longest gradient chain is:

$$\frac{\partial J_4}{\partial m_4} \cdot \frac{\partial m_4}{\partial m_3} \cdot \frac{\partial m_3}{\partial m_2} \cdot \frac{\partial m_2}{\partial m_1} \cdot \frac{\partial m_1}{\partial \Theta}, \tag{4}$$

which spans all chunks. Omitting any chunk would break this chain, leading to biased gradient estimation. However, in the Transformer [\(Vaswani et al.,](#page-8-14) [2017\)](#page-8-14) architecture, KV cache chunks within the same layer are independent and computed in parallel. As a result, errors propagate from one KV cache chunk to another only between adjacent layers [\(Dai et al.,](#page-8-12) [2019\)](#page-8-12), as shown in Figure [4.](#page-4-1) Thus, the maximum gradient chain length is bounded by the number of layers. Theoretically, unbiased gradient estimation is achievable if the number of selected chunks meets the number of layers, ensuring sufficient coverage.

Challenges in Unbiased Estimation. While bounded gradient chain length suggests theoretical feasibility, practical implementation faces significant hurdles. Consider a DAG with n nodes and n(n − 1)/2 edges, as shown in Figure [5](#page-5-0) (*Left*). The number of p-length paths follows combinatorial principles:

$$d_p = \binom{n}{p+1} = \frac{n!}{(p+1)!(n-p-1)!}.$$
 (5)

Let superscripts d and s denote dense (k chunks) versus sparse (t chunks) configurations respectively. The path count ratio between these two configurations exhibits:

$$\frac{d_p^{\mathsf{d}}}{d_p^{\mathsf{s}}} = \frac{k(k-1)(k-2)\cdots(k-p)}{t(t-1)(t-2)\cdots(t-p)},\tag{6}$$

for t ≫ p, this simplifies to:

$$\frac{d_p^{\mathsf{d}}}{d_p^{\mathsf{s}}} \approx \left(\frac{k}{t}\right)^{p+1}.\tag{7}$$

This exponentially decaying pattern highlights a crucial insight: graph sparsification disproportionately weakens longer gradient chains. Since removing a node multiplicatively affects all paths passing through it, longer chains suffer a greater cumulative impact. Consequently, naive sparsification introduces systematic bias.

To ensure unbiased estimation, we must incorporate compensation mechanisms that counteracts this attenuation. A viable solution is to strategically apply scaling factors to the preserved paths, effectively rebalancing gradient contributions across different chain lengths.

Compensation Factor. To analyze gradient propagation under sparsification, we consider all gradient chains of length p in Eq. [\(3\)](#page-2-0), denoted by zp:

$$\mathbf{z}_{p} = \sum_{i < t_{1} < \dots < t_{p-1} \leq j} \frac{\partial J_{j}}{\partial m_{t_{p-1}}} \cdot \frac{\partial m_{t_{p-1}}}{\partial m_{t_{p-2}}} \cdot \dots \cdot \frac{\partial m_{i}}{\partial \Theta}.$$
(8)

<span id="page-5-0"></span>![](_page_5_Figure_0.jpeg)

(a) Dense graph (k chunks). (b) Sparse graph (t chunks).

Figure 5: SpaCO sparsifies the gradient flow among KV caches. (*Left*) The original graph. k=6. (*Right*) Only the gradient flow between t=4 chunks is retained. By adding a factor k/t to each path, the gradient computed from this sparse graph remains an unbiased estimate.

A crucial observation is that any gradient chain in  $\mathbf{z}_p$  necessitates the sampling of all its constituent chunks. The survival probability of such a chain under t-out-of-k sparse sampling can be derived as  $(t/k)^p$  based on the following reasoning:

- The initial term  $\partial J_j/\partial m_{t_{p-1}}$  survives with probability t/k.
- Given that the previous term survives, each subsequent term (except the last) also survives with probability t/k.

Using this survival probability, we can express the expected value of  $\mathbf{z}_p$  after sparsification, denoted as  $\bar{\mathbf{z}}_p$ :

$$\bar{\mathbf{z}}_p = \left(\frac{t}{k}\right)^p \mathbf{z}_p + \left(1 - \frac{t^p}{k^p}\right) 0 = \left(\frac{t}{k}\right)^p \mathbf{z}_p. \tag{9}$$

Given the complete gradient  $\mathbf{Z} = \sum_{p=1}^{\infty} \mathbf{z}_p$  from Eq. (3), its expectation after sparsification becomes:

$$\bar{\mathbf{Z}} = \frac{t}{k}\mathbf{z}_1 + \left(\frac{t}{k}\right)^2\mathbf{z}_2 + \left(\frac{t}{k}\right)^3\mathbf{z}_3 + \dots \quad (10)$$

To achieve unbiased gradient estimation ( $\bar{\mathbf{Z}} = \mathbf{Z}$ ), each gradient chain  $\mathbf{z}_p$  requires compensation by factor  $(k/t)^p$ . This multiplicative scaling counteracts the exponential decay induced by sparsity.

**Implementation.** The compensation factor is implemented through modifying backpropagation, which occurs during gradient computation: When calculating  $\partial m_i/\partial \{m_j'\}_{j=1}^{i-1}$ , we scale the gradient by k/t. Through the nested structure of  $f(\cdot)$ , this creates compound compensation where each p-length chain automatically accumulates  $(k/t)^p$  scaling through successive operations.

<span id="page-5-1"></span>Algorithm 2 Sparse Chunk-wise Optimization

**Require:** Model f, data  $X = \{x_1, x_2, \dots, x_k\}$ , parameters  $\Theta$ , fixed budget t

Ensure:  $\nabla_{\Theta}$ 

1: **for** i = 1 to k **do** 

2: 
$$m'_i \leftarrow f(x_i; \{m'_i\}_{i=1}^{i-1}; \Theta)$$

3: end for

4: Randomly select t distinct indices from  $\{1, ..., k\}$ , denoted as  $\mathcal{I}$ .

5: **for** i in  $\mathcal{I}$  **do** 

6: 
$$J_i, m_i \leftarrow f(x_i; \{m'_j\}_{j=1}^{i-1}; \Theta)$$

7: 
$$m_i$$
.grad  $\leftarrow \left(\frac{k}{t}\right) \cdot m_i'$ .grad

- 8: backprop( $J_i$ )
- 9: end for

Figure 5 (*Right*) illustrates this dynamic scaling mechanism, with full implementation details provided in Algorithm 2. This approach enhances computational efficiency while maintaining the statistical accuracy of full backpropagation.

### 6 Experiment

We designed experiments to address two key questions:

- How do SeCO and SpaCO compare to mainstream training methods in time and memory efficiency?
- Can SpaCO provide reliable gradient estimation and maintain competitive performance compared to exact gradient training?

The following sections present our comprehensive analysis. Additional experiments, including the verification of gradient computation accuracy for SeCO, are presented in Appendix D.

#### 6.1 Experimental Setup

Our experiments utilize the LLaMA3-8B (Meta-AI, 2024) as the base model, implementing LoRA (Hu et al., 2022) fine-tuning with hyperparameters r=8 and  $\alpha=16$ . The training datasets comprises 1,000 instances sampled from the PG19 training split (Rae et al., 2018), with sequence lengths truncated to 16K tokens.

#### **6.2** Baseline Methods

We compare our methods against three mainstream training paradigms: DeepSpeed (MicroSoft, 2021), conventional layer-level gradient checkpointing,

<span id="page-6-1"></span>> **[图片提取文字 (无描述)]:**
> Training Time per Iteration Training Time per Iteration Max Memory Allocation Global View Local View 24000 30 SeCO [512] SeCO [64] SeCO [128] SpaCO [256] SeCO [256] Gd Checkpoint SeCO [512] Naive Parallel SpaCO [32] ZeRO1 SpaCO [64] ZeRO2 22125 22.5 SpaCO [128] ZeRO3 Offload SpaCO [256] Memory Allocation (MB) End-to-End Latency (s) 20250 15 2 SeCO [64] SeCO [128] SeCO [256] SeCO [512] SpaCO [32] 18375 SpaCO [64] 7.5 SpaCO [128] SpaCO [256] - Naive Parallel - Gd Checkpoint ■ ZeRO1 Local View ZeRO2 ZeRO3 Offload 16500 本。本、本、本、本、本、本、本、本、本、本、本、本、本、本、本、本、本、本、本 50 Sequence Length Sequence Length Sequence Length
![](_page_6_Figure_0.jpeg)

Figure 6: (*Left*) Compared to SeCO, SpaCO achieves lower training time with more favorable linear-like scaling as the sequence length increases. (*Middle*) A zoomed-in view of the left panel shows that SeCO only incurs ∼30% additional time overhead compared to naive parallel training, significantly outperforming ZeRO3 offload which suffers from GPU-CPU communication bottlenecks and demonstrates approximately 10× slower training speed. (*Right*) SeCO and SpaCO exhibit superior memory efficiency, achieving more than 4× memory reduction compared to standard gradient checkpointing and an order-of-magnitude improvement over naive parallel training.

and naive parallel training, all of which leveraging FlashAttention [\(Dao,](#page-8-5) [2024\)](#page-8-5).

DeepSpeed. DeepSpeed [\(MicroSoft,](#page-8-10) [2021\)](#page-8-10) is a high-performance distributed training framework based on *Fully Sharded Data Parallel* (FSDP). It provides three optimization levels, ZeRO1/2/3, which progressively reduce memory consumption by distributing optimizer states, gradients, and parameter across multiple GPUs. ZeRO3 offload further extends ZeRO3 by offloading these components to CPU for additional memory savings. We evaluate DeepSpeed on 8 RTX 3090 GPUs with default ZeRO1/2 configurations and a custom ZeRO3 setup (Appendix[C.2\)](#page-9-2). Notably, FSDP offers limited benefits in parameter-efficient scenarios, where its advantages may not fully emerge.

Gradient Checkpointing. A standard gradient checkpointing implementation on a single RTX 3090 GPU, placing checkpoints at each LLM layer's input hidden states to reduce memory usage.

Naive Parallel Training. A baseline implementation on a single RTX 3090 GPU, utilizing no memory-efficient techniques except for FlashAttention. This serves as a reference for time efficiency.

#### 6.3 Efficiency Analysis

We evaluate time and memory efficiency of SeCO and SpaCO implemented on a single RTX 3090 GPU with varying sequence lengths.

- Configuration: For SeCO, we evaluate chunk sizes {64, 128, 256, 512}. For SpaCO, we use a chunk budget of t = 8 and test with chunk sizes {32, 64, 128, 256}.[2](#page-6-0)
- Measurement Protocol: Peak memory usage recorded via PyTorch's memory profiler, and the minimum end-to-end iteration time among the first 10 iterations is reported.

Our findings are concluded in Figure [6.](#page-6-1)

Practical Guidance. Based on our findings, we recommend maximizing the chunk size within the available memory limits. This accelerates training while maintains the same memory scalability.

### 6.4 Effectiveness Analysis

While SpaCO theoretically ensures unbiased gradient estimation through compensation factors, its increased gradient estimation variance raises practical effectiveness concerns. To investigate this, we evaluate its performance using a common training recipe for context window extension: As outlined in the experimental setup, we extend LLaMA3- 8B's original 8K window to 16K via LoRA.

<span id="page-6-0"></span><sup>2</sup>We observe that t = 8 already achieves satisfactory performance in practice.

<span id="page-7-1"></span>> **[图片提取文字 (无描述)]:**
> $LR = \{1e^{-4}, 3e^{-4}1e^{-3}, 1.6e^{-3}, 2.3e^{-3}, 3e^{-3}, 1e^{-2}\}\$  $LR = 1e^{-3}$ 3.0 -- Exact Gradient Exact Gradient (Mean) --- SpaCO t=8 Exact\_Gradient (±1 Std Dev) → SpaCO t=16 — SpaCO t=8 (Mean) 2.8 — SpaCO t=32 Language Modeling Error SpaCO t=8 (±1 Std Dev) — SpaCO t=16 (Mean) 3.5 SpaCO t=16 (±1 Std Dev) — SpaCO t=32 (Mean) SpaCO  $t=32 (\pm 1 \text{ Std Dev})$ 3.0 2.5 2.0 2.0 10-4 10-3 10-2 200 400 600 800 Learning Rate Step (update  $\Theta$  every 4 steps)
![](_page_7_Figure_0.jpeg)

Figure 7: (*Left*) Performance comparison of SpaCO across t = {8, 16, 32} under varying learning rates, using the same training setup and random seed. (*Right*) EMA-smoothed learning curves (α=0.95) for SpaCO and exact gradient method, both trained with the optimal learning rate of 1e-3.

For SpaCO, we fix the chunk size to 128 and conduct experiments with budgets of t = {8, 16, 32}, corresponding to sparsity ratios of 1/16, 1/8 and 1/4 respectively. The training results using model parallelism combined with gradient checkpointing serves as the baseline reference. All training runs use a batch size of 4, allowing for a total of 250 parameter updates. To mitigate numerical instability and gradient vanishing or explosion, we limit the compensation factor to a maximum value of 2.[3](#page-7-0)

Performance Under Varying Learning Rates. SpaCO introduces additional noises to the training process, necessitating independent tuning of the learning rate. To this end, we perform grid search over seven learning rates {1e-4,3e-4,1e-3,1.6e-3,2.3e-3,3e-3,1e-2} to identify the optimal values. Results are shown in Figure [7](#page-7-1) (*Left*).

Learning Curves. We further record the learning curves for each configuration using the optimal LR of 1e-3, as determined from Figure [7](#page-7-1) (*Left*). Training is conducted across four random seeds (controlling dataset shuffling), and we record the mean trajectories along with ±1 standard deviation bands. Results are presented in Figure [7](#page-7-1) (*Right*). The results demonstrate that with proper hyperparameter tuning, SpaCO achieves comparable performance to exact gradient training.

Practical Guidance. We recommend setting an upper bound (*e.g.*, 2) on the compensation factor to reduce gradient estimation variance. While omitting this constraint does not compromise training stability, it may lead to sub-optimal results. Furthermore, although our experiments used fixed batch size and training iterations for fair comparisons, we suggest using larger batch size and more training epochs than exact gradient training. This adjustment can promote better results in practical applications.

# 7 Conclusion

To address the critical challenge of efficiency in long-context LLM training, we introduce two training paradigms: SeCO and its enhanced variant SpaCO. By partitioning the input sequence into smaller, manageable chunks and performing localized backpropagation for each chunk, SeCO achieves substantial memory savings. Building upon this foundation, SpaCO introduces a carefully designed sparsification mechanism that randomly selects few chunks for backpropagation, reducing computational overhead. The integration of a mathematically-grounded compensation factor ensures unbiased gradient estimation. Our methods achieve impressive memory efficiency, enabling the fine-tuning of 8B models with 16K tokens on a single RTX 3090 GPU. This represents a 16× memory reduction compared to naive parallel training. SeCO and SpaCO significantly lower the barrier for practitioners working with long-context LLMs.

<span id="page-7-0"></span><sup>3</sup>Even in the absence of the compensation factor, excessively long gradient chains often result in vanishing or exploding gradients, diminishing their overall impact.

## Limitations

SeCO and SpaCO each present unique advantages but also have exhibit distinct limitations. SeCO achieves accurate gradient computation and efficient memory usage but suffers from a quadratic increase in computation with sequence length, making it impractical for training on ultra-long sequences. In contrast, SpaCO significantly reduces computational cost and maintains comparable memory efficiency but sacrifices gradient accuracy, introducing substantial randomness that complicates convergence. Ultimately, no single training strategy perfectly balances the trade-offs in all training scenarios. A practical approach requires identifying an optimal balance among the "impossible triangle" of computation, memory efficiency, and gradient accuracy.

### Ethics Statement

By optimizing memory consumption and computational efficiency, our approach not only lowers the financial barriers to training such models but also reduces energy consumption, contributing to more sustainable AI practices.

However, as with any significant technological advancement, ethical concerns must be considered. Lowering the cost and resource requirements for training long-context models may inadvertently enable the misuse of these models, including the creation of harmful or malicious language systems. It is essential to address these risks through responsible research practices and the development of robust safeguards.

### References

- <span id="page-8-1"></span>Chenxin An, Fei Huang, Jun Zhang, Shansan Gong, Xipeng Qiu, Chang Zhou, and Lingpeng Kong. 2024. [Training-free long-context scaling of large language](https://arxiv.org/abs/2402.17463) [models.](https://arxiv.org/abs/2402.17463) In *ICML*.
- <span id="page-8-17"></span>Léon Bottou and Olivier Bousquet. 2007. [The tradeoffs](https://papers.nips.cc/paper_files/paper/2007/hash/0d3180d672e08b4c5312dcdafdf6ef36-Abstract.html) [of large scale learning.](https://papers.nips.cc/paper_files/paper/2007/hash/0d3180d672e08b4c5312dcdafdf6ef36-Abstract.html) In *NeurIPS*.
- <span id="page-8-8"></span>Yaroslav Bulatov. 2018. [Fitting larger networks into](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9) [memory.](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9) Medium.
- <span id="page-8-9"></span>Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. 2016. [Training deep nets with sublinear](https://arxiv.org/abs/1604.06174) [memory cost.](https://arxiv.org/abs/1604.06174) *arXiv*.
- <span id="page-8-0"></span>Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. 2024. [Lon](https://arxiv.org/abs/2309.12307)[gloRA: Efficient fine-tuning of long-context large](https://arxiv.org/abs/2309.12307) [language models.](https://arxiv.org/abs/2309.12307) In *ICLR*.

- <span id="page-8-15"></span>Soumith Chintala, Gregory Chanan, Dmytro Dzhulgakov, Edward Yang, and Nikita Shulga. 2016. [py](https://github.com/pytorch/pytorch/tree/v2.6.0)[torch/pytorch.](https://github.com/pytorch/pytorch/tree/v2.6.0) Github.
- <span id="page-8-12"></span>Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G. Carbonell, Quoc Viet Le, and Ruslan Salakhutdinov. 2019. [Transformer-xl: Attentive language models](https://arxiv.org/abs/1901.02860) [beyond a fixed-length context.](https://arxiv.org/abs/1901.02860) In *ACL*.
- <span id="page-8-5"></span>Tri Dao. 2024. [Flashattention-2: Faster attention with](https://github.com/Dao-AILab/flash-attention) [better parallelism and work partitioning.](https://github.com/Dao-AILab/flash-attention) In *ICLR*.
- <span id="page-8-4"></span>Harm de Vries. 2023. [In the long \(context\) run.](https://www.harmdevries.com/post/context-length/) Personal website.
- <span id="page-8-20"></span>DeepSpeed. 2021. [Deepspeed's flops profiler.](https://www.deepspeed.ai/tutorials/flops-profiler/#flops-measurement) Deepspeed documentation.
- <span id="page-8-16"></span>Google. 2015. [tensorflow/tensorflow.](https://github.com/tensorflow/tensorflow/tree/v2.18.0) Github.
- <span id="page-8-11"></span>Sylvain Gugger, Lysandre Debut, Thomas Wolf, Philipp Schmid, Zachary Mueller, Sourab Mangrulkar, Marc Sun, and Benjamin Bossan. 2022. [huggingface/ac](https://github.com/huggingface/accelerate/tree/v1.2.1)[celerate.](https://github.com/huggingface/accelerate/tree/v1.2.1) Github.
- <span id="page-8-7"></span>Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. [LoRA: Low-rank adaptation of](https://arxiv.org/abs/2106.09685) [large language models.](https://arxiv.org/abs/2106.09685) In *ICLR*.
- <span id="page-8-13"></span>Zhiyuan Hu, Yuliang Liu, Jinman Zhao, and other. 2024. [Longrecipe: Recipe for efficient long context gener](https://arxiv.org/abs/2409.00509)[alization in large language models.](https://arxiv.org/abs/2409.00509) *arXiv*.
- <span id="page-8-21"></span>Jared Kaplan. 2019. [Notes on contemporary machine](https://www.semanticscholar.org/paper/Notes-on-Contemporary-Machine-Learning-for-Kaplan/70a1e83b5c539eacfa972710c92ac4b6ac8d128d) [learning for physicists.](https://www.semanticscholar.org/paper/Notes-on-Contemporary-Machine-Learning-for-Kaplan/70a1e83b5c539eacfa972710c92ac4b6ac8d128d) Semantic Scholar.
- <span id="page-8-18"></span>Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. [Effi](https://arxiv.org/abs/2309.06180)[cient memory management for large language model](https://arxiv.org/abs/2309.06180) [serving with pagedattention.](https://arxiv.org/abs/2309.06180) In *SIGOPS*.
- <span id="page-8-3"></span>Jerry Liu. 2022. [run-llama/llama\\_index.](https://github.com/run-llama/llama_index/tree/v0.12.16) Github.
- <span id="page-8-6"></span>Meta-AI. 2024. [The llama 3 herd of models.](https://arxiv.org/abs/2407.21783) Technical report.
- <span id="page-8-10"></span>MicroSoft. 2021. [microsoft/deepspeed.](https://github.com/deepspeedai/DeepSpeed/tree/v0.16.3) Github.
- <span id="page-8-2"></span>Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. 2024. [YaRN: Efficient context window](https://arxiv.org/abs/2309.00071) [extension of large language models.](https://arxiv.org/abs/2309.00071) In *ICLR*.
- <span id="page-8-22"></span>Jack W Rae, Anna Potapenko, Siddhant M Jayakumar, Chloe Hillier, and Timothy P Lillicrap. 2018. [google](https://github.com/google-deepmind/pg19)[deepmind/pg19.](https://github.com/google-deepmind/pg19) Github.
- <span id="page-8-14"></span>Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. [Attention is all](https://arxiv.org/abs/1706.03762) [you need.](https://arxiv.org/abs/1706.03762) In *NeurIPS*.
- <span id="page-8-19"></span>Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, and Luis Ceze. 2025. [Flashinfer: Efficient and cus](https://arxiv.org/abs/2501.01005)[tomizable attention engine for llm inference serving.](https://arxiv.org/abs/2501.01005) *arXiv*.

<span id="page-9-0"></span>Jinman Zhao, Xueyan Zhang, et al. 2024. [Large lan](https://openreview.net/forum?id=wLQ3I0F1oj)[guage model is not a \(multilingual\) compositional](https://openreview.net/forum?id=wLQ3I0F1oj) [relation reasoner.](https://openreview.net/forum?id=wLQ3I0F1oj) In *CoLM*.

<span id="page-9-3"></span>Table 1: Arguments for DeepSpeed ZeRO3 offload

| Argument                    | Value |
|-----------------------------|-------|
| overlap_comm                | true  |
| contiguous_gradients        | true  |
| reduce_bucket_size          | 5e8   |
| stage3_max_live_parameters  | 1e9   |
| stage3_max_reuse_distance   | 1e9   |
| stage3_prefetch_bucket_size | 5e8   |

### A Experimental Datasets

PG19 Dataset. The PG19 corpus, an open-source long-text dataset released by DeepMind, is derived from books in the [Project Gutenberg](https://www.gutenberg.org) repository published prior to 1919. This collection is supplemented with metadata containing book titles and publication dates. For model training, we randomly selected 1,000 samples from the PG19 training partition. To ensure consistent sequence lengths, text samples exceeding 16K tokens were truncated to this threshold. The PG19 dataset is publicly available under the Apache License 2.0.

### B Language Models

LLaMA3-8B. The LLaMA3-8B model, an opensource large language model developed by Meta AI, serves as the foundational model in our experiments. This selection is motivated by its widespread adoption within the research community. The licensing terms for the LLaMA3 series models are governed by the [Meta Llama 3 Com](https://github.com/meta-llama/llama3/blob/main/LICENSE)[munity License Agreement,](https://github.com/meta-llama/llama3/blob/main/LICENSE) which notably permits academic and commercial use with specific attribution requirements.

### C Implementation Details

### C.1 Pseudocode

The workflows of SeCO and SpaCO primarily manage the KV cache, focusing on its updates and the relay of gradients during backpropagation. These operations require overriding the default backpropagation mechanism in deep learning frameworks, which poses implementation challenges. To clarify this process, we provide pseudocode below.

<span id="page-9-4"></span>Table 2: Training results of SeCO vs. Model Parallelism (Baseline) across different learning rates.

| Method   | LR   |      |      |
|----------|------|------|------|
|          | 1e-4 | 3e-4 | 1e-3 |
| Baseline | 2.52 | 2.16 | 2.13 |
| SeCO     | 2.53 | 2.18 | 2.15 |

### <span id="page-9-2"></span>C.2 ZeRO3 Offload

Detailed configurations are provided in Table [1.](#page-9-3)

### <span id="page-9-1"></span>D Additional Results

Direct Validation of Gradient Accuracy. To assess the accuracy of the computed gradients, we conducted experiments using Qwen2.5-0.5B with float64 precision. Gradients were obtained for sequences of 512 tokens using both naive parallel training and SeCO (with a chunk size of 64) and then compared element-wise. The results show that the gradients computed with SeCO achieve a precision exceeding 12 decimal places. The test code for this experiment is publicly available in our repository under the test\_estimate directory.

Indirect Validation of Gradient Accuracy. To evaluate SeCO's performance in real training scenarios, we follow the experimental setup described in the main text. We compare SeCO's training results with those obtained using model parallelism and gradient checkpointing. The results are summarized in Table [2.](#page-9-4)

The minor performance gap may be attributed to numerical issues arising from the increased number of operations in SeCO. For example, FlashAttention introduces randomness during backpropagation due to the use of atomic additions (see [Github](https://github.com/Dao-AILab/flash-attention/issues/414) [issue\)](https://github.com/Dao-AILab/flash-attention/issues/414). Since SeCO involves tens of times more such operations than parallel training, it exhibits greater numerical instability.

```
1 def update_kv_cache(kv_cache, keys, vals):
2 try:
3 return concat(kv_cache.keys, keys), concat(kv_cache.vals,
               vals)
4 finally:
5 if is_gradient_enabled():
6 kv_cache.keys.append(keys)
7 kv_cache.vals.append(vals)
8 else:
9 k_detach, v_detach = keys.detach(), vals.detach()
10 k_detach.requires_grad_(), v_detach.requires_grad_()
11 kv_cache.keys.append(k_detach)
12 kv_cache.vals.append(v_detach)
13
14 def grad_hook(grad, base, scaler=1):
15 return grad + base * scaler
16
17 def copy_grad(a, b):
18 for ak, av, bk, bv in zip(a.keys, a.vals, b.keys, b.vals):
19 bk.register_hook(partial(grad_hook, base=ak.grad))
20 bv.register_hook(partial(grad_hook, base=av.grad))
```