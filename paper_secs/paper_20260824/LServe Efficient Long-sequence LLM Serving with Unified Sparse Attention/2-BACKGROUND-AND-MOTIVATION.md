# 2 BACKGROUND AND MOTIVATION

#### 2.1 Background

LLM Inference. LLMs are transformer-based architectures with stacked identical layers, each containing attention blocks, feed-forward networks (FFN), and normalization components. LLM inference involves two stages: an initial *prefilling* stage that handles multiple tokens concurrently, followed by auto-regressive *decoding* stage where only one token will be processed for each request in a decoding step.

Attention. The attention mechanism exchanges information across tokens. It first transforms input x through linear projections to generate query vectors q ∈ R <sup>N</sup>×HD, and key-value pairs k, v ∈ R N×HDˆ , where Hˆ represents the key/value head count. Traditional multi-head attention (MHA) maintains H = Hˆ , and contemporary architectures [\(Touvron et al.,](#page-13-7) [2023;](#page-13-7) [Jiang et al.,](#page-11-8) [2023;](#page-11-8) [2024a\)](#page-12-13) employ grouped-query attention (GQA) [\(Ainslie et al.,](#page-11-5) [2023\)](#page-11-5) where

<span id="page-2-0"></span>> **[图片提取文字 (无描述)]:**
> Attention Others **GFMM** Attention Others **GFMM** 8K 8K 16K 16K Length ength 32K 32K Input 64K 64K 128K 128K 0.25 0.5 0.75 0.25 0.5 0.75 (a) Latency breakdown of LLM prefilling (b) Latency breakdown of LLM decoding
![](_page_2_Figure_2.jpeg)

Figure 2: Latency breakdown of LLM inference during prefilling and decoding stages. Attention dominates both stages as sequence length increases, due to its quadratic complexity in prefilling and linear complexity in decoding. GEMM exhibits linear complexity in prefilling and constant complexity in decoding. Measurements obtained with Llama-3-8B on NVIDIA A100 GPU.

 $H = n\hat{H}(n \in \mathbb{Z})$  to shrink the size of KV cache. The current  $\mathbf{k}$  and  $\mathbf{v}$  is then concatenated with KV cache from S preceding tokens, yielding  $\mathbf{K}, \mathbf{V} \in \mathbb{R}^{(S+N) \times \hat{H}D}$ . The attention computation can then be formulated as follows:

$$\mathbf{S}_{h} = \frac{\mathbf{q}_{h} \mathbf{K}_{\hat{h}}^{T}}{\sqrt{D}}, \quad \mathbf{o}_{h} = \operatorname{softmax}(\mathbf{S}_{h}) \mathbf{V}_{\hat{h}}, \quad \hat{h} = \left\lfloor \frac{h}{n} \right\rfloor$$
 (1)

Therefore, the complexity of attention can be expressed as O(N(S+N)HD), which increases quadratically in the prefilling stage and linearly in the decoding stage with respect to sequence length. When S is long, both decoding stage and prefilling stage are bounded by attention.

**Paged Attention.** In LLM serving, the generation length of each sequence is highly variable and unpredictable. Padding all sequences to the maximum length results in considerable memory waste and fragmentation. To address this, vLLM (Kwon et al., 2023b) introduces PagedAttention, a KV cache management algorithm inspired by operating systems' virtual memory. Instead of allocating a continuous memory buffer for each sequence's KV cache, PagedAttention partitions the cache into fixed-size blocks (or pages), each holding KV data for a set number of consecutive tokens (typically 16 to 64). A page block table records the physical address of each page, allowing the PagedAttention kernel to use indirect addressing to retrieve KV features. TensorRT-LLM (NVIDIA, 2023) and QServe (Lin et al., 2024b) implement quantized page attention to reduce memory bandwidth usage during the decoding stage, resulting in further generation speedups.

#### 2.2 Motivation

Serving long-sequence LLMs is challenging due to the high cost of attention. Figure 2 profiles the latency breakdown of Llama-3-8B with a batch size of 1 across various sequence lengths on the A100 GPU. In both the prefilling and decoding stages, attention kernels account for at least 50% of the runtime at sequence lengths over 64k, rising to 75% at 128k. According to QServe (Lin et al., 2024b), the ratio of attention kernels in end-to-end runtime will increase as the batch size scale up. Therefore, in real-world serving scenarios, optimizing the attention becomes increasingly

<span id="page-2-1"></span>> **[图片提取文字 (无描述)]:**
> Sequential Sequential  $Q_0$  $Q_0$ Parallel (prefil.) Sequential (dec.) Parallel (prefil.) Sequential (dec.  $Q_1$  $Q_1$  $Q_2$  $Q_2$  $Q_3$  $Q_3$  $Q_4$  $Q_4$  $K_0$  $K_1$   $K_2$   $K_3$   $K_4$  $K_3$   $K_4$  $K_2$  $K_0$  $K_2$  $K_4$  $K_4$  $K_0$  $K_2$  $K_3$  $Q_4$  $V_1$  $V_2$  $V_3$  $V_4$  $V_0$  $V_2$  $V_4$  $V_0$ Dense calculation Block sparse calculation (time=4) (time=6)
![](_page_2_Figure_10.jpeg)

Figure 3: **Attention calculation on GPUs**: In both the decoding and prefilling stages, each query token iterates over all key and value tokens sequentially in a *block-by-block* manner. Skipping KV blocks reduces the number of sequential iterations, directly accelerating attention.

critical.

Accelerating attention in long-sequence LLMs requires a deep understanding of attention kernel implementation on GPUs, as illustrated in Figure 3. During the prefilling stage, the attention kernel is parallelized across batch size, attention heads, and query tokens, with query tokens set to 1 in the decoding stage. In both stages, the computation along the KV token dimension remains sequential. In each iteration, a block (depicted as a grid with an orange contour in Figure 3) is computed collaboratively by all threads in the current thread block. Although skipping certain computation within each block is possible, it yields minimal speedup. This is due to the lockstep execution of threads within a GPU warp, where faster threads are forced to wait for slower ones.

That said, rather than focusing on sparsity within each iteration, a more effective way to accelerate attention is to **reduce the number of sequential iterations** along the KV token dimension. This approach leads to our unified *block sparse attention* formulation, where attention computation is skipped in a blockwise manner. In this scheme, aside from the most recent KV block, each block is either fully computed or entirely skipped during the prefilling stage. During decoding, each sequence contains only one query token, reducing the dimensionality of each orange-contoured grid to  $1 \times P$ , where P represents the page size (i.e., the number of KV tokens per page). We will detail LServe's sparsity pattern selection in Section 3.

Additionally, because the decoding stage is memory-bound, KV cache quantization also contributes to speed improvements. Quantization is orthogonal to block sparsity, as it reduces the *runtime of each iteration*, while sparsity reduces the *number of iterations*.

<span id="page-3-0"></span>> **[图片提取文字 (无描述)]:**
> (a) Dense Attention (b) Block-Sparse Attention Contextual Local Blocks History Selected Sink Blocks Pages (c) Block-Sparse: Streaming Heads (d) Block-Sparse: Page Pruning
![](_page_3_Figure_2.jpeg)

Figure 4: Unified block sparse attention pattern. LServe integrates various sparsity patterns into a unified framework.

