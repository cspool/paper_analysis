## Trainable Dynamic Mask Sparse Attention

Jingze Shi<sup>13</sup>, Yifan Wu<sup>13</sup>, Yiran Peng<sup>13</sup>, Bingheng Wu<sup>13</sup>, Liangdong Wang<sup>2</sup>, Guang Liu<sup>2</sup>, Yuyu Luo<sup>\*1</sup>

<sup>1</sup>HKUST(GZ), <sup>2</sup>BAAI, <sup>3</sup>SmallDoges

#### **Abstract**

The increasing demand for long-context modeling in large language models (LLMs) is bottlenecked by the quadratic complexity of the standard self-attention mechanism. The community has proposed sparse attention to mitigate this issue. However, position-aware sparse attention methods rely on static sparse structures that lack adaptability to diverse query contexts, while content-aware sparse attention methods depend on heuristic key-value selection, hindering full differentiability. We introduce a trainable dynamic mask sparse attention mechanism, a method that merges the advantages of both position-aware and content-aware approaches. Dynamic Mask Attention (DMA) achieves this through three key innovations: First, it leverages value vector representations to generate content-aware dynamic masks, enabling the model to adaptively identify and attend to critical information. Second, it computes position-aware sparse weights in a hardwarefriendly manner, efficiently skipping unnecessary computational regions. Finally, we demonstrate that the introduced dynamic mask and sparse weights do not obstruct gradients, supporting end-to-end training. We have validated the performance of DMA through comprehensive experiments. A large body of experimental evidence shows that DMA consistently holds a Pareto advantage over state-of-the-art sparse attention baselines in tasks including scaling laws, multiquery associative recall, standard benchmarks, and needle in a haystack tests, while also delivering up to a 10x overall speedup. These results highlight its ability to effectively balance model efficiency with long-context modeling capabilities. Our computational kernel code is now open-source at https://github.com/flash-algo/flash-sparse-attention to encourage further research and application by the community.

## 1 Introduction

Recent breakthroughs in large language models (LLMs) have yielded remarkable achievements in tasks requiring *long-context reasoning* (Snell et al. 2024), such as deep reasoning (HuggingFace 2025), codebase generation (K. Zhang et al. 2024), and multi-turn autonomous agents (Park et al. 2023). A key factor underpinning these successes is the ability to effectively model long-range dependencies, often spanning thousands of tokens (DeepMind 2025; Guo et al. 2025; Team 2025). However, the standard self-attention mechanism (Vaswani et al. 2017) employed in Transformer architectures inherently exhibits quadratic computational complexity (Zaheer et al. 2020), which severely restricts scalability to longer sequences. Consequently, designing attention mechanisms that enhance computational efficiency without sacrificing modeling accuracy has become a critical research frontier for advancing the capabilities of LLMs.

Limitations of Existing Methods. Current efficient attention strategies primarily leverage two types of sparsity: the position sparsity (Martins and Astudillo 2016), which facilitates the efficient computation of essential query-key pairs, and the content sparsity (Ge et al. 2023), which enables the selective computation of relevant tokens. The first category includes methods such as sliding window attention (Beltagy, Peters, and Cohan 2020), which employs static structures; multi-head latent attention (A. Liu et al. 2024), which uses low-rank approximations; and native sparse attention (Yuan et al. 2025), which utilizes learnable compression weights. Although these approaches can achieve efficient long-context modeling, they often struggle to maintain their effectiveness. The second category encompasses KV cache eviction methods (Y. Li et al. 2024; Zhenyu Zhang et al. 2023; Z. Zhou et al. 2024); block-wise KV cache selection strategies that dynamically choose cache blocks based on relevance predictions (Y. Gao et al. 2024; Jiaming Tang et al. 2024; C. Xiao et al. 2024); and filtering methods that employ sampling (Z. Chen et al. 2024), clustering (G. Liu et al. 2024), or hashing (Desai et al. 2024). Despite their conceptual appeal, these techniques often fail to realize their theoretical speedups in practical deployments due to the overhead from dynamic computations or inaccurate sparsification decisions.

<sup>\*</sup>Corresponding author: Yuyu Luo (E-mail: yuyuluo@hkust-gz.edu.cn).

<span id="page-1-1"></span>![](_page_1_Figure_0.jpeg)

Figure 1: **Workflow and Performance of Dynamic Mask Attention**. **Left:** Overall workflow of DMA. The first step projects the input into *Q*, *K*, and *V*. The second step generates content-aware dynamic masks. The third step computes sparse weights. Black solid arrows indicate the forward computation path, while gray dashed arrows represent the backward computation path. **Right:** Relative performance comparison between full attention and DMA on benchmark tests. DMA achieves higher recall rates and significantly faster speeds while maintaining competitive accuracy.

<span id="page-1-0"></span>Table 1: **Comparison of Different Attention Variants**. Comparison of different Self-Attention mechanisms. n denotes sequence length,  $d_h$  represents head dimension, w is window size,  $d_c$  is compressed dimension, B is compression block size, and k is selection budget. Complexities focus on attention weight computation and memory requirements. Trainable indicates whether the sparsity pattern can be learned end-to-end.

| MECHANISM | COMP. COMPLEXITY               | Mem. Complexity  | Sparsity                    | Trainable |
|-----------|--------------------------------|------------------|-----------------------------|-----------|
| MHA       | $O(n^2d_h)$                    | $O(n^2)$         | Static                      | Х         |
| SWA       | $O(nwd_h)$                     | O(nw)            | Position-aware              | X         |
| MLA       | $O(n^2d_c)$                    | $O(n^2)$         | Low-rank Approx             | ✓         |
| NSA       | $O(n^2d_c/B + nkBd_h + nwd_h)$ | $O(n^2/B + nkB)$ | Hybrid                      | ✓         |
| H2O       | $O(nkd_h)$                     | O(nk)            | Content-aware               | X         |
| InfLLM    | $O(nkd_h)$                     | O(nk)            | Content-aware               | X         |
| Quest     | $O(nkd_h)$                     | O(nk)            | Content-aware               | X         |
| DAM       | $O(nkd_h)$                     | O(nk)            | Content-aware               | X         |
| DMA       | $O(nwd_h)$                     | O(nw)            | Content-Position Dual-aware | <b>√</b>  |

**Key Challenges.** To overcome these limitations, an ideal sparse attention mechanism must simultaneously address two fundamental challenges: **leveraging position-aware sparsity for essential computations** (Child et al. 2019) and **exploiting content-aware sparsity for selective computation** (Z. Dai et al. 2019). Meeting both requirements is crucial for achieving efficient and effective long-context reasoning and training in practice. However, existing methods still exhibit limitations, often facing a trade-off between efficiency and effectiveness. This dilemma highlights the urgent need for attention mechanisms that can preserve information integrity while achieving computational efficiency.

**Our Method.** In order to address these challenges and achieve efficient and effective sparse attention mechanisms, we ingeniously integrate the strengths of both strategies, attempting to strike a balance between the two, and propose Dynamic Mask Attention (DMA) to tackle the challenges of long-context modeling. As shown in Table 1, compared to other attention variants, DMA is a trainable content-position dual-aware sparse attention mechanism. As illustrated in Figure 1, it leverages two core innovations: **generating dynamic masks using content-aware sparsity** and **computing sparse weights using position-aware sparsity**, allowing the model to focus on relevant tokens while ignoring irrelevant ones. Furthermore, all computational operations are designed to be continuously differentiable, enabling end-to-end training of dynamic mask attention via gradient descent.

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Figure 2: **Dynamic Mask Attention Architecture. Left: Content-Aware Mask Computation.** The mask computation part of dynamic mask attention. In the outer loop, the stride weight  $\Delta$  and gate weight A are loaded into high-speed SRAM, and in the inner loop, the zero-order hold method is used to loop through the V blocks loaded into SRAM, sampling from it to generate content-aware K masks. These masks are then causally broadcast to the length of Q in HBM. Finally, in the outer loop, all mask blocks are concatenated to form the final content-aware sparse dynamic mask. **Right: Position-Aware Weights Computation.** The weight computation part of dynamic mask attention, where in the outer loop, the K and K blocks are looped and loaded into SRAM, and in the inner loop, the K blocks are accessed, loaded into SRAM, and the output of the attention weight computation is written back to HBM. If the current position of the K block is designated as masked in the dynamic mask, the attention weight at that position is directly filled with 0, skipping the computation at that position, forming the final position-aware sparse attention weights.

**Kernel Design.** We implement a dedicated CUDA kernel that merges the memory efficiency of FlashAttention (Dao et al. 2022) with DMA's trainable sparsity, as illustrated in Figure 2, enabling hardware-level skipping of masked regions without incurring additional redundant computations. The kernel natively supports attention masks and biases with batch, head, and query broadcasting, ensuring flexible integration with diverse Transformer architectures. A block-level reduction determines the skip logic: tiles corresponding to all-zero masks bypass both computation and memory access, reducing the effective complexity from  $O(n^2)$  to  $O(n \cdot w)$  for  $w \ll n$ . The forward and backward passes share a unified skip logic, fetching K/V tiles only when necessary, thereby maintaining an O(n) memory footprint without materializing the full attention matrix. The backward pass incorporates a complete gradient chain with fused bias gradients, rendering the entire pipeline fully differentiable for end-to-end training. To maximize throughput, we employ shared memory aliasing, pipelined prefetching, and coalesced memory accesses to minimize bandwidth pressure and improve hardware occupancy. These design choices allow DMA to sustain high performance on extremely long contexts, such as 128K+ tokens, while preserving accuracy comparable to dense attention baselines.

**Contributions.** We make the following contributions: (i) We introduce Dynamic Mask Attention (DMA), a trainable, content- and position-aware sparse attention mechanism that decouples content-driven dynamic mask generation from position-aware sparse weight computation, reducing the effective time complexity from  $O(n^2)$  to  $O(n \cdot w)$  and the memory footprint to  $O(n \cdot w)$  for window size  $w \ll n$ . (ii) We develop a fully differentiable CUDA kernel that fuses FlashAttention-style tiling with hardware-efficient mask skipping. The kernel supports batch/head/query broadcasting for masks and biases, employs block-level skip logic to avoid unnecessary computation and memory traffic, and maintains an O(n) memory footprint without materializing the full attention weight matrix. (iii) We validate DMA across diverse application scenarios, showing consistent speedups over other optimized implementations, improved scaling laws (Hoffmann et al. 2022), higher accuracy on multi-query associative recall (Arora et al. 2024), competitive downstream benchmarks performance, and robust needle-in-a-haystack (Kamradt 2023) results.

## <span id="page-3-0"></span>2 Rethinking Sparse Attention

Since the advent of the Transformer (Vaswani et al. 2017), the attention mechanism has become central to sequence modeling, yet its  $O(n^2)$  computational and memory complexity remains a bottleneck for processing long sequences. To overcome this limitation, researchers have explored sparse attention from two core perspectives: **position-aware essential computation** and **content-aware selective computation**. The former reduces computational load through predefined sparse patterns, while the latter dynamically determines the scope of computation based on input content. This section reviews the evolution of sparse attention, analyzes the strengths and weaknesses of existing methods, and provides context for our proposed Dynamic Mask Attention.

#### 2.1 Position-Aware Essential Computation

To achieve hardware efficiency, early sparse attention methods predominantly employed fixed sparse patterns, aiming to simplify computation through structured sparsity.

**Static Locality.** Sliding Window Attention (Beltagy, Peters, and Cohan 2020) confines computation to a local neighborhood for each token, reducing complexity to  $O(n \cdot w)$ , where w is the window size. While simple and efficient, its fixed local window limits the model's ability to capture long-range dependencies. This limitation is particularly pronounced in tasks requiring information integration across window boundaries, such as long-form question answering or code analysis.

**Low-Rank Approximation.** Methods like Multi-Head Latent Attention (A. Liu et al. 2024) approximate the full attention matrix using low-rank decomposition to reduce computational and memory demands. While this approach is better at preserving global information than sliding window attention, it comes at the cost of precision loss due to information compression. Low-rank approximation can obscure fine-grained details crucial for specific tasks and, due to its global nature, cannot dynamically adjust its compression strategy based on context, lacking content adaptability.

Hardware-Aligned Sparsity. Work such as Native Sparse Attention (Yuan et al. 2025) designs regularized sparse patterns for modern accelerators, achieving high computational efficiency through hardware-friendly block-sparse structures. However, the core deficiency of such methods lies in their static nature. Fixed sparse patterns cannot adapt to the dynamic changes in input content, leading to potential misallocation of computational resources to non-critical regions while neglecting genuinely important information.

#### 2.2 Content-Aware Selective Computation

As model capabilities have advanced, research has shifted towards content-aware selective computation, enabling models to learn autonomously where to focus their attention.

**KV Eviction.** Methods like H2O (Zhenyu Zhang et al. 2023) and SnapKV (Y. Li et al. 2024) save memory and computation by evicting "unimportant" tokens from the KV cache. These approaches typically rely on heuristics such as attention scores or access frequency to decide which tokens to retain. While effective in some scenarios, these heuristics can lead to erroneous eviction decisions, permanently losing critical information, especially in complex reasoning tasks that require long-distance backtracking.

**Token Selection.** Another class of methods actively selects a small subset of tokens for attention computation through techniques like sampling (Z. Chen et al. 2024), hashing (Desai et al. 2024), or clustering (G. Liu et al. 2024). While theoretically appealing, these methods face two major challenges in practice. First, discrete selection operations (such as sampling and hashing) are often non-differentiable, which impedes end-to-end training and prevents the model from learning optimal sparse patterns. Second, token-granular selection strategies disrupt memory access continuity, rendering them incompatible with modern efficient attention implementations like FlashAttention, leading to low hardware utilization and a decrease in both training and inference speed.

In summary, existing sparse attention mechanisms face an inherent trade-off between efficiency and effectiveness. Position-based methods, while efficient, lack flexibility and content awareness. Content-based methods, while more intelligent, are often limited by non-differentiable operations and inefficient hardware implementations. This dilemma highlights the urgent need for a new attention paradigm that can leverage structured sparsity for efficient computation while enabling content-aware selection through a trainable, dynamic mechanism.

<span id="page-4-0"></span>![](_page_4_Figure_0.jpeg)

Figure 3: **Sparsity in Language Modeling Tasks**. The tasks of Copy, Select, and Induce are three essential tasks for language modeling. The Copy task requires maintaining a fixed distance between input elements and output elements, the Select task involves selectively remembering or ignoring certain elements based on the input, and the Induce task requires retrieving answers through associative recall based on context. Where the colored parts represent the tokens that the model needs to remember in the current time step of inference, the black parts represent the output tokens that the model needs to predict based on the input, and the white parts represent irrelevant tokens that can be filtered out.

## 3 Background

### 3.1 Language Modeling Tasks

**Sparsity in Language Modeling.** As illustrated in Figure 3, long-context language modeling can be decomposed into three fundamental tasks: Copying (Romero et al. 2021), Selecting (Arjovsky, Shah, and Bengio 2016), and Inducing (Olsson et al. 2022). The Copy task requires preserving fixed-distance relationships between input and output tokens. The Select task involves selectively retaining or discarding information based on its content. The Induce task necessitates retrieving information via associative recall from the context. Each of these tasks is characterized by a distinct sparsity pattern: the Copy task exhibits *positional sparsity*, attending only to tokens at fixed distances; the Select task demonstrates *content sparsity*, focusing on tokens with specific content; and the Induce task relies on *associative sparsity*, where attention is directed only to key-value pairs relevant to the query. These inherent sparsity patterns provide a strong theoretical foundation for designing more efficient attention mechanisms.

#### 3.2 Multi-Head Attention

**QKV Projection.** In the Transformer architecture (Vaswani et al. 2017), the input is first transformed into Q, K, V. For the hidden state of the t-th token in a sequence of length n, denoted as  $h_t \in \mathbb{R}^{d_{model}}$ , the linear projections are performed using weight matrices  $W^Q$ ,  $W^K$ , and  $W^V$  to obtain  $q_t$ ,  $k_t$ , and  $v_t$ , respectively, as shown in Equation 1. These projections map the input representation into distinct subspaces for each of the  $n_h$  attention heads, allowing each head to focus on different aspects of the input. The weight matrices shape the projections to have a dimension of  $d_h$  per head.

<span id="page-4-1"></span>
$$q_{t} = h_{t}W^{Q} \quad \text{where} \quad h_{t} \in \mathbb{R}^{d_{model}} \quad W^{Q} \in \mathbb{R}^{d_{model} \times n_{h} \times d_{h}} \quad q_{t} \in \mathbb{R}^{n_{h} \times d_{h}}$$

$$k_{t} = h_{t}W^{K} \quad \text{where} \quad h_{t} \in \mathbb{R}^{d_{model}} \quad W^{K} \in \mathbb{R}^{d_{model} \times n_{h} \times d_{h}} \quad k_{t} \in \mathbb{R}^{n_{h} \times d_{h}}$$

$$v_{t} = h_{t}W^{V} \quad \text{where} \quad h_{t} \in \mathbb{R}^{d_{model}} \quad W^{V} \in \mathbb{R}^{d_{model} \times n_{h} \times d_{h}} \quad v_{t} \in \mathbb{R}^{n_{h} \times d_{h}}$$

$$(1)$$

**Key-Value Concatenation.** During autoregressive generation, the key-value pairs of historical tokens are cached to prevent redundant computations. As shown in Equation 2, the cached key and value matrices from past tokens are concatenated with the key-value representations of the current token to form the complete key matrix K and value matrix V. By maintaining and updating this cache, a complete context window spanning all tokens from position 1 to the current position t is constructed, enabling the model to access and utilize the full sequence history.

<span id="page-4-2"></span>
$$k = \operatorname{concat}([k_1, \dots, k_t]) \quad \text{where} \quad k \in \mathbb{R}^{n_h \times n \times d_h}$$

$$v = \operatorname{concat}([v_1, \dots, v_t]) \quad \text{where} \quad v \in \mathbb{R}^{n_h \times n \times d_h}$$
(2)

<span id="page-5-0"></span>![](_page_5_Figure_0.jpeg)

Figure 4: **Dynamic Mask Attention Structure**. It demonstrates the mask structure and weight structure of Dynamic Mask Attention in the multi-head case. Unlike the same and redundant mask and weight structures in Self-Attention and State-Space, the mask structure of DMA is dynamically adjusted through content awareness, where each head's mask can be different. This allows DMA to achieve different attention weight distributions in each head, enabling the model to maximize the utilization of each subspace in multi-head attention and focus on different tokens in each head.

#### <span id="page-5-2"></span>4 Method

As discussed in Section 2, existing attention mechanisms are confronted with numerous challenges when processing long sequences, including high computational complexity, substantial memory requirements, the rigidity of static masks, and the presence of non-differentiable components that impede end-to-end training. To address these issues, we introduce Dynamic Mask Attention, a mechanism that strategically leverages the inherent sparsity patterns of language modeling. As illustrated in Figure 2, DMA is composed of two core components: content-aware dynamic sparse masking and position-aware sparse attention weight computation. The former utilizes value representations to dynamically generate masks that determine which historical tokens each attention head should attend to, while the latter performs efficient sparse attention computations guided by these masks. This dual-component design allows DMA to maintain focus on critical information while adapting to varying contextual dependencies. Furthermore, as shown in Figure 4, DMA generates a unique mask structure for each attention head, enabling the model to capture diverse content patterns across different representational subspaces. A sample PyTorch implementation is provided in Listing 1 for reference.

#### 4.1 Generate Content-Aware Dynamic Mask

The content-aware dynamic mask constitutes the central innovation of DMA. It operates by analyzing the content features embedded within the value representations to determine which historical information is relevant to the current query. For each attention head at the current time step, this mechanism generates a unique dynamic mask that directs the subsequent attention weight computation to focus exclusively on the most critical key positions.

To sample the value vector representations, we introduce a sampling weight matrix  $\Delta \in \mathbb{R}^{n_h \times d_h \times n_h}$ , a per-head gating coefficient  $A \in \mathbb{R}^{n_h}$ , and a non-negative activation function  $\tau(\cdot)$ . As detailed in Equation (3), the process begins with a tensor contraction,  $v \cdot \Delta$ , which projects each token's  $d_h$ -dimensional value vector into a scalar representation, serving as an initial estimate of its importance. Subsequently, the activation function  $\tau(\cdot)$  ensures these scores are non-negative, preventing signal cancellation. The gating coefficient A then scales the importance scores for each head, allowing the model to learn varying degrees of sparsity across heads, where A can be made query-dependent, enabling the sampling scores to adapt based on the input. Finally, an exponential function amplifies the differences between scores and maps them to a positive value space, which facilitates the learning of gating effects and yields the final scores,  $\delta \in \mathbb{R}^{n_h \times n}$ .

<span id="page-5-1"></span>
$$\delta = \exp(\tau(v \cdot \Delta) \times A) \quad \text{where} \quad \Delta \in \mathbb{R}^{n_h \times d_h \times n_h} \quad A \in \mathbb{R}^{n_h} \quad \delta \in \mathbb{R}^{n_h \times n}$$
 (3)

Subsequently, as defined in Equation (4), a sparsification function  $f(\cdot)$  is applied. This function identifies whether each score  $\delta_{n_h,j}$  ranks within the  $top_w$  for its respective head. Scores within the  $top_w$  are retained to preserve the gradient flow, while all other scores are set to  $-\infty$ , effectively nullifying their contribution in the subsequent softmax operation. For causal language modeling, this function can incorporate a causal mask via broadcasting to avoid additional memory overhead. This process yields the final dynamic mask,  $m_t \in \mathbb{R}^{n_h \times n}$ .

<span id="page-6-0"></span>
$$m_{t} = f(\delta) \quad \text{where} \quad m_{t} \in \mathbb{R}^{n_{h} \times n}$$

$$= \begin{bmatrix} f(\sum_{j=1}^{n} \delta_{1,j}) \\ f(\sum_{j=1}^{n} \delta_{2,j}) \\ \vdots \\ f(\sum_{j=1}^{n} \delta_{n_{h},j}) \end{bmatrix} \quad \text{where} \quad f(\delta_{n_{h},j}) = \begin{cases} \delta_{n_{h},j} & \text{if } \delta_{n_{h},j} \in top_{w}(\delta_{n_{h}}) \\ -\infty & \text{otherwise} \end{cases}$$

$$(4)$$

This approach offers three significant advantages. First, by sampling importance scores from value representations, the model can more accurately focus on semantically critical tokens, regardless of their distance. This mitigates the risk of overlooking important long-range dependencies, a common issue with methods relying solely on positional patterns. Second, the combination of the gating coefficient A and independent  $top_w$  selection allows different attention heads to specialize in distinct functions, such as local, long-range, and global, thereby improving the breadth of representational coverage. Third, the sparse selection mechanism is inherently effective during training, eliminating the need for post-hoc pruning and preserving the model's learned retrieval capabilities. The kernel implementation, illustrated in Figure 2 (left), is designed for efficiency. In an outer loop, the sampling weight  $\Delta$  and gating coefficient A are loaded into high-speed SRAM. In an inner loop, a zero-order hold method iteratively processes blocks of the value matrix V from SRAM to generate content-aware masks for the key matrix K. These masks are then causally broadcast to match the length of the query matrix Q in HBM, avoiding memory usage with quadratic complexity. Finally, the mask blocks are concatenated to form the complete content-aware dynamic mask.

## 4.2 Compute Position-Aware Sparse Weights

The second core component of DMA is the position-aware sparse weight computation. As outlined in Algorithm 1, this component leverages the previously generated dynamic mask to sparsify the computation of scaled dot-product attention weights, effectively reducing the per-step complexity from  $O(nd_h)$  to  $O(wd_h)$ .

For the query at step t,  $q_t \in \mathbb{R}^{n_h \times d_h}$ , and the complete key-value pairs,  $K, V \in \mathbb{R}^{n_h \times n \times d_h}$ , the entire computation flow is detailed in Equation (5). Initially, for each attention head  $n_h$ , the scaled dot-product between the query and keys,  $q_t K^{\top}$ , is computed and then element-wise multiplied by the previously constructed dynamic mask  $m_t$ . The scaling factor  $\sqrt{d_h}$  is crucial here as it prevents the dot products from becoming excessively large, which could push the softmax function into a saturated region with minimal gradients. After applying the mask, the softmax function normalizes the results to produce attention weights  $p_{n_h,j}$ . Notably, when a mask value  $m_{n_h,j} = -\infty$ , the corresponding attention weight  $p_{n_h,j} \approx 0$ , effectively skipping computations for masked positions and filling them with zeros. This ensures that the model focuses solely on relevant unmasked contexts. The attention weights for each head are then multiplied by the value vectors and summed to produce the final context vector  $o_t \in \mathbb{R}^{n_h \times d_h}$ , where each row captures different contextual patterns and dependencies. The multi-head mechanism, combined with dynamic masking, allows the model to attend to various patterns in parallel across the sequence. This output integrates information from all attention heads, forming a rich hierarchical context representation that effectively captures dependencies at varying distances within the sequence history. It is important to note that this method can approximate full attention when  $n_h \times w \leq n$ , while maintaining computational efficiency.

<span id="page-6-1"></span>
$$o_{t} = \operatorname{softmax}\left(\frac{q_{t}k^{\top}}{\sqrt{d_{h}}} \circ m_{t}\right)v \quad \text{where} \quad p_{t} \in \mathbb{R}^{n_{h} \times n} \quad o_{t} \in \mathbb{R}^{n_{h} \times d_{h}}$$

$$= \begin{bmatrix} \sum_{j=1}^{n} p_{1,j} \cdot v_{1,j} \\ \sum_{j=1}^{n} p_{2,j} \cdot v_{2,j} \\ \vdots \\ \sum_{i=1}^{n} p_{n_{h},j} \cdot v_{n_{h},j} \end{bmatrix} \quad \text{where} \quad p_{n_{h},j} = \begin{cases} \frac{\exp\left(\frac{q_{n_{h}} \cdot k_{n_{h},j}^{\top}}{\sqrt{d_{h}}} + m_{n_{h},j}\right)}{\sum_{j'=1}^{n} \exp\left(\frac{q_{n_{h}} \cdot k_{n_{h},j'}^{\top}}{\sqrt{d_{h}}} + m_{n_{h},j'}\right)} \quad \text{if } m_{n_{h},j} \neq -\infty \end{cases}$$

$$(5)$$

#### <span id="page-7-0"></span>Algorithm 1 Flash Dynamic Mask Attention Forward Pass Per Head

```
Require: Matrices Q, K, V \in \mathbb{R}^{N \times d_h}, M \in \mathbb{R}^N in HBM. Set block sizes B.
    Initialize \mathbf{O} = (0)_{N \times d_h} \in \mathbb{R}^{N \times d_h}, \ell = (0)_N \in \mathbb{R}^N, m = (-\infty)_N \in \mathbb{R}^N in HBM.
    Divide Q into T_r blocks of size B \times d_h each, and divide K, V into T_c blocks of size B \times d_h each.
    Divide M into T_c blocks of size B each.
    Divide O into T_r blocks of size B \times d_h each, divide \ell into T_r blocks of size B each, divide m into T_r blocks of size B each.
    for 1 \le j \le T_c do
         Load M_i from HBM to SRAM.
         Compute active _i = \text{Judge}(\mathbf{M}_i).
         if active i = 0 then
              Advance stream pointers for K_i, V_i, continue.
         Load K_i, V_i from HBM to SRAM.
         for 1 \le i \le T_r do
              Load Q_i, O_i, \ell_i, m_i from HBM to SRAM.
              Compute S_{ij} = Q_i \mathbf{K}_j^T \times d_h^{-0.5} + \mathbf{M}_j \in \mathbb{R}^{B \times B}.
              Compute \tilde{m}_{ij} = \operatorname{rowmax}(\mathbf{S}_{ij}) \in \mathbb{R}^B, \tilde{\mathbf{P}}_{ij} = \exp(\mathbf{S}_{ij} - \tilde{m}_{ij}) \in \mathbb{R}^{B \times B}, \tilde{\ell}_{ij} = \operatorname{rowsum}(\tilde{\mathbf{P}}_{ij}) \in \mathbb{R}^B. Compute m_i^{\text{new}} = \max(m_i, \tilde{m}_{ij}) \in \mathbb{R}^B, \ell_i^{\text{new}} = e^{m_i - m_i^{\text{new}}} \ell_{ij} + e^{\tilde{m}_{ij} - m_i^{\text{new}}} \tilde{\ell}_{ij} \in \mathbb{R}^B.
              Write O_i \leftarrow \operatorname{diag}(\ell_i^{\text{new}})^{-1}(\operatorname{diag}(\ell_i)e^{m_i-m_i^{\text{new}}}O_i + e^{\tilde{m}_{ij}-m_i^{\text{new}}}\tilde{\mathbf{P}}_{ij}\mathbf{V}_j) to HBM.
              Write \ell_i \leftarrow \ell_i^{\text{new}}, m_i \leftarrow m_i^{\text{new}} to HBM.
    end for
    Return O, \ell, m.
```

This method offers three key advantages. First, the mask prunes the set of candidate tokens before the matrix multiplication and softmax operations. This avoids the inefficiency of pseudo-sparsity, where computations are performed for all tokens only to be zeroed out afterward. Second, unlike methods that perform key-value selection by discarding tokens, our approach preserves the full sequence. This ensures that the complete global context remains available for all attention heads to access as needed. Third, the kernel implementation can perform block-level skipping by loading a block of the mask to check if all positions within it are masked. If so, the entire block is skipped, avoiding unnecessary memory loads and matrix multiplication operations. The kernel implementation, depicted in Figure 2 (right), is optimized for this process. In an outer loop, blocks of the K and K matrices are loaded into SRAM. In an inner loop, blocks of the K matrix are loaded, and if the corresponding K block is not entirely masked, the attention weights are computed and the output is written back to HBM. If a position in the K block is masked, its attention weight is set to zero, and the computation for that position is skipped, resulting in position-aware sparse attention weights.

### 4.3 Fully Gradient Flow

Finally, the entire backward process is outlined in Algorithm 2. We ensure that the introduced dynamic mask and sparse weights do not block gradients, and the gradients of the retained attention paths are strictly consistent with those of full attention. They can flow completely to all inputs and parameters without gradient discontinuity issues caused by discrete operations, supporting end-to-end training while minimizing redundant costs.

For clarity, our derivation considers a single attention head h at a single time step t. Let  $I_h \subset \{1, ..., n\}$  be the set of w indices selected for this head. For unselected indices  $j \notin I_h$ , the mask value is treated as  $m_{h,j} = -\infty$ . The key intermediate variables in the forward pass are defined in Equation (6).

<span id="page-7-1"></span>
$$s_{h,j} = \frac{q_h \cdot k_{h,j}}{\sqrt{d_h}} + m_{h,j} \quad p_{h,j} \quad = \begin{cases} \frac{\exp(s_{h,j})}{\sum_{j' \in \mathcal{I}_h} \exp(s_{h,j'})} & j \in \mathcal{I}_h \\ 0 & j \notin \mathcal{I}_h \end{cases} \quad o_h = \sum_{j \in \mathcal{I}_h} p_{h,j} v_{h,j}$$
(6)

In the backward pass, let the upstream gradient of the loss function L with respect to the head's output  $o_h$  be  $do_h = \frac{\partial L}{\partial o_h} \in \mathbb{R}^{d_h}$ . As shown in Equation (7), the gradient for v is computed by distributing  $do_h$  to the selected vectors  $v_{h,j}$  in proportion to their attention weights  $p_{h,j}$ , while the gradients for unselected positions are zero.

<span id="page-7-2"></span>
$$\frac{\partial L}{\partial v_{h,j}} = \begin{cases} p_{h,j} \, do_h & j \in \mathcal{I}_h \\ 0 & j \notin \mathcal{I}_h \end{cases} \tag{7}$$

#### <span id="page-8-0"></span>Algorithm 2 Flash Dynamic Mask Attention Backward Pass Per Head

```
Require: Matrices Q, K, V, O, dO \in \mathbb{R}^{N \times d_h}, M, dM \in \mathbb{R}^N in HBM, vectors \ell, m \in \mathbb{R}^N in HBM. Set block sizes B.
   Divide Q into T_r blocks of size B \times d_h each, and divide K, V in to T_c blocks of size B \times d_h each.
   Divide M into T_c blocks of size B each.
   Divide O, dO into T_r blocks of size B \times d_h each, divide \ell into T_r blocks of size B each, and divide m into T_r blocks of size B each.
   Initialize dQ = (0)_{N \times d_h} in HBM and divide it into T_r blocks of size B \times d_h each.
   Initialize d\mathbf{K} = (0)_{N \times d_h}, d\mathbf{V} = (0)_{N \times d_h} in HBM and divide d\mathbf{K}, d\mathbf{V} in to T_c blocks of size B \times d_h each.
   Initialize dM = (0)_N in HBM and divide it into T_c blocks of size B each.
   for 1 \le j \le T_c do
        Load M_i from HBM to SRAM.
        Compute active i = \text{Judge}(M_i).
        if active i = 0 then
             Advance stream pointers for K_i, V_i, continue.
        Load K_i, V_i from HBM to SRAM.
        Initialize \tilde{\mathbf{dK}}_j = (0)_{B \times d_h}, \tilde{\mathbf{dV}}_j = (0)_{B \times d_h}, \tilde{\mathbf{dM}}_j = (0)_B on SRAM.
        for 1 \le i \le T_r do
             Load Q_i, O_i, dO_i, dQ_i, \ell_i, m_i from HBM to SRAM.
             Compute S_{ij} = Q_i K_j^T \times d_h^{-0.5} + M_j \in \mathbb{R}^{B \times B}.
             Compute P_{ij} = \operatorname{diag}(\ell_i)^{-1} \exp(S_{ij} - m_i) \in \mathbb{R}^{B \times B}.
             Compute \mathbf{d}\tilde{\mathbf{V}}_j \leftarrow \mathbf{d}\tilde{\mathbf{V}}_j + \mathbf{P}_{ij}^{\top}\mathbf{d}\mathbf{O}_i \in \mathbb{R}^{B \times d_h}.
             Compute d\mathbf{P}_{ij} = d\mathbf{O}_i \mathbf{V}_i^{\top} \in \mathbb{R}^{B \times B}.
             Compute D_i = \text{rowsum}(\mathbf{dO}_i \circ \mathbf{O}_i) \in \mathbb{R}^B.
             Compute dS_{ij} = P_{ij} \circ (dP_{ij} - D_i) \in \mathbb{R}^{B \times B}.
             Compute \tilde{\mathbf{dM}}_j \leftarrow \tilde{\mathbf{dM}}_j + \operatorname{rowsum}(\mathbf{dS}_{ij}) \in \mathbb{R}^B.
             Write d\mathbf{Q}_i \leftarrow d\mathbf{Q}_i + d\mathbf{S}_{ij}\mathbf{K}_j \times d_h^{-0.5} \in \mathbb{R}^{B \times d_h} to HBM.
             Compute \tilde{\mathbf{dK}}_{i} \leftarrow \tilde{\mathbf{dK}}_{i} + \mathbf{dS}_{ii}^{\mathsf{T}} \mathbf{Q}_{i} \times d_{h}^{-0.5} \in \mathbb{R}^{B \times d_{h}}.
        Write d\mathbf{K}_i \leftarrow d\tilde{\mathbf{K}}_i, d\mathbf{V}_i \leftarrow d\tilde{\mathbf{V}}_i, d\mathbf{M}_i \leftarrow d\mathbf{M}_i + d\tilde{\mathbf{M}}_i to HBM.
   Return dQ, dK, dV, dM.
```

Next, we compute the gradient for the scores  $s_{h,j}$ . The gradient of the attention weights  $p_{h,j}$  with respect to their inputs is  $dp_{h,j} = v_{h,j} \cdot do_h$ . Using the standard softmax Jacobian, we can derive the gradient for  $s_{h,j}$ , denoted as  $ds_{h,j}$ , as shown in Equation (8). For masked positions where  $p_{h,j} = 0$ , the gradient  $ds_{h,j}$  is naturally zero.

<span id="page-8-1"></span>
$$ds_{h,j} = p_{h,j}(dp_{h,j} - \sum_{j' \in \bar{I}_h} p_{h,j'} \times dp_{h,j'})$$
(8)

Because  $s_{h,j}$  is an additive combination of  $q_h \cdot k_{h,j}$  and  $m_{h,j}$ , the gradient is distributed directly. The gradient for the mask  $m_{h,j}$  is simply  $ds_{h,j}$ , as shown in Equation (9). This ensures that gradients can flow directly to  $\Delta$  and A.

<span id="page-8-2"></span>
$$\frac{\partial L}{\partial m_{h,j}} = ds_{h,j} \tag{9}$$

Finally, as shown in Equation (10), the gradients for  $q_h$  and  $k_{h,j}$  are obtained by backpropagating  $ds_{h,j}$  through the computation path. Crucially, the gradient calculations only involve the selected index set  $I_h$ , thereby reducing computation.

<span id="page-8-3"></span>
$$\frac{\partial L}{\partial q_h} = \sum_{j \in I_h} ds_{h,j} \frac{k_{h,j}}{\sqrt{d_h}}, \qquad \frac{\partial L}{\partial k_{h,j}} = ds_{h,j} \frac{q_h}{\sqrt{d_h}}$$
(10)

Our approach has several significant advantages. First, for the selected positions, the gradients are identical to those of full attention, and DMA only prunes the operator chain for positions whose contributions can be ignored, ensuring expressiveness. Then, only second-order correlation information is propagated to  $I_h$ , improving bandwidth utilization. The gating parameter A and weight  $\Delta$  directly receive attention weights as gradients, quickly shaping head specialization. Finally, the equivalence relation dM = dS allows the kernel to only recompute the local S without storing additional intermediate mask gradient tensors.

<span id="page-9-1"></span>![](_page_9_Figure_0.jpeg)

Figure 5: **Scaling Laws**. The perplexity performance of different self-attention variants on SmolLMCorpus at different parameter scales. For suboptimal variants like SWA and MLA, we omit them for clarity. Compared to other variants, our Dynamic Mask Attention has a Pareto advantage in performance.

## 5 Experiments

We will validate the efficiency and effectiveness of Dynamic Mask Attention, as detailed in Section 4, in handling long contexts through its content-aware dynamic sparse mask and position-aware dynamic sparse weight computation.

## 5.1 Experimental Settings

**Baselines.** To thoroughly evaluate DMA, we benchmarked it against representative baselines surveyed in Section 2. First, we compared DMA with various mainstream attention variants in terms of pre-training perplexity at different model scales, further validating DMA's advantage in long-sequence information retrieval through the challenging multiquery associative recall task, and tested the efficiency of our kernel compared to other optimized implementations in Section 5.2. Second, we compared DMA, NSA, and MHA on a 1.7B parameter Transformer model pre-trained and supervisedly fine-tuned on 40B tokens, evaluating them on downstream benchmark tasks and needle-in-a-haystack tests in Section 5.3.

Training Settings. All experiments were conducted using the open-source PyTorch images (NVIDIA 2022) and the Transformers framework (Wolf et al. 2020). For model configuration, we consistently employed the NeoX tokenizer (Black et al. 2022), the AdamW optimizer (Loshchilov and Hutter 2017), and the WSD learning rate scheduler (Hägele et al. 2024), while strictly adhering to the Optimal Hyperparameter Scaling Law (H. Li et al. 2025) and the Chinchilla (Hoffmann et al. 2022) standard protocol throughout our training on the SmolLMCorpus (Ben Allal et al. 2024) dataset. For evaluation frameworks, we utilized the LM evaluation harness (L. Gao et al. 2021) from EleutherAI for perplexity tasks, and the lighteval (Fourrier et al. 2023) from HuggingFace for downstream tasks.

#### <span id="page-9-0"></span>5.2 Variants Comparison

**Scaling Perplexity.** First, we present the comparison of the perplexity performance of different self-attention variants at various parameter scales in Figure 5. This experiment includes the baseline <sup>1</sup>, sliding window attention <sup>2</sup> driven by static mask structures, multi-head latent attention <sup>3</sup> driven by low-rank decomposition approximations, native sparse attention <sup>4</sup> driven by hardware content adaptation, and our proposed Dynamic Mask Attention. These experiments were conducted on the SmolLMCorpus dataset, with model sizes ranging from 80M to 1.7B parameters, and the experimental configurations are detailed in Table 4. Our experimental results validate that Dynamic Mask Attention maintains the best performance across various scales. We speculate that this advantage primarily stems from DMA's ability to adaptively focus on key information in the input sequence, effectively avoiding the lost in middle (N. F. Liu et al. 2024) problem.

<span id="page-10-0"></span>![](_page_10_Figure_0.jpeg)

Figure 6: Multi-Query Associative Recall. This is a more challenging version of the original multi-query associative recall task (Arora et al. 2024), which includes longer sequence lengths and smaller model dimensions. Dynamic Mask Attention maintains good performance in most cases.

<span id="page-10-1"></span>![](_page_10_Figure_2.jpeg)

Figure 7: **Kernal Performance**. A performance comparison of efficient implementation kernels of different attention variants on an A100 GPU. Our DMA achieves significant acceleration in forward propagation, decoding, and backward propagation, maintaining the same efficiency level as SWA.

Associative Recall. To further validate the ability of different attention variants in long-sequence information retrieval, we designed a more challenging variant of the multi-query associative recall task (Arora et al. 2024), which includes longer sequence lengths and smaller model dimensions. This task assesses the ability of language models to retrieve information within their context. Specifically, it provides key-value pairs to the autoregressive model, prompting the model to generate the correct value when displaying previously seen keys. To increase the difficulty of the task, we used 512 key-value pairs in the experiment. We employ sliding window attention, native sparse attention, and dynamic mask attention, all with a window size of 512. This approach replaces non-query/key/value parts with random tokens, forcing the model to locate relevant information precisely rather than relying on contextual clues. The experimental dataset comprises 250,000 training samples and 1,000 test samples, with all models trained for 100 epochs to ensure sufficient convergence. As shown in Figure 6, Dynamic Mask Attention performs excellently across various sequence lengths, indicating its ability to intelligently identify and focus on tokens relevant to the current state while ignoring irrelevant tokens.

**Kernel Acceleration.** To analyze the performance of DMA within modern efficient operator frameworks, we benchmarked the forward, decoding, and backward performance of MHA, SWA, MLA, NSA, and DMA on an A100-SXM4-80GB GPU. The results represent the average of 1,000 runs after three warm-up iterations; specific configurations and implementations can be found in Table 5. As shown in Figure 7, for the Forward pass, compared to MHA, DMA achieves speedups of approximately 26.1×, 10.2×, and 21.5× at token lengths of 8192, 16384, and 32768, respectively. For the Decode phase, speedups against MHA at key lengths of 65536, 131072, 262144, and 524288 are approximately 49.6×, 92.7×, and 171.1×, respectively. For the Backward pass, speedups against MHA at lengths of 8192, 16384, and 32768 are approximately 2.5×, 4.4×, and 7.9×, respectively. DMA avoids large-scale redundant score and softmax backpropagation overhead by explicitly skipping masked blocks, thus providing strong acceleration capabilities across multiple critical stages.

<span id="page-11-2"></span>![](_page_11_Figure_0.jpeg)

Figure 8: **Needle in a Haystack**. Comparison of needle-in-a-haystack performance between MHA, NSA, and DMA in an apples-to-apples setting. The white dotted line indicates the sequence length of the model.

<span id="page-11-1"></span>Table 2: **Downstream Task Evaluations for Base Model**. The best results for each size are in bold, and the second-best results are unlined. DMA outperforms MHA and NSA, as well as other advanced inference sparse methods, on most tasks.

| Model      | LAMBADA<br>ppl↓ | LAMBADA<br>acc↑ | MMLU<br>acc↑ | TriviaQA<br>acc↑  | ARC<br>ACC ↑ | PIQA<br>acc↑ | HellaSwag<br>acc↑ | OBQA<br>acc↑ | WinoGrande<br>acc↑ | Avg<br>acc↑ |  |
|------------|-----------------|-----------------|--------------|-------------------|--------------|--------------|-------------------|--------------|--------------------|-------------|--|
| Zero-Shot  |                 |                 |              |                   |              |              |                   |              |                    |             |  |
| MHA        | 15.22           | 44.3            | 35.4         | 9.4               | 53.4         | 72.9         | 56.1              | 37.0         | 57.3               | 45.7        |  |
| H2O        | 15.38           | 44.2            | 34.8         | 7.4               | 53.3         | 72.8         | 55.6              | 36.6         | 56.9               | 45.2        |  |
| InfLLM     | 15.23           | 44.2            | 35.1         | 8.0               | 53.1         | 72.4         | 55.8              | 36.6         | 56.8               | 45.2        |  |
| Quest      | 15.43           | 43.9            | 35.1         | 7.6               | 53.1         | 72.6         | 56.1              | 36.8         | 57.2               | 45.3        |  |
| DAM        | 15.89           | 44.5            | 34.6         | 8.9               | 52.1         | 72.3         | 56.2              | 36.3         | 56.0               | 45.1        |  |
| Exact-Top  | 15.23           | 44.4            | 35.3         | 9.2               | 53.3         | 72.8         | 56.0              | 36.8         | 57.0               | 45.6        |  |
| NSA        | 14.91           | <u>45.2</u>     | 33.8         | 8.7               | 53.1         | 72.8         | 56.7              | 36.3         | <u>57.8</u>        | 45.5        |  |
| DMA (ours) | 14.42           | 45.9            | 37.0         | 9.1               | 55.6         | 73.4         | <u>56.4</u>       | 36.5         | 58.4               | 46.5        |  |
|            |                 |                 |              | Five-Sho          | ot           |              |                   |              |                    |             |  |
| MHA        | 19.40           | 40.4            | 36.8         | 13.2              | 56.8         | 73.2         | 56.8              | 38.0         | 58.6               | 46.7        |  |
| H2O        | 19.14           | 38.9            | 35.7         | $\overline{10.2}$ | 56.6         | 73.2         | 56.4              | 37.8         | 58.1               | 45.8        |  |
| InfLLM     | 19.13           | 41.3            | 35.9         | 11.7              | 56.7         | 73.3         | 56.1              | 38.0         | 57.7               | 46.3        |  |
| Quest      | 19.22           | 40.9            | 36.1         | 10.9              | 56.2         | 73.2         | 55.8              | 37.9         | 58.2               | 46.1        |  |
| DAM        | 19.47           | 41.2            | 35.2         | 13.3              | 55.1         | 71.0         | 54.4              | 38.0         | 57.2               | 45.6        |  |
| Exact-Top  | 18.22           | 39.7            | 36.4         | 13.1              | 56.3         | 73.4         | 56.5              | 38.2         | 58.5               | 46.5        |  |
| NSA        | 21.37           | 39.6            | 34.6         | 12.5              | 56.1         | 76.0         | 58.9              | 39.2         | 58.3               | 46.9        |  |
| DMA (ours) | 17.88           | 40.9            | 38.2         | 12.6              | 56.4         | 76.6         | <u>58.7</u>       | 39.6         | 60.4               | 47.9        |  |

### <span id="page-11-0"></span>5.3 Performance Comparison

Downstream Benchmark Evaluations for Base Model. We used the Qwen3 1.7B (Team 2025) model structure as a baseline, making only modifications to the self-attention part for comparison. We first pre-trained the model on a high-quality dataset covering four domains: Web, TextBook, Code, and Math, with a total of 32 billion tokens and a sequence length of 2,048, thereby providing the model with basic language skills and general knowledge. Subsequently, we carefully selected 8B tokens packaged into sequences of length 8K. We conducted a second phase of pre-training by adjusting the RoPE base frequency from 10K to 100K (Xiong et al. 2023), ensuring that the model could effectively handle longer inputs. Ultimately, we obtained three models: MHA, NSA, and DMA, and evaluated them on the following tasks: LLAMBADA (Paperno et al. 2016), MMLU (Hendrycks et al. 2021), TriviaQA (Joshi et al. 2017), ARC (P. Clark et al. 2018), PIQA (Bisk et al. 2020), HellaSwag (Zellers et al. 2019), OBQA (Mihaylov et al. 2018), Winogrande (Sakaguchi et al. 2021), and the English tasks of LongBench (Bai et al. 2023). We also compared several advanced inference sparse methods, including H2O (Zhenyu Zhang et al. 2023), infLLM (C. Xiao et al. 2024), Quest (Jiaming Tang et al. 2024), DAM (Hanzhi Zhang et al. 2025), and Exact-Top, which first computes full attention scores using MHA and then performs sparsification based on that. The results are shown in Table 2. In both zero-shot and five-shot settings, DMA outperforms the baseline on most tasks, achieving excellent overall performance.

<span id="page-12-0"></span>Table 3: Downstream Evaluations for Finetuned Model. The best results are in bold, and the second-best results are underlined. For models supervisedly fine-tuned at a 16K sequence length, DMA outperforms other methods in most tasks.

| Method     | MMLU<br>acc<br>↑ | BBH<br>acc<br>↑ | GSM8K<br>acc<br>↑ | MATH<br>acc<br>↑ | MBPP<br>acc<br>↑ | ARC<br>acc<br>↑ | PIQA<br>acc<br>↑ | LongBench<br>acc<br>↑ | RULER<br>acc<br>↑ | Avg<br>acc<br>↑ |
|------------|------------------|-----------------|-------------------|------------------|------------------|-----------------|------------------|-----------------------|-------------------|-----------------|
| MHA        | 46.4             | 37.7            | 46.3              | 11.7             | 40.0             | 59.8            | 76.2             | 30.2                  | 60.6              | 39.0            |
| H2O        | 42.4             | 34.5            | 44.8              | 10.5             | 38.5             | 57.6            | 74.1             | 28.5                  | 47.8              | 35.3            |
| InfLLM     | 44.2             | 36.0            | 45.5              | 11.0             | 39.2             | 58.9            | 75.6             | 29.1                  | 55.3              | 37.2            |
| Quest      | 44.0             | 34.8            | 45.0              | 10.8             | 38.8             | 56.2            | 73.7             | 28.9                  | 50.7              | 36.1            |
| DAM        | 45.0             | 36.5            | 46.0              | 11.5             | 37.2             | 56.9            | 72.4             | 26.4                  | 49.2              | 36.0            |
| Exact-Top  | 45.1             | 37.2            | 45.1              | 11.5             | 38.4             | 57.3            | 75.4             | 28.3                  | 44.8              | 35.7            |
| NSA        | 43.8             | 38.4            | 46.2              | 11.6             | 40.5             | 59.1            | 75.7             | 30.2                  | 59.6              | 38.6            |
| DMA (ours) | 46.2             | 38.2            | 46.8              | 11.6             | 40.6             | 59.6            | 76.6             | 30.7                  | 60.5              | 39.2            |

Downstream Benchmark Evaluations for Finetuned Model. We further fine-tuned all models at a 16K sequence length with an adjusted RoPE base frequency to 400K, to further enhance the models' long-context generalization capabilities. Ultimately, we obtained three fine-tuned models: MHA, NSA, and DMA, and evaluated them on the following tasks: MMLU (Hendrycks et al. [2021\)](#page-19-15), BBH (Suzgun et al. [2023\)](#page-20-17), GSM8K (Cobbe et al. [2021\)](#page-18-18), MATH (Hendrycks et al. [2020\)](#page-19-18), MBPP (Austin et al. [2021\)](#page-18-19), LongBench (Bai et al. [2023\)](#page-18-17), and RULER (Hsieh et al. [2024\)](#page-19-19). We used the same advanced inference sparse methods, which also compute full attention scores using MHA and then perform sparsification based on that. The results are shown in Table [3.](#page-12-0) DMA achieved the best average score, leading in GSM8K, MBPP, and LongBench, while remaining highly competitive in MMLU and RULER. NSA ranked first in BBH, with DMA closely following, while the full-attention MHA performed best in RULER but lagged behind DMA in average score. These results indicate that DMA's content-aware sparse mask effectively transfers even under longer context fine-tuning.

Extrapolated Content Retrieval. We further conducted an apples-to-apples comparison between MHA, NSA, and DMA using the needle-in-a-haystack task (Kamradt [2023\)](#page-19-7) to evaluate the models' ability to retrieve information accurately from long texts. In this synthetic retrieval task, a random and information-rich sentence is inserted into a lengthy document, and the model needs to retrieve the needle from the haystack to answer the question. As shown in Figure [8,](#page-11-2) as the context length increases, the advantage of DMA over NSA and MHA gradually expands. Notably, when the context length exceeds the pre-training sequence length, all three models exhibit a performance decline; however, the decrease in DMA's performance is significantly smaller than that of NSA and MHA, demonstrating stronger extrapolation capabilities and more effective retrieval of information in unseen length ranges. We speculate that trainable sparse attention inherently possesses stronger sequence length extrapolation. This experimental result has dual significance: on one hand, it validates DMA's intrinsic advantages in handling ultra-long documents, especially in practical application scenarios that require precise localization and extraction of key information; on the other hand, it reveals the structural advantages of DMA's content-aware dynamic mask mechanism in maintaining long-distance dependency modeling capabilities, even when the sequence length exceeds the pre-training range, thus maintaining relatively stable performance. This extrapolation capability is of great value for practical applications that require processing long documents.

Our comprehensive experimental results demonstrate the exceptional performance of Dynamic Mask Attention across various tasks and model scales. In scaling perplexity experiments, DMA consistently outperformed other attention variants across different parameter scales from 80M to 1.7B; in the multi-query associative recall task, DMA exhibited superior information retrieval capabilities and efficiency; in kernel implementations, DMA showed extremely high speedup ratios in various long-sequence application scenarios; in downstream benchmark evaluations, DMA models outperformed the original MHA and its various sparse variants on most tasks; in the needle-in-a-haystack task, DMA demonstrated significantly stronger length extrapolation capabilities. These results collectively validate the effectiveness of DMA as a sparse attention solution that simultaneously enhances computational efficiency and model performance.

<span id="page-13-0"></span>![](_page_13_Figure_0.jpeg)

Figure 9: **Weights Heatmaps of DMA**. The heatmaps show the attention weights of each head in the Dynamic Mask Attention mechanism, indicating which tokens each head focuses on. Full heatmaps can be found in Appendix C.

![](_page_13_Figure_2.jpeg)

Figure 10: **Adaptive Filtering of DMA**. **Top**: the noisy token signal retains its underlying low-frequency structure while the adaptive mask focuses on informative tokens compared with uniform sampling. **Bottom**: the learned content-aware scores and the resulting mask illustrate how DMA allocates compute to relevant regions while skipping redundant context.

## 6 Analysis

In this section, we analyze Dynamic Mask Attention, highlighting its distinct advantages in handling long-range dependencies and providing dynamic context awareness.

## 6.1 Head Specialization

As shown in Figure 9 11, our analysis of the attention patterns learned by the model reveals how DMA creates content-aware sparse structures that adapt to different contextual needs. Unlike the uniform patterns of traditional attention mechanisms, each DMA attention head develops a unique sparse pattern: some heads focus on the most recent tokens to capture local context, while others attend to specific distant positions for long-range dependencies, and additional heads maintain broader context awareness for global understanding. This diversity allows the model to capture various types of dependencies simultaneously while maintaining computational efficiency, making efficient use of each subspace.

Local Context Heads. For example, Head 0, Head 1, Head 8, Head 10, and Head 11 tend to focus on the most recent tokens, forming a local band attention pattern. These heads are primarily responsible for capturing syntactic structures, phrase-level semantics, and local dependencies, which are particularly important for tasks requiring precise local context handling.

Range Dependency Heads. For example, Head 2, Head 3, Head 4, Head 5, and Head 14 demonstrate the ability to attend to specific distant tokens. These heads are specialized in capturing long-range semantic associations, such as resolving coreference issues or tracking complex storylines in lengthy documents. They can skip over large amounts of intermediate information and directly connect distant but semantically related parts, which is crucial for deep reasoning and contextual understanding.

Global Context Heads. For example, Head 6, Head 7, Head 9, Head 12, Head 13, and Head 15 exhibit a sparser but broader attention distribution, sampling key information from the entire sequence to form an overall perception of the global context. These heads function similarly to summarizers, responsible for integrating information from different parts to create a coherent global representation. This capability is crucial for tasks that require a comprehensive understanding of the entire input to make accurate predictions.

Dynamic Adaptability. The most significant advantage of DMA lies in its dynamism. These attention patterns are not static; they are dynamically generated based on the input content. This means the model can adjust its attention strategy in real-time, activating the most appropriate combination of heads when processing different tasks or text types. For example, when processing code, it might rely more on long-range dependency heads to track variable definitions and usages, whereas in a conversation, it might focus more on local context heads to understand the current exchange. This content-aware adaptability is the core advantage of DMA over static sparse attention methods.

This naturally occurring specialization is a direct result of the content-aware mask mechanism, enabling the model to effectively handle various types of dependencies while maintaining computational efficiency, achieving effective integration of multi-scale information. This hierarchical integration mechanism can effectively handle multi-level semantic structures in complex texts. It is worth noting that head specialization may also occur in traditional MHA, but the specialization patterns in DMA are more pronounced and functionally clearer, which may be a key reason for its superior performance across various tasks.

## 6.2 Adaptive Filtering

From the perspective of modern signal processing, Trainable Dynamic Mask Sparse Attention essentially performs dynamic downsampling of the input sequence through learnable adaptive filters, i.e., masks, retaining only key information components. This allows for efficient extraction of low-frequency dependencies in long-distance signals, such as text, while suppressing noise redundancy. The core logic is to treat long texts as noisy low-frequency signals, where the mask acts as an adaptive filter, and the dynamically selected retained tokens are equivalent to intelligent downsampling based on signal relevance.

Learnable Content-aware Filters. Unlike traditional sparse methods that rely on fixed patterns, DMA's mask is dynamically generated based on the input content, making it a content-aware adaptive filter. This filter learns to identify and amplify key components in the signal, i.e., important tokens, while attenuating or completely filtering out noise, i.e., irrelevant tokens. This mechanism ensures that computational resources are precisely allocated to the most critical information for the current task, effectively avoiding information loss due to excessively long contexts in needle-in-a-haystack problems.

Multi-scale Signal Decomposition. The different attention heads in DMA learn various sparse patterns, which can be viewed as a set of parallel adaptive filters, each responsible for capturing different scales or types of signal features. This multi-scale decomposition allows the model to build a comprehensive and hierarchical understanding of the input signal with extremely high efficiency.

Recasting sparse attention as an adaptive filtering problem, DMA offers a new perspective for understanding and optimizing long text processing. It achieves intelligent filtering of information through content awareness and multi-scale decomposition, ensuring that the model learns the optimal sparse strategies.

### 7 Discussion

In this section, we discuss the core deficiencies of existing sparse attention methods, analyze how Dynamic Mask Attention addresses these issues, and explore its limitations and future development directions.

### 7.1 Limitations of Existing Approaches

Existing sparse attention methods exhibit three critical deficiencies that limit their practical effectiveness:

**Post-hoc Sparsification Degradation.** The performance degradation caused by post-hoc sparsification stems from the fundamental mismatch between existing methods and the optimization trajectory of pretrained models. As demonstrated by Chen et al. (Z. Chen et al. 2024), retaining only the top 20% of attention weights covers only 70% of the total attention scores. This forced sparsification strategy compels models to deviate from the optimal parameter configurations learned on large-scale corpora. More critically, this approach causes irreversible damage to key structural components in pretrained models, such as retrieval heads and copy heads, as these specialized attention heads are misidentified as "unimportant" and pruned during inference. This structural destruction directly leads to significant performance degradation in tasks that require precise information retrieval and copying.

**Training-Inference Efficiency Gap.** Most sparse attention methods optimize only for inference, neglecting training-phase computational demands. This creates bottlenecks across LLM development: pretraining on long documents, long-context fine-tuning, and reinforcement learning. Without effective training-time sparsity support, these crucial phases remain constrained by  $O(n^2)$  computational complexity, limiting development of more capable long-context models.

Non-differentiable Components and Inefficient Backpropagation. Non-differentiable components and inefficient backpropagation problems reveal the technical shortcomings of existing methods in terms of trainability. The discrete operations in methods like ClusterKV (G. Liu et al. 2024) and MagicPIG (Z. Chen et al. 2024) introduce discontinuities in computational graphs, which block gradient flow and hinder the learning of optimal sparse patterns. Even trainable methods like HashAttention (Desai et al. 2024) suffer from memory access inefficiencies due to token-granular selection, which is incompatible with the contiguous memory access and block-wise computation requirements of efficient attention techniques, such as FlashAttention. Consequently, these implementations are forced to revert to naive implementations with low hardware utilization, significantly degrading training efficiency.

#### 7.2 How Dynamic Mask Attention Addresses Core Issues

Dynamic Mask Attention systematically addresses the aforementioned fundamental issues through three core innovations, achieving unified, efficient, and sparse computation for both training and inference phases.

Native Trainable Sparsity. Native trainable sparsity is DMA's key innovation for addressing post-hoc sparsification issues. Unlike traditional methods, DMA embeds sparsity into the model architecture from the ground up, ensuring that sparse attention patterns are fully aligned with the model's optimization trajectory. Specifically, DMA retains complete, uncompressed KV caches  $k = \text{concat}([k_1, \ldots, k_t])$  and  $v = \text{concat}([v_1, \ldots, v_t])$ , ensuring the original fidelity of historical information and precise recall capabilities, avoiding information bottlenecks that may arise from fixed-state compression in State Space Models. This comprehensive information retention mechanism enables DMA to precisely access any token in the historical context at any moment, without losing critical information due to lossy compression methods like Mamba. More importantly, DMA's sparsification occurs during the attention weight computation phase, rather than in post-training processing, ensuring that models do not deviate from pre-trained parameter configurations during sparsification, thereby protecting key structural components, such as retrieval heads and copy heads, from damage.

Unified Training-Inference Architecture. The unified training-inference architecture eliminates the fundamental gap in training-inference efficiency that exists in existing methods. DMA's dynamic weight computation  $\delta = \exp(\tau(v\Delta) \times A)$  uses identical sparsification strategies during both training and inference phases. This consistency ensures that models can learn optimal sparse patterns during training and seamlessly apply these patterns during inference. This unified architecture particularly benefits three critical stages of modern LLM development: the pretraining stage can efficiently process long document sequences; the long-context fine-tuning stage can adapt to specific task requirements; the reinforcement learning stage can effectively update attention weights through policy gradients. DMA reduces computational complexity from  $O(n^2)$  to  $O(n \cdot w)$ , enabling the training of larger-scale long-context models.

Fully Differentiable Design. The fully differentiable design ensures that DMA maintains gradient flow continuity throughout the entire computation process. The computation of dynamic mask weights is based entirely on differentiable operations, including linear transformations of value representations, non-negative activation functions (·), and exponential functions, thereby avoiding gradient interruptions caused by discrete operations such as k-means clustering and SimHash. Although the mask generation process involves a top-k operation, it is not the core learning objective of DMA but merely a tool for sparse selection; thus, we only use this discrete operation in the forward pass. Moreover, the attention weight computation part is designed such that the gradients for masked positions should naturally be zero, so skipping computation and setting gradients to zero is the correct behavior. This design enables the model to learn optimal attention patterns that are sparse in an end-to-end manner, dynamically adjusting which historical positions are most critical for current reasoning, thereby achieving truly content-aware, selective computation. Additionally, each head in a multi-head attention mechanism can independently generate different sparse patterns, thereby the representational capabilities of the multi-head architecture by focusing on different information segments in distinct subspaces.

## 7.3 Limitations and Future Works

Despite Dynamic Mask Attention's significant progress in addressing the core issues of existing methods, several limitations remain that warrant further exploration and improvement in future work.

Adaptive Window Size Selection. Adaptive window size selection is the primary challenge facing DMA. While the current fixed window size design provides predictable computational complexity, it may not optimally adapt to the dynamic demands of different tasks and contexts. For instance, code generation tasks may require larger windows to capture longrange structural dependencies, while simple question-answering tasks may only need smaller windows. Future research directions include developing adaptive window size selection mechanisms based on task complexity, sequence length, and content features, potentially through reinforcement learning or meta-learning approaches to dynamically optimize window parameters. Alternatively, designing hierarchical multi-scale attention structures can be considered to capture dependencies across different ranges simultaneously.

Position Encoding Enhancement. Our needle-in-a-haystack experiments revealed an intriguing phenomenon: trainable sparse attention mechanisms, such as DMA, exhibit stronger length extrapolation capabilities compared to dense attention when context lengths exceed the pretraining bounds. This finding suggests that the fundamental bottleneck for extrapolation may lie in the position encoding method rather than the attention mechanism itself. Current RoPE-based position encodings struggle with out-of-distribution sequence lengths, but DMA's dynamic sampling architecture offers a potential alternative pathway for encoding positional information. Specifically, the zero-order hold sampling values that are added as attention biases can be explored to explicitly incorporate positional information into these sampling values, potentially replacing or complementing RoPE to create a more extrapolation-friendly encoding scheme. Such an approach might leverage the inherent advantages of sparse attention's selective computation to create position representations that scale more naturally to unseen lengths. This direction could help address one of the most persistent challenges in longcontext modeling: maintaining consistent positional understanding across arbitrary sequence lengths without requiring length-specific fine-tuning.

Multi-Modal Extension. Multi-modal extension represents an essential direction for DMA development. The current DMA design is primarily optimized for text sequences; however, modern AI systems increasingly require processing mixed inputs of text, images, audio, and video. Attention sparsity in multi-modal scenarios exhibits more complex patterns: interactions between different modalities may require different attention distributions, temporally aligned multi-modal information may need synchronized attention mechanisms, and modality-specific long-range dependencies may require specialized sparse patterns. Future research can explore modality-aware dynamic mask generation, coordination mechanisms for cross-modal attention weights, and specialized sparse pattern designs for different modal characteristics.

Integration with Modern Frameworks. Seamless integration of DMA into mainstream deep learning frameworks, such as PyTorch and the Hugging Face Transformers library, is crucial for its widespread adoption and practical impact. This requires developing a user-friendly and highly optimized implementation that can be easily incorporated into existing model architectures and training pipelines. A key aspect of this integration is the development of efficient, low-level kernels, potentially using Triton or CUDA, to ensure that the performance benefits of DMA are fully realized on modern hardware. Providing a plug-and-play module compatible with the Hugging Face Transformers library would significantly lower the barrier for researchers and practitioners to apply DMA to their models and tasks, thereby fostering further innovation and comparative studies in the field of sparse attention.

## 8 Conclusion

In this paper, we introduced Dynamic Mask Attention, a novel trainable sparse attention mechanism that effectively addresses the key challenges in long-context modeling for large language models. By integrating content-aware dynamic sparse masks with position-aware sparse attention weight computations, Dynamic Mask Attention successfully balances computational efficiency while preserving the ability to retrieve information from long contexts precisely.

Our approach makes several key contributions to the field of efficient attention mechanisms. First, Dynamic Mask Attention achieves computational efficiency comparable to sliding window attention while maintaining the information retrieval capabilities of full attention by retaining a complete, uncompressed key-value cache. Second, by dynamically generating attention masks from value representations, our method enables models to learn which tokens are relevant to the current reasoning process, effectively leveraging both content-aware and position-aware sparsity patterns inherent in language modeling tasks. Third, our specialized hardware-optimized kernel for Dynamic Mask Attention efficiently handles sparse mask regions, translating theoretical computational gains into practical speed improvements.

The comprehensive experimental evaluation demonstrates that Dynamic Mask Attention consistently outperforms existing attention mechanisms across various scales and tasks. In scaling law studies, Dynamic Mask Attention exhibited superior perplexity compared to other attention variants. On challenging tasks like multi-query associative recall, Dynamic Mask Attention demonstrated both effectiveness in information retrieval and computational efficiency. Most significantly, our 1.7B parameter model with Dynamic Mask Attention outperformed the vanilla attention counterpart on standard benchmarks and showed remarkably stronger extrapolation capabilities on the needle-in-a-haystack task when context lengths exceeded the pre-training sequence length.

Dynamic Mask Attention represents a significant step forward in developing efficient and effective attention mechanisms for long-context modeling. By maintaining the full expressive power of attention while reducing computational complexity, our approach enables the development of more capable language models that can effectively process lengthy documents, complex reasoning chains, and rich contextual information. This capability is particularly valuable for applications requiring deep reasoning, code generation, and multi-turn autonomous agents.

Future work could explore adaptive window size selection based on content complexity, create more extrapolation-friendly positional encoding schemes, extend it to multimodal contexts, and develop further theoretical analyses of its properties. We believe that Dynamic Mask Attention provides a promising direction for future research in efficient transformer architectures and will facilitate the development of more powerful and computationally efficient language models.

#### Acknowledgments

We would like to express our gratitude to the OpenSeek project team at Beijing Academy of Artificial Intelligence for their support in developing the hardware kernels. We would also like to thank Professor Albert Gu from Carnegie Mellon University for his valuable guidance and suggestions in connecting the concepts of state-space and self-attention. Additionally, we extend our thanks to all the friends in the community who provided feedback and suggestions, your support has been instrumental in the continuous improvement of this work.

### References

- <span id="page-18-10"></span>[1] Martin Arjovsky, Amar Shah, and Yoshua Bengio. "Unitary Evolution Recurrent Neural Networks". In: *The International Conference on Machine Learning (ICML)*. 2016, pp. 1120–1128.
- <span id="page-18-9"></span>[2] Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra, and Christopher Ré. "Zoology: Measuring and Improving Recall in Efficient Language Models". In: *The International Conference on Learning Representations*. 2024.
- <span id="page-18-19"></span>[3] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. "Program synthesis with large language models". In: arXiv preprint arXiv:2108.07732 (2021).
- <span id="page-18-17"></span>[4] Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, et al. "Longbench: A bilingual, multitask benchmark for long context understanding". In: *arXiv* preprint arXiv:2308.14508 (2023).
- <span id="page-18-2"></span>[5] Iz Beltagy, Matthew E Peters, and Arman Cohan. "Longformer: The long-document transformer". In: *arXiv preprint arXiv:2004.05150* (2020).
- <span id="page-18-12"></span>[6] Loubna Ben Allal, Anton Lozhkov, Guilherme Penedo, Thomas Wolf, and Leandro von Werra. *SmolLM-Corpus*. July 2024.
- <span id="page-18-16"></span>[7] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. "PIQA: Reasoning about Physical Commonsense in Natural Language". In: *Proceedings of the AAAI conference on Artificial Intelligence*. Vol. 34. 2020.
- <span id="page-18-11"></span>[8] Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, et al. "Gpt-NeoX-20B: An Open-source Autoregressive Language Model". In: *arXiv* preprint arXiv:2204.06745 (2022).
- <span id="page-18-4"></span>[9] Zhuoming Chen, Ranajoy Sadhukhan, Zihao Ye, Yang Zhou, Jianyu Zhang, Niklas Nolte, Yuandong Tian, Matthijs Douze, Leon Bottou, Zhihao Jia, et al. "Magicpig: Lsh sampling for efficient llm generation". In: *arXiv preprint* arXiv:2410.16179 (2024).
- <span id="page-18-6"></span>[10] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. "Generating long sequences with sparse transformers". In: *arXiv preprint arXiv:1904.10509* (2019).
- <span id="page-18-15"></span>[11] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. "Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge". In: *arXiv preprint arXiv:1803.05457* (2018).
- <span id="page-18-18"></span>[12] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. "Training verifiers to solve math word problems". In: *arXiv* preprint arXiv:2110.14168 (2021).
- <span id="page-18-7"></span>[13] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V Le, and Ruslan Salakhutdinov. "Transformer-xl: Attentive language models beyond a fixed-length context". In: *arXiv preprint arXiv:1901.02860* (2019).
- <span id="page-18-8"></span>[14] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. "FLASHATTENTION: fast and memory-efficient exact attention with IO-awareness". In: *Proceedings of the 36th International Conference on Neural Information Processing Systems*. 2022.
- <span id="page-18-0"></span>[15] Google DeepMind. *Gemini 2.5 Pro.* Mar. 2025. URL: https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025.
- <span id="page-18-5"></span>[16] Aditya Desai, Shuo Yang, Alejandro Cuadron, Ana Klimovic, Matei Zaharia, Joseph E Gonzalez, and Ion Stoica. "HashAttention: Semantic Sparsity for Faster Inference". In: *arXiv preprint arXiv:2412.14468* (2024).
- <span id="page-18-14"></span>[17] Clémentine Fourrier, Nathan Habib, Hynek Kydlíček, Thomas Wolf, and Lewis Tunstall. *LightEval: A lightweight framework for LLM evaluation*. Version 0.7.0. 2023. URL: https://github.com/huggingface/lighteval.
- <span id="page-18-13"></span>[18] Leo Gao, Jonathan Tow, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Kyle McDonell, Niklas Muennighoff, Jason Phang, Laria Reynolds, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. *A Framework for Few-shot Language Model Evaluation*. Version v0.0.1. Sept. 2021. DOI: 10.5281/zenodo.5371628. URL: https://doi.org/10.5281/zenodo.5371628.
- <span id="page-18-3"></span>[19] Yizhao Gao, Zhichen Zeng, Dayou Du, Shijie Cao, Peiyuan Zhou, Jiaxing Qi, Junjie Lai, Hayden Kwok-Hay So, Ting Cao, Fan Yang, et al. "Seerattention: Learning intrinsic sparse attention in your llms". In: *arXiv preprint* arXiv:2410.13276 (2024).
- <span id="page-18-1"></span>[20] Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. "Model tells you what to discard: Adaptive kv cache compression for llms". In: *arXiv preprint arXiv:2310.01801* (2023).

- <span id="page-19-1"></span>[21] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. "Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning". In: *arXiv* preprint arXiv:2501.12948 (2025).
- <span id="page-19-11"></span>[22] Alex Hägele, Elie Bakouch, Atli Kosson, Leandro Von Werra, Martin Jaggi, et al. "Scaling laws and compute-optimal training beyond fixed training durations". In: *Advances in Neural Information Processing Systems* 37 (2024), pp. 76232–76264.
- <span id="page-19-18"></span>[23] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. "Measuring massive multitask language understanding". In: *arXiv preprint arXiv:2009.03300* (2020).
- <span id="page-19-15"></span>[24] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. "Measuring Massive Multitask Language Understanding". In: *International Conference on Learning Representations*. 2021.
- <span id="page-19-6"></span>[25] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. "An Empirical Analysis of Compute-Optimal Large Language Model Training". In: *Advances in Neural Information Processing Systems (NeurIPS)* 35 (2022), pp. 30016–30030.
- <span id="page-19-19"></span>[26] Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. "RULER: What's the Real Context Size of Your Long-Context Language Models?" In: arXiv preprint arXiv:2404.06654 (2024).
- <span id="page-19-0"></span>[27] HuggingFace. Open R1: A fully open reproduction of DeepSeek-R1. 2025. URL: https://github.com/huggingface/open-r1.
- <span id="page-19-16"></span>[28] Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. *TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension*. 2017. arXiv: 1705.03551 [cs.CL].
- <span id="page-19-7"></span>[29] GKamradt. LLMTest NeedleInAHaystack. 2023. URL: https://github.com/gkamradt/LLMTest\_NeedleInAHaystack.
- <span id="page-19-12"></span>[30] Houyi Li, Wenzhen Zheng, Qiufeng Wang, Hanshan Zhang, Zili Wang, Shijie Xuyang, Yuantao Fan, Shuigeng Zhou, Xiangyu Zhang, and Daxin Jiang. *Predictable Scale: Part I – Optimal Hyperparameter Scaling Law in Large Language Model Pretraining*. 2025. arXiv: 2503.04715 [cs.LG]. URL: https://arxiv.org/abs/2503.04715.
- <span id="page-19-4"></span>[31] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. "Snapkv: Llm knows what you are looking for before generation". In: *Advances in Neural Information Processing Systems* 37 (2024), pp. 22947–22970.
- <span id="page-19-3"></span>[32] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. "Deepseek-v3 technical report". In: *arXiv preprint arXiv:2412.19437* (2024).
- <span id="page-19-5"></span>[33] Guangda Liu, Chengwei Li, Jieru Zhao, Chenqi Zhang, and Minyi Guo. "Clusterkv: Manipulating llm kv cache in semantic space for recallable compression". In: *arXiv preprint arXiv:2412.03213* (2024).
- <span id="page-19-13"></span>[34] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. "Lost in the middle: How language models use long contexts". In: *Transactions of the Association for Computational Linguistics* 12 (2024), pp. 157–173.
- <span id="page-19-10"></span>[35] Ilya Loshchilov and Frank Hutter. "Fixing Weight Decay Regularization in Adam". In: *ArXiv* abs/1711.05101 (2017). URL: https://api.semanticscholar.org/CorpusID: 3312944.
- <span id="page-19-2"></span>[36] Andre Martins and Ramon Astudillo. "From softmax to sparsemax: A sparse model of attention and multi-label classification". In: *International conference on machine learning*. PMLR. 2016, pp. 1614–1623.
- <span id="page-19-17"></span>[37] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. "Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering". In: arXiv preprint arXiv:1809.02789 (2018).
- <span id="page-19-9"></span>[38] Meta NVIDIA. *PyTorch Container Image*. https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch. 2022.
- <span id="page-19-8"></span>[39] Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. "In-context Learning and Induction Heads". In: *Transformer Circuits Thread* (2022). https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html.
- <span id="page-19-14"></span>[40] Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc-Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. "The LAMBADA Dataset: Word Prediction Requiring a Broad Discourse Context". In: *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*. 2016, pp. 1525–1534.

- <span id="page-20-2"></span>[41] Joon Sung Park, Joseph O'Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. "Generative agents: Interactive simulacra of human behavior". In: *Proceedings of the 36th annual acm symposium on user interface software and technology.* 2023, pp. 1–22.
- <span id="page-20-11"></span>[42] David W Romero, Anna Kuzina, Erik J Bekkers, Jakub M Tomczak, and Mark Hoogendoorn. "CKConv: Continuous Kernel Convolution For Sequential Data". In: *arXiv preprint arXiv:2102.02611* (2021).
- <span id="page-20-15"></span>[43] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. "Winogrande: An Adversarial Winograd Schema Challenge at Scale". In: *Communications of the ACM* 64.9 (2021), pp. 99–106.
- <span id="page-20-0"></span>[44] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. "Scaling llm test-time compute optimally can be more effective than scaling model parameters". In: *arXiv preprint arXiv:2408.03314* (2024).
- <span id="page-20-17"></span>[45] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, et al. "Challenging big-bench tasks and whether chain-of-thought can solve them". In: Findings of the Association for Computational Linguistics: ACL 2023. 2023, pp. 13003–13051.
- <span id="page-20-9"></span>[46] Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. "Quest: Query-aware sparsity for efficient long-context llm inference". In: *arXiv preprint arXiv:2406.10774* (2024).
- <span id="page-20-3"></span>[47] Qwen Team. Qwen3. Apr. 2025. URL: https://qwenlm.github.io/blog/qwen3.
- <span id="page-20-4"></span>[48] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. "Attention Is All You Need". In: *Advances in Neural Information Processing Systems*. 2017.
- <span id="page-20-12"></span>[49] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. "Transformers: State-of-the-Art Natural Language Processing". In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. Online: Association for Computational Linguistics, Oct. 2020, pp. 38–45.
- <span id="page-20-10"></span>[50] Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan Xiao, Yankai Lin, Zhengyan Zhang, Zhiyuan Liu, and Maosong Sun. "Infllm: Training-free long-context extrapolation for llms with an efficient context memory". In: *arXiv* preprint *arXiv*:2402.04617 (2024).
- <span id="page-20-13"></span>[51] Wenhan Xiong, Jingyu Liu, Igor Molybog, Hejia Zhang, Prajjwal Bhargava, Rui Hou, Louis Martin, Rashi Rungta, Karthik Abinav Sankararaman, Barlas Oguz, et al. "Effective long-context scaling of foundation models". In: *arXiv* preprint arXiv:2309.16039 (2023).
- <span id="page-20-6"></span>[52] Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, YX Wei, Lean Wang, Zhiping Xiao, et al. "Native sparse attention: Hardware-aligned and natively trainable sparse attention". In: arXiv preprint arXiv:2502.11089 (2025).
- <span id="page-20-5"></span>[53] Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. "Big bird: Transformers for longer sequences". In: *Advances in neural information processing systems* 33 (2020), pp. 17283–17297.
- <span id="page-20-14"></span>[54] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. "HellaSwag: Can a Machine Really Finish Your Sentence?" In: *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*. 2019.
- <span id="page-20-16"></span>[55] Hanzhi Zhang, Heng Fan, Kewei Sha, Yan Huang, and Yunhe Feng. DAM: Dynamic Attention Mask for Long-Context Large Language Model Inference Acceleration. 2025. arXiv: 2506.11104 [cs.CL]. URL: https://arxiv.org/abs/2506.11104.
- <span id="page-20-1"></span>[56] Kechi Zhang, Jia Li, Ge Li, Xianjie Shi, and Zhi Jin. "Codeagent: Enhancing code generation with tool-integrated agent systems for real-world repo-level coding challenges". In: *arXiv preprint arXiv:2401.07339* (2024).
- <span id="page-20-7"></span>[57] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, et al. "H2o: Heavy-hitter oracle for efficient generative inference of large language models". In: Advances in Neural Information Processing Systems 36 (2023), pp. 34661–34710.
- <span id="page-20-8"></span>[58] Zihan Zhou, Chong Li, Xinyi Chen, Shuo Wang, Yu Chao, Zhili Li, Haoyu Wang, Rongqiao An, Qi Shi, Zhixing Tan, et al. "LLM MapReduce: Simplified Long-Sequence Processing using Large Language Models". In: *CoRR* (2024).

## A Dynamic Mask Attention Implementation

The following listing provides a sample implementation of the Dynamic Mask Attention algorithm in PyTorch, as described in Section [4.](#page-5-2)

Listing 1: Dynamic Mask Attention implementation in PyTorch

```
def dynamic_mask_attention(h_t, position_embeddings, causal_m, past_key_value,
   W_Q, W_K, W_V, W_dt, A, W_O,
   num_heads, scaling, keep_window_size):
   input_shape = h_t.shape[:-1] # [b, q_len]
   hidden_shape = (*input_shape, -1, h_t.shape[-1] // num_heads)
   # linear projections
   q_t = W_Q(h_t).view(hidden_shape).transpose(1, 2) # [b, n_h, q_len, d_h]
   k_t = W_K(h_t).view(hidden_shape).transpose(1, 2) # [b, n_h, q_len, d_h]
   v_t = W_V(h_t).view(hidden_shape).transpose(1, 2) # [b, n_h, q_len, d_h]
   o_t = torch.zeros_like(q_t) # [b, n_h, q_len, d_h]
   # apply rotary position embeddings
   q_t, k_t = apply_rotary_pos_emb(q_t, k_t, *position_embeddings)
   # concatenate past key and value states
   k, v = past_key_value.update(k_t, v_t) # [b, n_h, k_len, d_h]
   # calculate dynamic mask
   dt = W_dt(v.transpose(1, 2).reshape(v.shape[0], v.shape[-2], -1)) # [b, k_len, n_h]
   dt = torch.exp(A * F.softplus(dt)).transpose(-1, -2) # [b, n_h, k_len]
   m_t = dt[:, :, None, :].expand(-1, -1, h_t.shape[1], -1) # [b, n_h, q_len, k_len]
   active_m = torch.zeros_like(m_t)
   m_t = m_t.masked_fill(causal_m != 0, -float('inf'))
   topk_indices = torch.topk(m_t, keep_window_size, dim=-1, sorted=False).indices
   active_m = active_m.scatter(-1, topk_indices, 1.0)
   m_t = m_t.masked_fill(active_m == 0.0, -float('inf'))
   # calculate sparse attention weight
   for b_idx in range(hidden_shape[0]): # b
      for h_idx in range(num_heads): # n_h
          for q_idx in range(hidden_shape[1]): # q_len
             q_elem = q_t[b_idx, h_idx, q_idx, :] # [d_h]
             indices = topk_indices[b_idx, h_idx, q_idx] # [w]
             k_vecs = k[b_idx, h_idx, indices, :] # [w, d_h]
             v_vecs = v[b_idx, h_idx, indices, :] # [w, d_h]
             a_elem = torch.sum(q_elem.unsqueeze(0) * k_vecs, dim=-1) # [w]
             a_elem = a_elem * scaling + m_t[b_idx, h_idx, q_idx, indices]
             a_elem = F.softmax(a_elem, dim=-1)
             o_elem = torch.sum(a_elem.unsqueeze(1) * v_vecs, dim=0) # [d_h]
             o_t[b_idx, h_idx, q_idx, :] = o_elem
   o_t = o_t.transpose(1, 2).contiguous() # [b, q_len, n_h, d_h]
   h_t = W_O(o_t.view(*input_shape, -1)) # [b, q_len, d_model]
   return h_t
```

The implementation demonstrates the core computational flow of the Dynamic Mask Attention mechanism. First, the query, key, and value matrices are computed through linear projections, followed by the application of rotary position embeddings. The core innovation of the algorithm is then reflected in the dynamic mask generation process: dynamic weights are calculated from the value vectors, and a sparse mask is generated using the topk operation, retaining only the most relevant key-value pairs. Finally, in the sparse attention computation phase, the algorithm computes attention weights only for the selected key-value pairs, significantly reducing computational complexity. In actual kernel implementations, it is possible to check if there are any active tokens in the MMA block; if not, the computation for that block can be skipped.

## **B** Experiment Setup

To make the comparison between attention variants fair and reproducible, we standardize the model, data pipeline, optimization, and evaluation, only changing the attention module and its related hyperparameters. All experiments were conducted using the open-source PyTorch images (NVIDIA 2022) and the Transformers framework (Wolf et al. 2020). We use SmolLMCorpus (Ben Allal et al. 2024) as training data. For evaluation frameworks, we utilized the LM evaluation harness (L. Gao et al. 2021) from EleutherAI for perplexity tasks, and the lighteval (Fourrier et al. 2023) from HuggingFace for downstream tasks. Table 4 lists the model sizes and key hyperparameters used at each scale. We keep the depth, width, and number of heads consistent across variants under the same parameter budget, only changing the specific parameters of the attention variants, with parameter symbols consistent with those in the original papers of the attention variants.

We summarize the meaning of the columns in Table 4 and clarify which hyperparameters are used by each attention variant.

- Params: total number of model parameters.
- Steps: training steps.
- Batch: tokens per step.
- LR: peak learning rate.
- *n*<sub>layers</sub>: number of Transformer layers.
- $d_{model}$ : model hidden size.
- $n_h$ : number of attention heads.
- $n_{h_{kv}}$ : number of KV heads. In all our configurations we set  $n_{h_{kv}} = n_h/2$ .
- MHA (full attention): standard scaled dot-product attention. Sparse-specific columns  $(w, d_c, B, B', k)$  are not used.
- SWA (sliding-window attention): w is the sliding window size; other sparse-specific columns are not used.
- MLA (multi-head latent attention):  $d_c$  is the latent/compression dimension; other columns are not used.
- NSA (native sparse attention):  $d_c$  is the compression dimension, w is the sliding window size, B is the compressing block size, B' is the selection block size, and B' is the num selected blocks. All settings following the original paper.
- **DMA** (dynamic mask attention): w is the per-head top-w kept keys for dynamic masks; other sparse-specific columns are not used.

<sup>&</sup>lt;sup>1</sup>The implementation code for MHA is available at https://github.com/Dao-AILab/flash-attention.

<sup>&</sup>lt;sup>2</sup>The implementation code for SWA is available at https://github.com/Dao-AILab/flash-attention.

<sup>&</sup>lt;sup>3</sup>The implementation code for MLA is available at https://github.com/deepseek-ai/FlashMLA.

<sup>&</sup>lt;sup>4</sup>The implementation code for NSA is available at https://github.com/lucidrains/native-sparse-attention-pytorch.

<span id="page-23-0"></span>Table 4: Self-Attention Variants Scaling Laws Configurations. The model and hyperparameter configurations used in our self-attention variants scaling laws experiments.

| Algos | Params    | Steps  | Batch         | LR   | 𝑛𝑙𝑎𝑦𝑒𝑟𝑠 | 𝑑𝑚𝑜𝑑𝑒𝑙 | 𝑛ℎ | 𝑛ℎ𝑘𝑣 | 𝑤    | 𝑑𝑐  | 𝐵  | 𝐵′ | 𝑘  |
|-------|-----------|--------|---------------|------|---------|--------|----|------|------|-----|----|----|----|
| MHA   | 80M<br>≈  | 13,500 | 0.128M tokens | 3e-3 | 12      | 768    | 6  | 3    | -    | -   | -  | -  | -  |
| SWA   | 80M<br>≈  | 13,500 | 0.128M tokens | 3e-3 | 12      | 768    | 6  | 3    | 1024 | -   | -  | -  | -  |
| MLA   | 80M<br>≈  | 13,500 | 0.128M tokens | 3e-3 | 12      | 768    | 6  | 3    | -    | 192 | -  | -  | -  |
| NSA   | 80M<br>≈  | 13,500 | 0.128M tokens | 3e-3 | 12      | 768    | 6  | 3    | 512  | 192 | 32 | 64 | 16 |
| DMA   | 80M<br>≈  | 13,500 | 0.128M tokens | 3e-3 | 12      | 768    | 6  | 3    | 1024 | -   | -  | -  | -  |
| MHA   | 200M<br>≈ | 20,800 | 0.192M tokens | 2e-3 | 16      | 1024   | 8  | 4    | -    | -   | -  | -  | -  |
| SWA   | 200M<br>≈ | 20,800 | 0.192M tokens | 2e-3 | 16      | 1024   | 8  | 4    | 1024 | -   | -  | -  | -  |
| MLA   | 200M<br>≈ | 20,800 | 0.192M tokens | 2e-3 | 16      | 1024   | 8  | 4    | -    | 256 | -  | -  | -  |
| NSA   | 200M<br>≈ | 20,800 | 0.192M tokens | 2e-3 | 16      | 1024   | 8  | 4    | 512  | 192 | 32 | 64 | 16 |
| DMA   | 200M<br>≈ | 20,800 | 0.192M tokens | 2e-3 | 16      | 1024   | 8  | 4    | 1024 | -   | -  | -  | -  |
| MHA   | 680M<br>≈ | 35,000 | 0.392M tokens | 1e-3 | 24      | 1536   | 12 | 6    | -    | -   | -  | -  | -  |
| NSA   | 680M<br>≈ | 35,000 | 0.392M tokens | 1e-3 | 24      | 1536   | 12 | 6    | 512  | 192 | 32 | 64 | 16 |
| DMA   | 680M<br>≈ | 35,000 | 0.392M tokens | 1e-3 | 24      | 1536   | 12 | 6    | 1024 | -   | -  | -  | -  |
| MHA   | 1.7B<br>≈ | 40,000 | 1M tokens     | 1e-3 | 28      | 2048   | 16 | 8    | -    | -   | -  | -  | -  |
| NSA   | 1.7B<br>≈ | 40,000 | 1M tokens     | 1e-3 | 28      | 2048   | 16 | 8    | 512  | 256 | 32 | 64 | 16 |
| DMA   | 1.7B<br>≈ | 40,000 | 1M tokens     | 1e-3 | 28      | 2048   | 16 | 8    | 2048 | -   | -  | -  | -  |

<span id="page-23-1"></span>Table 5: Speed Benchmark Configurations. The common settings for running time curves and the sparse hyperparameters for each method.

| Algo | warmups | runs  | 𝑛ℎ | 𝑛ℎ𝑘𝑣 | 𝑑ℎ  | 𝑤    | 𝑑𝑐  | 𝐵  | 𝐵′ | 𝑘  | precision |
|------|---------|-------|----|------|-----|------|-----|----|----|----|-----------|
| MHA1 | 3       | 1,000 | 32 | 8    | 128 | -    | -   | -  | -  | -  | bf16      |
| SWA2 | 3       | 1,000 | 32 | 8    | 128 | 1024 | -   | -  | -  | -  | bf16      |
| MLA3 | 3       | 1,000 | 32 | 8    | 128 | -    | 192 | -  | -  | -  | bf16      |
| NSA4 | 3       | 1,000 | 32 | 8    | 128 | 512  | 256 | 32 | 64 | 16 | bf16      |
| DMA  | 3       | 1,000 | 32 | 8    | 128 | 1024 | -   | -  | -  | -  | bf16      |

# <span id="page-24-0"></span>C Attention Heatmaps

<span id="page-24-1"></span>![](_page_24_Figure_1.jpeg)

Figure 11: **Full Heatmaps of Dynamic Mask Attention**. The heatmaps show the attention weights of each head in the Dynamic Mask Attention mechanism, indicating which tokens each head focuses on.