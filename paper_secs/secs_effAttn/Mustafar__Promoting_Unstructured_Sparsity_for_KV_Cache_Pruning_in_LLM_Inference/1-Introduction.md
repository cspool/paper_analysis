# 1 Introduction

In the age of Large Language Models (LLMs), advances in the machine learning domain [\[41,](#page-13-0) [2,](#page-10-0) [6\]](#page-10-1) and the fast and efficient computing systems [\[21,](#page-11-0) [35\]](#page-13-1) have led to the emergence of highly capable LLMs that can summarize a book [\[22\]](#page-11-1), write a compelling story [\[18\]](#page-11-2), code a library [\[53\]](#page-14-0), and generally reason over longer contexts than ever before [\[7\]](#page-10-2). As LLMs are increasingly tasked with processing longer sequences, the memory overhead associated with key-value (KV) caching has emerged as a critical bottleneck to scaling context length.

Prior work has approached the challenge of KV cache memory overhead through techniques such as quantization [\[30,](#page-12-0) [15,](#page-11-3) [48,](#page-14-1) [52\]](#page-14-2), low-rank approximation [\[47,](#page-14-3) [4,](#page-10-3) [37,](#page-13-2) [50,](#page-14-4) [26\]](#page-12-1), token-wise eviction [\[51,](#page-14-5) [29,](#page-12-2) [25,](#page-12-3) [8,](#page-10-4) [1,](#page-10-5) [11\]](#page-10-6), and structured pruning (e.g., channel-wise removal [\[44,](#page-13-3) [31\]](#page-12-4)). The need to improve individual compression techniques has become increasingly important, especially as joint applications of multiple methods, such as pruning combined with token eviction [\[44\]](#page-13-3), quantization with token-wise eviction [\[52\]](#page-14-2), and low-rank approximation with quantization [\[4\]](#page-10-3), gain popularity. However, previous work on KV cache pruning have been limited to structured pruning, primarily due to the difficulty of efficiently leveraging finer-grained (i.e., unstructured) sparsity during execution. Effective pruning of the KV cache entails two core challenges: (1) achieving substantial reduction in KV cache size while preserving model accuracy, and (2) ensuring that the runtime pruning and compression processes are sufficiently efficient (i.e., the associated overhead must not outweigh the latency gains introduced by the resulting sparsity).

In this paper, we find that removing any constraint on the sparsity pattern, effectively unstructured sparsity can ensure that compressed KV cache perform with minimal model accuracy degradation while being pruned to a higher sparsity. In Section 2 (green region of Figure 1), we first present our journey to find the optimal pruning algorithm for the key and value cache, based on the element magnitude distributions of the KV cache. We explore the feasibility of various pruning algorithms on both KV cache to conclude that applying a simple per-token magnitude-based pruning on both Key and Value caches is capable of preserving the model accuracy at a high sparsity, while also demonstrating strong compatibility with orthogonal compression techniques.

<span id="page-1-1"></span>![](_page_1_Figure_1.jpeg)

Figure 1: High-level overview of Mustafar. Green region describes the pruning algorithm of Section 2, pink region describes the custom sparse attention kernel of Section 3.

Section 3 (pink region of Figure 1) discusses the next step: having induced sparsity in the KV

cache, the challenge becomes leveraging the unstructured sparsity to reduce memory footprint and accelerate computation. To this end, we adopt a bitmap-based sparse format that serves two purposes. First, the bitmap enables maximal compression of matrices with arbitrary sparsity patterns. Second, this maximal compression of matrix operands translates into computational speedup of the attention operation, which is severely memory-bound on GPUs. Alongside the sparse format, we introduce the custom attention kernel tailored to operate on the bitmap-based sparse format. We see that the speedup of our attention kernel overshadows the latency introduced by runtime pruning and compression, meanwhile effectively compressing the KV cache to high sparsity with minimal accuracy degradation.

In summary, we demonstrate that adopting unstructured sparsity in the KV cache without imposing constraints on the pruning pattern enables higher degrees of sparsity while preserving model accuracy. Furthermore, we introduce the necessary computational tools to support unstructured sparsity efficiently, ensuring that the derived high sparsity leads to gains in memory compression and end-to-end inference throughput.

### <span id="page-1-0"></span>2 Pruning Algorithm for Unstructured Sparsity

**Question**: Does removing structural constraints in KV cache pruning allow for higher sparsity while preserving model accuracy more effectively than structured pruning methods?

We explore the potential unstructured sparsity on KV cache pruning by considering the two factors for Key and Value cache pruning: pruning direction and output-awareness. **Pruning Direction** refers to the axis along which sparsity is induced when selecting elements for removal. Since both the Key and Value caches are represented as matrices with dimensions [ $tokens \times channels$ ], we consider two primary pruning directions: per-channel pruning, which determines target sparsity across each channel (i.e., across tokens for each channel), and per-token pruning, which determines target sparsity across each token's cache (i.e., across model dimensions for each token). Output-Awareness refers to the use of a scoring metric that serves as a proxy for estimating each element's contribution to the operation's output. Commonly employed in LLM weight pruning [38] and structured KV cache pruning [44], this technique involves computing a score for each pruning unit such as a channel or an element by taking the product of the corresponding element with its associated input. This approach effectively captures the element's influence on the final output, guiding more informed pruning decisions. For a fair and effective comparison between pruning strategies, we uniformly employ a **local dense window**, where the recent 32 tokens remain untouched during the decode phase. Previous works [51, 44] have shown that this is effective in preserving model accuracy, meanwhile being small enough in size to not severely impact the compression.

#### 2.1 Pruning Key Cache

In deciding the pruning direction, we build on top of the observation of KIVI [30], that Key cache exhibits distinct channel-wise outliers, where "channel" refers to the head dimension (Figure 2a). This leads us to focus on per-token pruning for key cache, as it can effectively capture the elements in the outlier channel.

<span id="page-2-0"></span>![](_page_2_Figure_2.jpeg)

- (a) Magnitude distribution of Key cache
- (b) Magnitude distribution of Value cache

Figure 2: Visualization of the KV cache in LLaMA-27B. Color intensity indicates element magnitude. The figure was generated using the visualization code from KIVI [30].

Based on the same observation to perform structured pruning of individual channels, ThinK [44] incorporates output-awareness by using a per-channel score of the accumulation of last 32 query, multiplied by each channel. To this end we compare the accuracy of ThinK [44], per-token magnitude-based unstructured pruning, and output-aware unstructured pruning of our design.

<span id="page-2-1"></span>![](_page_2_Figure_7.jpeg)

Figure 3: Per-token, output-aware pruning of Key cache

Figure 3 elaborates the per-token output-aware unstructured pruning score of Key cache. The element-wise  $L_1$  accumulation of the current and next 31 Query vector (blue) is multiplied element-wise across each token's key vector (pink) to derive the pruning score (green). The absolute value of the score element in the corresponding position of each Key cache element is used to decide the elements to be pruned within a token's Key vector. In other words, we formulate the per-token output-aware unstructured pruning score S of a Key cache K to be:

$$S = |K| \odot broadcast \left( \sum_{t=T}^{T+31} |Q_t| \right), \quad \text{where } Q_t \text{ is the query at time } t$$

For Group Query Attention (GQA) [2], where multiple queries correspond to a KV cache pair, we sum the pruning score of all queries mapped to each KV cache.

<span id="page-2-2"></span>Table 1: Comparison of ThinK [44] structured pruning, per-token magnitude-based unstructured pruning, and per-token output-aware unstructured pruning on LongBench [3] with Llama-3-8B-Instruct Key cache.  $K_s$  denotes Key cache sparsity.

|               |       |              | $K_s = 0.5$  |              |              | $K_s = 0.7$  |              |
|---------------|-------|--------------|--------------|--------------|--------------|--------------|--------------|
| Task          | Dense | ThinK        | Unstructured | Unstructured | ThinK        | Unstructured | Unstructured |
|               |       | (Structured) | Output-aware | Magnitude    | (Structured) | Output-aware | Magnitude    |
| Average       | 43.19 | 38.53        | 43.23        | 42.84        | 26.55        | 42.13        | 41.55        |
| SingleDoc QA  | 36.66 | 35.61        | 36.57        | 36.90        | 25.26        | 35.78        | 35.53        |
| MultiDoc QA   | 36.09 | 34.99        | 35.92        | 35.77        | 29.75        | 35.55        | 35.40        |
| Summarization | 26.75 | 24.96        | 26.87        | 26.45        | 17.70        | 25.16        | 25.18        |
| Few-shot      | 68.96 | 66.54        | 68.82        | 68.75        | 44.88        | 67.22        | 67.84        |
| Synthetic     | 37.25 | 35.50        | 37.00        | 36.75        | 16.86        | 35.25        | 35.00        |
| Code          | 55.58 | 29.56        | 56.61        | 54.14        | 19.15        | 56.19        | 51.47        |

In Table 1, we compare Llama-3-8B-Instruct accuracy of different pruning methods on LongBench [3]. For structured pruning, we see that even at a moderate sparsity, model accuracy retention is dismal compared to pruning to an unstructured sparsity pattern. Notably, unstructured pruning is capable of outperforming structured pruning even without the memory footprint of pruning scores involved with output-awareness. Applying output-awareness to unstructured pruning results in a slight improvement in the LongBench total average score, while individual task performance is mixed with each method outperforming the other on different tasks.

**Key Cache Verdict:** While the existence of outlier channels with exceptionally high magnitudes show promise for per-channel structured pruning, unstructured sparsity achieves higher accuracy at greater sparsity levels, even without output-awareness.

### 2.2 Pruning Value Cache

As shown in Figure 2b, Value cache exhibits more uniform distribution of activations, making it challenging to apply the same channel-wise pruning without incurring substantial degradation in model accuracy. This difficulty has led recent Value cache pruning approaches to be more susceptible to accuracy degradation.

With no discernible outliers in certain direction, we explore all possible combinations of (pruning direction, magnitude/output-aware) pairs. However, we are able to rule out per-token outputaware pruning, as the attention formulation  $AttentionScore \times Value$  involves a multiplyand-accumulate operation along the token dimension. As seen in Figure 4, every element of a token's Value cache is multiplied by the same element of the attention score, with each element's impact on the output proportionate to the magnitude of each value. That is, for Value cache pruning, per-token magnitudebased pruning is already output-aware. For per-channel pruning, we prune each channel to the target sparsity in groups of 32 tokens, for compatibility with the local window size. For

<span id="page-3-0"></span>![](_page_3_Figure_5.jpeg)

Figure 4: Output-aware per-channel (red) and magnitude-based per-token (pink) pruning of Value cache. Magnitude-based per-token pruning is equal to output-aware per-token pruning (yellow).

per-channel output-aware pruning, we accumulate the current and subsequent 31 attention score  $\alpha$  of each token, which is then element-wise multiplied to the corresponding Value Cache (V) element. The following formula describes the pruning score S of per-channel output-aware pruning:

$$S = |V| \odot broadcast\left(\sum_{t=T}^{T+31} |\alpha_t|\right), \quad \text{where } \alpha_t \text{ is the attention score at time } t$$

<span id="page-3-1"></span>Table 2: Comparison of ThinK [44] structured pruning, per-channel magnitude-based unstructured pruning, per-channel output-aware unstructured pruning, and per-token magnitude-based pruning on LongBench [3] with Llama-3-8B-Instruct Value Cache.  $V_s$  denotes Value cache sparsity.

|               |       |              | $V_s =$       | : 0.5         |             |              | $V_s =$       | 0.7           |             |
|---------------|-------|--------------|---------------|---------------|-------------|--------------|---------------|---------------|-------------|
| Task          | Dense | ThinK        | Magnitude     | Output-aware  | Magnitude   | ThinK        | Magnitude     | Output-aware  | Magnitude   |
|               |       | (Structured) | (Per-channel) | (Per-channel) | (Per-token) | (Structured) | (Per-channel) | (Per-channel) | (Per-token) |
| Average       | 43.19 | 38.45        | 42.50         | 42.84         | 43.04       | 30.60        | 41.69         | 42.67         | 42.78       |
| SingleDoc QA  | 36.66 | 34.92        | 36.56         | 36.24         | 36.75       | 25.05        | 36.11         | 36.05         | 36.96       |
| MultiDoc QA   | 36.09 | 34.74        | 35.45         | 36.07         | 36.22       | 23.90        | 35.11         | 36.20         | 35.82       |
| Summarization | 26.75 | 23.31        | 24.74         | 25.79         | 26.34       | 20.41        | 22.72         | 24.75         | 25.19       |
| Few-shot      | 68.96 | 67.18        | 67.66         | 68.65         | 68.91       | 60.16        | 67.39         | 68.23         | 68.08       |
| Synthetic     | 37.25 | 35.43        | 38.31         | 37.00         | 36.25       | 29.63        | 38.75         | 37.25         | 35.50       |
| Code          | 55.58 | 31.97        | 55.07         | 55.57         | 55.77       | 20.85        | 52.65         | 56.17         | 57.62       |

As shown in the Table 2, we first see that applying structured pattern to Value cache pruning incurs significant accuracy degradation even in 50% sparsity. This is concurrent with ThinK [44] findings, which points to 30% sparsity as the upper-bound on acceptable accuracy. In contrast, per-token

magnitude pruning is capable of preserving model accuracy even at 70% sparsity. For per-channel pruning, we see that incorporating output-awareness boasts model accuracy retention almost to the level of per-token pruning. However, we prefer per-token magnitude-based pruning for the following two reasons. First, output-aware per-channel value cache pruning requires access to the attention score which requires additional recomputation when used alongside FlashAttention [6], where the full attention score matrix does not materialize in the global memory. Second, per-token magnitude-based pruning allows smooth compatibility with orthogonal compression method tokenwise eviction [24, 51], where the retained token's KV cache can be pruned individually. We examine the accuracy of joint application in Section 4.2.

**Value Cache Verdict**: All unstructured pruning methods explored outperform structured pruning. Among unstructured pruning methods, token-wise pruning, which is inherently output-aware by matrix multiplication formulation, best preserves model accuracy even at high sparsity levels. While channel-wise pruning with output-awareness can achieve comparable accuracy, token-wise pruning offers advantages in both efficiency and modularity.

With the two verdicts in Key and Value caches, on Table 3 we finally validate the model accuracy retention of per-token magnitude-based pruning with both Key and Value caches pruned. Not only can Value cache be pruned to high sparsity with unstructured sparsity, but both KV cache can be pruned to 70% sparsity while showing similar or better accuracy than Key-only 50% structured pruning of ThinK [44]. In Appendix A.1, methodology of this section is applied on Llama-2 7B to reinforce the effectiveness of per-token magnitude-based KV cache pruning.

<span id="page-4-1"></span>Table 3: Longbench Score of Llama-3-8B-Instruct and Mistral-7B-Instruct-v0.2 with KV Cache Per-Token Magnitude-based Pruning.

|               | Lla   | ma-3-8B-In  | struct      | Mistral-7B-Instruct-v0.2 |             |             |  |  |  |
|---------------|-------|-------------|-------------|--------------------------|-------------|-------------|--|--|--|
| Task          | Dense | $K_s = 0.5$ | $K_s = 0.7$ | Dense                    |             | $K_s = 0.7$ |  |  |  |
|               | Dense | $V_s = 0.5$ | $V_s = 0.7$ | Dense                    | $V_s = 0.5$ | $V_s = 0.7$ |  |  |  |
| Average       | 43.19 | 42.65       | 40.96       | 42.65                    | 42.30       | 40.95       |  |  |  |
| SingleDoc QA  | 36.66 | 36.67       | 35.28       | 36.21                    | 36.22       | 36.08       |  |  |  |
| MultiDoc QA   | 36.09 | 36.23       | 35.11       | 29.93                    | 30.42       | 29.40       |  |  |  |
| Summarization | 26.75 | 26.05       | 23.57       | 28.10                    | 27.77       | 26.72       |  |  |  |
| Few-shot      | 68.96 | 68.18       | 66.10       | 66.68                    | 66.70       | 66.24       |  |  |  |
| Synthetic     | 37.25 | 36.00       | 34.13       | 44.85                    | 41.92       | 36.13       |  |  |  |
| Code          | 55.58 | 54.50       | 53.49       | 54.98                    | 54.83       | 53.84       |  |  |  |

