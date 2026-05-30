# <span id="page-4-0"></span>2 Related Works

Generation Speed-up with Long Context Input. One effective technique to accelerate autoregressive generation is KV cache compression/eviction. During generation, LLMs store the previous key and value matrices to reduce computational complexity. However, when the input context is long (e.g., 128K tokens), the memory consumption and running time associated with the KV cache dominate iterative generation. Many studies have focused on KV cache eviction. For instance, [\[GZL](#page-14-5)+23] evict long-range contexts on attention heads to prioritize local contexts, using the KV cache only for heads that broadly attend to all tokens. Streaming LLM [\[XTC](#page-15-2)+23] introduces an attention sink that retains only the first few tokens and the latest k tokens in the KV cache to enable fast streaming generation. LOOK-M [\[WWL](#page-15-3)+24] applies KV eviction in the multimodality so that the model only needs to look once for the image. LongWriter [\[BZL](#page-13-6)+24] uses KV eviction to enable LLMs to generate coherent outputs exceeding 20,000 words. MInference 1.0 [\[JLZ](#page-14-6)+24] determines the optimal KV cache pattern for each attention head offline and dynamically builds sparse indices based on the assigned query during inference. QuickLLaMA [\[LSJ](#page-14-7)+24] classifies the KV cache to many subsets, e.g., query tokens, context tokens, global tokens, and local tokens, and only preserves some types of tokens in the KV cache. ThinK [\[XJD](#page-15-4)+24] proposes a query-dependent KV cache pruning method by pruning the least significant channel dimensions of the KV cache. H2O [\[ZSZ](#page-15-1)+23] retains only tokens contributing to cumulative attention. SnapKV [\[LHY](#page-14-4)+24] evicts non-essential KV positions for each attention head based on observation windows. While the aforementioned studies focus on eviction and compression of the KV cache during the prompt computation phase to optimize the iterative generation phase, they do not reduce the running time or GPU memory usage during the prompt computation phase. In contrast, our method, GemFilter, achieves both reduced running time and GPU memory usage in the prompt computation phase, as well as during the iterative generation phase. We provide a more detailed comparison in Section [3.4.](#page-8-0)

More related to our work, [LDLG23] compress input sequences by pruning redundancy in the context, making inputs more compact. However, they need to keep 50% of input tokens to keep the LLMs' performance, whereas GemFilter achieves comparable performance by only reserving 1% of input tokens. For further details, we refer the reader to Section 4.1.

### <span id="page-5-0"></span>3 Method

### <span id="page-5-1"></span>3.1 Notations and Preliminary

While the Transformer and self-attention architecture [VSP<sup>+</sup>17] have already become overwhelmingly popular, we first introduce certain preliminary definitions to provide a better methodological connection to our proposed GemFilter method in Section 3.2.

For any positive integer n, we use [n] to denote the set  $\{1, 2, \cdots, n\}$ . We use  $\circ$  to denote function composition and  $\odot$  to denote the Hardamard product. Let n be the input token/prompt length, d the hidden feature dimension, and  $\mathcal V$  the vocabulary set. We now introduce the key concept of attention and transformers. We first define the query, key, and value matrices. It is important to note that during text generation, the key and value matrices are also referred to as the KV cache, as they are stored in GPU memory to reduce running time during the iterative prediction of the next token.

**Definition 3.1** (Single layer self-attention). Let  $Q \in \mathbb{R}^{n \times d}$  be the query matrix,  $K \in \mathbb{R}^{n \times d}$  the key cache, and  $V \in \mathbb{R}^{n \times d}$  the value cache. Let  $M_c \in \{0,1\}^{n \times n}$  be the causal attention mask, where  $(M_c)_{i,j}$  is 1 if  $i \geq j$  and 0 otherwise. The self-attention function Attn is defined as:

$$\mathsf{Attn}(Q,K,V) = M_c \odot \mathsf{Softmax}(QK^\top/\sqrt{d}) \cdot V$$

<span id="page-5-3"></span>**Definition 3.2** (Multi-layer transformer). Let  $T \in \mathcal{V}^n$  represent the input tokens, and let m denote the number of transformer layers. Let  $g_i$  represent components in the i-th transformer layer other than self-attention, such as layer normalization, residual connections, and the MLP block, where  $g_i : \mathbb{R}^{n \times d} \to \mathbb{R}^{n \times d}$  for any  $i \in \{0, 1, ..., m\}$ . Let  $\mathsf{Attn}_i$  denote the self-attention module in the i-th transformer layer. We define an m-layer transformer  $\mathsf{F}_{1:m} : \mathcal{V}^n \to \mathbb{R}^{n \times d}$  as

$$\mathsf{F}_{1:m}(T) := g_m \circ \mathsf{Attn}_m \circ g_{m-1} \circ \cdots \circ g_1 \circ \mathsf{Attn}_1 \circ g_0 \circ \mathcal{E}(T) \quad \in \mathbb{R}^{n \times d},$$

where  $\mathcal{E}$  is the input embedding function mapping the input tokens to hidden features using the vocabulary dictionary, i.e.,  $\mathcal{E}(T) \in \mathbb{R}^{n \times d}$ .

Note that the above definitions use a single attention head for simplicity, but in practice, multihead attention is used [VSP<sup>+</sup>17].

#### <span id="page-5-2"></span>3.2 Our Algorithm: GemFilter

We present our method, GemFilter, in Algorithm 1. We also present PyTorch code in Appendix C.1 for the reader's interests. The high-level idea is to run the LLM twice. In the first pass, we run only the early layers of the LLM to select the key input tokens. This corresponds to the prompt computation phase (Line 4-7 of Algorithm 1). This process selects the top k tokens that receive the most attention from the last query token. In the second pass, we feed the selected tokens to the full LLM and run the generation function, corresponding to the iterative generation phase (Line 8). Below, we explain Algorithm 1 step by step.

#### <span id="page-6-1"></span>Algorithm 1 GemFilter: Generation with Token Selection Based on Early Layers

```
1: procedure SELECTIONGEN(\mathsf{F}_{1:m}, T \in [\mathcal{V}]^n, r \in [m], k \in [n])
2: \triangleright \mathsf{F}_{1:m} : \mathsf{An} \ m-layer transformer network; T: input sequence of tokens
3: \triangleright r: filter layer index for token selection; k: number of selected tokens
4: \mathsf{Get} \ Q^{(r)}, K^{(r)} \ \mathsf{by} \ \mathsf{doing} \ \mathsf{a} \ r-layer forward pass: \mathsf{F}_{1:r}(T)
5: \triangleright Q^{(r)}, K^{(r)} \in \mathbb{R}^{n \times d}: the r-th layer query, key
6: J \leftarrow \mathsf{topk\_index}(Q_n^{(r)} K^{(r)}, k) \quad \triangleright Q_n^{(r)}: the last row of Q^{(r)}; Q_n^{(r)} K^{(r)} \in \mathbb{R}^n are attn scores
7: Sort the indices in J \quad \triangleright J \subseteq [n] \ \mathsf{and} \ |J| = k
8: \mathsf{return} \ \mathsf{Gen}(\mathsf{F}_{1:m}, T_J) \quad \triangleright \ \mathsf{Gen} \ \mathsf{is} \ \mathsf{generation} \ \mathsf{function}, T_J \in [\mathcal{V}]^k \ \mathsf{is} \ \mathsf{a} \ \mathsf{sub\text{-sequence}} \ \mathsf{of} \ T \ \mathsf{on} \ J
9: \mathsf{end} \ \mathsf{procedure}
```

The input of the algorithm is an m-layer transformer  $F_1$  (Definition 3.2), an input token sequence  $T \in \mathcal{V}^n$ , and two hyperparameters  $r \leq m, k \leq n$ , where r represents the index of the filter layer for context token selection and k denotes the number of tokens to select. For example, in the case of LLaMA 3.1 8B Instruct (Figure 1), we have m = 32, r = 13, and k = 1024.

In the first step (Line 4), we run only the first r layers forward to serve as a filter, obtaining the r-th layer's query and key matrices,  $Q^{(r)}$  and  $K^{(r)}$ . Note that we do not need to run all layers of the LLM on a long context input, thereby saving both computation time and memory (see detailed analysis in Section 3.3). In Line 6, we select token indices based on the r-th layer attention matrix. The selection is made by identifying the k largest values from the last row of the attention matrix, i.e., the inner product between the last query token  $Q_n^{(r)}$  and all key tokens  $K^{(r)}$ . For multi-head attention, the top-k indices are selected based on the summation of the last row across the attention matrices of all heads. For instance, suppose we have h attention heads, and let  $Q^{(r,j)}, K^{(r,j)} \in \mathbb{R}^{n \times d}$  represent the query and key matrices for the r-th layer and j-th attention head. Then, we compute  $J \leftarrow \text{topk\_index}(\sum_{j=1}^h Q_n^{(r,j)} K^{(r,j)^\top}, k)$ , where J is a set of top k index selection. Note that our method uses a single index set J, whereas SnapKV [LHY+24] and H2O [ZSZ+23] use different index sets for each layer and attention head, resulting in  $m \cdot h$  index sets in total. A detailed discussion is provided in Section 3.4.

In Line 6, J is sorted by inner product values. However, we need to re-sort J so that the selected tokens follow their original input order, ensuring, for example, that the  $\langle bos \rangle$  token is placed at the beginning. Line 7 performs this reordering operation. Finally, in Line 8, we can run any language generation function using the selected tokens  $T_J$ , which is a sub-sequence of T on the index set J, across all layers. This generation is efficient as the input context length is reduced from n to k, e.g., from 128K to 1024 tokens in Figure 1. Below, we provide a formal time complexity analysis.

#### <span id="page-6-0"></span>3.3 Running Time and Memory Complexity Analysis

The results of our analysis on time complexity and GPU memory consumption are presented in Theorem 3.3 below, with the proof deferred to Appendix B.

<span id="page-6-2"></span>**Theorem 3.3** (Complexity analysis). Let n be the input sequence (prompt) length and d the hidden feature dimensions. In our Algorithm 1, GemFilter uses the r-th layer as a filter to select k input tokens. Let SnapKV and H2O also use k as their cache size. Assume the LLM has m attention layers, each with h attention heads, and each transformer layer's parameters consume w GPU memory. Assuming that we generate t tokens with the GEN function and  $n \ge \max\{d, k, t\}$ , the following table summarizes the complexity for standard attention, SnapKV and H2O, and GemFilter:

| Co          | mplexity                      | Standard attention                           | SnapKV and H2O                          | GemFilter                                     |
|-------------|-------------------------------|----------------------------------------------|-----------------------------------------|-----------------------------------------------|
| Time        | Prompt Comp. Iter. generation | $\frac{\Theta(mhn^2d)}{\Theta(mh(nt+t^2)d)}$ | $\Theta(mhn^2d) \\ \Theta(mh(kt+t^2)d)$ | $\frac{\Theta(rhn^2d)}{\Theta(mh(k^2+t^2)d)}$ |
| $GPU\ mem.$ | Prompt Comp. Iter. generation | mw + 2mhnd $mw + 2mh(n+t)d$                  |                                         |                                               |

Recall that there are two phases in text generation. The first phase is prompt computation, which involves attention computation on the long context input tokens and generating the KV cache. The second phase is iterative generation, where auto-regressive generation occurs based on the pre-computed KV cache. Theorem 3.3 demonstrates that GemFilter is faster and consumes less GPU memory than SnapKV/H2O and standard attention during the prompt computation phase. Additionally, during the iterative generation phase, GemFilter has the same running time and GPU memory consumption as SnapKV/H2O, which is significantly better than standard attention. This conclusion aligns with our experimental results in Section 4.4.

Case Study. Let us consider the case  $n \gg k \approx t$ , e.g., n = 128K, k = t = 1024 and r < m. During the prompt computation phase, we have the running time:

```
Standard attention: SnapKV/H2O: GemFilter = \Theta(m:m:r),
```

and the GPU memory consumption:

```
Standard attention: SnapKV/H2O: GemFilter \approx mw + mhnd: mw + hnd: rw + hnd,
```

We see that GemFilter has a lower time complexity and less GPU memory consumption than standard attention, SnapKV, and H2O. During the iterative generation phase, we have the running time:

```
Standard attention: SnapKV/H2O: GemFilter = \Theta(n:k:k),
```

and the GPU memory consumption:

```
Standard attention: SnapKV/H2O: GemFilter \approx w/hd + 2n : w/hd + 4k : w/hd + 4k,
```

As such, GemFilter has the same time complexity and GPU memory consumption as SnapKV/H2O, while significantly outperforming the standard attention.

The running time bottleneck for all methods occurs during prompt computation, which takes  $\Theta(mhn^2d)$  for standard attention, SnapKV, and H2O. In contrast, GemFilter only requires  $\Theta(rhn^2d)$  for prompt computation, as it only processes the early layers of the LLMs to select and compress the input tokens during the first run. See detailed proof in Appendix B.

Note that the GPU memory bottleneck for standard attention occurs during iterative generation, while for other methods, the memory bottleneck arises during prompt computation due to the reduced KV cache. GemFilter consumes less GPU memory than SnapKV and H2O because it only requires loading some layer model weights when processing the long context input in its first run. Our empirical results in Section 4.4 support our complexity analysis findings.

### <span id="page-8-0"></span>3.4 Comparison with Other Methods

GemFilter reduces both running time and GPU memory usage in both the prompt computation and iterative generation phases, whereas SnapKV [\[LHY](#page-14-4)+24] and H2O [\[ZSZ](#page-15-1)+23] focus only on the iterative generation phase. During the prompt computation phase, standard attention computes and stores the entire KV cache for all layers in GPU memory, which is used during the generation phase. SnapKV and H2O, on the other hand, compute the entire KV cache for all layers but only store a portion of it in GPU memory (e.g., k = 1024). They use the selected KV cache for memory-efficient generation. SnapKV selects important clustered positions of the KV cache from an 'observation' window located at the end of the prompt, while H2O greedily drops tokens based on cumulative attention scores to retain only a small portion of the KV cache. In contrast, GemFilter avoids computing the KV cache for all layers during the prompt computation phase.

Compared to SnapKV and H2O, there are two additional differences. First, SnapKV and H2O maintain separate index sets for each layer and attention head, resulting in m·h index sets in total. This leads to different behaviors across attention heads, making their intermediate mechanisms more difficult to interpret. On the other hand, GemFilter uses a single index set, J, allowing for easier interpretability by enabling the printing of the selected sequence for human review before the second run (see a real example in Figure [1\)](#page-3-0). Another distinction lies in how positional embeddings are handled. In SnapKV and H2O, the maximum positional embedding distance is n + t, as the same positional embedding is used in both the prompt computation and iterative generation phases. However, in GemFilter's second run, the maximum positional embedding distance is reduced to k+t because the input token length is reduced from n to k, and the RoPE function[1](#page-8-3) is re-computed. This reduction makes GemFilter more efficient, as the model can better handle shorter input sequences, as demonstrated in Figure [4](#page-9-0) (a).

