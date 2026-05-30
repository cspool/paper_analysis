# 1 Introduction

Large Language Models (LLMs) have recently demonstrated remarkable success across diverse text processing tasks, including document retrieval [\(Laban et al.,](#page-9-0) [2023\)](#page-9-0), code generation [\(Gu,](#page-8-0) [2023\)](#page-8-0), and mathematical reasoning (like R1 model [\(DeepSeek-AI et al.,](#page-8-1) [2025\)](#page-8-1)). The Scaling law [\(Kaplan et al.,](#page-8-2) [2020\)](#page-8-2) suggests that larger models generally achieve superior performance. The R1-like models further indicates that longer generation sequences with additional 'thinking tokens' can enhance reasoning capabilities. However, these improvements comes at a significant cost: the growing KV cache size poses a major challenge for efficient LLM inference. Many efforts try to mitigate this challenge.

Most of LLMs are totally relying on Self-Attention mechanism [\(Vaswani et al.,](#page-9-1) [2023\)](#page-9-1) to determine which historical tokens are important in the next token prediction. Therefore, many KV compression approaches are based on it to drop unimportant ones [\(Zhang et al.,](#page-9-2) [2024;](#page-9-2) [Liu et al.,](#page-9-3) [2024b;](#page-9-3) [Li et al.,](#page-9-4) [2024;](#page-9-4) [Tang et al.,](#page-9-5) [2024b;](#page-9-5) [NVIDIA,](#page-9-6) [2024\)](#page-9-6). This kind of algorithms keeps a remarkable performance even when the compression ratio is high. However, most of these importance-based token-dropping approaches depend on the ending query question (Instruction Dependence) to achieve such a performance [\(Li et al.,](#page-9-7) [2025;](#page-9-7) [Feng et al.,](#page-8-3) [2024;](#page-8-3) [Tang et al.,](#page-9-8) [2024a\)](#page-9-8).

Another prominent direction in KV cache optimization involves quantization techniques [\(Yang](#page-9-9) [et al.,](#page-9-9) [2024;](#page-9-9) [Liu et al.,](#page-9-10) [2024c\)](#page-9-10), which aim to compress the memory footprint of KV states by representing them with reduced precision. These methods achieve significant memory savings—often by 4× or more—while preserving model performance through careful error mitigation strategies. Beyond memory efficiency, quantization also reduces the bandwidth overhead of transferring KV cache across devices in distributed inference scenarios, accelerating multi-GPU or memory-bound workloads. However, a critical limitation of pure quantization approaches is that they retain all historical tokens, leaving the computational cost of attention unchanged. For long-context tasks, this means the quadratic complexity of attention persists despite the reduced memory usage.

The simple but with limited performance methods are usually based on the sliding window tokens eviction. Sliding window-based eviction methods—such as those used in Infinite-LLM [\(Han](#page-8-4) [et al.,](#page-8-4) [2024\)](#page-8-4) and StreamingLLM [\(Xiao et al.,](#page-9-11) [2023\)](#page-9-11)—retain only the initial cache tokens and those within a fixed sliding window, discarding the

rest. However, this indiscriminate eviction strategy often leads to a notable degradation in generation quality.

Recent work by (Liu et al., 2024a,c) addresses the statistical properties of KV states, revealing distinct distribution patterns for keys and values. Their findings suggest that per-channel quantization for keys (which exhibit consistent variance across feature dimensions) and per-token quantization for values (which vary more significantly across sequence positions) yield better fidelity. This observation motivates our key insight: token importance for eviction—traditionally derived from attention weights—can instead be inferred from token- and channel-wise distribution patterns in the KV space. By leveraging these structural properties, we can design a pruning criterion, LagKV, that is both hardware-friendly (compatible with Flash Attention (Dao, 2023)) and instruction independent, enabling compute savings alongside memory reduction.

#### 2 Methodology

In this section, we formally introduce our KV compression method, LagKV. We begin by looking at the autoregressive process of the LLMs. Inspired by this, we propose a simple yet effective strategy to use the subsequent tokens to compress the previous ones.

## 2.1 Preliminaries

LLMs' next token prediction relies on the previous tokens. First, in the prefill stage, the model uses its tokenizer to convert the words to n indices of the embedding metrics  $E \in \mathbb{R}^{V \times d}$  of the model and collects the representations to form a input matrix,  $X \in \mathbb{R}^{n \times d}$ . This matrix is the initial tokens of the first layer of LLM and then each layer will output a same shape matrix as next layer's input. To depict the operations in each layer, we follow the notation system from (Liu et al., 2023) with h attention heads. For each head  $i \in [1,h]$  and head dimension  $d_h$ , we focus on the Query, Key, and Value states, which are converted from tokens by three linear transformation matrices  $W_i^Q$ ,  $W_i^K$ ,  $W_i^V \in \mathbb{R}^{d \times d_h}$  separately:

$$Q_i = XW_i^Q, K_i = XW_i^K, V_i = XW_i^V \tag{1}$$

The output  $Y \in \mathbb{R}^{n \times d}$  is computed using the attention weights  $A_i \in \mathbb{R}^{n \times n}$  and the final output matrix  $W^O \in \mathbb{R}^{d \times d}$ :

$$Y = Concat_{i \in [1,h]}(A_i V_i) W^O$$
 (2)

where

$$A_i = \operatorname{softmax}(\frac{Q_i K_i^T}{\sqrt{d_h}}). \tag{3}$$

When the new tokens are generated subsequently in the autoregressive inference, which named as decode stage, the embedding of generated token  $\boldsymbol{x}$  is mapped to its respective Query, Key, and Value states for each head, and the previous KV cache is updated accordingly:

$$q_i = xW_i^Q, k_i = xW_i^K, v_i = xW_i^V$$
 (4)

$$K_i = Cat[K_i:k_i], V_i = Cat[V_i:v_i]$$
 (5)

$$A_i = \operatorname{softmax}(\frac{q_i K_i^T}{\sqrt{d_h}}) \tag{6}$$

Since  $q_i \in \mathbb{R}^{1 \times d_h}$ , the computation will be much faster because of the KV cache.

#### 2.2 LagKV

Since the intrinsic property of autoregressive model, the next token representation will not change abruptly from the previous one. As observed in (Liu et al., 2024a), the called token-wise locality will show that the tokens in closer proximity have more similar K/V tensor values compared to tokens that are further apart.

And also, the StreamingLLM method (Xiao et al., 2023) has demonstrated that the head portion and sliding window of the KV cache are crucial. This suggests that cache compression should use subsequent tokens to assess whether prior tokens remain in the cache, rather than relying on the competition between them—as done in many attention-weight-based methods.

Inspired by above insights, we proposed our LagKV method as:

- After the prefill is done, start to apply the compression dynamically.
- Always keep the attention sink with size S and the already compressed part if had unchanged.
- Skip the compression if the length of the rest KV after the static part is less than 2L, where we denote the lag size as L.
- Partition the rest KV with L. If it's not divisible by L, the modulo of it will be added to the sliding window.

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Figure 1: LagKV recursively compression process: partition the KV cache and use the next joint chunk as reference to compress the current one. Keep the rest of them as the sliding window.

• Recursively compute the KV cache score. Use the next partition as a reference, calculate token-wise max and min from the reference then use max-min to normalize the Key and Value states respectively. After the KV are normalized, calculate the channel-wise standard deviation then softmax. The equations are formally like:

<span id="page-2-2"></span>
$$min_i^{p,Z} = min_{seq}(Z_i^{p+1}) \tag{7}$$

<span id="page-2-3"></span>
$$max_i^{p,Z} = max_{seq}(Z_i^{p+1}) \tag{8}$$

$$\bar{Z_i^p} = \frac{Z_i^p - min_i^{p,Z}}{max_i^{p,Z} - min_i^{p,Z}}$$
(9)

$$score(Z_i) = Softmax(Std.(\bar{Z}_i))$$
 (10)

where Z is one of  $\{K,V\}$ , p denotes the partition index, i represents the head index and seq for the sequence axis. Since the last partition has no reference can be used, our method will naturally have a sliding window with at least size L.

• Sum the scores of Key and Value to get the final score of each token:

<span id="page-2-4"></span>
$$score_i = score(K_i) + score(V_i)$$
 (11)

 Base on the score<sub>i</sub>, use the top-K strategy to select tokens in each partition and each head and add them to the compressed part.

The max-min normalization is applied along the sequence dimension, meaning each channel is normalized using statistics from lag-L tokens. Due to token-wise locality, the channel-specific norms of  $K_i$  and  $V_i$  are largely eliminated. The resulting normalized representations,  $\bar{K}_i$  and  $\bar{V}_i$ , retain the original channel-wise variance, allowing the

standard deviation to serve as a measure of token importance. The softmax operation then identifies and separates outliers, while the summed scores  $score(K_i)$  and  $score(V_i)$  determine their relative contributions.

As showed in Fig. 1, our method is recursively compressing KV cache in both prefill and decode parts, which is essential for the token-wise locality as mentioned above. It requires relative short distance to keep the similarity among the KV states. Subsequently, another benefit, it also avoids the bias from the long context with length much larger than L and the case when the question is at the end of the prompt.

We do not compare the LagKV score to the attention weights here. The attention weights vary on different incoming queries. But our scoring method does not depend on the query states or the tokens after the next joint chunk. It mainly finds the tokens that are not coherent to the next chunk and keep them in the cache. As in KIVI (Liu et al., 2024c) quantization method, we need a rightful mean to find the correct variance and then prune the small ones. However, we use this strategy to evict tokens instead of quantizing them.

To calculate the compression ratio, we set the token retention ratio as r in each partition. In the partition chunk, only rL tokens will be kept and others are evicted. Therefore, the compression ratio C for the token sequence length  $L_s \geq S + 2L$  can be expressed as:

$$L_R = S + rL(\lfloor \frac{L_s - S}{L} \rfloor - 1) + L + Mod(L_s - S, L)$$
(12)

<span id="page-2-1"></span>
$$C = 1 - \frac{L_R}{L_s} \tag{13}$$

Where  $L_R$  is the length of the KV cache after compression. For the case  $L_s < S + 2L$ , the compression ratio is zero.

#### 3 Comparisons

#### 3.1 Base Models

We employ two open-source base models: Llama-3.1-8B-Instruct [\(Grattafiori et al.,](#page-8-6) [2024\)](#page-8-6) and Qwen2.5-7B-Instruct [\(Qwen et al.,](#page-9-14) [2025\)](#page-9-14). These models are main stream LLMs with moderate size and both leverage the GQA [\(Ainslie et al.,](#page-8-7) [2023\)](#page-8-7) technique to reduce the KV cache size.

### 3.2 Results of RULER

We use the RULER [\(Hsieh et al.,](#page-8-8) [2024\)](#page-8-8) benchmark to compare our approach to SnapKV [\(Li et al.,](#page-9-4) [2024\)](#page-9-4) and StreamingLLM [\(Xiao et al.,](#page-9-11) [2023\)](#page-9-11) in various compression ratios. To fairly compare different methods, we integrate our approach into the framework KVPress [\(NVIDIA,](#page-9-6) [2024\)](#page-9-6) and adapt their versions of other approaches. This framework applies compression without question to avoid the query-aware bias. In this task, we set the lag size to be L = 128 for LagKV and the retention ratio of each recursive window will be adaptively changed by Eq. [13](#page-2-1) for different compression ratios.

The results are present in Table [1](#page-4-0) and [2](#page-4-1) with best scores of each compression ratio shown in bold. The average scores of RULER tasks show that LagKV outperforms SnapKV and StreamingLLM across all compression ratios.

