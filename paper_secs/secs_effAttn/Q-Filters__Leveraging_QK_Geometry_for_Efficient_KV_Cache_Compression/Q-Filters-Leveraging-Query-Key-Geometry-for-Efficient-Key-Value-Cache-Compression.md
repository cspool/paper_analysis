# Q-Filters: Leveraging Query-Key Geometry for Efficient Key-Value Cache Compression

Nathan Godey <sup>12</sup> Alessio Devoto <sup>3</sup> Yu Zhao <sup>4</sup> Simone Scardapane <sup>3</sup>
Pasquale Minervini <sup>45</sup> Éric de la Clergerie <sup>2</sup> Benoît Sagot <sup>2</sup>

Q github.com/NathanGodey/qfilters

#### **Abstract**

Autoregressive language models rely on a Key-Value (KV) Cache, which avoids re-computing past hidden states during generation, making it faster. As model sizes and context lengths grow, the KV Cache becomes a significant memory bottleneck, which calls for compression methods that limit its size during generation. In this paper, we discover surprising properties of Query (Q) and Key (K) vectors that allow us to efficiently approximate attention scores without computing the attention maps. We propose Q-Filters, a trainingfree KV Cache compression method that filters out less crucial Key-Value pairs based on a single context-agnostic projection. Contrarily to many alternatives, Q-Filters is compatible with FlashAttention, as it does not require direct access to attention weights. Experimental results in long-context settings demonstrate that Q-Filters is competitive with attention-based compression methods such as SnapKV in retrieval tasks while consistently outperforming efficient compression schemes such as Streaming-LLM in generation setups. Notably, O-Filters achieves a 99% accuracy in the needle-in-a-haystack task with a  $\times 32$ compression level while reducing the generation perplexity drop by up to 65% in text generation compared to Streaming-LLM.

### 1. Introduction

The performance of Large Language Models (LLMs) as autoregressive text-generation systems relies on the effectiveness of the Transformer architecture (Vaswani et al., 2017). Recently, long-context models such as Gemini-Pro-1.5 (Reid et al., 2024), Claude-3 (Anthropic, 2024), GPT-4 (Achiam et al., 2023), and Llama3.1 (Dubey et al., 2024)

![](_page_0_Figure_8.jpeg)

Figure 1: Accuracy vs Time to First Token (TTFT) tradeoff for Llama-3.1-70B-Instruct, measured on the Ruler dataset with  $\times 8$  compression. The TTFT is measured using 2 A100 GPUs on 8192-tokens sequences.

have demonstrated the ability to process hundreds of thousands of tokens. However, processing such long sequences comes with significant challenges, as it may lead to higher decoding latency and memory saturation. As the context length grows, each inference step involves storing an increasingly large context from GPU memory in the form of the KV Cache, creating a memory bottleneck that hinders efficient inference (Fu, 2024). To address this issue, KV Cache compression methods aim to reduce the size of this past-context representations storage by removing or merging Key-Value pairs, thereby alleviating memory bottlenecks. While KV Cache compression techniques have gained popularity, many approaches require fine-tuning or re-training the underlying models (Nawrot et al., 2024; Ainslie et al., 2023; DeepSeek-AI et al., 2024), which limits their applicability in real-world deployment scenarios. Training-free methods have also been proposed, but they often rely on access to attention weights to evaluate the importance of Key-Value pairs (Xiao et al., 2024; Li et al., 2024), making them incompatible with the widely adopted efficient attention algorithm FlashAttention (Dao, 2024). These methods usually require a partial re-computation of the attention matrices, which

<sup>&</sup>lt;sup>1</sup>Sorbonne Université, Paris, France <sup>2</sup>Inria, Paris, France <sup>3</sup>Sapienza University of Rome <sup>4</sup>University of Edinburgh <sup>5</sup>Miniml.AI. Correspondence to: Nathan Godey <nathan.godey@inria.fr>.

<span id="page-1-0"></span>![](_page_1_Figure_1.jpeg)

Figure 2: Left and center: distributions of the projections of  $Q^h$  and  $K^h$  on  $u^h$  for Llama-3.1-8B. Right: estimates of  $\left|\mathbb{E}_i(\langle Q_i^h, v_m\rangle)\right|$  where  $v_m$  are the right vectors from the SVD of a set of  $Q^h$  representations from different Llama models, averaged over all layers and heads.

leads to a time and memory overhead. Hence, these algorithms are often used to compress prompts before generating answers and are not ideally suited for memory-constrained generation.

In this work, we propose Q-Filters, a training-free KV Cache compression method that uses the geometric properties of **Q**uery-Key to **filter** out the less important Key-Value pairs. Our approach achieves competitive results across synthetic tasks and pure generation cases while maintaining compatibility with FlashAttention and, thus, better time efficiency.

Analysing the properties of queries (Q) and Keys (K) distributions, we find that a single direction, spanned by the principal eigenvector of Q, encodes an input selection process for each head. Identifying this direction allows us to efficiently estimate which inputs are mostly ignored by a given head and can thus be discarded with minimal performance loss. Interestingly, we find that this direction is context-agnostic, i.e., the directions we identify in different contexts are highly consistent. Leveraging this property, we calculate lightweight projections, which we refer to as Q-Filters, based on a small held-out calibration dataset only once for every model, incurring minimal computational overhead. At inference time, we use Q-Filters to project Keys in the pre-computed direction to estimate the importance of Key-Value pairs without accessing attention scores, and we prune the KV Cache accordingly. This makes our method faster than most KV Cache compression alternatives that use attention scores to estimate the importance of the KV pairs.

Additionally, our method is training-free, requiring only a very short initial calibration, and we show it can be easily applied to a variety of decoder-only language models. We validate our method on a wide set of tasks, ranging from language modelling to in-context learning and long-context tasks, achieving competitive performance even with 32x compression ratios.

## 2. Background

#### 2.1. Key-Value Cache

We first introduce the relevant notation for our analysis and the role of the KV Cache in efficient LLM inference. Consider a transformer model with a hidden dimension  $d_m$  and  $n_l$  layers, processing a sequence of length L. Each transformer layer processes the input sequence via Multi-Head Self-Attention (MHA).

In MHA, the model transforms the input features  $X \in \mathbb{R}^{L \times d_m}$  into three distinct representations for each attention head  $h \in [1, H]$ . These representations, known as queries  $Q^h$ , Keys  $K^h$ , and Values  $V^h$ , each belong to  $\mathbb{R}^{L \times d_h}$ , where  $d_H = d_m/H$  represents the dimension per head, and h denotes the head index. The second step computes the attention output  $O^h$  for each head using the following equation:

$$O^h = \operatorname{softmax} \left( \frac{Q^h (K^h)^T}{\sqrt{d_H}} \right) V^h.$$

In causal modelling, where the model generates text sequentially, we ensure that each token only attends to previous tokens and itself. This causality constraint means that when generating the t-th token, its output  $O^h_t$  depends only on the current and previous inputs, as expressed by:

$$O_t^h = \operatorname{softmax} \left( \frac{Q_t^h(K_{\leq t}^h)^T}{\sqrt{d_H}} \right) V_{\leq t}^h.$$

The Key and Value representations  $K^h_{\leq t}$ ,  $V^h_{\leq t}$ , which combine previous Keys and Values with the current ones  $K^h_t$ ,  $V^h_t$ , reuse information from previous generation steps. By storing these representations in a KV Cache during the generation process, we can avoid the computational cost of recalculating them at each step, thereby significantly improving efficiency at the cost of the memory occupied by stored KV pairs.

<span id="page-2-1"></span>![](_page_2_Figure_1.jpeg)

Figure 3: Projection of  $Q^h$  and  $K^h$  vectors in the first two components of the SVD of  $Q^h$  for different heads in Llama-3.2-1B. The colour on the K projections represents the log-average attention at the corresponding index for the current head. The x-axis and y-axis indicate the results of a projection of the representations on  $v_1$  and  $v_2$ , respectively.

However, this memory-compute tradeoff introduces a new challenge: as the context length grows, decoding latency increases due to the frequent transfers of large KV Cache states between high-bandwidth memory (HBM) and streaming multiprocessors (SM) (Fu, 2024). For this reason, KV Cache compression methods have become essential to allow inference in long contexts.

#### 2.2. Geometry of Multi-Head Attention

In Devoto et al. (2024), the authors examined a relationship between basic characteristics of the Key representations and attention score distributions. Notably, they observe a negative correlation between the average attention weight given to a position and the  $L_2$ -norm of the  $K_t^h$  vector at that position. Leveraging this observation, they propose to compress the KV Cache by selecting the KV pairs for which  $||K_t^h||_2$  is the smallest. Using this simple heuristic, they are able to reach  $\times 2$  compression ratios without altering

the retrieval and modelling performance of the models they study. In their paper, while they relate this approach to the well-known oulier dimension phenomenon (Kovaleva et al., 2021), they do not provide a grounded explanation as to the strength of the observed correlation.

A promising path towards a better explanation of the  $L_2$ -norm observation consists in systematically exploring the geometry of the representations involved in the attention score computation, namely  $Q^h$  and  $K^h$ .

Godey et al. (2024) show that the distributions of  $Q^h_t$  and  $K^h_t$  are anisotropic, i.e. they do not uniformly occupy  $\mathbb{R}^{d_H}$ . They observe that both distributions "drift away" from the origin as training progresses. Crucially, this drift occurs along parallel directions in  $\mathbb{R}^{d_H}$ , so that the dot product between mean  $Q^h_t$  and mean  $K^h_t$  representations tends to increase in absolute value, and to be either positive or negative for different heads. In the paper, it is argued that this drift could be linked to the sparsity of attention patterns, but the authors do not propose a clear interpretation of this phenomenon from the perspective of the attention mechanism.

In this paper, we bridge the gap between the two aforementioned observations; namely, we explain the effectiveness of the  $L_2$ -norm heuristic introduced in Devoto et al. (2024) by leveraging the (jointly) anisotropic nature of Query-Key representations, and we explore a stronger heuristic that exploits this finding to refine the  $L_2$ -norm approximation by projecting Keys onto the drift directions, that we refer to as Q-Filters.

### <span id="page-2-2"></span>3. Method

## 3.1. Exploring the Query-Key Geometry

Motivated by Devoto et al. (2024) and Godey et al. (2024), we propose to further explore some geometrical properties of  $Q^h$  and  $K^h$  vectors and their implications for unnormalized attention logits  $Q^h(K^h)^T$ .

First, we formalize the findings from Godey et al. (2024) into our theoretical framework. The authors shed light on the existence of a favored common normalized direction for both  $Q^h$  and  $K^h$  distributions. We denote such direction as  $u^h \in \mathbb{S}^{d_H-1}$  where  $\mathbb{S}^{d_H-1}$  is the  $d_H$ -dimensional hypersphere (i.e.  $\mathbb{S}^{d_H-1} = \{x \in \mathbb{R}^{d_H} \text{ s.t. } ||x||_2 = 1\}$ ). As a consequence, the projection of  $Q^h$  and  $K^h$  distributions on  $u^h$  is usually non-null but can take opposite signs in  $Q^h$  and  $K^h$ . Hence, we use  $\epsilon = \pm 1$  to account for the possible sign discrepancy and formulate the following Observation 3.1 in terms of expectation.

<span id="page-2-0"></span>**Observation 3.1** (Joint anisotropy). There exist  $u^h \in \mathbb{S}^{d_H-1}$  and  $\epsilon = \pm 1$  such that

$$\mathbb{E}\left(\langle Q_i^h, u^h\rangle\right) > 0 \quad \text{and} \quad \mathbb{E}\left(\langle K_j^h, \epsilon u^h\rangle\right) > 0,$$

<span id="page-3-2"></span>![](_page_3_Figure_1.jpeg)

Figure 4: Spearman rank correlation between KV compression scoring metrics and the observed attention  $S^h$  for Llama-3.2-1B, for K-norm (top) and Q-Filters (bottom).

where  $\langle \cdot, \cdot \rangle$  denotes the dot product.

To validate Observation 3.1, we compute the Singular Value Decomposition (SVD) of a set of  $Q^h$  representations taken from various sequences for Llama-3.1-8B. We find that the first right-vector of the SVD verifies Observation 3.1 for all tested heads, and we display examples of projection distributions in Figures 2a and 2b. The intuitive consequence of this observation regarding attention weights is that, if a given  $K^h_t$  has a strong projection along  $\epsilon u_h$ , then future queries  $Q^h_{\geq t}$  can be expected to have a stronger dot-product with  $K^h_t$  in average.

However, it is not clear a priori that this effect is unidirectional, i.e. that there exists a unique direction  $u^h$  (up to a sign) that verifies Observation 3.1. Hence, identifying one such direction may not suffice to characterize the anisotropy of  $Q^h$  representations and to derive estimations of the dot-products used in attention. The uni-directional nature of the Query-Key anisotropy can be formalized as in Observation 3.2.

<span id="page-3-0"></span>**Observation 3.2.** Let  $u^h = \arg \max_{u \in \mathbb{S}^{d_H-1}} \mathbb{E}\left(\langle Q_i^h, u \rangle\right)$  and  $B = (u^h, u_2, ..., u_{d_H})$  an orthonormal basis of  $\mathbb{R}^{d_H}$ .

Then for all attention inputs X:

$$\forall m \in [2, d_H], \mathbb{E}\left(\langle Q_i^h, u_m \rangle\right) \approx 0$$

In Figure 2c, we observe that only the first singular component of the SVD of  $Q^h$  representations carries an anisotropic behavior, as the projections on all other components have a null mean. Hence, by taking the SVD right-vector basis as B, we can show that the first component of the SVD empirically verifies Observation 3.2. This lets us derive a basic estimation for the average unnormalized attention logits  $\langle Q_i^h, K_j^h \rangle$ .

<span id="page-3-1"></span>**Theorem 3.3** (proof in Appendix A). *Under Observation 3.1 and Observation 3.2, we have:* 

$$\mathbb{E}_{Q_i^h}(\langle Q_i^h, K_i^h \rangle) \approx \kappa^h \langle K_i^h, u^h \rangle$$

where  $\kappa^h$  is a positive constant.

Intuitively, projecting  $K_t^h$  along the anisotropic direction  $u^h$  gives us an estimate of the attention logits that involve  $K_t^h$  up to a positive multiplicative constant  $\kappa^h$ .

This result provides a justification for the method developed in Devoto et al. (2024). As a matter of fact, Observation 3.1 implies that  $\mathbb{E}_j\left(\cos(K_j^h,u^h)\right)$  should have the same sign as  $\epsilon$ . In practice, we observe  $\epsilon=-1$  for a vast majority of heads in trained causal LMs. Hence, we can derive a looser estimation from Theorem 3.3:

$$\mathbb{E}_{i,X}(\langle Q_i^h, K_j^h \rangle) \approx -\kappa^h \left| \mathbb{E}_{j,X} \left( \cos(K_j^h, u^h) \right) \right| ||K_j^h||_2$$

This estimation shows that the  $L_2$ -norm of  $K_j^h$  vectors is negatively correlated with the corresponding mean attention logits and can therefore be used to approximate them. However, only using the  $L_2$ -norm to estimate the attention score as done in Devoto et al. (2024) is suboptimal, as it ignores the angular component of the  $\langle K_j^h, u^h \rangle$  product. In practice, one can approximate  $u^h$  as defined in Observation 3.2 using the SVD of concatenated representations  $Q^h$  extracted by passing samples through the model. Formally, we collect a batch of Query activations  $Q^h = \{Q_1^h, Q_2^h, ..., Q_n^h\}$  by passing documents sampled from pre-training corpora and using the right-vectors V as the orthonormal basis B:

<span id="page-3-3"></span>
$$Q^h = U \Sigma V^{\top}, \text{ with } V = (v_1, v_2, ..., v_{d_H})$$
 (1)

The resulting  $v_1$  vectors are, up to a sign, what we refer to as Q-Filters, as they allow to estimate which Key-Value pairs are worth storing for each head along generation. Figure 3 also displays information about attention levels for the corresponding indices. For a given input X, we measure the average attention at position t as:

$$\mathcal{S}_t^h = \frac{1}{L - t + 1} \sum_{i=t}^L A_{it}^h,$$

where  $A^h$  is the attention map for head h. It appears clearly from Figure 3 that there exists a strong correlation between the average attention at a given index and the projection of  $K^h$  on the  $v_1$  component.

We observe that the projection of  $K^h$  on the  $v_1$  component has a consistent sign for a given head, e.g., it is consistently positive in Figure 3a and consistently negative in Figure 3b, while the projection results on  $v_2$  have a near-zero expectation, further validating Observation 3.1 and Observation 3.2.

#### 3.2. Q-Filters

Based on Theorem 3.3, we can design a KV Cache compression scheme that consists of the following steps:

- 1. For a given model, retrieve its Q-Filters, which can be obtained with the following procedure:
  - (a) Gather  $Q^h$  representations by passing samples through the model;
  - (b) Compute the SVD of the gathered representations at each layer and head;
  - (c) Obtain the *positive* right vector (or Q-Filter) for each head  $v_1^+ = \operatorname{sgn}(\mathbf{1}u_1^T)v_1$ .
- 2. At inference, for each head, discard the  $K_t^h$  with the lowest  $\langle K_t^h, v_1^+ \rangle$  value.

In the case of Grouped-Query Attention or GQA (Ainslie et al., 2023), we simply average the Q-Filters for each group of Query representations.

We bring the attention of the reader to the fact that this method only requires a single preparation step following training for a given model. The Q-Filters are entirely context-agnostic and rely on inherent properties of the Query and Key latent spaces. In the rest of this article, we use a subset of the Pile dataset (Gao et al., 2020) to compute the Q-Filters and discuss the choice of the dataset and of the number of necessary SVD samples in Section 4.1.

In Figure 4, we observe that the Q-Filters heuristic is noticeably more correlated with the attention score  $S^h$  for most heads compared to the  $L_2$ -norm metric. As such, ordering the Key-Value pairs using the Q-Filters heuristic should allow us to select more relevant pairs than using the method from Devoto et al. (2024) - that we will call K-norm for the sake of simplicity.

### 4. Experiments

We validate our method both on memory-constrained language modelling and on long-context retrieval tasks (e.g. needle-in-a-haystack). Additionally, we test our method on the Ruler dataset (Hsieh et al., 2024), which is specifically designed to test the model's long context modelling abilities. We test Q-Filters on Llama-3.1-8B, Llama-3.1-70B (Dubey et al., 2024) and Qwen-2.5-7B (Qwen et al., 2025), but the method can be easily adapted to any pre-trained decoderonly LLM. We compare Q-Filters with several KV Cache compression methods. These include StreamingLLM (Xiao et al., 2024), which focuses on language modeling by always retaining the initial tokens of the sequence. We also compare with SnapKV (Li et al., 2024), which performs compression based on attention scores from the final portion of the prompt, making it particularly suitable for compression of large prompts. Additionally, we compare against preserving low- $L_2$  norm tokens (Devoto et al., 2024) and the recent ExpectedAttention (Jegou & Jeblick, 2024).

Language Modelling To evaluate the performance of Q-Filters in the language modelling setup, we perform generation on the Pile dataset (Gao et al., 2020). We let the KV Cache grow up until a certain threshold, after which we start evicting the KV pairs so that the total size never exceeds the maximum threshold. We measure performance by tracking the model perplexity computed on past tokens in 20 sequences. We report results for a maximum KV Cache size of 512 pairs in Figure 5. We observe that Q-Filters consistently achieves the lowest perplexity among compression schemes, even for very long contexts. This observation scales to the 70B model, where Q-Filters significantly reduces the perplexity gap. This improvement is more pronounced in the latter portions of the sequences, suggesting better retention of relevant contextual information.

**Needle in a Haystack** The Needle-in-a-Haystack task embeds a key piece of information (the "needle") within a long sequence of distractors (the "haystack"), followed by a question that requires retrieving the needle. This evaluates the model's ability to handle long-range dependencies and tests how well KV Cache compression retains critical information. If important KV pairs are evicted, the model fails to answer correctly.

We evaluate Q-Filters by placing the needle at depths from 1k to 64k tokens and measuring retrieval accuracy. Similarly to (Devoto et al., 2024), we do not compress key-value pairs in the first two layers of the models in this experiment. As shown in Figure 6, Q-Filters outperforms K-Norm (Devoto et al., 2024), preserving crucial information even in extremely long contexts.

**Ruler Tasks** We evaluate the proposed method on the Ruler dataset (Hsieh et al., 2024), which comprises several sub-tasks that test the model long context modelling abilities, including Multi-hop Tracing, Long Context Aggregation, Long Context Retrieval and Question Answer-

<span id="page-5-1"></span>![](_page_5_Figure_1.jpeg)

Figure 5: Generation performance for a KV Cache size limited to 512 items for Llama-3.1-8B (top) and Llama-3.1- 70B (bottom).

ing. The dataset offers 3 variants with different sequence lengths: 4096, 8192, and 16384. We compare the score on Ruler with several other KV Cache compression methods and show average results in [Figure 7a.](#page-6-0) We report detailed per-task results in [Table 1](#page-6-1) and in [Appendix C.](#page-11-1) We test the model's score for several compression factors ranging from 2× to 32×. While for some lower compression factors, we find performance on par with other methods, Q-Filters achieve the highest score with the strongest compression factor of 32×, demonstrating the method's effectiveness at high compression rates.

## <span id="page-5-0"></span>4.1. Robustness of the Calibration Dataset

In [Figure 8,](#page-6-2) we analyse how the calibration dataset size impacts the performance of our Q-Filters computation. Our experimental results demonstrate that increasing the number of samples in the calibration dataset leads to an improvement in average perplexity, although the marginal benefits diminish beyond a certain point, namely around 1k samples. This suggests that while larger calibration datasets generally produce more robust Q-Filters, there exists a practical trade-off balancing computational cost and performance

<span id="page-5-2"></span>![](_page_5_Figure_6.jpeg)

![](_page_5_Figure_7.jpeg)

(b) Q-filters (average accuracy: 91%)

Figure 6: Needle-in-a-haystack performance for Llama-3.1- 8B using 64x KV Cache compression.

benefits. Based on these empirical findings and computational efficiency considerations, we standardized our experimental protocol to utilize 3,000 samples for computing the Q-Filters across all subsequent experiments. Another important consideration in the development of robust Q-Filters is the choice of calibration dataset. To investigate this aspect, we conducted a systematic analysis using multiple diverse datasets and model versions in [Figure 9.](#page-7-0) Our experiments revealed that the Q-Filter vectors exhibit remarkable stability across different calibration datasets, with a high average cosine similarity between vectors computed from distinct sources. This finding suggests that our method is relatively insensitive to the specific choice of calibration data, provided it maintains sufficient diversity and quality. Based on these results, we opted to use a carefully curated subset of the Pile dataset [\(Gao et al.,](#page-9-7) [2020\)](#page-9-7) for all Q-Filter computations.

## 4.2. Q-Filters Estimation Overhead

It could be argued that our method introduces a memory overhead as we need to store the Q-Filters on-device. Nevertheless, for a model using l layers and n<sup>H</sup> heads, storing the Q-Filters requires l × n<sup>H</sup> × d<sup>H</sup> parameters. For Llama-

<span id="page-6-1"></span>

| Compression method | FA-compatible | CWE  | FWE  | Multi-Key | Multi-Query | Multi-Value | Single | QA          | VT   | Average |
|--------------------|---------------|------|------|-----------|-------------|-------------|--------|-------------|------|---------|
| SnapKV             | X             | 88.7 | 89.0 | 15.1      | 29.6        | 28.8        | 68.7   | 42.8        | 83.2 | 50.5    |
| Expected Attention | ×             | 70.0 | 79.3 | 12.0      | 59.7        | <u>37.8</u> | 31.2   | <u>44.2</u> | 96.3 | 43.2    |
| Streaming-LLM      | ✓             | 53.8 | 93.4 | 14.1      | 16.8        | 16.7        | 15.7   | 62.3        | 15.8 | 31.6    |
| K-Norm             | $\checkmark$  | 22.9 | 74.8 | 8.7       | 16.6        | 25.8        | 55.9   | 20.6        | 32.0 | 31.3    |
| Q-Filters (ours)   | $\checkmark$  | 82.5 | 80.2 | 22.9      | 49.1        | 60.6        | 71.1   | 37.6        | 100  | 56.1    |

Table 1: Results on the Ruler-4096 dataset for Llama-3.1-70B-Instruct with an  $8 \times$  compression ratio. The second column indicates compatibility with FlashAttention.

<span id="page-6-0"></span>![](_page_6_Figure_3.jpeg)

![](_page_6_Figure_4.jpeg)

![](_page_6_Figure_5.jpeg)

(b) Average performance on Loogle (Short Dependency QA)

Figure 7: Average score for different long-context benchmarks using Llama-3.1-8b with different methods and compression ratios

3.2-1B, this is  $36k \times$  smaller than the total parameter count and  $196k \times$  smaller in the case of Llama-3.2-405B. Another source of overhead could be attributed to the initial computation of the filters that are required for every new model. We find that passing 20 samples of length 2048 through the model and performing the SVD on 3k randomly sampled representations for each head is sufficient to obtain strong performance. In our experiments with Llama-3.2-70B, computing the filters took less than 3 minutes on two A100-80GB GPUs. This cost is thus negligible when compared with the cost of inference.

<span id="page-6-2"></span>![](_page_6_Figure_9.jpeg)

Figure 8: Perplexity after 1024 tokens for Q-Filters obtained using different sizes of  $Q^h$  (Eq. (1)) to calculate the SVD.

## 4.3. Throughput and Scalability

In this section, we analyze the time and memory overhead induced by the Q-Filters method. Our approach is more efficient than many KV Cache compression methods, as it estimates the relevance of a  $K^h$  representation without materializing the attention maps. This property makes it compatible with memory-efficient self-attention implementations such as FlashAttention (Dao, 2024). During inference, Q-Filters maintains the same theoretical time complexity as the K-norm method (Devoto et al., 2024), since computing a norm and a scalar product require a comparable number of floating-point operations.

By avoiding the explicit computation of attention scores, our method achieves lower inference latency compared to existing approaches. To quantify this efficiency, we measure the *Time to First Token* across different methods in Figure 10. Time to First Token (TTFT) refers to the latency between submitting a prompt and receiving the first generated token. This metric is particularly relevant in scenarios where fast response times are critical, such as interactive AI applications. Compressing the KV Cache directly impacts TTFT: by reducing the memory footprint of the KV Cache, it allows a larger portion of the prompt context to fit within fast-access memory, minimizing memory swapping overhead. As a result, compression techniques that efficiently manage the KV Cache should significantly reduce

<span id="page-7-0"></span>![](_page_7_Figure_1.jpeg)

Figure 9: Cosine-similarity between Q-Filters computed on datasets coming from different domains and languages and on pre-trained and post-trained models. The scores are averaged over all layers and heads.

<span id="page-7-1"></span>![](_page_7_Figure_3.jpeg)

Figure 10: First token latency across KV Cache compression methods of Llama-3.2-8B with a length of 64k prompt.

initial response latency. Notably, our experiments show that Q-Filters maintain this performance advantage even as the sequence length increases, suggesting better scalability compared to methods that require explicit attention computation.

## 5. Limitations

In Appendix B, we run generation experiments on Qwen-2.5-7B-Instruct (Qwen et al., 2025), and we observe that, although the results still favour the Q-Filters method, the gap is less clear compared to the Llama models. Our main hypothesis for this discrepancy lies in the slightly different attention mechanism used in Qwen-2.5 suite, which adds a bias to the QKV projection. Hence, it is likely that the geometrical observations made in Section 3 are not accurate in that case. Similarly, initial experiments with Olmo-2 models (OLMo et al., 2025) were unsuccessful, which can

be explained by their use of the QK-normalization technique (Dehghani et al., 2023). These different tricks would most likely require an adaptation of our analysis to yield a better approximation of the attention distributions.

## 6. Related Works

After the success of long-context models (Reid et al., 2024; Anthropic, 2024; Achiam et al., 2023), compressing the KV Cache has become a key research focus to enable processing of long-context inputs.

Some methods reduce the KV Cache size by modifying the model architecture. For example, Ainslie et al. (2023) and Shazeer (2019) reuse the same Keys for multiple queries, thereby reducing redundancy in storage. Nawrot et al. (2024) propose a dynamic token-merging strategy, learning which KV pairs to merge. While these approaches achieve significant compression, they require training or fine-tuning, making them less practical in real-world scenarios where retraining the model from scratch is not feasible. In contrast, our method requires only a short, computationally inexpensive calibration step, avoiding parameter updates entirely. Recently DeepSeek-AI et al. (2024) introduced a Multi-Head Latent Attention, a modification to the standard attention mechanism that performs a low-rank reduction of the KV Cache during pre-training.

Training-free approaches aim to compress the KV Cache without modifying the model, typically by approximating the attention score over long sequences and prioritizing tokens with higher importance. Among these, Xiao et al. (2024) focus on language modelling tasks and propose always retaining the first token(s) (as an attention sink) and the last n tokens in a sliding window. Also, Zhang et al. (2024) focuses on generation tasks and introduces a policy that evicts tokens during generation based on a scoring function derived from cumulative attention. In contrast, other works focus on the task of compressing a large prompt provided by the user. Li et al. (2024) uses attention from the last part of the prompt to estimate KV pairs importance. With the same goal, Cai et al. (2024) assigns more cache budget to lower layers and less to higher layers. Finally, Guo et al. (2024) proposes to rescale the KV score of other methods by the  $L_1$  norm of the Values.

In contrast, our approach is not tailored to a specific use case but provides competitive performance across both synthetic tasks and real-world scenarios, including in-context learning and chat-based interactions. Additionally, many of these approaches are incompatible with FlashAttention Dao (2024) due to their reliance on accessing the full attention weights, which FlashAttention does not expose.

