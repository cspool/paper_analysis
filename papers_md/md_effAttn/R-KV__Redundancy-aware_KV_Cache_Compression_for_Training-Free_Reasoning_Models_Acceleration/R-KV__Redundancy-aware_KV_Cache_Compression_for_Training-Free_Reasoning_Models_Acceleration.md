# R-KV: Redundancy-aware KV Cache Compression for Reasoning Models

Zefan Cai<sup>1⊠</sup>, Wen Xiao<sup>2⊠</sup>, Hanshi Sun<sup>3</sup>, Cheng Luo<sup>4</sup>, Yikai Zhang<sup>1</sup>, Ke Wan<sup>5</sup>, Yucheng Li<sup>6</sup>, Yeyang Zhou<sup>5</sup>, Li-Wen Chang, Jiuxiang Gu, Zhen Dong<sup>7</sup>, Anima Anandkumar<sup>4</sup>, Abedelkadir Asi<sup>2</sup>, Junjie Hu<sup>1⊠</sup>

<sup>1</sup>University of Wisconsin - Madison <sup>2</sup>Microsoft <sup>3</sup>Carnegie Mellon University <sup>4</sup>California Institute of Technology <sup>5</sup>University of California - San Diego <sup>6</sup>University of Surrey <sup>7</sup>University of California - Berkeley https://zefan-cai.github.io/R-KV.page/ https://github.com/Zefan-Cai/R-KV

## Abstract

Reasoning models have demonstrated impressive performance in self-reflection and chain-of-thought reasoning. However, they often produce excessively long outputs, leading to prohibitively large key-value (KV) caches during inference. While chain-of-thought inference significantly improves performance on complex reasoning tasks, it can also lead to reasoning failures when deployed with existing KV cache compression approaches. To address this, we propose Redundancyaware KV Cache Compression for Reasoning models (R-KV), a novel method specifically targeting redundant tokens in reasoning models. Our method preserves nearly 100% of the full KV cache performance using only 10% of the KV cache, substantially outperforming existing KV cache baselines, which reaches only 60% of the performance. Remarkably, R-KV even achieves 105% of full KV cache performance with 16% of the KV cache. This KV-cache reduction also leads to a 90% memory saving and a 6.6× throughput over standard chain-ofthought reasoning inference. Experimental results show that R-KV consistently outperforms existing KV cache compression baselines across two mathematical reasoning datasets.

# 1 Introduction

Recent advancements in large language models (LLMs) have demonstrated remarkable capabilities in complex reasoning and self-reflection. However, reasoning models (e.g., DeepSeek-R1 [1]) exhibit a critical deployment challenge: their tendency to produce excessively lengthy and redundant reasoning traces results in unsustainable memory demands [2], primarily due to the rapid growth of the key-value (KV) cache during autoregressive generation. For instance, a DeepSeek-R1-Distill-Llama-8B model may generate 32K tokens to solve a complex math problem, consuming 15.5GB of memory to load the model weight and 4.1GB of memory to store the KV cache. This paradigm of long chain-of-thought (CoT) reasoning generation necessitates the development of KV cache compression.

Outputs from current reasoning models, especially during complex chain-of-thought generation, are fundamentally marked by pervasive redundancy. This inherent characteristic means they are often filled with superfluous content, including unnecessary reflections, iterative re-evaluations, and verbose self-dialogue, all of which add little new semantic value while significantly inflating the

 $<sup>\</sup>boxtimes Corresponding \ to \ Zefan \ Cai \ {\tt zefncai@gmail.com}, \ Wen \ Xiao \ {\tt wxiao@microsoft.com} \ and \ Junjie \ Hujunjie.hu@wisc.edu$ 

![](_page_1_Figure_0.jpeg)

Figure 1: R-KV: (1) Decoding-Time Compression ([§3.1\)](#page-3-0); (2) KV Cache Selection with Importance and Redundancy Estimation ([§3.2,](#page-3-1) [§3.3\)](#page-4-0) ; (3) KV Cache Compression by joint selection ([§3.4\)](#page-4-1).

length of the generation beyond what is needed for concise, effective reasoning. Our analysis ([§2.1\)](#page-1-0) shows that over half of the tokens in R1's reasoning chains contribute minimally to task performance, indicating that repetitive self-verification steps or intermediate calculations could be substantially condensed by KV cache compression methods without compromising reasoning accuracy.

However, existing KV cache compression works [\[3,](#page-10-2) [4,](#page-10-3) [5,](#page-10-4) [6,](#page-10-5) [7\]](#page-10-6) primarily handle long input prompts but do not explore extensively for long generation outputs. Furthermore, based on our observation ([§2.2\)](#page-2-0), standard KV-cache compression methods that rely on simple attention-based importance filtering often fail because the repetitive sections generate high attention signals for themselves. Naively pruning tokens with "low attention weight" may remove crucial but scattered bits of reasoning, or over-retain duplicative self-reflections that appear to have high attention. This observation motivates our exploration of redundancy-aware compression strategies, which selectively retain "important and non-repetitive context" during decoding to preserve the model's critical reasoning ability.

In this work, we propose Redundancy-aware KV cache compression for reasoning models (i.e., R-KV). Our approach consists of three key components: (1) an attention-based importance scoring mechanism that selects critical tokens for retention, (2) a dynamic redundancy scoring mechanism that identifies repetitive tokens through real-time analysis of key vectors, and (3) a joint eviction mechanism that balances both redundancy and importance to optimize cache efficiency.

In our experiments on popular math reasoning benchmarks ([§4\)](#page-5-0), by selectively retaining only 10-34% of the original KV cache, R-KV achieves comparable performance parity with the uncompressed reasoning model, outperforming state-of-the-art compression baselines with only 60% of the performance. Remarkably, R-KV even achieves 105% accuracy of the full KV baseline with around 16% of the KV cache using DeepSeek-R1-Distill-Llama-8B on the AIME-24 dataset.

This advancement addresses a fundamental tension in deploying state-of-the-art LLMs—balancing reasoning capabilities with practical memory constraints. Our contributions extend beyond technical optimization: we provide systematic evidence that redundancy in CoT generation can be strategically compressed without compromising reasoning abilities. As a training-free and model-agnostic method, R-KV can be used in the rollout process in reinforcement learning (RL) and LLM serving.

# 2 Observation

## <span id="page-1-0"></span>2.1 Redundancy in Reasoning Models

As noted in [\[2\]](#page-10-1), reasoning models often generate a detailed chain of thoughts and multiple reflection steps, resulting in significantly longer responses than standard models. [Figure 2](#page-2-1) shows that both reasoning models (i.e., DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Qwen-14B) generate more than 8× longer generation output compared to the ground truth on two popular math reasoning datasets. However, not all of the additional tokens contribute meaningful content, as much of the decoded context is dominated by repetition. [Figure 2](#page-2-1)

<span id="page-2-1"></span>![](_page_2_Figure_0.jpeg)

Figure 2: Comparison of generation length and average 1-/2-gram frequency for reasoning models and ground truth of MATH-500 [8] and AIME 2024 [9]. Reasoning models generate substantially longer responses with  $8-14 \times$  more tokens, and show higher word repetition with  $5-7 \times$  higher frequency.

also shows that the average frequency of 1- to 2-grams is consistently higher in the generation output of reasoning models than in ground truth, indicating greater repetitions in the generated outputs of reasoning models.

## <span id="page-2-0"></span>2.2 Failure of Existing KV Compression Methods to Handle Redundancy

Most existing KV cache compression methods prioritize token selection based primarily on tokens' contextual importance, typically measured through attention scores between key and query tokens [3, 4]. While this approach effectively retains critical context, it fails to account for redundancy—particularly problematic in reasoning models. In such models, we find that repetitive content often receives disproportionately high attention scores, as it closely mirrors previously generated repetitive text. As a result, redundant tokens are excessively retained, unnecessarily inflating the KV cache size without providing additional meaningful new information. In Figure 3, we visualize the cached tokens (inside red boxes) selected by a popular attention-based KV cache method (i.e., SnapKV), showing many repetitions related to self-reflection and conclusion to the final answer.

```
You are given a math problem. Problem: In Mr. Roper's class of 30 students,
 [Question and Instruction - 102 words]
First, the problem says that there are 30 students in total in the class. Out of
[Think - 203 words]
[Reflection for 13 times and 581 words in total]
 But wait, the ... So, 10% of 30 is 3. So 3 students are leaving early.
 Think - 36 words]
But in the initial problem...So
[Think - 42 words]
    wait, the ...30 is 3. So 3
 But in the initial problem, the... So 3
             .. early?" So, 10% of 30 is 3. So 3 students are leaving early.
                    So. 10% of 30 is 3. So
But wait, the user wrote: ...10% of 30 is 3, So, 3 students are leaving early.
                   of 30 is 3. Therefore, 3 students are leaving early.
I think that's all. The calculation is straightforward: 10% of 30 is 3.
```

Figure 3: KV selected by SnapKV. SnapKV suffers from redundancy in reasoning models. Black tokens are not selected by SnapKV; brighter colors reflect higher attention scores. Blue tokens are omitted output.

# 3 Redundancy-aware KV Cache Compression (R-KV)

To address the redundant thinking issue, we propose a *redundancy-aware decoding-time KV cache compression method* (**R-KV**) that explicitly targets the compression of redundant tokens in reasoning models. Our approach balances *importance* and *non-redundancy* in token selection, ensuring that KV cache storage is allocated to both highly informative and diverse content. By incorporating

redundancy estimation into the selection process, our method effectively mitigates unnecessary KV cache growth while preserving the model's reasoning capabilities.

Specifically, R-KV consists of three key components: (1) an *importance scoring mechanism* (§3.2) leveraging attention weights, (2) a *redundancy estimation mechanism* (§3.3) based on semantic similarity of key vectors, and (3) a *joint selection strategy* (§3.4) that optimizes cache efficiency by balancing redundancy and importance.

### <span id="page-3-0"></span>3.1 Decoding-time Compression

Different from existing KV cache compression methods[3, 5, 4] that focus on the *prefilling stage* to manage long-context inputs, our R-KV focuses on the *decoding stage* for reasoning models—a distinctive setting where the generated output is significantly longer than the prompt.

Specifically, R-KV allocates memory for two components: a cache of budget size  $B_{\text{budget}}$  to store retained KV tokens, and a buffer of size  $B_{\text{buffer}}$  for newly generated text tokens. The total memory requirement is thus  $B_{\text{total}} = B_{\text{budget}} + B_{\text{buffer}}$ . After the model generates each fixed-length text segment in the buffer, R-KV performs KV cache compression. At the end of each text segment, the last  $\alpha$  tokens are always retained in the cache as **observation tokens**, following prior work [3]. Next, we concatenate the existing  $B_{\text{budget}}$  tokens in the cache with the first  $B_{\text{buffer}} - \alpha$  tokens in the buffer, resulting in  $n = B_{\text{budget}} + B_{\text{buffer}} - \alpha$  candidate KV tokens. Each candidate is assigned a selection score (§3.4), and we select the top  $k = B_{\text{budget}} - \alpha$  tokens to fit in the rest of the cache budget, in addition to the  $\alpha$  observation tokens. This process compresses the KV cache while preserving critical context, enabling efficient memory utilization during autoregressive decoding.

## <span id="page-3-1"></span>3.2 Importance Scoring via Attention Weights

Following attention-based methods (e.g., SnapKV [3], PyramidKV [5]), R-KV estimates token importance using attention weights, leveraging the intuition that tokens receiving higher attention contribute more to decoding and are thus more critical for preserving model performance. Specifically, we compute each key token's attention scores received from the last  $\alpha$  **observation tokens** during decoding. In addition to the standard multi-head attention mainly adopted by the prior works [3], we also propose the importance score estimation using the grouped-query attention. Below, we detail the estimation on top of these two popular attention mechanisms used by current LLMs.

**Multi-Head Attention (MHA).** Given the last  $\alpha$  observation tokens as query  $\mathbf{Q}^h \in \mathbb{R}^{\alpha \times d}$  and n key states  $\mathbf{K}^h \in \mathbb{R}^{n \times d}$  for each attention head h, the attention scores  $\mathbf{A}^h \in \mathbb{R}^{\alpha \times n}$  are computed as:

<span id="page-3-3"></span><span id="page-3-2"></span>
$$\mathbf{A}^{h} = \operatorname{softmax}(\mathbf{Q}^{h} \cdot (\mathbf{K}^{h})^{\top} / \sqrt{d}). \tag{1}$$

**Grouped-Query Attention (GQA).** In GQA, each key/value head h is shared among a group of G distinct query heads indexed by  $g \in [0, G)$ . Correspondingly, we denote the shared key/value states as  $K^h, V^h \in \mathbb{R}^{n \times d}$ , and the G query states as  $Q^{h,0}, \ldots, Q^{h,G-1} \in \mathbb{R}^{\alpha \times d}$  within the head group indexed by h, where n is the number of key/value states, d is the head hidden dimension. The attention score for each of the G query heads within the group is computed as:

$$\mathbf{A}_{\text{group}}^{h,g} = \mathbf{Q}^{h,g} \cdot (\mathbf{K}^h)^{\top} / \sqrt{d} \in \mathbb{R}^{\alpha \times n}, \quad \text{for } g = 0, \dots, G - 1.$$
 (2)

These G individual matrices are then aggregated into a single consolidated matrix  $\boldsymbol{A}_{\text{group}}^h$  for the head group h using a max-pooling operation across the group dimension. The final attention weight  $\boldsymbol{A}^h$  for the head group h is then obtained by renormalizing  $\boldsymbol{A}_{\text{group}}^h$  along the key token dimension.

$$\boldsymbol{A}_{\text{group}}^{h} = \text{maxpool}\left(\left[\boldsymbol{A}_{\text{group}}^{h,0}, \ldots, \boldsymbol{A}_{\text{group}}^{h,G-1}\right]\right) \in \mathbb{R}^{\alpha \times n}, \quad \boldsymbol{A}^{h} = \text{softmax}\left(\boldsymbol{A}_{\text{group}}^{h}\right) \in \mathbb{R}^{\alpha \times n} \quad (3)$$

**Stabilization and Importance Estimation.** We use  $A^h$  hereafter to denote the attention weights calculated using either MHA or GQA. Note that the per-token importance scores derived from  $A^h$  may contain outliers with excessively high values, resulting in unstable estimation of importance scores. To mitigate this influence, we follow the prior work [3] and apply a max-pooling operation to these per-token importance scores over a sliding window of size 2W across recent tokens. Specifically, we denote  $A^h_{j,i}$  as the attention score from the j-th query to the i-th key in  $A^h$ . We obtain the stabilized

attention score  $\tilde{A}^h$  by computing its (i, j) entry, and finally obtain the importance score of retaining the *i*-th token in the KV cache as  $I_i^h$  for each attention head h, as shown below:

<span id="page-4-3"></span>
$$\tilde{A}_{j,i}^{h} = \max\left(A_{j,i-W}^{h}, \dots, A_{j,i}^{h}, \dots, A_{j,i+W-1}^{h}\right), \quad I_{i}^{h} = \frac{1}{\alpha} \sum_{j=0}^{\alpha-1} \tilde{A}_{j,i}^{h} \in \mathbb{R}.$$
 (4)

#### <span id="page-4-0"></span>3.3 Redundancy Estimation via Semantic Similarity

To identify redundant tokens, we measure the semantic similarity between key states using cosine similarity. Tokens with high similarity to others are considered potentially redundant and can be selectively removed to optimize KV cache memory.

Cosine Similarity between Key Tokens: Given the key tokens  $\boldsymbol{K}^h \in \mathbb{R}^{n \times d}$  for a specific head h, We first normalize each key vector  $\boldsymbol{K}_i^h, \forall i \in [0,1)$  into  $\overline{\boldsymbol{K}}_i^h$ , and then compute the cosine similarity matrix  $\boldsymbol{S}^h$  using the normalized key vectors.

$$\overline{\boldsymbol{K}}_{i}^{h} = \frac{\boldsymbol{K}_{i}^{h}}{\|\boldsymbol{K}_{i}^{h}\|_{2} + \epsilon} \in \mathbb{R}^{d}, \quad \boldsymbol{S}^{h} = \overline{\boldsymbol{K}}^{h}(\overline{\boldsymbol{K}}^{h})^{\top} \in \mathbb{R}^{n \times n}, \quad S_{i,i}^{h} \leftarrow 0, \forall i \in [0, n),$$
 (5)

where  $\|\cdot\|_2$  is the L2 norm and  $\epsilon$  is a small constant (e.g.,  $10^{-8}$ ) for numerical stability. To prevent tokens from being marked as redundant with themselves, we zero out the diagonal elements  $S_{i,i}^h$ .

Enforce Retention of Recent Tokens. While redundant, such tokens may still carry meaningful information. Thus, naively removing all redundant tokens can impair model performance. To address this, we retain only the  $\beta$  most recently generated tokens among those exhibiting high similarity, as these later tokens tend to better support the model's decoding than earlier ones. To enforce this, we further zero out the similarity scores in  $S^h$  corresponding to these  $\beta$  most recent similar tokens. Formally, for each token  $i \in [0, n)$ , we identify the set of indices of highly similar tokens:  $\mathcal{I}_i^h = \{j \mid S_{j,i}^h > T, j \in [0, n)\}$ , where T is a fixed hyperparameter for similarity threshold. For this set, we extract the subject  $\mathcal{I}_{i,\beta}^h \subseteq \mathcal{I}_i^h$ , containing up to the  $\beta$  largest indices—i.e., the  $\beta$  most recent similar tokens to token i, or fewer if not enough such tokens exist. We then suppress their influence by zeroing out their similarity scores with token i in  $S^h$ , i.e.,  $S_{j,i}^h \leftarrow 0$ ,  $\forall j \in \mathcal{I}_{i,\beta}^h$ . This modification effectively nullifies the direct similarity links from token i to its  $\beta$  most recent highly similar tokens.

**Redundancy Score Estimation:** Finally, we compute normalized redundancy scores for all key tokens in Eq. (6). First, for each key token  $i \in [0,n)$  in each head h, we compute its average similarity score  $\bar{S}_i^h$ . Intuitively,  $\bar{S}_i^h$  measures how similar token i is, on average, to all other key tokens in the sequence. A high  $\bar{S}_i^h$  indicates that the semantic content of token i is largely shared with other tokens, suggesting potential redundancy. Next, to obtain per-token redundancy scores  $R_i^h$  within a fixed numerical range for each head h, we normalize  $\bar{S}_i^h$  using a softmax operation. The resulting score  $R_i^h$  reflects the redundancy of token i for head h, with higher values indicating greater redundancy.

$$\bar{S}_{i}^{h} = \frac{1}{n} \sum_{i=0}^{n-1} S_{j,i}^{h}, \quad R_{i}^{h} = \left( \text{softmax} \left( [\bar{S}_{0}^{h}, \dots, \bar{S}_{n-1}^{h}] \right) \right)_{i}$$
 (6)

#### <span id="page-4-1"></span>3.4 Joint Selection Strategy for KV Cache Retention

To efficiently manage KV cache storage while retaining essential context, we employ a joint selection strategy that integrates both importance and redundancy scores. Given a predefined token budget  $B_{budget}$  per attention head, our goal is to retain tokens that maximize information diversity while minimizing redundancy. The final selection score  $Z_i^h$  for each token i in head h is computed as:

<span id="page-4-2"></span>
$$Z_i^h = \lambda I_i^h - (1 - \lambda) R_i^h, \tag{7}$$

where the importance score  $I_i^h$  and the redundancy score  $R_i^h$  are computed in Eq. (4) and Eq. (6) respectively. A higher  $I_i^h$  indicates that a token is more important and should ideally be retained, while a higher  $R_i^h$  suggests higher token redundancy. The hyperparameter  $\lambda$  controls the trade-off

<span id="page-5-1"></span>![](_page_5_Figure_0.jpeg)

Figure 4: Results of R-KV compared with SnapKV and FullKV on the MATH-500 and AIME24 datasets for R1-Llama-8B (**top**) and R1-Qwen-14B (**bottom**). Results are reported as pass@1 based on 64 generated responses per question.

between prioritizing important tokens and reducing redundant tokens. We discuss the rationale for choosing  $\lambda$  through a sensitivity analysis in §5.1. This strategy ensures that the KV cache prioritizes storing tokens that are both important and semantically diverse, thereby improving memory efficiency without compromising model performance.

# <span id="page-5-0"></span>4 Experiment

### 4.1 Experimental Setup

**Models and Datasets** In our experiments, we use variants of the DeepSeek-R1 distilled model: DeepSeek-R1-Distill-Llama-8B, and DeepSeek-R1-Distill-Qwen-14B [1], which we refer to as R1-Llama-8B and R1-Qwen-14B, respectively, for brevity throughout the paper.

We evaluate the models' mathematical reasoning capabilities using three benchmarks: MATH-500 [8] and AIME 2024 [9].

**Hyperparameters** We set  $B_{\text{buffer}} = 128$ ,  $\alpha = 8$  and  $\lambda = 0.1$ , with an analysis of  $\lambda$  in §5.1.

**Baselines** We compare our method against SnapKV [3], originally designed for long prefilling. To adapt it for decoding, we apply the same compression interval as our method, i.e., compressing the KV cache every 128 decoding steps using identical  $B_{\text{budget}}$  and  $B_{\text{buffer}}$ . Our approach focuses on improving KV cache eviction through a hybrid strategy, and we therefore restrict comparison to state-of-the-art attention-based eviction methods. Budget allocation techniques (e.g., head-level [6] and layer-level [5]) are orthogonal to our work and not included. We also report results for FullKV, which retains the full KV cache and serves as the gold standard for decoding quality.

**Evaluation Setup** We set the maximum generation length to 16,384 tokens for MATH-500 and 32,768 tokens for AIME 2024 and AIME 2025, because further increasing the generation length has shown no improvement on model performance on these datasets from our attempts. We find that using greedy decoding to evaluate long-output reasoning models results in significant variability across different setups. Following existing works [1], we utilize pass@k evaluation [10] and report

pass@1 using a non-zero temperature. We use the recommended sampling temperature and top-p value for each model, i.e., sampling temperature of 0.6 and a top-p value of 0.95 for DeepSeek-R1 Distilled models. We generate 64 responses for each question. Pass@1 is then calculated as Pass@1 =  $\frac{1}{k}\sum_{i=1}^{k}p_i$ , where  $p_i$  denotes the correctness of the i-th response. This method provides more reliable performance estimates.

#### <span id="page-6-1"></span>4.2 Results

The accuracy performance of R-KV compared with all baselines is shown in Figure 4, with detailed accuracy numbers in Appendix B.2. The KV cache budget ratio is calculated based on the KV cache budget and the average generation length of tokens, i.e., R1-Llama-8B: 2,979.1 on MATH-500 and 15,535.8 on AIME24; R1-Qwen-14B: 2,833.04 on MATH-500 and 12,402 on AIME24. Our method significantly outperforms the baseline SnapKV, achieving up to 40% Acc. improvement. We provide two KV cache budget and performance analysis. Fixed budget analysis is more practical because when the model outputs longer (i.e., from 2,979.1 on MATH-500 to 15,535.8 on AIME24), the KV cache budget needed for lossless compression increases less (i.e., 512). In the KV cache budget ratio perspective, the changes of lossless compression ratio is dominated by generation length.

**Ratio Budget** For R1-Llama-8B, R-KV achieves lossless compression with 34% KV cache budget on the MATH-500 dataset and with 10% KV cache budget on the AIME-2024 dataset. Given 16% KV cache budget, our method even surpasses the FullKV baseline, reaching 105% of its accuracy. Similarly, for R1-Qwen-14B, R-KV achieves lossless compression with 54% KV cache budget on the MATH-500 dataset and with 25% KV cache budget on the AIME-2024 dataset. Given 33% KV cache budget, our method achieves 105% of FullKV accuracy.

**Fixed Budget** For R1-Llama-8B, R-KV achieves lossless compression with 1024 KV cache budget on the MATH-500 dataset and with 1536 KV cache budget on the AIME-2024 dataset. For R1-Llama-8B, R-KV achieves lossless compression with 1536 KV cache budget on the MATH-500 dataset and with 3072 KV cache budget on AIME-2024.

### 5 Discussion

### <span id="page-6-0"></span>5.1 How to Choose $\lambda$ ?

Figure 5 shows the distributions of the Importance Score ( $I^h$ ) and Redundancy Estimation ( $R^h$ ) for head h=0 at the top layer ( $N_{\text{layer}}=31$ ). The figure reveals that  $I^h$  is sparse and dominated by a few outlier values, while the similarity distributions (which inform  $R^h$ ) are relatively dense. When  $\lambda=0$ , the token retention strategy is overned entirely by Redundancy Estimation ( $R^h$ ). As shown in Figure 5, the initial four tokens are not guaranteed to be preserved. As highlighted by prior work [7], evicting these initial tokens can severely impair the generative capabilities of LLMs. Therefore, it is crucial to select a  $\lambda$  value that starts from at least 0.01. On the other hand, as  $\lambda$  increases beyond 0.1, the selection metric becomes increasingly dominated by attention scores. These observations suggest that an optimal  $\lambda$  lies within the range of  $0.01 \le \lambda \le 0.1$ , effectively balancing the contributions of Importance Score and Redundancy Estimation.

Figure 6 presents the accuracy (Acc.) performance of R-KV on the DeepSeek-Distill-R1-Llama-8B model using the MATH-500 dataset. The results further guide the choice of  $\lambda$  for optimal performance. The figure demonstrates that  $\lambda=0.1$  yields the highest accuracy. In contrast, strategies relying solely on redundancy ( $\lambda=0$ ) or solely on attention ( $\lambda=1$ ) exhibit the poorest performance, underscoring the complementary nature of these two metrics and the importance of a balanced approach. Thus, based on this finding, we select  $\alpha=0.1$  for all evaluations detailed in Figure 4.

## 5.2 Failure of Attention-Based Methods to Capture Redundancy

To thoroughly investigate the advantages of R-KV's hybrid selection metrics (combining attention and redundancy) over pure attention-based importance metrics, we compared the tokens selected by R-KV against those chosen by a pure attention-based method (SnapKV). We present a case where R-KV correctly completes the task while the comparison method fails. As illustrated in Figure 7,

<span id="page-7-0"></span>![](_page_7_Figure_0.jpeg)

![](_page_7_Figure_1.jpeg)

Figure 5: KV sekection score comparison of attention-only metric v.s. redundency-only metric v.s. R-KV with different  $\lambda$ . When  $\lambda \geq 0.1$ , the selection score starts to be dominated by attention score.

Figure 6: Performance Comparison of the same methods as Figure 5.

<span id="page-7-1"></span>![](_page_7_Figure_4.jpeg)

Figure 7: Comparison of selected key-value (KV) tokens for an example between SnapKV (left) and R-KV (right). Grey tokens are unselected, while the gradient from light to dark red indicates the number of attention heads selecting each token (darker = more heads). R-KV selects a more diverse and broadly distributed set of tokens, capturing richer contextual information.

grey tokens represent unselected tokens, while the gradient from light orange to red indicates the number of heads selecting each token, with darker red signifying selection by more heads.

When considering the tokens selected by all heads, we observe that R-KV selects a more diverse set of tokens that cover a broader range and contain more effective information. These selections are more evenly distributed throughout the decoded output, capturing a more comprehensive context representation. In contrast, SnapKV's selected tokens exhibit more limited coverage. It tends to favor tokens positioned close to the query token, which are often selected multiple times by various heads, indicating a concentration of attention in localized areas. Furthermore, SnapKV also selects tokens that are not in close proximity to the query but still constitute largely redundant and unimportant segments (i.e., "3 students are leaving early." and "But in the initial").

## 5.3 Efficiency Analysis

**Memory Saving** R-KV achieves improved memory efficiency by allocating fixed-size buffers for both the retained KV cache and newly generated tokens. Unlike FullKV, which scales memory linearly with sequence length, R-KV 's memory footprint remains constant, enabling substantial savings during long-form generation. Detailed memory accounting is provided in Appendix C.1.

<span id="page-8-0"></span>

| Gen. Length | Method | Budget                                                            | Mem. Saving (%)         | Batch                               | Throughput (tok/s)               | Tokens Gen.                       | Dec. Time (s)                    |
|-------------|--------|-------------------------------------------------------------------|-------------------------|-------------------------------------|----------------------------------|-----------------------------------|----------------------------------|
| 8K          | FullKV | -                                                                 | -                       | 1<br>62 (max)                       | 75.44<br>849.13                  | 8 094<br>501 828                  | 107.30<br>590.99                 |
|             | R-KV   | Fixed – 1024<br>Fixed – 1024<br>Fixed – 1536                      | 87.50<br>87.50<br>81.25 | 1<br>402 (max)<br>287 (max)         | 80.46<br>3 251.52<br>2 525.75    | 8 094<br>3 253 788<br>6 546 972   | 100.60<br>1 000.70<br>919.72     |
|             |        | Ratio – 10% – 819<br>Ratio – 34% – 2785<br>Ratio – 54% – 4423     | 90.00<br>66.00<br>46.00 | 479 (max)<br>167 (max)<br>105 (max) | 3 809.15<br>1 608.01<br>1 257.83 | 3 877 026<br>1 351 698<br>849 870 | 1 017.82<br>840.61<br>675.66     |
| 16K         | FullKV | -<br>-                                                            | -<br>-                  | 1<br>30 (max)                       | 69.41<br>347.03                  | 16 286<br>488 580                 | 234.65<br>1 407.89               |
|             | R-KV   | Fixed – 1024<br>Fixed – 1024<br>Fixed – 1536                      | 93.75<br>93.75<br>90.63 | 1<br>402 (max)<br>287 (max)         | 80.95<br>3 188.82<br>2 447.61    | 16 286<br>6 546 972<br>4 674 082  | 201.18<br>2 053.10<br>1 909.65   |
|             |        | Ratio – 10% – 1 638<br>Ratio – 34% – 5 570<br>Ratio – 54% – 8 847 | 90.00<br>66.00<br>46.00 | 271 (max)<br>82 (max)<br>46 (max)   | 2 300.28<br>797.43<br>584.77     | 4 413 506<br>1 335 452<br>749 156 | 1 918.68<br>1 674.70<br>1 281.12 |

Table 1: Memory saving, throughput, and decoding-time comparison for Llama3-8B under various generation length and KV cache compression budget settings.

**Computation Overhead** While R-KV introduces additional computation for importance and redundancy scoring, the total overhead is modest and often outweighed by the reduced attention cost over a compressed KV cache. This trade-off becomes increasingly favorable as sequence length grows. Complexity comparisons can be found in Appendix C.1

**Real-time analysis** We present the real-time analysis of memory saving and end-to-end throughput improvement in Table 1. When the batch size is 1, R-KV exhibits a slight throughput advantage over FullKV. This suggests that the acceleration achieved by R-KV through reduced attention computation outweighs computational overhead of R-KV. However, this direct speedup constitutes a minor portion of the overall benefit. The primary throughput improvement from R-KV stems from enabling significantly larger inference batch sizes due to KV cache compression.

We evaluate end-to-end throughput under both ratio-based and fixed KV cache budgets. R-KV consistently enables much larger batch sizes and higher throughput than FullKV, with benefits becoming more pronounced at longer sequence lengths. For example, at a sequence length of 16K, R-KV achieves up to  $9\times$  larger batch sizes and over  $6.6\times$  higher throughput under a 10% compression ratio, and  $13.4\times$  larger batch sizes with  $9.2\times$  throughput under a fixed budget of 1024. Detailed analysis are provided in Appendix C.2.

## 6 Related Work

**KV Cache Compression** The optimization of KV cache memory efficiency in LLMs has garnered increasing attention as model sizes and context windows expand. Existing approaches primarily fall into three categories: dynamic token eviction[3, 11, 12], quantization[13, 14, 15], merging[16, 17, 18], and low-rank decomposition[19, 20, 21]. Previous eviction methods like SnapKV[3], PyramidKV[5], Ada-KV[22], HeadKV[6] dynamically prune tokens based on attention scores, but mainly focus on evicting tokens for prefilling stage. StreamingLLM[7] and H2O[4] are proposed for decoding. However, these general-purpose techniques often struggle with reasoning-intensive tasks, where aggressive eviction risks disrupting critical intermediate steps in CoT, and suffers from reasoning models' inherent redundency.

**Efficient Reasoning** Recent works in efficient reasoning focus on training the model to generate less CoT without sacrificing performance. [23, 24, 25] use RL optimization with length penalty rewards to encourage models to produce more concise chains-of-thought (CoT). [26, 27] employs variable-length CoT datasets to supervised fine-tune (SFT) the LLM to reduce token usage while preserving reasoning correctness. Both RL and SFT methods require additional training. [27, 28, 29] use test-time prompting to reduce generation length, but these methods may hurt the performance. As a KV cache compression work for reasoning models, R-KV is able to achieve lossless compression without extensive training and prompting.

# 7 Conclusion

We introduced R-KV, a novel decoding-time KV cache compression method tailored to the challenges of complex reasoning in large language models (LLMs). Reasoning models often generate long, redundant outputs that impose substantial memory and computational burdens during inference. R-KV addresses this by jointly scoring token importance and redundancy, enabling the retention of essential reasoning content while discarding repetitive or uninformative tokens. This dynamic and attention-guided strategy allows R-KV to preserve nearly full model performance using only 10–34% of the original KV cache—substantially outperforming prior compression methods.

Extensive throughput and efficiency analysis demonstrate that R-KV enables up to 13× larger batch sizes and over 9× speedup in long-sequence generation scenarios compared to FullKV, with particularly strong gains under constrained memory budgets. With its training-free and modelagnostic design, R-KV provides a scalable and deployment-ready solution for reasoning LLMs, especially in streamlining the rollout phase of reinforcement learning workflows.

# 8 Acknowledgement

Research reported in this publication was partially supported by the National Science Foundation under Award Number IIS-2449768. The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Science Foundation.

# References

- <span id="page-10-0"></span>[1] DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025.
- <span id="page-10-1"></span>[2] Mehdi Fatemi, Banafsheh Rafiee, Mingjie Tang, and Kartik Talamadupula. Concise reasoning via reinforcement learning, 2025.
- <span id="page-10-2"></span>[3] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, T sianle Cai, Patrick Lewis, and Deming Chen. Snapkv: Llm knows what you are looking for before generation, 2024.
- <span id="page-10-3"></span>[4] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, and Beidi Chen. H2o: Heavy-hitter oracle for efficient generative inference of large language models, 2023.
- <span id="page-10-4"></span>[5] Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong, Baobao Chang, Junjie Hu, et al. Pyramidkv: Dynamic kv cache compression based on pyramidal information funneling. *arXiv preprint arXiv:2406.02069*, 2024.
- <span id="page-10-5"></span>[6] Yu Fu, Zefan Cai, Abedelkadir Asi, Wayne Xiong, Yue Dong, and Wen Xiao. Not all heads matter: A head-level kv cache compression method with integrated retrieval and reasoning, 2024.
- <span id="page-10-6"></span>[7] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks, 2024.
- <span id="page-10-7"></span>[8] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. *arXiv preprint arXiv:2103.03874*, 2021.
- <span id="page-10-8"></span>[9] MAA. American invitational mathematics examination - aime. In *American Invitational Mathematics Examination - AIME 2024*, February 2024.

- <span id="page-11-0"></span>[10] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pondé de Oliveira Pinto, Jared Kaplan, Harrison Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. *CoRR*, abs/2107.03374, 2021.
- <span id="page-11-1"></span>[11] Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. Model tells you what to discard: Adaptive kv cache compression for llms. *arXiv preprint arXiv:2310.01801*, 2023.
- <span id="page-11-2"></span>[12] Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, and Anshumali Shrivastava. Scissorhands: Exploiting the persistence of importance hypothesis for llm kv cache compression at test time. *Advances in Neural Information Processing Systems*, 36:52342–52364, 2023.
- <span id="page-11-3"></span>[13] Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W Mahoney, Sophia Shao, Kurt Keutzer, and Amir Gholami. Kvquant: Towards 10 million context length llm inference with kv cache quantization. *Advances in Neural Information Processing Systems*, 37:1270–1303, 2024.
- <span id="page-11-4"></span>[14] Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. Kivi: A tuning-free asymmetric 2bit quantization for kv cache. *arXiv preprint arXiv:2402.02750*, 2024.
- <span id="page-11-5"></span>[15] Yuxuan Yue, Zhihang Yuan, Haojie Duanmu, Sifan Zhou, Jianlong Wu, and Liqiang Nie. Wkvquant: Quantizing weight and key/value cache for large language models gains more, 2024.
- <span id="page-11-6"></span>[16] Yuxin Zhang, Yuxuan Du, Gen Luo, Yunshan Zhong, Zhenyu Zhang, Shiwei Liu, and Rongrong Ji. Cam: Cache merging for memory-efficient llms inference. In *Forty-first International Conference on Machine Learning*, 2024.
- <span id="page-11-7"></span>[17] Jang-Hyun Kim, Junyoung Yeom, Sangdoo Yun, and Hyun Oh Song. Compressed context memory for online language model interaction. *arXiv preprint arXiv:2312.03414*, 2023.
- <span id="page-11-8"></span>[18] Piotr Nawrot, Adrian Łancucki, Marcin Chochowski, David Tarjan, and Edoardo M Ponti. ´ Dynamic memory compression: Retrofitting llms for accelerated inference. *arXiv preprint arXiv:2403.09636*, 2024.
- <span id="page-11-9"></span>[19] Hanshi Sun, Li-Wen Chang, Wenlei Bao, Size Zheng, Ningxin Zheng, Xin Liu, Harry Dong, Yuejie Chi, and Beidi Chen. Shadowkv: Kv cache in shadows for high-throughput long-context llm inference, 2025.
- <span id="page-11-10"></span>[20] Utkarsh Saxena, Gobinda Saha, Sakshi Choudhary, and Kaushik Roy. Eigen attention: Attention in low-rank space for kv cache compression, 2024.
- <span id="page-11-11"></span>[21] Rongzhi Zhang, Kuang Wang, Liyuan Liu, Shuohang Wang, Hao Cheng, Chao Zhang, and Yelong Shen. Lorc: Low-rank compression for llms kv cache with a progressive compression strategy, 2024.
- <span id="page-11-12"></span>[22] Yuan Feng, Junlin Lv, Yukun Cao, Xike Xie, and S Kevin Zhou. Ada-kv: Optimizing kv cache eviction by adaptive budget allocation for efficient llm inference. *arXiv preprint arXiv:2407.11550*, 2024.
- <span id="page-11-13"></span>[23] Chen Li, Nazhou Liu, and Kai Yang. Adaptive group policy optimization: Towards stable training and token-efficient reasoning, 2025.

- <span id="page-12-0"></span>[24] Junjie Yang, Ke Lin, and Xing Yu. Think when you need: Self-adaptive chain-of-thought learning, 2025.
- <span id="page-12-1"></span>[25] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, Chuning Tang, Congcong Wang, Dehao Zhang, Enming Yuan, Enzhe Lu, Fengxiang Tang, Flood Sung, Guangda Wei, Guokun Lai, Haiqing Guo, Han Zhu, Hao Ding, Hao Hu, Hao Yang, Hao Zhang, Haotian Yao, Haotian Zhao, Haoyu Lu, Haoze Li, Haozhen Yu, Hongcheng Gao, Huabin Zheng, Huan Yuan, Jia Chen, Jianhang Guo, Jianlin Su, Jianzhou Wang, Jie Zhao, Jin Zhang, Jingyuan Liu, Junjie Yan, Junyan Wu, Lidong Shi, Ling Ye, Longhui Yu, Mengnan Dong, Neo Zhang, Ningchen Ma, Qiwei Pan, Qucheng Gong, Shaowei Liu, Shengling Ma, Shupeng Wei, Sihan Cao, Siying Huang, Tao Jiang, Weihao Gao, Weimin Xiong, Weiran He, Weixiao Huang, Wenhao Wu, Wenyang He, Xianghui Wei, Xianqing Jia, Xingzhe Wu, Xinran Xu, Xinxing Zu, Xinyu Zhou, Xuehai Pan, Y. Charles, Yang Li, Yangyang Hu, Yangyang Liu, Yanru Chen, Yejie Wang, Yibo Liu, Yidao Qin, Yifeng Liu, Ying Yang, Yiping Bao, Yulun Du, Yuxin Wu, Yuzhi Wang, Zaida Zhou, Zhaoji Wang, Zhaowei Li, Zhen Zhu, Zheng Zhang, Zhexu Wang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Ziyao Xu, and Zonghan Yang. Kimi k1.5: Scaling reinforcement learning with llms, 2025.
- <span id="page-12-2"></span>[26] Yingqian Cui, Pengfei He, Jingying Zeng, Hui Liu, Xianfeng Tang, Zhenwei Dai, Yan Han, Chen Luo, Jing Huang, Zhen Li, Suhang Wang, Yue Xing, Jiliang Tang, and Qi He. Stepwise perplexity-guided refinement for efficient chain-of-thought reasoning in large language models, 2025.
- <span id="page-12-3"></span>[27] Tingxu Han, Zhenting Wang, Chunrong Fang, Shiyu Zhao, Shiqing Ma, and Zhenyu Chen. Token-budget-aware llm reasoning, 2025.
- <span id="page-12-4"></span>[28] Yule Liu, Jingyi Zheng, Zhen Sun, Zifan Peng, Wenhan Dong, Zeyang Sha, Shiwen Cui, Weiqiang Wang, and Xinlei He. Thought manipulation: External thought can be efficient for large reasoning models, 2025.
- <span id="page-12-5"></span>[29] Wenjie Ma, Jingxuan He, Charlie Snell, Tyler Griggs, Sewon Min, and Matei Zaharia. Reasoning models can be effective without thinking, 2025.
- <span id="page-12-6"></span>[30] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzmán, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao

Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vítor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. The llama 3 herd of models, 2024.

- <span id="page-14-1"></span>[31] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report, 2024.
- <span id="page-14-2"></span>[32] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. GQA: Training generalized multi-query transformer models from multi-head checkpoints. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 4895–4901, Singapore, December 2023. Association for Computational Linguistics.

# A Method

# A.1 Algorithm

The pseudo-code of the method is shown in Algorithm [1.](#page-15-0)

## A.2 Implementation Details

Max Pooling of Attention Weights Latest open-source LLMs [\[30,](#page-12-6) [31\]](#page-14-1) have widely adopted Grouped-Query Attention (GQA) [\[32\]](#page-14-2), where multiple query heads share a common pair of keyvalue heads to substantially reduce memory access overhead during inference. In key-value (KV) cache eviction strategies, it's thus often necessary to downscale attention scores from (Q\_head, seq\_len, seq\_len) to (KV\_head, seq\_len, seq\_len). While previous works such as SnapKV [\[3\]](#page-10-2) have predominantly employed mean pooling to aggregate attention scores across query head groups, we hypothesize that max pooling could better preserve the most critical tokens for each query head. Our empirical results demonstrate that max pooling leads to improved performance, and we adopt it for all main experiments.

# B Experiment

# B.1 Devices

We use NVIDIA A100 80G to finish all the experiments.

# <span id="page-14-0"></span>B.2 Main Results

See [Table 2.](#page-16-2)

<span id="page-15-0"></span>**Algorithm 1** R-KV:  $Q_{\text{obs}}$  are query states for  $\alpha$  observation tokens,  $K_{\text{full}}$ ,  $V_{\text{full}}$  are the full KV cache states of length  $L_{\text{full}}$ .

```
1: procedure R-KV((K_{\text{full}}, V_{\text{full}}), L_{\text{full}}, L_{\text{budget}}, Q_{\text{obs}}, \alpha, B_{\text{budget}}, B_{\text{buffer}}, T, \beta, \lambda, \epsilon, H, d_k)
                                                                                                                 ▷ Check if compression is triggered
  2:
              if L_{\text{full}} - L_{\text{budget}} < B_{\text{buffer}} then
  3:
                     return (\boldsymbol{K}_{\text{full}}, \boldsymbol{V}_{\text{full}})
  4:
  5:
               (\mathbf{K}_{\text{obs}}, \mathbf{V}_{\text{obs}}) \leftarrow \text{last } \alpha \text{ tokens of } (\mathbf{K}_{\text{full}}, \mathbf{V}_{\text{full}})
  6:
               (\mathbf{K}_{\text{cand}}, \mathbf{V}_{\text{cand}}) \leftarrow \text{first } (L_{\text{full}} - \alpha) \text{ tokens of } (\mathbf{K}_{\text{full}}, \mathbf{V}_{\text{full}})
              N_c \leftarrow L_{\text{full}} - \alpha
  7:

    Number of candidate tokens

  8:
              if N_c \leq B_{\text{budget}} then
  9:
                     return (K_{\text{full}}, V_{\text{full}})
                                                                                         ▶ Not enough candidates to prune beyond budget
10:
              for each head h = 0 \dots H - 1 do
11:
                     Compute attention matrix A^h \in \mathbb{R}^{\alpha \times N_c} using Q^h_{obs} and K^h_{cand} \triangleright Handles MHA/GQA as
12:
       per Eqs. (1)-(3) from text
                    \begin{array}{c} \textbf{for } k = 0 \dots N_c - 1 \textbf{ do} \\ I'_{k,h} \leftarrow \frac{1}{\alpha} \sum_{q=0}^{\alpha-1} (\boldsymbol{A}^h)_{qk} \end{array}
                                                                                                                             \triangleright For each candidate token k
13:
14:
                                                                                                     \triangleright q: observation token, k: candidate token
                     end for \{I_{k,h}\}_{k=0}^{N_c-1} \leftarrow \text{1D-Pooling}(\{I'_{k,h}\}_{k=0}^{N_c-1})
15:
16:
17:
               \begin{aligned} & \textbf{for} \text{ each head } h = 0 \dots H - 1 \textbf{ do} \\ & \boldsymbol{K}^h_{\text{norm}} \in \mathbb{R}^{N_c \times d_k}; \text{For } k = 0 \dots N_c - 1, \boldsymbol{K}^h_{\text{norm},k} \leftarrow \boldsymbol{K}^h_{cand,k} / (\|\boldsymbol{K}^h_{cand,k}\|_2 + \epsilon) \end{aligned}
18:
19:
                     \boldsymbol{S}^h \leftarrow \boldsymbol{K}^h_{\text{norm}} (\boldsymbol{K}^h_{\text{norm}})^{\top}
                                                                       ▷ Cosine Similarity Matrix Computation, similarity matrix
20:
       \boldsymbol{S}^h \in \mathbb{R}^{N_c \times N_c}
21:
                    for k = 0 ... N_c - 1 do
                                                                                                                               ▶ Prevent Self-Redundancy
                          (\mathbf{S}^h)_{kk} \leftarrow 0
22:
                     end for
23:
                     B^h_{uv} \leftarrow ((\mathbf{S}^h)_{uv} > T?1:0) for u,v \in \{0,\dots,N_c-1\} \triangleright Identify Highly Similar Pairs
24:
                     for u = 0 ... N_c - 1 do T_u^h \leftarrow \{v \mid B_{uv}^h = 1, v \in \{0, ..., N_c - 1\}\}
25:

26:
                           T_{u,\beta}^h \leftarrow \text{subset of } T_u^h \text{ with up to } \beta \text{ largest indices } v.
27:
                           for v' \in T_{u,\beta}^h do
28:
                                  (\mathbf{S}^h)_{u,v'} \leftarrow 0
29:
                                                                                                                                         \triangleright S^h is now modified
30:
                           end for
31:
                     end for
                    Let \bar{\mathbf{S}}^h \in \mathbb{R}^{N_c} where (\bar{\mathbf{S}}^h)_u \leftarrow \frac{1}{N_c} \sum_{v=0}^{N_c-1} (\mathbf{S}^h)_{uv}
32:
                     for u = 0 \dots N_c - 1 do
33:
                           R_{u,h} \leftarrow (\operatorname{softmax}(\bar{\mathbf{S}}^h))_u
34:
35:
36:
              end for
              for each head h = 0 \dots H - 1 do
37:
                     for k = 0 ... N_c - 1 do
38:
                           Score_{k,h} \leftarrow \lambda I_{k,h} - (1-\lambda)R_{k,h}
39:
40:
                     end for
              end for
41:
              Let AggScore \in \mathbb{R}^{N_c}
42:
              for k = 0 ... N_c - 1 do
43:
44:
                     AggScore_k \leftarrow mean_h(Score_{k,h})

    ▶ Aggregate scores across heads

45:
46:
              Idx_{sel} \leftarrow \text{indices of top-} B_{budget} \text{ tokens from } \{0, \dots, N_c - 1\} \text{ based on AggScore}
              K_{cand\_sel} \leftarrow K_{cand}[Idx_{sel}]; V_{cand\_sel} \leftarrow V_{cand}[Idx_{sel}]
47:
              K_{comp} \leftarrow \text{concatenate}(K_{cand\_sel}, K_{obs})
                                                                                                                                              ▷ Order might vary
48:
              V_{comp} \leftarrow \text{concatenate}(V_{cand\_sel}, V_{obs})
49:

    □ Update length for next cycle

50:
              L_{prev\_comp} \leftarrow B_{budget} + \alpha
              return (\boldsymbol{K}_{comp},\boldsymbol{V}_{comp})
51:
52: end procedure
```

<span id="page-16-2"></span>

| Model     | Benchmark | Method | 128   | 256   | 512   | 768   | 1 024 | 1 536 | 2 048 | 2 5 6 0 | 3 072 | 4 096 |
|-----------|-----------|--------|-------|-------|-------|-------|-------|-------|-------|---------|-------|-------|
| Llama3-8B | МАТН      | FullKV | 82.38 | 82.38 | 82.38 | 82.38 | 82.38 | 82.38 | 82.38 | _       | _     | _     |
|           |           | R-KV   | 51.08 | 67.39 | 76.92 | 80.21 | 81.34 | 82.34 | 82.65 | _       | _     | _     |
|           |           | SnapKV | 32.53 | 50.07 | 64.03 | 70.81 | 74.43 | 78.43 | 80.50 | -       | _     | -     |
|           | AIME24    | FullKV | 49.79 | 49.79 | 49.79 | 49.79 | 49.79 | 49.79 | 49.79 | 49.79   | 49.79 | _     |
|           |           | R-KV   | 0.42  | 10.21 | 29.48 | 40.31 | 45.26 | 51.56 | 52.29 | 53.85   | 53.13 | _     |
|           |           | SnapKV | 0.16  | 0.94  | 4.53  | 11.20 | 15.73 | 26.04 | 32.76 | 39.43   | 41.93 | -     |
| Qwen-14B  | МАТН      | FullKV | 94.58 | 94.58 | 94.58 | 94.58 | 94.58 | 94.58 | 94.58 | _       | _     | _     |
|           |           | R-KV   | 56.21 | 73.33 | 84.77 | 88.79 | 90.72 | 92.72 | 93.62 | -       | _     | -     |
|           |           | SnapKV | 26.32 | 43.93 | 77.93 | 82.52 | 86.63 | 90.86 | 92.73 | -       | -     | -     |
|           | AIME24    | FullKV | 65.68 | 65.68 | 65.68 | 65.68 | 65.68 | 65.68 | 65.68 | _       | 65.68 | 65.68 |
|           |           | R-KV   | 0.57  | 7.92  | 24.53 | 36.25 | 42.66 | 55.00 | 56.09 | _       | 64.32 | 67.45 |
|           |           | SnapKV | 0.26  | 2.86  | 12.86 | 16.30 | 25.00 | 36.41 | 46.56 | _       | 52.86 | 54.32 |

Table 2: Accuracy (%) of **Llama3-8B** and **Qwen-14B** on the MATH and AIME24 benchmarks under different memory-optimization methods across context lengths. "—" denotes configurations that were not evaluated.

# **C** Efficiency

## <span id="page-16-0"></span>C.1 Complexity Analysis of Memory and Computation

**Memory Saving** As discussed in §3.1, we need to allocate memory for the KV cache budget  $M_{\text{budget}} \in \mathbb{R}^{b \times B_{\text{budget}} \times N_{\text{layer}} \times N_{\text{head}} \times d}$  to retain  $B_{\text{budget}}$  KV cache tokens, and for the buffer  $M_{\text{buffer}} \in \mathbb{R}^{b \times B_{\text{buffer}} \times N_{\text{layer}} \times N_{\text{head}} \times d}$  to store  $B_{\text{buffer}}$  newly generated KV cache tokens during the generation of a text segment. Here, b is the batch size,  $N_{\text{layer}}$  is the number of Transformer layers,  $N_{\text{head}}$  is the number of attention heads, and d is the dimension of attention heads. In addition, we also need to allocate memory for the model weight  $M_{\theta}$ . During decoding, the previous query states are typically discarded by default, so we use a query cache to store the last  $\alpha$  tokens in the query state, consuming memory of  $M_{\alpha} \in \mathbb{R}^{b \times \alpha \times N_{\text{layer}} \times N_{\text{head}} \times d}$ . In summary, R-KV requires memory of  $M_{\text{total}} = M_{\theta} + M_{\text{budget}} + M_{\text{buffer}} + M_{\alpha}$  during generation. In comparison to FullKV without KV cache compression, generating  $B_{\text{full}}$  tokens requires memory of  $M_{\text{full}} \in \mathbb{R}^{b \times B_{\text{full}} \times N_{\text{layer}} \times N_{\text{head}} \times D_{\text{head}}}$  to retain  $B_{\text{full}}$  KV tokens, and memory of the model weight  $M_{0}$ . Therefore, the memory saved by our method w.r.t. FullKV is:  $M_{\text{saving}} = M_{\text{full}} - M_{\text{budget}} - M_{\text{buffer}} - M_{\alpha}$ .

Computation Overhead The computational complexity of importance scoring (See §3.2) is  $O(\alpha B_{\mathrm{budget}})$  while redundancy estimation (see §3.3) has complexity  $O(B_{\mathrm{budget}}^2)$ . Thus, the total overhead incurred during each generation segment is  $O(\alpha B_{\mathrm{budget}} + B_{\mathrm{budget}}^2)$ . The generation complexity without KV cache compression is  $O(B_{\mathrm{full}}B_{\mathrm{buffer}})$ , whereas the complexity with KV cache compression is  $O((B_{\mathrm{budget}} + B_{\mathrm{buffer}})B_{\mathrm{buffer}})$ . For reasoning models,  $B_{\mathrm{full}}$  tends to be large because of the long generation length, and using a relatively small  $B_{\mathrm{budget}}$  value can efficiently reduce computation cost. The effectiveness of this approach depends on depends on whether the speedup gained by attending over a reduced KV cache outweighs the overhead of computing the compression scores—i.e., the combined cost of importance and redundancy scores,  $(O(\alpha B_{\mathrm{budget}}) + O(B_{\mathrm{budget}}^2))$ .

## <span id="page-16-1"></span>C.2 Detailed Analysis of Throughput Results

We analyze the end-to-end throughput from two perspectives: ratio budget and fixed budget.

**Ratio Budget:** section 4.2 indicates that for DeepSeek-R1-Distill-Llama-8B, lossless compression (i.e., model performance equivalent to no KV compression) is achievable when the KV budget ratio, relative to the output length, is between 10% and 34%. For DeepSeek-R1-Distill-Qwen-14B, this range for lossless compression is 25% to 54% of the output length. Consequentlywe investigated the maximum achievable batch size and corresponding throughput for R-KV at compression ratios of 10%, 34%, and 54%, comparing these against the maximum batch size and throughput of FullKV using DeepSeek-R1-Distill-Llama-8B. In 8K sequence length setting, at a 54% compression ratio, R-KV allows for a batch size  $1.7 \times larger$  than FullKV, resulting in  $1.5 \times the$  throughput. At a  $10\% to the compression ratio, R-KV achieves a <math>1.7 \times the compression$  ratio, at  $1.5 \times the$  throughput compared to FullKV. For a  $1.5 \times the$  sequence length setting, at  $1.5 \times the$  compression, the batch size is  $1.5 \times the$  compression, the batch size is  $1.5 \times the$  compression, the batch size is  $1.5 \times the$  compression, the batch size is  $1.5 \times the$  compression, the batch size is  $1.5 \times the$  compression, the batch size is  $1.5 \times the$  compression, the batch size is  $1.5 \times the$  compression.

<span id="page-17-0"></span>

| Gen. Length | Method | Budget              | Mem. Saving (%) | Batch     | Throughput (tok/s) | Tokens Gen. | Dec. Time (s) |
|-------------|--------|---------------------|-----------------|-----------|--------------------|-------------|---------------|
|             | FullKV | -                   | -               | 1         | 75.44              | 8 094       | 107.30        |
|             |        | -                   | -               | 62 (max)  | 849.13             | 501 828     | 590.99        |
|             | SnapKV | Fixed - 1024        | 87.50           | 1         | 81.26              | 8 094       | 99.60         |
| 8K          |        | Fixed - 1024        | 87.50           | 402 (max) | 3 253.93           | 3 253 788   | 999.96        |
|             |        | Fixed - 1536        | 81.25           | 287 (max) | 2 525.25           | 2 322 978   | 919.90        |
|             |        | Fixed - 3072        | 62.50           | 150 (max) | 1 527.67           | 1 214 100   | 794.74        |
|             |        | Ratio - 10% - 819   | 90.00           | 479 (max) | 3 808.81           | 3 877 026   | 1017.91       |
|             |        | Ratio - 34% - 2785  | 66.00           | 167 (max) | 1 625.46           | 1 351 698   | 831.58        |
|             |        | Ratio - 54% - 4423  | 46.00           | 105 (max) | 1 269.68           | 849 870     | 669.36        |
|             | R-KV   | Fixed - 1024        | 87.50           | 1         | 80.46              | 8 094       | 100.60        |
|             |        | Fixed - 1024        | 87.50           | 402 (max) | 3 251.52           | 3 253 788   | 1 000.70      |
|             |        | Fixed - 1536        | 81.25           | 287 (max) | 2 525.75           | 6 546 972   | 919.72        |
|             |        | Fixed - 3072        | 62.50           | 150 (max) | 1 520.99           | 1 214 100   | 798.23        |
|             |        | Ratio - 10% - 819   | 90.00           | 479 (max) | 3 809.15           | 3 877 026   | 1017.82       |
|             |        | Ratio - 34% - 2785  | 66.00           | 167 (max) | 1 608.01           | 1 351 698   | 840.61        |
|             |        | Ratio - 54% - 4423  | 46.00           | 105 (max) | 1 257.83           | 849 870     | 675.66        |
|             | FullKV | _                   | _               | 1         | 69.41              | 16 286      | 234.65        |
| 16K         |        | -                   | -               | 30 (max)  | 347.03             | 488 580     | 1 407.89      |
|             | SnapKV | Fixed - 1024        | 87.50           | 1         | 81.03              | 16 286      | 200.99        |
|             |        | Fixed - 1024        | 87.50           | 402 (max) | 3 202.17           | 6 546 972   | 2 044.54      |
|             |        | Fixed - 1536        | 81.25           | 287 (max) | 2 449.02           | 4 674 082   | 1 908.56      |
|             |        | Fixed - 3072        | 81.25           | 150 (max) | 1 413.84           | 2 442 900   | 1 727.84      |
|             |        | Ratio - 10% - 1638  | 90.00           | 271 (max) | 2 306.26           | 4413506     | 1913.71       |
|             |        | Ratio - 34% - 5 570 | 66.00           | 82 (max)  | 798.42             | 1 335 452   | 1 672.61      |
|             |        | Ratio - 54% - 8 847 | 46.00           | 46 (max)  | 586.43             | 749 156     | 1 277.48      |
|             | R-KV   | Fixed - 1024        | 93.75           | 1         | 80.95              | 16 286      | 201.18        |
|             |        | Fixed - 1024        | 93.75           | 402 (max) | 3 188.82           | 6 546 972   | 2 053.10      |
|             |        | Fixed - 1536        | 90.63           | 287 (max) | 2 447.61           | 4 674 082   | 1 909.65      |
|             |        | Fixed - 3072        | 81.25           | 150 (max) | 1 406.28           | 2 442 900   | 1 737.13      |
|             |        | Ratio - 10% - 1638  | 90.00           | 271 (max) | 2 300.28           | 4413506     | 1918.68       |
|             |        | Ratio - 34% - 5570  | 66.00           | 82 (max)  | 797.43             | 1 335 452   | 1 674.70      |
|             |        | Ratio - 54% - 8847  | 46.00           | 46 (max)  | 584.77             | 749 156     | 1 281.12      |

Table 3: Memory-saving, throughput, and decoding-time comparison for LLAMA3-8B under various generation lengths and KV-cache compression budgets.

that of FullKV, and the throughput is  $1.7 \times$  higher. At 10% compression, R-KV supports a  $9 \times$  larger batch size, delivering  $6.6 \times$  the throughput. We observe that for smaller batch sizes (e.g., less than 128), throughput scales nearly linearly with increasing batch size. However, for larger batch sizes this linear scaling diminishes as inference on the NVIDIA A100 GPU becomes compute-bound.

**Fixed Budget:** We also conducted an analysis under a fixed KV cache budget. With an output length of 8K and a fixed budget  $B_{\rm budget} = 1024$ , R-KV enables a batch size  $6.48 \times$  larger than FullKV, yielding  $3.8 \times$  the throughput. At  $B_{\rm budget} = 1536$ , the batch size is  $4.6 \times$  larger, and throughput is  $3 \times$  that of FullKV. For an output length of 16K and  $B_{\rm budget} = 1024$ , R-KV achieves a  $13.4 \times$  increase in batch size and a  $9.19 \times$  increase in throughput. With  $B_{\rm budget} = 1536$ , the batch size is  $9.6 \times$  larger, and throughput is  $7.1 \times$  higher. In the fixed budget scenario, the advantage of R-KV becomes more pronounced with longer generation lengths. This is because the KV cache size for R-KV under a fixed budget does not increase with the sequence length, unlike FullKV where the memory footprint grows linearly with the generation length, thus more severely limiting its maximum batch size.

#### C.3 Results

Full results could be found at Table 3. While R-KV incurs a minor computational overhead for redundancy estimation compared with SnapKV, this results in a throughput that is only slightly lower, with a negligible difference of less than 1%.

### **D** Limitations

One limitation of our proposed KV cache compression method is its current compatibility with certain advanced attention mechanisms, such as paged attention. Adapting our compression technique to seamlessly integrate with such mechanisms presents a non-trivial challenge and may require further investigation. Additionally, the implementation of KV cache compression within existing serving frameworks can encounter practical difficulties, particularly if these frameworks lack native support or flexible interfaces for KV cache compression. In serving frameworks that do not offer specialized KV cache compression interfaces, the performance benefits of our method might be less pronounced. Without such interfaces, implementing KV cache compression may necessitate reallocating memory

to store the compressed KV cache and subsequently deallocating the memory used for the original, uncompressed cache. This process of memory reallocation can introduce significant overhead, potentially offsetting some of the acceleration gains. In contrast, serving frameworks equipped with dedicated KV compression interfaces can handle these operations much more efficiently, avoiding such costly memory management tasks.