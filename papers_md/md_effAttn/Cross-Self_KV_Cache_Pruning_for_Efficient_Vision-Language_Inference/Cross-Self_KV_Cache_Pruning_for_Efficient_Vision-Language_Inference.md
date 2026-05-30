# <span id="page-0-1"></span>Cross-Self KV Cache Pruning for Efficient Vision-Language Inference

Xiaohuan Pei Tao Huang Chang Xu School of Computer Science, Faculty of Engineering, The University of Sydney

{xiaohuan.pei, t.huang, c.xu}@sydney.edu.au

## Abstract

*KV cache pruning has emerged as a promising technique for reducing memory and computation costs in long-context auto-regressive generation. Existing methods for visionlanguage models (VLMs) typically rely on self-attention scores from large language models (LLMs) to identify and prune irrelevant tokens. However, these approaches overlook the inherent distributional discrepancies between modalities, often leading to inaccurate token importance estimation and the over-pruning of critical visual tokens. To address this, we propose decomposing attention scores into intra-modality attention (within the same modality) and inter-modality attention (across modalities), enabling more precise KV cache pruning by independently managing these distinct attention types. Additionally, we introduce an n-softmax function to counteract distribution shifts caused by pruning, preserving the original smoothness of attention scores and ensuring stable performance. Our final training-free method, Cross-Self Pruning (CSP), achieves competitive performance compared to models with full KV caches while significantly outperforming previous pruning methods. Extensive evaluations on MileBench, a benchmark encompassing 29 multimodal datasets, demonstrate CSP's effectiveness, achieving up to a 41% performance improvement on challenging tasks like conversational embodied dialogue while reducing the KV cache budget by 13.6%. The code is available at [TerryPei/CSP.](https://github.com/TerryPei/CSP)*

## 1. Introduction

The success of large language models (LLMs) [\[1,](#page-8-0) [4,](#page-8-1) [6,](#page-8-2) [38,](#page-9-0) [41,](#page-9-1) [44\]](#page-9-2) has propelled the advancement of large vision-language models (VLMs) [\[5,](#page-8-3) [10,](#page-8-4) [22,](#page-8-5) [24,](#page-8-6) [26,](#page-9-3) [37,](#page-9-4) [42\]](#page-9-5), enabling powerful integration and reasoning over multimodal inputs that combine both text and visual tokens. Unlike single-modal contexts, multimodal samples often comprise numerous images alongside text instructions, creating extended context lengths that challenge efficient inference.

To address the challenges of long-context generation, KV caching has become a standard technique, where previously computed keys and values in the attention layers are stored in

<span id="page-0-0"></span>![](_page_0_Figure_9.jpeg)

Figure 1. Distribution gap between self-attention and crossattention during the decoding process in VLM tasks: (a) Kernel Density Estimation (KDE) of the attention weight distributions, and (b) Jensen-Shannon (JS) divergence scores between cross-attention and self-attention across all layers.

memory for reuse during subsequent generation steps. However, this approach still faces significant memory limitations, particularly for GPU and machine memory constraints. Recent works [\[8,](#page-8-7) [21,](#page-8-8) [23,](#page-8-9) [31,](#page-9-6) [46\]](#page-9-7) have explored pruning unimportant tokens within the KV cache to alleviate memory demands, primarily by leveraging attention scores. Methods such as SnapKV [\[21\]](#page-8-8) and H2O [\[46\]](#page-9-7) apply this strategy to vision-language modeling (VLM) tasks by treating visual and text tokens uniformly across long sequences during pruning. Unfortunately, these methods rely on original attention scores that mix different modalities, potentially leading to suboptimal pruning outcomes.

In this paper, we identify a critical limitation in previous KV cache pruning methods: the distributional discrepancy between visual and textual modalities leads to inaccurate token importance estimation. Specifically, as illustrated in Figure [1a,](#page-0-0) self-attention scores (within a single modality) and cross-attention scores (across modalities) exhibit distinct and non-overlapping distributions. This divergence highlights that each attention type captures unique aspects of the input space, reflecting modality-specific priorities during decoding. Furthermore, as shown in Figure [1b,](#page-0-0) the Jensen-Shannon (JS) divergence between cross-attention and self-attention distributions reveals substantial variation across layers in LLaVA-7b. Relying solely on these mixed distributions for pruning introduces a selection bias: the pruned tokens tend

<span id="page-1-0"></span>to cluster within a single region or modality. This imbalance disrupts cross-modal interactions, ultimately degrading the model's performance.

Inspired by these observations, we hypothesize that independently selecting important tokens from distinct distributions can provide a more balanced and effective pruning strategy. To achieve this, we propose decomposing the attention matrix into intra-modality attention (within the same modality) and inter-modality attention (across different modalities), and performing token selection independently within each category. This approach ensures accurate preservation of critical tokens from both modalities while maintaining the efficiency and simplicity of prior methods. Additionally, we observe that pruning inherently shifts the distribution of attention scores, leading to degraded performance. To mitigate this, we introduce an n-softmax function that smooths the post-pruning distribution, effectively restoring the original smoothness of the attention scores and improving overall performance.

Our final method, Cross-Self Pruning (CSP), effectively addresses the challenges of KV cache pruning by leveraging independent intra-modality and inter-modality token selection along with n-softmax smoothing. CSP achieves a superior balance between performance and memory efficiency, significantly reducing the KV cache size while preserving or even enhancing model accuracy. We conduct extensive experiments on various VLMs, including LLaVA-v1.5-7b [24], InternVL [10], and MobileVLM [12]. The results demonstrate that CSP consistently outperforms existing methods like SnapKV [21] and H2O [46], achieving up to a 41% improvement in performance on complex tasks such as conversational embodied dialogue [33], while reducing the KV cache budget by up to 13.6%.

#### 2. Related Work

#### 2.1. Vision-language models

Following the remarkable success of large language models (LLMs) [1, 4, 6, 38, 41, 44], recent research has focused on generative large vision-language models (VLMs) [5, 10, 22, 24, 26, 37, 42] to enhance multimodal comprehension and generation by leveraging the generalization capabilities of LLMs. Using the multi-modal pre-trained visual foundation models such as CLIP [30] as the visual encoder, existing methods commonly utilize extensive imagetext data to align the visual encoder with LLMs, enabling the LLM to process and interpret visual inputs. For example, Flamingo [3] incorporates visual features into the LLM through gated attention, while LLaVA [26] connects the vision encoder and LLM via MLPs, demonstrating strong performance in multimodal dialogues.

#### 2.2. KV cache optimization

Recent advancements in large language models (LLMs) have achieved notable success in optimizing KV cache for efficient long-context processing. Existing work on KV cache optimization [2, 8, 14, 21, 23, 31, 46] primarily utilizes attention scores to retain important tokens and improve memory efficiency in long-context processing. For example, SnapKV [21] introduces a technique for identifying attention allocation patterns and compressing the KV cache by pooling key tokens. H2O [46] presents a dynamic eviction policy that balances recent and frequently accessed tokens, identifying heavy hitters based on attention scores alone. ReCo [31] improves existing eviction strategies through refined importance scoring and structured eviction scopes. Keyformer [2] introduces gumbel softmax to relieve the impact of pruning tokens. More recent works [39, 45] have shifted towards adapting KV cache inference to multimodal contexts, where modality-specific properties are considered. For instance, LOOK-M [39] applies modality awareness to selectively evict image tokens and merge them with text tokens, prioritizing the retention of text-based KV pairs.

#### 3. Method

Our approach decomposes the attention scores into intramodality and inter-modality attention. We apply top-k selection within each region, ranking the tokens by summing attention scores along the query dimension for each key. Finally, we concatenate the selected tokens with recent tokens to form the key and value cache.

#### 3.1. Preliminary

The auto-regressive generation of text yields a multi-step generation process. At each step i, the current token  $x_i$  is predicted by the LLM based on the input of prompt and previously-generated tokens  $\{x_j\}_{j=1}^{i-1}$ .

For reducing the computational cost and avoiding duplicated computations, current inference usually adopts KV cache technique, where the keys  $(K_j)$  and values  $(V_j)$  in self-attention of each previous tokens  $x_j$  are cached in memory and reused in subsequent steps.

However, when the context length is long, the storage of KV values still poses significant challenges in memory size and memory access speed. Therefore, to reduce the memory cost and run LLMs on resource-limited devices, KV cache pruning methods[2, 8, 14, 21, 23, 31, 46] are proposed to remove the less-important tokens.

Generally, the method contains two main components: (1) an importance estimation function f to measure the importance of each token  $x_j$ ; (2) a selecting strategy to keep important tokens in KV and remove the rest based on their importance. Formally, with KV sequences to be cached, and T denotes the maximum cache length, it first measures the

<span id="page-2-2"></span>![](_page_2_Figure_0.jpeg)

Figure 2. This illustration depicts the Cross-Self Pruning (CSP) KV cache process. The input sequence {#image, #text,#image, #text, ...} is projected onto query and key representations across multiple modalities. The n-softmax attention weights serve as the selection function, which is decomposed into intra- and cross-modality. Summation is performed along the query axis within each region, and top-k keys are selected along the key dimension to retain tokens for pruning.

importance scores of each token using f, then generate the selective mask M, M<sup>i</sup> ∈ {0, 1}, ∀i = {1, . . . , T} to select the tokens with top-T highest scores, where the tokens with zero values in M are omitted.

### 3.2. Cross-self pruning

Previous KV cache pruning methods [\[8,](#page-8-7) [21,](#page-8-8) [23,](#page-8-9) [31,](#page-9-6) [39,](#page-9-10) [46\]](#page-9-7) usually use the self-attention scores to indicate the importance of each token, as the attention scores determine the contributions of tokens to the attention output. Nevertheless, some recent works [\[39,](#page-9-10) [45\]](#page-9-11) simply adopt the same strategy for pruning vision-language hybrid tokens, neglecting the distribution gap between different modalities. As validated in Figure [1,](#page-0-0) the discrepancy between attention scores within the same modality and different modalities are significant, resulting in overestimate or underestimate of the importance when considering both modalities together[1](#page-2-0) .

Motivated by this, for a more accurate estimation for

VLMs, we aim to decompose the attention scores into two parts: intra-modality attention and inter-modality attention. The intra-modality attention denotes the attention scores between tokens within the same modality itself, and intermodality attention is the ones across different modalities.

Mathematically, for an attention matrix A ∈ [0, 1]L×<sup>L</sup> (we average over the head axis if there are multiple heads in attention), where L = L<sup>t</sup> + L<sup>v</sup> is the text-visual sequence length, we denote Ast ∈ [0, 1]<sup>L</sup>t×L<sup>t</sup> and Asv ∈ [0, 1]<sup>L</sup>v×L<sup>v</sup> as the self-attention scores between text tokens and visual tokens, Act ∈ [0, 1]<sup>L</sup>v×L<sup>t</sup> and A(cv) ∈ [0, 1]<sup>L</sup>t×L<sup>v</sup> as the visual-text (text as key) and text-visual cross-attention scores, respectively. Then, we can sum over the query axis to indicate the intra and inter importance of all the keys:

<span id="page-2-1"></span>
$$A^{s} = \sum_{k=1}^{L_{t}} A_{k}^{st} \oplus \sum_{k=1}^{L_{v}} A_{k}^{sv}, \quad A^{c} = \sum_{k=1}^{L_{t}} A_{k}^{ct} \oplus \sum_{k=1}^{L_{v}} A_{k}^{cv},$$
 (1)

where ⊕ denotes concatenation. Then the selective masks M(s) and M(c) of each matrix A<sup>∗</sup> are obtained by the top-K

<span id="page-2-0"></span><sup>1</sup>Specifically, we find that on VLMs such as LLaVA, the attention scores of text tokens are usually larger than that of visual tokens, potentially leading to the loss of important visual information after KV pruning.

<span id="page-3-2"></span>of the scores:

$$M^* = \{M_i\}_{i=1}^L, \quad \text{with } M_i^* = \begin{cases} 0, & i \notin \text{top-K}(A^*) \\ 1, & i \in \text{top-K}(A^*) \end{cases}. \tag{2}$$

We set different K values  $K^s$  and  $K^v$  for intra attention and inter attention, which we find better and will be discussed in experiment section.

Finally, the mask for selecting tokens is from the tokens both important and selected by  $M^s$  and  $M^c$ :

$$M = M^s \wedge M^c. \tag{3}$$

Also note that for better efficiency and accuracy, we trim the attention matrix by a number O of the most recent query tokens as an observation window and a number R of the previous key tokens, i.e., A[-O:,:-R], to reflect the actual needs in the generation of recent context. Figure 1 presents the cross-self selection approach. The green part refers to intra-modality, and the orange part to cross-modality. For both regions, we sum along the query axis, rank the score indices, and treat these as the selected key and value indices to retain. Due to potential overlap between the selected candidates  $M^s$  and  $M^c$ , the cache budget is significantly optimized across most modality scenarios, resulting in substantial savings in KV cache space. Finally, the combination of cross-self pruned tokens with recent tokens constitutes the optimized KV cache tokens:

$$K = (K \odot M) \oplus K[-R :]$$

$$V = (V \odot M) \oplus V[-R :],$$
(4)

where  $\odot$  is element wise operation. The pruned key and value are stored in the cache for later inference decoding. Algorithm 1 presents the cross-self decoding procedure during inference.

### 3.3. Smoothness recovery of attention scores

By using KV cache pruning, we can obtain reduced KV sequences for smaller costs. However, we find there is a sharpness-shift issue in the new pruned attention scores. Let us first consider the original computation of attention scores:

$$A = softmax(O), \text{ with } O = (\frac{QK^T}{\sqrt{d}}),$$
 (5)

where d is a factor for stabilizing the values.

Comparing the attention score  $A_i$  between original and post-pruning ones, the difference occurs in the denominator of softmax, i.e.,

$$\frac{e^{(O_i)}}{\sum_{j \in I^+} e^{(O_j)} + \sum_{j \in I^-} e^{(O_j)}} \to \frac{e^{(O_i)}}{\sum_{j \in I^+} e^{(O_j)}}, \quad (6)$$

where  $I^+$  and  $I^-$  denote the indices of tokens to be kept and pruned, respectively. It is clear to see that the  $A_i$  before

### <span id="page-3-0"></span>Algorithm 1 Cross-Self Pruning (CSP) Procedure.

```
1: Input: O \in \mathbb{R}^{H \times L_q \times L_k}, current keys and values in the cache
     K, Q, budget T, recent size R, observation window Q
 2: for iteration i \leftarrow 1 to N do
          L_k \leftarrow K

 3:
         if L_k < T then:
 4:
 5:
              Return: K, V
 6:
 7:
          A \leftarrow n\text{-}Softmax(O)
                                                 ⊳ Select function (Eq. 7).
          A^{st}, A^{sv}, A^{ct}, A^{cv} \leftarrow A
 8:

          A^s, A^c \leftarrow A^{st}, A^{sv}, A^{ct}, A^{cv}
 9:
10:
                ▶ Acquire cross-attention and intra-attention (Eq. 1).
          M^s \leftarrow \text{Topk}(A^s), M^c \leftarrow \text{Topk}(A^c)
11:
12:
                  ▶ Get Top-K masks from intra- and cross-modality.
13:
          M \leftarrow M^s \wedge M^c \triangleright Tokens that important both from intra-
     and cross-modality
          K = (K \odot M) \oplus K[-R:],
14:
          V = (V \odot M) \oplus V[-R:]
15:
                      ▷ Concatenate pruned tokens and recent tokens.
16:
17:
                                                ▶ Pruned Keys and Values
18: end for
```

pruning (left ) is smaller than the one after pruning (right), indicating that the original A is smoother. These changes would affect the attention outputs by overly enlarging the contributions of tokens with high attention scores, and therefore weaken the performance.

To address this issue, we propose a simple and effective function to recover the original smoothness of attention scores, which we call n-softmax, and  $A_i$  becomes:

<span id="page-3-1"></span>
$$A_i = n\text{-softmax}(O_i) = \frac{e^{(O_i)}}{n + \sum_{j \in I^+} e^{(O_j)}},$$
 (7)

where n is a hyper-parameter to control the smoothness of the distribution, we set n=1 in all experiments.

#### 4. Experiments

#### 4.1. Benchmark

We evaluate our method on MileBench [34] (MLLM), a benchmark specifically designed to assess long-context capabilities in multimodal language models. It collects widely-used datasets, providing a versatile and realistic foundation for evaluating model inference performance through two evaluation sets, *diagnostic* and *realistic-crafted*, which systematically measure inference performance in multimodality scenarios. The tasks in the benchmark organized with:

- Temporal Multi-Image Tasks (T1-T4): Temporal tasks involve understanding and predicting sequential events across images, and the methods in handling in action recognition, object tracking and spatial navigation.
- Semantic Multi-Image Tasks (S1-S4): Semantic tasks focus on interpreting multimodal information, requiring

<span id="page-4-1"></span><span id="page-4-0"></span>

| Method                     | T-1  | T-2  | T-3  | T-4            | S-1  | S-2  | S-3  | S-4  | S-5  | NH   | IR  |
|----------------------------|------|------|------|----------------|------|------|------|------|------|------|-----|
|                            |      |      |      | LLaVA-v1.5-7b  |      |      |      |      |      |      |     |
| Full Cache                 | 39.8 | 46.0 | 32.2 | 37.8           | 56.9 | 33.3 | 12.6 | 23.4 | 60.5 | 4.7  | 4.3 |
| H2O [46]                   | 39.5 | 46.6 | 31.8 | 38.5           | 55.0 | 33.0 | 13.0 | 23.0 | 60.0 | 1.4  | 3.7 |
| SnapKV [21]                | 39.6 | 46.0 | 31.5 | 40.6           | 54.6 | 33.6 | 13.0 | 20.0 | 60.0 | 1.4  | 3.7 |
| ReCo [31]                  | 39.7 | 46.1 | 31.8 | 38.5           | 55.0 | 33.0 | 12.6 | 22.8 | 60.0 | 4.7  | 4.3 |
| LOOK-M (A-Merge) [39]      | 39.7 | 46.1 | 32.2 | 39.1           | 54.9 | 34.0 | 12.4 | 21.4 | 60.5 | 1.6  | 3.7 |
| LOOK-M (W-Merge) [39]      | 39.6 | 46.1 | 31.8 | 39.1           | 55.1 | 34.0 | 13.2 | 24.0 | 60.5 | 1.4  | 3.7 |
| LOOK-M (P-Merge) [39]      | 39.7 | 46.1 | 32.5 | 39.9           | 57.0 | 34.0 | 12.8 | 23.9 | 60.5 | 5.3  | 3.8 |
| LOOK-M (TP+A-Merge) [39]   | 39.7 | 46.1 | 32.0 | 39.0           | 56.5 | 33.8 | 12.9 | 23.1 | 60.5 | 5.1  | 3.5 |
| LOOK-M (TP+W-Merge) [39]   | 39.8 | 46.1 | 32.5 | 39.9           | 57.0 | 34.0 | 12.8 | 23.9 | 60.5 | 5.3  | 3.8 |
| LOOK-M (TP+P-Merge) [39]   | 39.8 | 46.1 | 32.5 | 39.9           | 57.0 | 34.0 | 12.8 | 23.9 | 60.5 | 5.3  | 3.8 |
| CSP (Ours)                 | 39.9 | 46.8 | 32.5 | 41.6           | 57.5 | 34.1 | 13.7 | 27.8 | 61.0 | 1.4  | 6.3 |
|                            |      |      |      | LLaVA-v1.5-13b |      |      |      |      |      |      |     |
| Full Cache                 | 40.0 | 46.0 | 32.2 | 37.8           | 56.9 | 33.3 | 12.6 | 23.4 | 60.5 | 4.7  | 4.3 |
| H2O [46]                   | 39.5 | 45.9 | 30.4 | 47.9           | 64.1 | 48.7 | 13.9 | 25.1 | 59.7 | 3.6  | 0.0 |
| SnapKV [21]                | 39.6 | 46.0 | 30.6 | 47.8           | 64.2 | 48.2 | 13.4 | 22.9 | 59.8 | 4.2  | 1.0 |
| ReCo [31]                  | 39.7 | 45.9 | 30.5 | 48.0           | 64.3 | 48.3 | 13.8 | 24.9 | 59.7 | 3.5  | 0.0 |
| LOOK-M (A-Merge) [39]      | 39.7 | 46.1 | 30.7 | 48.0           | 64.6 | 48.0 | 13.3 | 22.1 | 59.8 | 4.6  | 1.0 |
| LOOK-M (W-Merge) [39]      | 39.6 | 46.1 | 30.6 | 47.9           | 64.5 | 48.4 | 13.4 | 23.4 | 59.9 | 4.7  | 1.0 |
| LOOK-M (P-Merge) [39]      | 39.7 | 46.0 | 30.6 | 48.0           | 64.6 | 48.0 | 13.3 | 25.7 | 59.8 | 5.8  | 1.0 |
| LOOK-M (TP + A-Merge) [39] | 39.7 | 46.2 | 30.7 | 48.0           | 65.4 | 48.3 | 13.7 | 26.6 | 60.0 | 11.2 | 1.0 |
| LOOK-M (TP + W-Merge) [39] | 39.8 | 46.1 | 30.8 | 48.1           | 64.8 | 48.2 | 13.9 | 26.9 | 60.0 | 11.4 | 1.0 |
| LOOK-M (TP + P-Merge) [39] | 39.8 | 46.2 | 30.8 | 48.1           | 65.2 | 48.5 | 14.1 | 26.6 | 60.0 | 11.7 | 1.0 |
| CSP (Ours)                 | 40.0 | 46.5 | 32.4 | 49.0           | 65.4 | 48.3 | 14.4 | 27.0 | 60.5 | 3.1  | 9.6 |

Table 1. Comparison of various KV cache management methods on the multiple multimodal tasks. Tasks are grouped as follows: Temporal Multi-Image Tasks (T1-T4), Semantic Multi-Image Tasks (S1-S4), Needle in a Haystack (NH), and Image Retrieval (IR). For fair comparison, we set the same KV cache size for all cache methods. Scores are calculated as the average performance across datasets within each subtask. For fair comparison, we set the same KV cache size for all inference methods.

inference methods to reason the knowledge-based QA, text-rich image analysis, visual relationship inference, dialogue understanding, and spatial reasoning.

- Needle in a Haystack (NH): Retrieval tasks designed to find a preset password from a long context, which test kv cache inference methods in precise password extraction.
- Image Retrieval (IR): Focused on identifying target images from candidates, which assess KV cache methods' effectiveness in perceptual and conceptual recognition.

Details of these challenging and comprehensive multimodal tasks, which include the corresponding datasets and evaluation metrics, are provided in the Appendix.

## 4.2. Baselines

We compare our kv cache method with previous mainstream baselines include SnapKV [\[21\]](#page-8-8), H2O [\[46\]](#page-9-7), ReCo [\[31\]](#page-9-6), and LOOK-M [\[39\]](#page-9-10), each offering unique strategies for managing KV cache in long-context scenarios. SnapKV [\[21\]](#page-8-8) introduces a method for intelligently identifying attention allocation pat-

terns and compressing the KV cache by pooling essential tokens for extended sequences. H2O [\[46\]](#page-9-7) presents a KV cache eviction policy that dynamically balances recent and frequently accessed tokens, identifying "heavy hitters" solely based on attention scores. ReCo [\[31\]](#page-9-6) focuses on enhancing the efficacy of existing eviction policies through refined importance score calculations and carefully constructed eviction scopes, proposing a robust cache omission policy rooted in temporal attention scores and robustness measures. The LOOK-M family [\[39\]](#page-9-10) considers modality awareness to evict image tokens and merge them with text tokens. This method prioritizes the retention of text-based KV pairs while evicting image-based KV pairs. The merging strategies include Average Merging (A), which computes the mean value of tokens within the similarity matrix for merging; Weighted Merging (W), which dynamically adjusts token weights based on similarity scores for adaptive merging; and Pivotal Merging (P), which enhances the importance of key conserved tokens during the merging process.

### <span id="page-5-1"></span>4.3. Setup

We employed LLaVA-v1.5-7b [\[25\]](#page-9-13) on RTX-4090 GPUs with *flash-attn-2.4.3post1* and LLaVA-v1.5-13b [\[25\]](#page-9-13) on A100 GPUs with *flash-attn-2.6.3* [2](#page-5-0) to conduct our experiments. To maintain consistency in generation, we set the sampling method to deterministic with a fixed temperature of 0, and the maximum context length was configured to 4096 tokens. Batch sizes were dynamically set based on dataset characteristics to balance computational load and memory constraints. Specifically, for datasets MMCoQA [\[20\]](#page-8-14), NeedleInAHaystack [\[34\]](#page-9-12), and GPR1200 [\[32\]](#page-9-14), the batch size was set to 1, while for all other datasets, a batch size of 24 was employed. Additionally, we calibrated the top-k value selection ratio for self-attention and cross-attention based on the sample mean ratio of cross-self region, with ablation studies showing the efficacy of adjusting this ratio. We applied biases of 0.5 and 1.5 in EgocentricNavigation[\[19\]](#page-8-15) and SlideVQA [\[36\]](#page-9-15), respectively, while keeping the default settings for other datasets. Our pre-processing and evaluation pipeline follows the standards of the benchmark, ensuring consistent assessment across the widely-used 29 datasets.

### 4.4. Main results

We use the widely-adopted open-source vision-language model LLaVA [\[25\]](#page-9-13) to test KV cache performance on the benchmark. We present the results of our experiments in Table [1](#page-4-0) and summarize our findings as follows.

For the LLaVA-v1.5-7b model, our approach achieves notable improvements across several tasks. By independently selecting top-k tokens for cross-attention and self-attention, our method effectively retains key tokens specific to each modality. This separation enables the model to capture essential temporal sequences in tasks with sequential dependencies while simultaneously focusing on relevant multimodal content in tasks requiring semantic understanding. This result reveals that separating cross and self attentions allows for better retention of modality-specific cues, enhancing performance in tasks highly relevant to visual and textual data, with improvements of 4.5%, 7.2%, and 9.8% in T-3, S-5, and 4.5%, 7.2%, 9.8% improvement in NH task respectively.

For the larger model, LLaVA-v1.5-13b, our approach shows even more pronounced improvements, especially on tasks T3, T4, and IR. These tasks share a common demand for precise handling of spatial and sequential elements across visual and textual modalities. By separating cross-attention and self-attention during top-k selection, our method effectively retains modality-specific tokens, which is crucial for tasks requiring spatial alignment and temporal tracking. This selective retention allows the model to preserve essential visual details for spatial localization (T3) with a performance boost of 8.3%, state transition (T4) with a 7.2% increase,

![](_page_5_Figure_7.jpeg)

Figure 3. The impact of the cross-self ratio. and accurate retrieval in IR with a 9.6% improvement.

## 5. Ablation Study

We delve into ablation analysis of the KV cache approach to comprehensively assess our method. First, we present the hyper-parameters selection of K<sup>c</sup> and K<sup>s</sup> . Next, we assess speed latency and GPU memory usage to examine efficiency. We also test the pruning selection function, and record the impact of varying budget sizes on performance. Finally, we present the influence of model architectures by introducing other vision-language models.

### 5.1. Influence of the hyper-parameters

The remaining tokens in the cache are composed of the top-k indices selected independently from self-attention and cross-attention, which are then concatenated with recent tokens. Consequently, the hyperparameters of our method include the ratio of top-k selections between cross-attention and self-attention. We observe that performance reaches an optimal level when both cross-attention and self-attention tokens are selected in balanced proportions (i.e., the ratio is neither 0 nor 1) across 29 datasets. This configuration suggests that integrating both types of attention improves performance across datasets, as it allows the model to capture important tokens from multiple perspectives. However, there are differences between datasets; for instance, tasks focused on temporal alignment or sequential event detection (e.g. ALFRED), tend to benefit more from cross-attention, where context from multiple image frames is critical. On the other hand, datasets emphasizing individual object identification or attribute-focused reasoning are more reliant on self-attention, as they require maintaining a focused view on specific elements within a single frame.

In extreme cases (ratio = 1 for only self-attention K<sup>s</sup>

<span id="page-5-0"></span><sup>2</sup>https://github.com/Dao-AILab/flash-attention

and ratio = 0 for only cross-attention K<sup>c</sup> ), we find distinct effects on task performance. With self-attention, static tasks like visual recognition perform well due to precise framespecific representations, though this setup struggles with tasks needing cross-frame integration. Conversely, crossattention supports tasks requiring sequence alignment, like scene understanding, but can dilute focus on high-resolution details crucial for object-specific tasks.

## 5.2. Efficiency analysis

We analyze the efficiency of our proposed method in terms of decoding and GPU memory usage. Table [2](#page-6-0) presents the results using LLaVA-v1.5-7b tested on a single RTX-4090 GPU with 100 warm-up iterations.The samples tested for the latency and memory usage is sampled from the AL-FRED dataset in the benchmark. At a 60% budget, decoding latency is reduced to 24.377 ms/token, which provides a modest improvement over the 26.023 ms observed with a full cache, while memory usage drops by approximately 23% to 1.207GiB. As the cache budget decreases further, the benefits become more pronounced: at a 30% budget, latency reaches 21.027 ms per token and memory usage is nearly halved to 0.523 GiB. With the most aggressive reduction, using only 10% of the original cache, decoding latency improves to 16.287 ms per token, achieving a 37% speed increase over the full cache setup and an 87% reduction in memory usage to just 0.208 GiB. These findings illustrate that our method allows for a flexible balance between memory efficiency and processing speed. Even with significantly reduced cache budgets, our approach retains acceptable latency and memory performance, offering a scalable solution for resource-constrained environments.

<span id="page-6-0"></span>

| Method     | Budget | Decoding Latency | GPU Mem   |
|------------|--------|------------------|-----------|
| Full Cache | 100%   | 26.023 ms/token  | 1.571 GiB |
| CSP        | 60%    | 24.377 ms/token  | 1.207 GiB |
| CSP        | 30%    | 21.027 ms/token  | 0.523 GiB |
| CSP        | 20%    | 19.736 ms/token  | 0.476 GiB |
| CSP        | 10%    | 16.287 ms/token  | 0.208 GiB |

Table 2. The efficiency of latency speed and memory usage. We utilized LLaVA-v1.5-7b to test speed performance on a single RTX-4090 with 100 warm-up iterations.

### 5.3. Influence of n-softmax

In this ablation study, we compare two KV cache pruning methods: one selects the top-k tokens directly from the overall selection function, while the other applies top-k selection separately within cross-attention and self-attention regions. We evaluate both approaches with standard Softmax and n-Softmax scoring functions to assess their impact on performance. Experimental results reveal that n-Softmax consistently provides a slight performance improvement over

![](_page_6_Figure_7.jpeg)

![](_page_6_Figure_8.jpeg)

(a) n-Softmax on Top-k selection. (b) n-Softmax on cross-self.

Figure 4. The benefit of n-softmax. We conduct the experiments on the ALFRED dataset by LLaVA-v1.5-7b.

Softmax, indicating that the smooth transition positively impacts decoding performance. Specifically, selection-based approaches demonstrate clear benefits by focusing retention on high-value tokens, which enhances model efficiency. This effect is evident across tasks, as the separate top-k selection in cross and self-attention regions improves performance by capturing modality-specific important tokens more effectively. Furthermore, we find that this transition is particularly effective for tasks requiring both temporal coherence and fine-grained feature retention, as it allows for the selective pruning of large, less critical tokens under limited KV cache budgets. These results underscore the advantages of selection-based methods for KV cache pruning, especially when integrating cross-self separation with n-Softmax.

### 5.4. Impact of cache budget

As the cache budget increases, we observe consistent performance gains across all tasks, indicating that larger cache budgets enhance model accuracy and retrieval quality. For each dataset, performance improves steadily with an increase in cache size, moving closer to or exceeding the baseline set by full cache. In tasks with complex sequence dependencies like ALFRED, our method (CSP) achieves a significant boost in accuracy at higher cache budgets (60%), outperforming other methods and reaching a level above the full cache baseline. This pattern suggests that a larger cache budget is especially beneficial in scenarios where maintaining temporal coherence is crucial for task success. In tasks requiring fine-grained visual differentiation, such as Spot-the-Diff and CLEVR-Change, performance gains with increased cache are less pronounced but still evident, indicating that these tasks can benefit from a moderate cache size. These findings support that our method consistently outperforms other methods across all budget size in the cache, suggesting that separate top-k selections from cross and self regions could effectively balance tokens selection and avoid collapse in the inference process.

### 5.5. Influence of Model Architectures

In this section, we evaluate the influence of model architecture on the performance of our proposed KV cache method across selected tasks (T-2, T-4, S-4, and IR) in the bench-

<span id="page-7-1"></span>![](_page_7_Figure_0.jpeg)

Figure 5. The impact of the cache size budget.

mark. We introduce two more architectures: InternVL-v1.5- 7B [\[11\]](#page-8-16), which scales up the vision encoder with cross-modal integration, and MobileVLM-V2-3B [\[12\]](#page-8-10), which features a lighter structure with an efficient downsampling projector (LDPv2) and focuses on intra-modal processing. Table [3](#page-7-0) demonstrates our test results of our methods compared with baselines. For InternVL-v1.5-7B, we observe that CSP achieves the highest performance in most tasks, particularly in T-2 (22.8) and S-4 (25.4), indicating that our KV cache pruning method benefits from InternVL's largescale vision encoder. This architecture supports a robust cross-modal alignment, which CSP leverages by retaining crucial tokens independently from cross-attention and selfattention regions, maintaining contextual richness in visualtextual integration tasks. Regarding MobileVLM-V2-3B, CSP also demonstrates superior performance, especially in IR (5.3), where precise image retrieval benefits from MobileVLM's lightweight, modality-aware processing enabled by LDPv2. The efficiency of this architecture allows CSP to perform well by focusing on high-salience tokens in the vision domain, as seen in improved scores across tasks. This shows that MobileVLM's intra-modal alignment complements CSP's motivations of avoiding the collapse of token selection.

## 6. Conclusion

In this work, we propose Cross-Self Pruning (CSP), a simple and training-free KV cache method designed to in-

<span id="page-7-0"></span>

| Method                | T-2                 | T-4  | S-4  | IR  |  |  |  |  |
|-----------------------|---------------------|------|------|-----|--|--|--|--|
| InternVL-v1.5-7B [11] |                     |      |      |     |  |  |  |  |
| Full Cache            | 19.2                | 21.3 | 19.1 | 0.0 |  |  |  |  |
| H2O [46]              | 20.0                | 20.4 | 19.6 | 0.5 |  |  |  |  |
| SnapKV [21]           | 19.9                | 19.5 | 19.4 | 0.2 |  |  |  |  |
| RoCo [31]             | 20.0                | 18.5 | 19.6 | 0.5 |  |  |  |  |
| LOOK-M [31]           | 22.0                | 19.6 | 22.9 | 0.5 |  |  |  |  |
| CSP (Ours)            | 22.8                | 20.7 | 25.4 | 0.6 |  |  |  |  |
|                       | MobileVLM-V2-3B[12] |      |      |     |  |  |  |  |
| Full Cache            | 46.2                | 38.5 | 33.0 | 4.7 |  |  |  |  |
| H2O [46]              | 46.4                | 38.2 | 28.2 | 4.5 |  |  |  |  |
| SnapKV [21]           | 46.4                | 38.5 | 27.2 | 4.7 |  |  |  |  |
| RoCo [31]             | 46.6                | 38.0 | 28.9 | 4.6 |  |  |  |  |
| LOOK-M [39]           | 47.0                | 38.7 | 32.8 | 4.8 |  |  |  |  |
| CSP (Ours)            | 47.3                | 39.0 | 33.1 | 5.3 |  |  |  |  |

Table 3. Comparison of KV cache methods across tasks (T-2, T-4, S-4, IR) on InternVL-v1.5-7B [\[11\]](#page-8-16) and MobileVLM-V2-3B [\[12\]](#page-8-10).

dependently select top-k tokens from cross-attention and self-attention regions. We evaluate pruning method on a range of multimodal tasks, demonstrating that CSP achieves competitive performance across all tasks while reducing cache budgets. Furthermore, our ablation study highlights the method's robustness and effectiveness in optimizing token selection through intra- and cross-modality pruning, offering a lightweight solution for improving computational efficiency without compromising model accuracy.

## References

- <span id="page-8-0"></span>[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. *arXiv:2303.08774*, 2023. [1,](#page-0-1) [2](#page-1-0)
- <span id="page-8-12"></span>[2] Muhammad Adnan, Akhil Arunkumar, Gaurav Jain, Prashant Nair, Ilya Soloveychik, and Purushotham Kamath. Keyformer: Kv cache reduction through key tokens selection for efficient generative inference. *Proceedings of Machine Learning and Systems*, 6:114–127, 2024. [2](#page-1-0)
- <span id="page-8-11"></span>[3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. *Advances in neural information processing systems*, 35:23716–23736, 2022. [2](#page-1-0)
- <span id="page-8-1"></span>[4] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. *arXiv:2309.16609*, 2023. [1,](#page-0-1) [2](#page-1-0)
- <span id="page-8-3"></span>[5] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-VL: A frontier large vision-language model with versatile abilities. *arXiv:2308.12966*, 2023. [1,](#page-0-1) [2](#page-1-0)
- <span id="page-8-2"></span>[6] Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, et al. Deepseek LLM: Scaling open-source language models with longtermism. *arXiv:2401.02954*, 2024. [1,](#page-0-1) [2](#page-1-0)
- <span id="page-8-23"></span>[7] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 11621–11631, 2020. [13](#page-12-0)
- <span id="page-8-7"></span>[8] Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong, Baobao Chang, Junjie Hu, et al. Pyramidkv: Dynamic kv cache compression based on pyramidal information funneling. *arXiv preprint arXiv:2406.02069*, 2024. [1,](#page-0-1) [2,](#page-1-0) [3](#page-2-2)
- <span id="page-8-19"></span>[9] Yingshan Chang, Mridu Narang, Hisami Suzuki, Guihong Cao, Jianfeng Gao, and Yonatan Bisk. Webqa: Multihop and multimodal qa. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 16495– 16504, 2022. [13](#page-12-0)
- <span id="page-8-4"></span>[10] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. *arXiv:2312.14238*, 2023. [1,](#page-0-1) [2](#page-1-0)
- <span id="page-8-16"></span>[11] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 24185–24198, 2024. [8](#page-7-1)
- <span id="page-8-10"></span>[12] Xiangxiang Chu, Limeng Qiao, Xinyu Zhang, Shuang Xu, Fei Wei, Yang Yang, Xiaofei Sun, Yiming Hu, Xinyang Lin,

- Bo Zhang, et al. Mobilevlm v2: Faster and stronger baseline for vision language model. *arXiv preprint arXiv:2402.03766*, 2024. [2,](#page-1-0) [8](#page-7-1)
- <span id="page-8-17"></span>[13] Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. Tall: Temporal activity localization via language query. In *Proceedings of the IEEE international conference on computer vision*, pages 5267–5275, 2017. [13](#page-12-0)
- <span id="page-8-13"></span>[14] Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. Model tells you what to discard: Adaptive kv cache compression for llms. *arXiv preprint arXiv:2310.01801*, 2023. [2](#page-1-0)
- <span id="page-8-22"></span>[15] Mehrdad Hosseinzadeh and Yang Wang. Image change captioning by learning from an auxiliary task. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 2725–2734, 2021. [13](#page-12-0)
- <span id="page-8-18"></span>[16] Qingqiu Huang, Yu Xiong, Anyi Rao, Jiaze Wang, and Dahua Lin. Movienet: A holistic dataset for movie understanding. In *Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IV 16*, pages 709–727. Springer, 2020. [13](#page-12-0)
- <span id="page-8-21"></span>[17] Harsh Jhamtani and Taylor Berg-Kirkpatrick. Learning to describe differences between pairs of similar images. *arXiv preprint arXiv:1808.10584*, 2018. [13](#page-12-0)
- <span id="page-8-20"></span>[18] Aniruddha Kembhavi, Minjoon Seo, Dustin Schwenk, Jonghyun Choi, Ali Farhadi, and Hannaneh Hajishirzi. Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension. In *Proceedings of the IEEE Conference on Computer Vision and Pattern recognition*, pages 4999–5007, 2017. [13](#page-12-0)
- <span id="page-8-15"></span>[19] Jacob Krantz, Erik Wijmans, Arjun Majumdar, Dhruv Batra, and Stefan Lee. Beyond the nav-graph: Vision-and-language navigation in continuous environments. In *Computer Vision– ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXVIII 16*, pages 104– 120. Springer, 2020. [6,](#page-5-1) [13](#page-12-0)
- <span id="page-8-14"></span>[20] Yongqi Li, Wenjie Li, and Liqiang Nie. Mmcoqa: Conversational question answering over text, tables, and images. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 4220–4231, 2022. [6,](#page-5-1) [13](#page-12-0)
- <span id="page-8-8"></span>[21] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. Snapkv: Llm knows what you are looking for before generation. *arXiv preprint arXiv:2404.14469*, 2024. [1,](#page-0-1) [2,](#page-1-0) [3,](#page-2-2) [5,](#page-4-1) [8](#page-7-1)
- <span id="page-8-5"></span>[22] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. *arXiv:2403.18814*, 2024. [1,](#page-0-1) [2](#page-1-0)
- <span id="page-8-9"></span>[23] Akide Liu, Jing Liu, Zizheng Pan, Yefei He, Gholamreza Haffari, and Bohan Zhuang. Minicache: Kv cache compression in depth dimension for large language models. *arXiv preprint arXiv:2405.14366*, 2024. [1,](#page-0-1) [2,](#page-1-0) [3](#page-2-2)
- <span id="page-8-6"></span>[24] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. *arXiv:2310.03744*, 2023. [1,](#page-0-1) [2](#page-1-0)

- <span id="page-9-13"></span>[25] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 26296–26306, 2024. [6](#page-5-1)
- <span id="page-9-3"></span>[26] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. *Advances in neural information processing systems*, 36, 2024. [1,](#page-0-1) [2](#page-1-0)
- <span id="page-9-21"></span>[27] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In *Proceedings of the IEEE/CVF winter conference on applications of computer vision*, pages 2200–2209, 2021. [13](#page-12-0)
- <span id="page-9-20"></span>[28] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In *2019 international conference on document analysis and recognition (ICDAR)*, pages 947–952. IEEE, 2019. [13](#page-12-0)
- <span id="page-9-18"></span>[29] Viorica Patraucean, Lucas Smaira, Ankush Gupta, Adria Recasens, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Mateusz Malinowski, Yi Yang, Carl Doersch, et al. Perception test: A diagnostic benchmark for multimodal video models. *Advances in Neural Information Processing Systems*, 36, 2024. [13](#page-12-0)
- <span id="page-9-9"></span>[30] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *International conference on machine learning*, pages 8748–8763. PMLR, 2021. [2](#page-1-0)
- <span id="page-9-6"></span>[31] Siyu Ren and Kenny Q Zhu. On the efficacy of eviction policy for key-value constrained generative language model inference. *arXiv preprint arXiv:2402.06262*, 2024. [1,](#page-0-1) [2,](#page-1-0) [3,](#page-2-2) [5,](#page-4-1) [8](#page-7-1)
- <span id="page-9-14"></span>[32] Konstantin Schall, Kai Uwe Barthel, Nico Hezel, and Klaus Jung. Gpr1200: a benchmark for general-purpose contentbased image retrieval. In *International Conference on Multimedia Modeling*, pages 205–216. Springer, 2022. [6,](#page-5-1) [13](#page-12-0)
- <span id="page-9-8"></span>[33] Mohit Shridhar, Jesse Thomason, Daniel Gordon, Yonatan Bisk, Winson Han, Roozbeh Mottaghi, Luke Zettlemoyer, and Dieter Fox. Alfred: A benchmark for interpreting grounded instructions for everyday tasks. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 10740–10749, 2020. [2,](#page-1-0) [13](#page-12-0)
- <span id="page-9-12"></span>[34] Dingjie Song, Shunian Chen, Guiming Hardy Chen, Fei Yu, Xiang Wan, and Benyou Wang. Milebench: Benchmarking mllms in long context. *arXiv preprint arXiv:2404.18532*, 2024. [4,](#page-3-2) [6,](#page-5-1) [13](#page-12-0)
- <span id="page-9-19"></span>[35] Alon Talmor, Ori Yoran, Amnon Catav, Dan Lahav, Yizhong Wang, Akari Asai, Gabriel Ilharco, Hannaneh Hajishirzi, and Jonathan Berant. Multimodalqa: Complex question answering over text, tables and images. *arXiv preprint arXiv:2104.06039*, 2021. [13](#page-12-0)
- <span id="page-9-15"></span>[36] Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, and Kuniko Saito. Slidevqa: A dataset for document visual question answering on multiple images. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 13636–13645, 2023. [6,](#page-5-1) [13](#page-12-0)
- <span id="page-9-4"></span>[37] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalk-

- wyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. *arXiv:2312.11805*, 2023. [1,](#page-0-1) [2](#page-1-0)
- <span id="page-9-0"></span>[38] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste ´ Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. ` Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023. [1,](#page-0-1) [2](#page-1-0)
- <span id="page-9-10"></span>[39] Zhongwei Wan, Ziang Wu, Che Liu, Jinfa Huang, Zhihong Zhu, Peng Jin, Longyue Wang, and Li Yuan. Look-m: Lookonce optimization in kv cache for efficient multimodal longcontext inference. *arXiv preprint arXiv:2406.18139*, 2024. [2,](#page-1-0) [3,](#page-2-2) [5,](#page-4-1) [8](#page-7-1)
- <span id="page-9-16"></span>[40] Bo Wu, Shoubin Yu, Zhenfang Chen, Joshua B Tenenbaum, and Chuang Gan. Star: A benchmark for situated reasoning in real-world videos. *arXiv preprint arXiv:2405.09711*, 2024. [13](#page-12-0)
- <span id="page-9-1"></span>[41] Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, et al. Baichuan 2: Open large-scale language models. *arXiv preprint arXiv:2309.10305*, 2023. [1,](#page-0-1) [2](#page-1-0)
- <span id="page-9-5"></span>[42] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. *arXiv:2304.14178*, 2023. [1,](#page-0-1) [2](#page-1-0)
- <span id="page-9-17"></span>[43] Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B Tenenbaum. Clevrer: Collision events for video representation and reasoning. *arXiv preprint arXiv:1910.01442*, 2019. [13](#page-12-0)
- <span id="page-9-2"></span>[44] Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Heng Li, Jiangcheng Zhu, Jianqun Chen, Jing Chang, et al. Yi: Open foundation models by 01. ai. *arXiv preprint arXiv:2403.04652*, 2024. [1,](#page-0-1) [2](#page-1-0)
- <span id="page-9-11"></span>[45] Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, et al. Sparsevlm: Visual token sparsification for efficient vision-language model inference. *arXiv preprint arXiv:2410.04417*, 2024. [2,](#page-1-0) [3](#page-2-2)
- <span id="page-9-7"></span>[46] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Re, Clark Barrett, et al. H2o: Heavy-hitter oracle ´ for efficient generative inference of large language models. *Advances in Neural Information Processing Systems*, 36, 2024. [1,](#page-0-1) [2,](#page-1-0) [3,](#page-2-2) [5,](#page-4-1) [8](#page-7-1)

## A. Appendix Section

### A.1. Ratio Selection for Dataset

The table reveals varying strategies for attention configuration across datasets, reflecting task-specific priorities. Datasets like *Spot-the-Diff* and *WebQA* emphasize crossattention by assigning 90% of the top-k selection to crossmodal interactions. Conversely, tasks such as *ActionPrediction* rely entirely on intra-modal attention, with no top-k selection allocated to cross-attention. Overall, most datasets adopt a balanced approach, allocating 50% of the top-k selection to cross-attention and maintaining a recency bias of 1, indicating a general preference for equal weighting of intraand cross-modal attention in multi-modal inference.

| Dataset | LLaVA-7B | LLaVA-13B |
|---------|----------|-----------|
| T1      | 0.5      | 1         |
| T2      | 0.5      | 0.5       |
| T3      | 0.5      | 0.5       |
| T4      | 0.1      | 0.5       |
| S1      | 0.5      | 0.5       |
| S2      | 0.5      | 0.5       |
| S3      | 0.5      | 0.5       |
| S4      | 0.5      | 0.5       |
| NH      | 0.5      | 0.5       |
| IR      | 0.9      | 0.9       |

Table 4. Cross Ratio Selection for Different Tasks.

### A.2. Distribution Visualization

In this section, we visualize the distribution differences between intra- and cross-attention using kernel density estimation (KDE). From the visualizations, we observe that certain datasets exhibit a stronger reliance on cross-attention, while others depend more heavily on self-attention. Here we apply the Kernel Density Estimation (KDE) and Jensen-Shannon (JS) divergence to analysis the difference of distributions.

In terms of KDE, figure [6](#page-11-0) demonstrates the attention weight distributions for both self-attention (blue) and crossattention (red) across eight datasets. It is evident that the two types of attention exhibit distinct patterns depending on the dataset, which directly impacts the pruning strategies during the KV cache process. For datasets such as *CLEVR-Change* and *CounterfactualInference*, cross-attention weights show a significantly concentrated and dominant peak at very low values, while self-attention demonstrates broader coverage. This suggests that cross-attention contributes heavily to the model's decision-making in these tasks, emphasizing token dependencies between modalities (e.g., image-text). Pruning strategies in these cases might inadvertently eliminate crucial cross-attention connections, leading to incomplete information transfer and subsequent degradation in inference accuracy. Conversely, datasets such as *DocVQA* and

*EgocentricNavigation* reveal more dispersed and substantial self-attention weights, while cross-attention peaks remain narrow. This indicates a reliance on intra-modal token interactions, such as contextual reasoning within the same modality. Aggressive pruning of self-attention tokens in such cases risks losing key intra-modal context, adversely impacting downstream predictions.

To further quantify the discrepancy between self-attention and cross-attention distributions, we compute the Jensen-Shannon (JS) divergence for each dataset. Higher divergence values suggest a stark imbalance between the two attention mechanisms, indicating that naive pruning may disproportionately affect one type of attention. In Figure [7,](#page-11-1) tasks like *CLEVR-Change* and *ActionPrediction* exhibit high divergence, implying the necessity for task-specific pruning thresholds to retain balanced contributions from both self- and cross-attention, whereas tasks like *DocVQA* show lower divergence, where self- and cross-attention operate more harmoniously, and uniform pruning strategies may suffice. These distribution discrepancies highlight several challenges and insights. Uniform pruning approaches that prioritize magnitude-based filtering may disproportionately affect regions with dense cross-attention peaks (e.g., *CLEVR-Change*), hindering the model's ability to encode cross-modal dependencies, particularly in visual reasoning tasks. For datasets where self-attention dominates (e.g., *EgocentricNavigation*), pruning strategies that overly prioritize low-weight tokens may reduce contextual coherence, especially for long-context sequences. The analysis underscores the necessity for adaptive pruning mechanisms that balance retention across self- and cross-attention regions, guided by the specific reliance patterns observed in the KDE and JS divergence results. Furthermore, the differential pruning of self- and cross-attention tokens influences the effectiveness of KV cache utilization. Over-pruning either region may cause a skewed representation in the cache, reducing its utility for long-context inference.

The observed disparities in self- and cross-attention distributions across datasets necessitate an adaptive pruning framework that dynamically adjusts based on task-specific ratio, effectively balancing the preservation of essential intraand cross-modal dependencies while optimizing KV cache utilization.

<span id="page-11-0"></span>![](_page_11_Figure_0.jpeg)

Figure 6. Kernel Density Estimation (KDE) of the attention weight distributions.

<span id="page-11-1"></span>![](_page_11_Figure_2.jpeg)

Figure 7. Jensen-Shannon (JS) divergence scores between cross-attention and self-attention across all layers.

Table 5. Detailed Statistics and Taxonomy from MILEBench [34].

<span id="page-12-0"></span>

| Category              | Task                                             | Dataset                          | Data Source            | Count | Metric   |
|-----------------------|--------------------------------------------------|----------------------------------|------------------------|-------|----------|
|                       |                                                  | Action Localization              | STA [13]               |       |          |
| T 114 16 1            | Action Understanding and Prediction (T-1)        | Action Prediction                | STAR [40]              | 200   | Accuracy |
|                       |                                                  | Action Sequence                  | STAR [40]              |       |          |
| Temporal Multi-image  |                                                  | Object Existence                 | CLEVRER [43]           |       |          |
|                       | Object and Scene Understanding (T-2)             | Object Interaction               | STAR [40]              | 200   | Accuracy |
|                       | Object and Scene Orderstanding (1-2)             | Moving Attribute                 | CLEVRER [43]           | 200   |          |
|                       |                                                  | Object Shuffle                   | Perception Test [29]   |       |          |
|                       | Visual Navigation and Spatial Localization (T-3) | Egocentric Navigation            | VLN-CE [19]            | 200   | Accuracy |
|                       | visuai Navigation and Spatiai Localization (1-5) | Moving Direction                 | CLEVRER [43]           | 200   |          |
|                       |                                                  | Counterfactual Inference         | CLEVRER [43]           |       |          |
|                       | Counterfactual Reasoning and State Change (T-4)  | State Change                     | Perception Test [29]   | 200   | Accuracy |
|                       | Counterfactual Reasoning and State Change (1-4)  | Character Order                  | Perception Test [29]   | 200   |          |
|                       |                                                  | Scene Transition                 | MovieNet [16]          |       |          |
|                       |                                                  | Webpage QA                       | WebQA [9]              |       | Accuracy |
|                       | Variable County to OA (C 1)                      | Textbook QA                      | TQA [18]               | 200   |          |
|                       | Knowledge Grounded QA (S-1)                      | Complex Multimodal QA            | MultiModalQA [35]      |       |          |
| Semantic Multi-image  |                                                  | Long Text with Images QA         | WikiVQA                |       |          |
| Semantic Muni-image – |                                                  | Slide QA                         | SlideVQA [36]          |       |          |
|                       | Text-Rich Images QA (S-2)                        | OCR QA                           | OCR-VQA [28]           | 200   | Accuracy |
|                       |                                                  | Document QA                      | DocVQA [27]            |       |          |
|                       | Visual Relation Inference (S-3)                  | Visual Change Captioning         | Spot-the-Diff [17]     | 200   | ROUGE-L  |
| , ,                   | visual Relation inference (3-3)                  | Visual Relationship Expressing   | CLEVR-Change [15]      | 200   |          |
|                       | Dialogue (S-4)                                   | Multimodal Dialogue              | MMCoQA [20]            | 200   | Accuracy |
|                       | Dialogue (5-4)                                   | Conversational Embodied Dialogue | ALFRED [33]            | 200   |          |
|                       | Space Understanding (S-5)                        | nuScenes                         | nuScenes [7]           | 200   | Accuracy |
| Diagnostic Evaluation | Needle In A Haystack (N-1)                       | Text Needle In A Haystack        | TextNeedleInAHaystack  | 320   | Accuracy |
|                       | Needle In A Haystack (N-2)                       | Image Needle In A Haystack       | ImageNeedleInAHaystack | 320   | Accuracy |
|                       | Image Retrieval (I-1)                            | Image Retrieval                  | GPR1200 [32]           | 600   | Accuracy |