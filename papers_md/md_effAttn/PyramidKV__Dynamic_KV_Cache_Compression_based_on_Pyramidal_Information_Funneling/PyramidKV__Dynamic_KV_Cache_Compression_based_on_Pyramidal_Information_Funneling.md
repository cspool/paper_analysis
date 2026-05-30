# **PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling**

**Zefan Cai**<sup>1</sup> **, Yichi Zhang**<sup>2</sup> **, Bofei Gao**<sup>2</sup> **, Yuliang Liu**<sup>3</sup> **, Yucheng Li**<sup>4</sup> **, Tianyu Liu**<sup>5</sup> **, Keming Lu**<sup>5</sup> , **Wayne Xiong**<sup>7</sup> , **Yue Dong**<sup>6</sup> , **Junjie Hu**<sup>1</sup> , **Wen Xiao**<sup>7</sup> ,

<sup>1</sup>University of Wisconsin - Madison <sup>2</sup>Peking University <sup>3</sup>Nanjing University <sup>3</sup>University of Surrey <sup>5</sup>Qwen <sup>6</sup>University of California - Riverside <sup>7</sup>Microsoft zefncai@gmail.com

<https://github.com/Zefan-Cai/PyramidKV>

# **Abstract**

In this study, we investigate whether attention-based information flow inside large language models (LLMs) is aggregated through noticeable patterns for long context processing. Our observations reveal that LLMs aggregate information through **Pyramidal** Information Funneling where attention is scattering widely in lower layers, progressively consolidating within specific contexts, and ultimately focusing on critical tokens (a.k.a massive activation or attention sink) in higher layers. Motivated by these insights, we developed PyramidKV, a novel and effective KV cache compression method. This approach dynamically adjusts the KV cache size across different layers, allocating more cache in lower layers and less in higher ones, diverging from traditional methods that maintain a uniform KV cache size. Our experimental evaluations, utilizing the LongBench benchmark, show that PyramidKV matches the performance of models with a full KV cache while retaining only 12% of the KV cache, thus significantly reducing memory usage. In scenarios emphasizing memory efficiency, where only 0.7% of the KV cache is maintained, PyramidKV surpasses other KV cache compression techniques, achieving up to a 20.5 absolute accuracy improvement on TREC dataset. In the Needle-in-a-Haystack experiment, PyramidKV outperforms competing methods in maintaining long-context comprehension in LLMs; notably, retaining just 128 KV cache entries enables the LLAMA-3-70B model to achieve 100.0 Acc. performance.

# **1 Introduction**

Large language models (LLMs) [\(Achiam et al., 2023;](#page-9-0) [Touvron et al., 2023a](#page-11-0)[;b;](#page-11-1) [Jiang et al.,](#page-10-0) [2023\)](#page-10-0) are integral to various natural language processing applications, including dialogue systems [\(Chiang et al., 2023\)](#page-9-1), document summarization [\(Fabbri et al., 2019a\)](#page-9-2), and code completion [\(Roziere et al., 2023\)](#page-10-1). These models have recently been scaled up to handle long contexts [\(Fu et al., 2024;](#page-9-3) [Ding et al., 2024;](#page-9-4) [Zhu et al., 2023;](#page-11-2) [Chen et al., 2023\)](#page-9-5), with GPT-4 processing up to 128K tokens and Gemini-pro-1.5 handling 1M tokens. However, scaling LLMs to extremely long contexts naturally leads to a significant delay due to the quadratic computation of attention over long contexts. A common solution to mitigate such inference delays involves caching the key and value states (KV) of previous tokens [\(Waddington](#page-11-3) [et al., 2013\)](#page-11-3), with the trade-off of requiring extensive GPU memory storage. For instance, maintaining a KV cache for 100K tokens in LLaMA-2 7B requires over 50GB of memory, while a 2K context requires less than 1GB of memory [\(Wu et al., 2024\)](#page-11-4).

To tackle these memory constraints, recent studies have explored the optimization of KV caching, including approaches such as low-rank decomposition of the KV cache [\(Dong et al.,](#page-9-6) [2024\)](#page-9-6) or pruning non-essential KV cache [\(Zhang et al., 2024;](#page-11-5) [Li et al., 2024;](#page-10-2) [Ge et al., 2023\)](#page-10-3). Notably, it has been shown that maintaining merely 20% of the KV cache can preserve a substantial level of performance [\(Zhang et al., 2024\)](#page-11-5). Moreover, extreme compression of the

![](_page_1_Figure_1.jpeg)

<span id="page-1-0"></span>Figure 1: Illustration of PyramidKV compared with existing KV cache compression methods. (a) Full KV has all tokens stored in the KV cache in each layer; cache size increases as the input length increases. (b) StreamingLLM [\(Xiao et al., 2023\)](#page-11-6) only keeps few initial tokens with a fixed cache size in each layer. (c) SnapKV [\(Li et al., 2024\)](#page-10-2) and H2O [\(Zhang et al.,](#page-11-5) [2024\)](#page-11-5) keep a fixed cache size across Transformer layers, and their selection is based on the attention score. (d) PyramidKV maintains pyramid-like cache sizes, allocating more cache budget to lower layers and less to higher layers. This approach to KV cache selection better aligns with the increasing attention sparsity observed in multi-layer Transformers ([§3\)](#page-2-0).

KV cache for tasks of longer contexts (e.g., retrieval augmented generation or RAG for short) can drastically improve efficiency and further reduce resource use. However, questions about the universal applicability of these strategies across all layers of an LLM remain open. (1) *Are these KV cache strategies applicable to all layers?* (2) *Is it computationally efficient to use the same KV cache size across layers as previous studies have done?* These considerations suggest a need for an in-depth, more nuanced understanding of KV cache optimization in LLMs.

To examine these questions, we aim to systematically investigate the design principles of the KV cache compression across different layers, specifically tailored to the behaviors of the attention mechanism. We first investigate how information flow is aggregated via attention mechanisms across different layers in multi-document question answering (QA), a classic task involving long contexts. Our analysis identifies a notable transition of attention distribution from a broad coverage of global contexts to a narrow focus of local tokens over layers in LLMs. This pattern suggests an aggregated information flow where information is initially gathered broadly and subsequently narrowed down to key tokens, epitomizing the massive attention phenomenon. Our findings provide unique insights beyond the previously documented "massive activation" [\(Sun et al., 2024\)](#page-11-7) that very few activations exhibit significantly larger values than others when calculating multi-head attention in LLMs and "attention sink" [\(Xiao et al., 2023\)](#page-11-6) that keeping the KV of initial tokens will largely recover the performance of window attention.

Building on these insights on how information flows are aggregated through a pyramid pattern, we design a novel and effective KV cache pruning approach that mirrors the geometric shape, named PyramidKV. As shown in [Figure 1,](#page-1-0) unlike the fixed-and-same length KV cache pruning common in prior works [\(Zhang et al., 2024;](#page-11-5) [Ge et al., 2023;](#page-10-3) [Li et al.,](#page-10-2) [2024\)](#page-10-2), PyramidKV allocates more KV cache to the lower layers where information is more dispersed and each KV holds less information while reducing the KV cache in higher layers where information becomes concentrated in fewer key tokens. To the best of our knowledge, PyramidKV is the first KV cache compression method with varied cache retention across layers, tailoring cache amounts to the informational needs of each layer.

We conducted comprehensive experiments on LongBench [\(Bai et al., 2023\)](#page-9-7) using 17 datasets across various tasks and domains with three backbone models (LLaMa-3-8B-Instruct, LLaMa-3-70B-Instruct and Mistral-7B [\(Jiang et al., 2023\)](#page-10-0)). The results show that PyramidKV preserves performance using just 12.0% of the KV cache (KV Cache size = 2048) on the LongBench benchmark and significantly outperforms other methods in extreme conditions, retaining only 0.7% of the KV cache. Moreover, PyramidKV outperforms baseline models (H2O [\(Zhang et al., 2024\)](#page-11-5), SnapKV [\(Li et al., 2024\)](#page-10-2), StreamingLLM [\(Xiao et al., 2023\)](#page-11-6)) across all tested cache sizes (64, 96, 128, 256), with its advantages most pronounced at smaller

![](_page_2_Figure_1.jpeg)

<span id="page-2-1"></span>Figure 2: Attention patterns of retrieval-augmented generation across layers in LlaMa [\(Tou](#page-11-0)[vron et al., 2023a](#page-11-0)[;b\)](#page-11-1) reveal that in the lower layers, the model exhibits a broad-spectrum mode of attention, distributing attention scores uniformly across all content. In the middle layers, attention becomes more localized within each document, indicating refined information aggregation (dotted red triangular shapes in layers 6 and 10). This culminates in the upper layers, where "massive attention" focuses on a few key tokens (concentrated attention bars after layer 18), efficiently extracting essential information for answers.

cache sizes. In the Needle In A Haystack experiment, PyramidKV effectively maintains the long-context comprehension in LLMs, outperforming than competing methods. Remarkably, with PyramidKV, retaining only 128 KV cache entries allows the LLaMa-3-70B-Instruct model to achieve 100.0 Acc. performance, matching the performance of a full KV cache.

# **2 Related Work**

There has been a growing interest in addressing LLMs' memory constraints on processing long context inputs. FastGen [\(Ge et al., 2023\)](#page-10-3) introduces an adaptive KV cache management strategy that optimizes memory use by tailoring retention tactics to the specific nature of attention heads. SnapKV [\(Li et al., 2024\)](#page-10-2) improves efficiency by compressing KV caches via selecting/clustering significant KV positions based on their attention scores. Heavy Hitter Oracle (H2O) [\(Zhang et al., 2024\)](#page-11-5) implements a dynamic eviction policy that effectively balances the retention of recent and historically significant tokens, optimizing memory usage while preserving essential information. StreamingLLM [\(Xiao et al., 2023\)](#page-11-6) enables LLMs trained on finite attention windows to handle infinite sequence lengths without fine-tuning, thus expanding the models' applicability to broader contexts.

# <span id="page-2-0"></span>**3 Pyramidal Information Funneling**

To systematically understand the attention mechanism over layers in LLMs for long-context inputs, we conduct a fine-grained study focusing on the multi-document question answering (QA) task. The model is presented with multiple interrelated documents and prompted to generate an answer for the given query. The main target is to investigate how the model aggregates dispersed information within these retrieved documents for accurate responses. In particular, we focus on our analysis of the LLaMa (Touvron et al., 2023a;b) and visualize the distribution and behavior of attention scores over layers. To assess the distinct behaviors of each multi-head self-attention layer, we compute the average attention from all heads within each layer. Figure 2 shows the attention patterns of one QA example over six different layers (i.e., 0, 6, 12, 18, 24, and 30).

We identify an approximately uniform distribution of attention scores from the lower layers (e.g., the 0th layer). This suggests that the model operates in a broad-spectrum mode at the lower layers, aggregating information globally from all available content without prioritizing its attention on specific input segments. Notably, a distinct transition to a more localized attention pattern within each document emerges, as the model progresses to encode information at the middle layers (6th to 18th layers). In this phase, attention is predominantly directed towards tokens within the same document, suggesting a more refined aggregation of information within individual contexts.

This trend continues and intensifies in the upper layers (from the 24th to the 30th layer), where we observed the emergence of 'massive attention' phenomena. In these layers, the attention mechanism concentrates overwhelmingly on a few key tokens. This pattern of attention allocation, where extremely high attention scores are registered, signifies that the model has aggregated the essential information into these focal tokens. Such behavior underscores a sophisticated mechanism by which LLMs manage and streamline complex and voluminous information, culminating in the efficient extraction of the most pertinent data points necessary for generating accurate answers.

### 4 PyramidKV

#### 4.1 Preliminaries and Problem Formulation

In an autoregressive transformer LLM, the generation of the i-th token requires that the attention module computes the query, key, and value vectors for all previous i-1 tokens. To speed up inference process and avoid duplicate computations, the key and value matrices are typically stored in the GPU memory. While the KV cache enhances inference speed and reduces redundant computations, it can consume significant memory when dealing with long input contexts. To optimize memory usage, a strategy called KV cache compression is proposed (Zhang et al., 2024; Xiao et al., 2023; Li et al., 2024), which involves retaining only a minimal amount of KV cache while preserving as much information as possible.

In a LLM with m transformer layers, we denote the key and value matrices in the l-th attention layer respectively as  $K^l, V^l \in \mathbb{R}^{n \times d}, \forall l \in [0, m-1]$  when encoding a sequence of n tokens. The goal of KV cache compression is to seek two sub-matrices  $K^l_s, V^l_s \in \mathbb{R}^{k^l \times d}$  from the full matrices  $K^l_s$  and  $V^l_s$  given a cache budget  $k^l < n$  for each layer  $l \in [0, m-1]$  while maximizing performance preservation. A LLM with KV cache compression only uses  $K^l_s$  and  $V^l_s$  in the GPU memory for inference on a dataset  $\mathcal{D}$ , and obtains a similar result to a full model according to an evaluation scoring metric, i.e.,  $\operatorname{score}(K^l, V^l, \mathcal{D}) \approx \operatorname{score}(K^l_s, V^l_s, \mathcal{D})$ .

#### 4.2 Proposed Method

In this section, we introduce our method, PyramidKV, based on the pyramidal information funneling observed across different layers in §3. PyramidKV consists of two steps: (1) Dynamically allocating different KV cache sizes/budgets across different layers (§4.2.1); and (2) Selecting important KV vectors in each attention head for caching (§4.2.2).

#### <span id="page-3-0"></span>4.2.1 KV Cache Size/Budget Allocation

Previous work on KV cache compression (Li et al., 2024; Zhang et al., 2024; Xiao et al., 2023) often allocates a fixed KV cache size across LLM layers. However, as our analysis in §3 demonstrates, attention patterns are not identical across different layers. Particularly dense attention is observed in the lower layers, and sparse attention in higher layers. Therefore,

using a fixed KV cache size across layers may lead to suboptimal performance. These approaches may retain many unimportant tokens in the higher layers of sparser attentions while potentially overlooking many crucial tokens in the lower layers of denser attentions.

Thus, we propose to increase compression efficiency by dynamically allocating the cache budgets across layers to reflect the aggregated information flow based on attention patterns. Specifically, PyramidKV allocates more KV cache to the lower layers where information is more dispersed and each KV state contains less information, while reducing the KV cache in higher layers where information becomes concentrated in a few key tokens.

Following the common practice in KV cache compression (Li et al., 2024; Xiao et al., 2023), we first retain the KV cache for the last  $\alpha$  tokens of the input across all layers, as these tokens have been shown to contain the most immediate task-related information, where  $\alpha$  is a hyperparameter, controlling the number of last few tokens being included in the KV cache. For simplicity, we call these tokens "instruction tokens", which is also referred to as "local window" in previous literature (Zhang et al., 2024; Li et al., 2024; Xiao et al., 2023).

Subsequently, given the remaining total cache budget  $k^{\text{total}} = \sum_{l \in [0, m-1]} k^l$  that can be used over all transformer layers (noted as m), we first determine the cache sizes for the top and bottom layers, and use an arithmetic sequence to compute the cache sizes for the intermediate layers to form the pyramidal shape. The key intuition is to follow the attention pattern in aggregated information flow, reflecting a monotonically decreasing pattern of important tokens for attention from lower layers to upper layers. We allocate  $k^{m-1} = k^{\text{total}}/(\beta \cdot m)$  for the top layer and  $k^0 = (2 \cdot k^{\text{total}})/m - k^{m-1}$  for the bottom layer, where  $\beta$  is a hyperparameter to adjust the pyramid's shape. The hyperparameter  $\beta$  is still required to determine the top layer. Once the top layer is identified, the budget of the bottom layer can be calculated by summing the budgets across all layers and equating this sum to the total budget. Once the cache sizes of the bottom and top layers are determined, the cache sizes for all intermediate layers are set according to an arithmetic sequence, defined as

$$k^{l} = k^{0} - \frac{k^{0} - k^{m-1}}{m-1} \times l. \tag{1}$$

#### <span id="page-4-0"></span>4.2.2 KV Cache Selection

Once the KV cache budget is determined for each layer, our method needs to select specific KV states for caching within each layer in LLMs. As described in the previous section, the KV cache of the last  $\alpha$  tokens, referred to as instruction tokens, are retained across all layers. Following SnapKV (Li et al., 2024), the selection of the remaining tokens is then guided by the attention scores derived from these instruction tokens—tokens receiving higher attention scores are deemed more relevant to the generation process and are thus their KV states are prioritized for retention in the GPU cache.

In a typical LLM, the attention mechanism in each head h is calculated using the formula:

$$A^{h} = \operatorname{softmax}(\mathbf{Q}^{h} \cdot (\mathbf{K}^{h})^{\top} / \sqrt{d_{k}}), \tag{2}$$

where  $d_k$  denotes the dimension of the key vectors. Following (Li et al., 2024), we utilize a pooling layer at  $A^h$  to avoid the risk of being misled by some massive activation scores.

To quantify the importance of each token during the generation process, we measure the level of attention each token receives from the instruction tokens, and use this measurement to select important tokens for KV caching. Specifically, we compute the score of selecting i-th token for retention in the KV cache as  $s_i^h$  in each attention head h by:

$$s_i^h = \sum_{j \in [n-\alpha,n]} A_{ij}^h \tag{3}$$

where  $[n - \alpha, n]$  is the range of the instruction tokens. In each layer l and for each head h, the top  $k^l$  tokens with the highest scores are selected, and their respective KV caches are retained. All other KV caches are discarded and will not be utilized in any subsequent computations throughout the generation process.

# **5 Experiment**

We conduct comprehensive experiments to evaluate the effectiveness of PyramidKV on performance preserving and memory reduction.

### **5.1 Experiment Setup**

We maintain a fixed constant KV cache size for each layer for the baseline methods. In contrast, PyramidKV employs varying KV cache sizes across different layers. To ensure a fair comparison, we adjusted the average KV cache size in PyramidKV to match that of the baseline models, to keep the total memory consumption of all methods the same. We set *β* = 20 and *α* = 8. We use the same prompt for each dataset in all experiments.

### *5.1.1 Backbone LLMs*

We compare PyramidKV against baselines using state-of-the-art open-sourced LLMs, namely LLaMa-3-8B-Instruct, Mistral-7B-Instruct [\(Jiang et al., 2023\)](#page-10-0) and LLaMa-3-70B-Instruct. Testing examples are evaluated in a generative format, with answers generated by greedy decoding across all tasks to ensure a fair comparison.

### *5.1.2 Datasets*

We use LongBench [\(Bai et al., 2023\)](#page-9-7) to assess the performance of PyramidKV on tasks involving long-context inputs. LongBench is a meticulously designed benchmark suite that tests the capabilities of language models in handling extended documents and complex information sequences. This benchmark was created for comprehensive multi-task evaluation of long context inputs. It includes 17 datasets covering tasks such as single-document QA [\(Koˇcisky et al., 2018;](#page-10-4) [Dasigi et al., 2021\)](#page-9-8), multi-document QA [\(Yang et al., 2018;](#page-11-8) [Ho](#page-10-5) ` [et al., 2020\)](#page-10-5), summarization [\(Huang et al., 2021;](#page-10-6) [Zhong et al., 2021;](#page-11-9) [Fabbri et al., 2019b\)](#page-9-9), few-shot learning [\(Li and Roth, 2002;](#page-10-7) [Gliwa et al., 2019;](#page-10-8) [Joshi et al., 2017\)](#page-10-9), synthetic, and code generation [\(Guo et al., 2023;](#page-10-10) [Liu et al., 2023b\)](#page-10-11). The datasets feature an average input length ranging from 1,235 to 18,409 tokens (detailed average lengths can be found in [Ta](#page-6-0)[ble 1\)](#page-6-0), necessitating substantial memory for KV cache management. For all these tasks, we adhered to the standard metrics recommended by LongBench [\(Bai et al., 2023\)](#page-9-7) (i.e., F1 for QA, Rouge-L for summarization, Acc. for synthetic and Edit Sim. for code generation.) We refer readers to more details at [Appendix F.](#page-16-0)

#### *5.1.3 Baselines*

We compare PyramidKV with three baselines, all of which keep the same KV cache size across different layers, with different strategies for KV cache selection.

- **StreamingLLM (SLM)** [\(Xiao et al., 2023\)](#page-11-6) is an efficient framework that enables LLMs to accept infinite input length.
- **Heavy Hitter Oracle (H2O)** [\(Zhang et al., 2024\)](#page-11-5) is a KV cache compression policy that dynamically retains a balance of recent and Heavy Hitter (H2) tokens.

**SnapKV (SKV)** [\(Li et al., 2024\)](#page-10-2) automatically compresses KV caches by selecting clustered important tokens for each attention head.

**FullKV (FKV)** caches all keys and values for each input token in each layer. All methods are compared to the FullKV simultaneously.

# **5.2 Main Results**

The evaluation results from LongBench [\(Bai et al., 2023\)](#page-9-7) are shown in [Table 1](#page-6-0) and [Figure 3.](#page-6-1) In [Figure 3,](#page-6-1) we report the average score across datasets for 64, 96, 128, and 256 case sizes. In [Table 1,](#page-6-0) we report the results for two different KV cache sizes with 64 and 2048. These two sizes represent two distinct operational scenarios—the memory-efficient scenario and the

<span id="page-6-1"></span>![](_page_6_Figure_1.jpeg)

Figure 3: The evaluation results from LongBench (Bai et al., 2023) across 64, 96, 128 and 256 cache sizes at LLaMa-3-8B-Instruct (Left), Mistral-7B-Instruct (Middle) and LLaMa-3-70B-Instruct (Right). The evaluation metrics are the average score of LongBench across datasets. PyramidKV outperforms H2O (Zhang et al., 2024), SnapKV (Li et al., 2024) and StreamingLLM (Xiao et al., 2023), especially in small KV cache sizes.

|             | Single                | -Docum                |                       |                      | i-Documer      |                       |                | nmariza        |                |                | -shot Le              | 0                     | Synt  | hetic          | Co    | ode    |       |
|-------------|-----------------------|-----------------------|-----------------------|----------------------|----------------|-----------------------|----------------|----------------|----------------|----------------|-----------------------|-----------------------|-------|----------------|-------|--------|-------|
| Method      | NrtvQA                | Qasper                | MEen                  | HorpotQ <sup>A</sup> | 2WikiMQ!       | Musique               | GovReport      | QMSum          | MultiNew       | TREC           | TriviaQA              | SAMSum                | PCoun | PRE            | rec   | RB-P   | Avg.  |
|             | 18409                 | 3619                  | 4559                  | 9151                 | 4887           | 11214                 | 8734           | 10614          | 2113           | 5177           | 8209                  | 6258                  | 11141 | 9289           | 1235  | 4206   |       |
|             |                       |                       |                       |                      |                | LlaMa                 | -3-8B-Instr    | uct, KV S      | ize = Full     |                |                       |                       |       |                |       |        |       |
| FKV         | 25.70                 | 29.75                 | 41.12                 | 45.55                | 35.87          | 22.35                 | 25.63          | 23.03          | 26.21          | 73.00          | 90.56                 | 41.88                 | 4.67  | 69.25          | 58.05 | 50.77  | 41.46 |
|             |                       |                       |                       |                      |                | LlaM                  | a-3-8B-Inst    | ruct, KV       | Size = 64      |                |                       |                       |       |                |       |        |       |
| SKV         | 19.86                 | 9.09                  | 27.89                 | 37.34                | 28.35          | 18.17                 | 15.86          | 20.80          | 16.41          | 38.50          | 85.92                 | 36.32                 | 5.22  | 69.00          | 51.78 | 48.38  | 33.05 |
| H2O         | 20.80                 | 11.34                 | 27.03                 | 37.25                | 30.01          | 17.94                 | 18.29          | 21.49          | 19.13          | 38.00          | 84.70                 | 37.76                 |       | 69.33          |       |        |       |
| SLM         | 17.44                 | 8.68                  | 22.25                 | 35.37                | 31.51          | 15.97                 | 15.46          | 20.06          | 14.64          | 38.00          | 72.33                 | 29.10                 |       | 69.50          |       |        |       |
| Ours        | 21.13                 | 14.18                 | 30.26                 | 35.12                | 23.76          | 16.17                 | 18.33          | 21.65          | 19.23          | 58.00          | 88.31                 | 37.07                 | 5.23  | 69.50          | 52.61 | 45.74  | 34.76 |
|             |                       |                       |                       |                      |                | LlaMa                 | -3-8B-Instri   |                |                | 1              |                       |                       |       |                |       |        |       |
| SKV         | 25.86                 | 29.55                 | 41.10                 | 44.99                | 35.80          | 21.81                 | 25.98          | 23.40          | 26.46          | 73.50          | 90.56                 | 41.66                 |       | 69.25          |       |        |       |
| SLM         | 21.71                 | 25.78                 | 38.13                 | 40.12                | 32.01          | 16.86                 | 23.14          | 22.64          | 26.48          | 70.00          | 83.22                 | 31.75                 |       | 68.50          |       |        |       |
| H2O<br>Ours | 25.56<br>25.40        | 26.85<br><b>29.71</b> | 39.54<br>40.25        | 44.30<br>44.76       | 32.92<br>35.32 | 21.09<br><b>21.98</b> | 24.68<br>26.83 | 23.01<br>23.30 | 26.16<br>26.19 | 53.00<br>73.00 | 90.56<br>90.56        | 41.84<br><b>42.14</b> |       | 69.25<br>69.25 |       |        |       |
|             |                       |                       |                       |                      |                |                       | al-7B-Instru   | ict. KV S      | ize = Full     |                |                       |                       |       |                |       |        |       |
| FKV         | 26.90                 | 33.07                 | 49.20                 | 43.02                | 27.33          | 18.78                 | 32.91          | 24.21          | 26.99          | 71.00          | 86.23                 | 42.65                 | 2.75  | 86.98          | 56 96 | 54 52  | 42 71 |
| 1111        | 20.70                 | 00.07                 | 17.20                 | 10.02                | 27.00          |                       | ral-7B-Instr   |                |                | 71.00          | 00.20                 | 12.00                 | 2 0   | 00.70          | 00.70 | 01.02  | 12.71 |
| SKV         | 16.94                 | 17.17                 | 39.51                 | 36.87                | 22.26          | 15.18                 | 14.75          | 20.35          | 21.45          | 37.50          | 84.16                 | 37.28                 | 4.50  | 61.13          | 12.40 | 28 11  | 20.72 |
| SLM         | 15.01                 | 13.84                 | 28.74                 | 30.97                | 24.50          | 13.42                 | 13.25          | 19.46          | 19.17          | 35.50          | 76.91                 | 29.61                 |       | 27.33          |       |        |       |
| H2O         | 18.19                 | 19.04                 | 37.40                 | 30.18                | 22.22          | 13.77                 | 16.60          | 21.52          | 21.98          | 37.00          | 81.02                 | 38.62                 |       | 66.03          |       |        |       |
| Ours        | 20.91                 | 20.21                 | 39.94                 | 33.57                | 22.87          | 15.70                 | 17.31          | 21.23          | 21.41          | 54.00          | 81.98                 | 36.96                 |       | 60.83          |       |        |       |
|             |                       |                       |                       |                      |                | Mistra                | ıl-7B-Instru   | ct, KV Si      | ze = 2048      |                |                       |                       |       |                |       |        |       |
| SKV         | 25.89                 | 32.93                 | 48.56                 | 42.96                | 27.42          | 19.02                 | 26.56          | 24.47          | 26.69          | 70.00          | 86.27                 | 42.57                 | 5.50  |                |       |        | 41.56 |
| SLM         | 20.31                 | 26.64                 | 45.72                 | 35.25                | 24.31          | 12.20                 | 27.47          | 21.57          | 24.51          | 68.50          | 71.95                 | 31.19                 |       | 22.56          |       |        |       |
| H2O         | 25.76                 | 31.10                 | 49.03                 | 40.76                | 26.52          | 17.07                 | 24.81          | 23.64          | 26.60          | 55.00          | 86.35                 | 42.48                 |       | 88.15          |       |        |       |
| Ours        | 25.53                 | 32.21                 | 48.97                 | 42.26                | 27.50          | 19.36                 | 26.60          | 23.97          | 26.73          | 71.00          | 86.25                 | 42.94                 | 4.50  | 87.90          | 55.12 | 47.21  | 41.63 |
|             |                       |                       |                       |                      |                |                       | 3-70B-Inst     |                |                |                |                       |                       |       |                |       |        |       |
| FKV         | 27.75                 | 46.48                 | 49.45                 | 52.04                | 54.90          | 30.42                 | 32.37          | 22.27          | 27.58          | 73.50          | 92.46                 | 45.73                 | 12.50 | 72.50          | 40.96 | 63.91  | 46.55 |
|             |                       |                       |                       |                      |                |                       | -3-70B-Inst    |                |                |                |                       |                       |       |                |       |        |       |
| SKV         | 23.92                 | 31.09                 | 36.54                 | 46.66                | 50.40          | 25.30                 | 18.05          | 21.11          | 19.79          | 41.50          | 91.06                 | 40.26                 |       | 72.50          |       |        |       |
| SLM         | 22.07                 | 23.53                 | 27.31                 | 43.21                | 51.66          | 23.85                 | 16.62          | 19.74          | 15.20          | 39.50          | 76.89                 | 33.06                 |       | 72.50          |       |        |       |
| H2O         | 25.45                 | 34.64                 | 33.23                 | 48.25                | 50.30          | 24.88                 | 20.03          | 21.50          | 21.39          | 42.00          | 90.36                 | 41.58                 |       | 71.50          |       |        |       |
| Ours        | 25.47                 | 36.71                 | 42.29                 | 47.08                | 46.21          | 28.30                 | 20.60          | 21.62          | 21.62          | 64.50          | 89.61                 | 41.28                 | 12.50 | 72.50          | 45.54 | 36.30  | 42.01 |
| CICI        | 07.50                 | 45.10                 | 47.01                 | <b>=2</b> 00         |                |                       | 3-70B-Instr    |                |                |                | 02.20                 | 45.50                 | 10.00 | <b>50.5</b> 0  | 44 85 | co. a= | 16.00 |
| SKV         | 26.73                 | 45.18                 | 47.91                 | 52.00                | 55.24          | 30.48                 | 28.76          | 22.35          | 27.31          | 72.50          | 92.38                 | 45.58                 |       | 72.50          |       |        |       |
| SLM<br>H2O  | 26.69<br><b>27.67</b> | 41.01<br><b>46.51</b> | 35.97<br><b>49.54</b> | 46.55<br>51.49       | 52.98<br>53.85 | 25.71<br>29.97        | 27.81<br>28.57 | 20.81<br>22.79 | 27.16<br>27.53 | 69.00<br>59.00 | 91.55<br><b>92.63</b> | 44.02<br>45.94        |       | 72.00<br>72.50 |       |        |       |
| Ours        | 27.22                 | 46.19                 | 48.72                 | 51.49                | 54.56          | 31.11                 | 29.76          | 22.79          | 27.33          | 73.50          | 91.88                 | 45.47                 |       | 72.50          |       |        |       |
|             |                       | 10.17                 | 10.72                 | 31.02                | 01.00          | 51.11                 |                | 00             |                |                | ,1.00                 | 10.17                 | 12.00 |                | 11.00 | 07.12  |       |

<span id="page-6-0"></span>Table 1: Performance comparison of PyramidKV (Ours) with SnapKV (SKV), H2O, StreamingLLM (SLM) and FullKV (FKV) on LongBench for LlaMa-3-8B-Instruct, Mistral-7B-Instruct and LlaMa-3-70B-Instruct. PyramidKV generally outperforms other KV Cache compression methods across various KV Cache sizes and LLMs. The performance strengths of PyramidKV are more evident in small KV Cache sizes (i.e. KV Size = 64).

performance-preserving scenario, respectively for a trade-off between memory and model performance. In Appendix N, we report results of KV cache sizes with 64, 96, 128 and 2048.

Overall, PyramidKV preserves the performance with only 12% of the KV cache and it consistently surpasses other method across a range of KV cache sizes and different backbone models, with its performance advantages becoming particularly pronounced in memory-

![](_page_7_Figure_1.jpeg)

<span id="page-7-0"></span>Figure 4: Results of the Fact Retrieval Across Context Lengths ("Needle In A HayStack") test in **LlaMa-3-70B-Instruct** with **8k** context size in **128** KV cache size. The vertical axis of the table represents the depth percentage, and the horizontal axis represents the length.

constrained environments where only about 0.8% of the KV cache from the prompt is retained. Upon examining specific tasks, PyramidKV demonstrates a notably superior performance on the TREC task, a few-shot question answering challenge. This suggests that the model effectively aggregates information from the few-shot examples, highlighting the potential for further investigation into in-context learning tasks.

Notably, we initially observe the pyramidal attention patterns from the visualization analysis on the multi-document QA task [\(Figure 2\)](#page-2-1), but the pyramid heuristic has demonstrated its effectiveness on a range of other LongBench tasks (e.g., single-document QA, In-Context Learning), suggesting its promising generalizability beyond multi-document QA.

The performance advantage of PyramidKV increases as the KV cache memory decreases. By focusing on optimizing budget allocation across layers, PyramidKV accurately allocates resources in memory-constrained scenarios, ensuring that retained information is effectively preserved to maintain model performance. Moreover, as in long bench results shown in

[Table 1,](#page-6-0) even in the performance-preserving scenario (i.e., KV cache size = 2048), PyramidKV improves the performance over baseline methods and even outperforms FullKV.

Among the 16 datasets, the tasks where our proposed method performs slightly worse than the baseline are mostly saturated (e.g., HotpotQA, Musique, etc under the LlaMa-3- 8B-Instruct setting with KV Size = 64, as shown in Table 1). In these cases, our method is only marginally inferior to the baseline and remains competitive. Conversely, on tasks with greater potential for improvement (e.g., Qasper, MF-en, TREC, TriviaQA, etc under the same setting), our method significantly outperforms the baseline. Consequently, the overall average performance of our method surpasses that of the baselines. Notably, these tasks include several In-Context Learning tasks (i.e., TREC), our method enjoys best performance gain at In-Context Learning tasks.

### **5.3 Discussion and Insights**

### *5.3.1 PyramidKV Preserves the Long-Context Understanding Ability*

We conduct the "Fact Retrieval Across Context Lengths" (Needle In A Haystack) experiment [\(Liu et al., 2023a;](#page-10-12) [Fu et al., 2024\)](#page-9-3), which is a dataset designed to test whether a model can find key information in long input sequences, to evaluate the in-context retrieval capabilities of LLMs when utilizing various KV cache compression methods. For this purpose, we employ **LlaMa-3-70B-Instruct** as our base, with context lengths extending up to 8k. We compared several KV cache compression techniques (PyramidKV, SnapKV [\(Li et al., 2024\)](#page-10-2), and H2O [\(Zhang et al., 2024\)](#page-11-5)) at cache sizes of **128** and full cache. The results, presented in [Figure 4](#page-7-0) [1](#page-8-0) . The results demonstrate that with only 128 KV cache retained, PyramidKV effectively maintains the model's ability to understand short contexts, and shows only modest degradation for longer contexts. In contrast, other KV cache compression methods significantly hinder the performance of LLMs. Notably, for the larger model (**LlaMa-3-70B-Instruct**), PyramidKV achieves 100.0 Acc. performance, matching the results of FullKV, thereby demonstrating its ability to preserve long-context comprehension with a substantially reduced KV cache. We adopt the haystack setting of haystack formed from a long corpus for the Needle In A Haystack task as [Wu et al.](#page-11-4) [\(2024\)](#page-11-4).

#### *5.3.2 PyramidKV Significantly Reduces Memory with Limited Performance Drop*

In this section, we study how sensitive the methods are with different sizes of KV cache. We report the KV cache memory reduction in Table [2.](#page-8-1) We evaluate the memory consumption of LLaMa-3-8B-Instruct. Specifically, we evaluate the memory consumption of all methods with a fixed batch size of 1, a sequence length of 8192, and model weights in fp16 format. We observe that PyramidKV substantially reduces the KV cache memory across different numbers of cache sizes. We also present that the allocation strategy and score-based selection add minimal complexity in the inference phase as [Appendix L.](#page-20-1)

| cache size | Memory | Compression Ratio | QMSum | TREC  | TriviaQA | PCount | PRe   | Lcc   |
|------------|--------|-------------------|-------|-------|----------|--------|-------|-------|
| 512        | 428M   | 6.3%              | 22.80 | 71.50 | 90.61    | 5.91   | 69.50 | 58.16 |
| 1024       | 856M   | 12.5%             | 22.55 | 71.50 | 90.61    | 5.91   | 69.50 | 58.16 |
| 2048       | 1712M  | 25.0%             | 22.55 | 72.00 | 90.56    | 5.58   | 69.25 | 56.79 |
| Full       | 6848M  | 100.0%            | 23.30 | 73.00 | 90.56    | 5.22   | 69.25 | 58.76 |

<span id="page-8-1"></span>Table 2: Memory reduction effect and benchmark result by using PyramidKV. We conducted a comparison of memory consumption between the Llama-3-8B-Instruct model utilizing the Full KV cache and the Llama-3-8B-Instruct model compressed with the PyramidKV.

<span id="page-8-0"></span><sup>1</sup>Additional results with 64, 96 and 128 KV cache sizes with **LlaMa-3-8B-Instruct** at 8k context length, **LlaMa-3-70B-Instruct** at 8k context length, and **Mistral-7B-Instruct** [\(Jiang et al., 2023\)](#page-10-0) at 32k context length are available in [Appendix P](#page-24-0)

# **6 Conclusion**

In this study, we investigate Pyramidal Information Funneling, the intrinsic attention patterns of Large Language Models (LLMs) when processing long context inputs. Motivated by this discovery, we design a novel KV cache compression approach PyramidKV that utilizes this information flow pattern. Our method excels in memory-constrained settings, preserves long-context understanding ability, and significantly reduces memory usage.

# **References**

- <span id="page-9-0"></span>•Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023.
- <span id="page-9-7"></span>Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, et al. Longbench: A bilingual, multitask benchmark for long context understanding. *arXiv preprint arXiv:2308.14508*, 2023.
- <span id="page-9-10"></span>Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. *arXiv preprint arXiv:2403.06764*, 2024a.
- <span id="page-9-5"></span>Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. Longlora: Efficient fine-tuning of long-context large language models. *arXiv preprint arXiv:2309.12307*, 2023.
- <span id="page-9-11"></span>Zhuoming Chen, Ranajoy Sadhukhan, Zihao Ye, Yang Zhou, Jianyu Zhang, Niklas Nolte, Yuandong Tian, Matthijs Douze, Leon Bottou, Zhihao Jia, and Beidi Chen. Magicpig: Lsh sampling for efficient llm generation, 2024b. URL <https://arxiv.org/abs/2410.16179>.
- <span id="page-9-1"></span>Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality, March 2023. URL <https://lmsys.org/blog/2023-03-30-vicuna/>.
- <span id="page-9-8"></span>Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A Smith, and Matt Gardner. A dataset of information-seeking questions and answers anchored in research papers. In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 4599–4610, 2021.
- <span id="page-9-4"></span>Yiran Ding, Li Lyna Zhang, Chengruidong Zhang, Yuanyuan Xu, Ning Shang, Jiahang Xu, Fan Yang, and Mao Yang. Longrope: Extending llm context window beyond 2 million tokens. *arXiv preprint arXiv:2402.13753*, 2024.
- <span id="page-9-6"></span>Harry Dong, Xinyu Yang, Zhenyu Zhang, Zhangyang Wang, Yuejie Chi, and Beidi Chen. Get more with less: Synthesizing recurrence with kv cache compression for efficient llm inference. *arXiv preprint arXiv:2402.09398*, 2024.
- <span id="page-9-2"></span>Alexander Fabbri, Irene Li, Tianwei She, Suyi Li, and Dragomir Radev. Multi-news: A large-scale multi-document summarization dataset and abstractive hierarchical model. In Anna Korhonen, David Traum, and Lluís Màrquez, editors, *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 1074–1084, Florence, Italy, July 2019a. Association for Computational Linguistics. doi: 10.18653/v1/P19-1102. URL <https://aclanthology.org/P19-1102>.
- <span id="page-9-9"></span>Alexander Richard Fabbri, Irene Li, Tianwei She, Suyi Li, and Dragomir Radev. Multi-news: A large-scale multi-document summarization dataset and abstractive hierarchical model. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 1074–1084, 2019b.
- <span id="page-9-3"></span>Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Hannaneh Hajishirzi, Yoon Kim, and Hao Peng. Data engineering for scaling language models to 128k context. *arXiv preprint arXiv:2402.10171*, 2024.

- <span id="page-10-3"></span>Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. Model tells you what to discard: Adaptive kv cache compression for llms. *arXiv preprint arXiv:2310.01801*, 2023.
- <span id="page-10-8"></span>Bogdan Gliwa, Iwona Mochol, Maciej Biesek, and Aleksander Wawer. Samsum corpus: A human-annotated dialogue dataset for abstractive summarization. *EMNLP-IJCNLP 2019*, page 70, 2019.
- <span id="page-10-10"></span>Daya Guo, Canwen Xu, Nan Duan, Jian Yin, and Julian McAuley. Longcoder: A long-range pre-trained language model for code completion. *arXiv preprint arXiv:2306.14893*, 2023.
- <span id="page-10-13"></span>Chi Han, Qifan Wang, Wenhan Xiong, Yu Chen, Heng Ji, and Sinong Wang. Lm-infinite: Simple on-the-fly length generalization for large language models. *arXiv preprint arXiv:2308.16137*, 2023.
- <span id="page-10-5"></span>Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. In *Proceedings of the 28th International Conference on Computational Linguistics*, pages 6609–6625, 2020.
- <span id="page-10-6"></span>Luyang Huang, Shuyang Cao, Nikolaus Parulian, Heng Ji, and Lu Wang. Efficient attentions for long document summarization. In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 1419–1436, 2021.
- <span id="page-10-0"></span>Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. *arXiv preprint arXiv:2310.06825*, 2023.
- <span id="page-10-15"></span>Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir H Abdi, Dongsheng Li, Chin-Yew Lin, et al. Minference 1.0: Accelerating pre-filling for long-context llms via dynamic sparse attention. *arXiv preprint arXiv:2407.02490*, 2024.
- <span id="page-10-9"></span>Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1601–1611, 2017.
- <span id="page-10-4"></span>Tomáš Koˇcisky, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, Gábor ` Melis, and Edward Grefenstette. The narrativeqa reading comprehension challenge. *Transactions of the Association for Computational Linguistics*, 6:317–328, 2018.
- <span id="page-10-14"></span>Wonbeom Lee, Jungi Lee, Junghwan Seo, and Jaewoong Sim. Infinigen: Efficient generative inference of large language models with dynamic kv cache management, 2024. URL <https://arxiv.org/abs/2406.19707>.
- <span id="page-10-7"></span>Xin Li and Dan Roth. Learning question classifiers. In *COLING 2002: The 19th International Conference on Computational Linguistics*, 2002.
- <span id="page-10-2"></span>Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. Snapkv: Llm knows what you are looking for before generation. *arXiv preprint arXiv:2404.14469*, 2024.
- <span id="page-10-12"></span>Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts, 2023a.
- <span id="page-10-11"></span>Tianyang Liu, Canwen Xu, and Julian McAuley. Repobench: Benchmarking repository-level code auto-completion systems, 2023b.
- <span id="page-10-1"></span>Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. Code llama: Open foundation models for code. *arXiv preprint arXiv:2308.12950*, 2023.

- <span id="page-11-7"></span>Mingjie Sun, Xinlei Chen, J Zico Kolter, and Zhuang Liu. Massive activations in large language models. *arXiv preprint arXiv:2402.17762*, 2024.
- <span id="page-11-0"></span>Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023a.
- <span id="page-11-1"></span>Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023b.
- <span id="page-11-3"></span>Daniel Waddington, Juan Colmenares, Jilong Kuang, and Fengguang Song. Kv-cache: A scalable high-performance web-object cache for manycore. In *2013 IEEE/ACM 6th International Conference on Utility and Cloud Computing*, pages 123–130. IEEE, 2013.
- <span id="page-11-10"></span>Lean Wang, Lei Li, Damai Dai, Deli Chen, Hao Zhou, Fandong Meng, Jie Zhou, and Xu Sun. Label words are anchors: An information flow perspective for understanding in-context learning. *arXiv preprint arXiv:2305.14160*, 2023.
- <span id="page-11-4"></span>Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, and Yao Fu. Retrieval head mechanistically explains long-context factuality. *arXiv preprint arXiv:2404.15574*, 2024.
- <span id="page-11-6"></span>Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. *arXiv preprint arXiv:2309.17453*, 2023.
- <span id="page-11-11"></span>Dongjie Yang, XiaoDong Han, Yan Gao, Yao Hu, Shilin Zhang, and Hai Zhao. Pyramidinfer: Pyramid kv cache compression for high-throughput llm inference. *arXiv preprint arXiv:2405.12532*, 2024.
- <span id="page-11-8"></span>Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 2369–2380, 2018.
- <span id="page-11-5"></span>Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, et al. H2o: Heavy-hitter oracle for efficient generative inference of large language models. *Advances in Neural Information Processing Systems*, 36, 2024.
- <span id="page-11-9"></span>Ming Zhong, Da Yin, Tao Yu, Ahmad Zaidi, Mutethia Mutuma, Rahul Jha, Ahmed Hassan, Asli Celikyilmaz, Yang Liu, Xipeng Qiu, et al. Qmsum: A new benchmark for querybased multi-domain meeting summarization. In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 5905–5921, 2021.
- <span id="page-11-2"></span>Dawei Zhu, Nan Yang, Liang Wang, Yifan Song, Wenhao Wu, Furu Wei, and Sujian Li. Pose: Efficient context window extension of llms via positional skip-wise training. *arXiv preprint arXiv:2309.10400*, 2023.

# **A Limitations**

Our experiments were limited to three base models: LLAMA-3-8B-Instruct, LLAMA-3-70B-Instruct and Mistral-7B-Instruct. While these models demonstrated consistent trends, the robustness of our findings could be enhanced by testing a broader array of model families, should resources permit. Additionally, our research was conducted exclusively in English, with no investigations into how these findings might be transferred to other languages. Expanding the linguistic scope of our experiments could provide a more comprehensive understanding of the applicability of our results globally. Based on our results at LongBench and Needle-in-a-HayStack experiment, PyramidKV generally works decently in most of the language tasks (i.e., Single-Document QA, Multi-Document QA, Summerization, Few-Shot In-Context Learning, etc.). Although we observe that PyramidKV performs better in some tasks (i.e., Few-Shot In-Context Learning) compared with some other tasks (i.e., Summerization), we have not observed cases that the decoding result collapses at some tasks. This remains a new topic for future work to explore.

# **B Future Work**

Our investigation on PyramidKV highlights considerable opportunities for optimizing KV cache compression by adjusting the number of KV caches retained according to the distinct attention patterns of each layer (or even for each head). For instance, the retention of KV cache for each layer could be dynamically modified based on real-time analysis of the attention matrices, ensuring that the compression strategy is consistently aligned with the changing attention dynamics within LLMs. Furthermore, our experiments indicate that PyramidKV significantly surpasses other methods in few-shot learning tasks, suggesting promising applications of KV cache in in-context learning. This approach could potentially enable the use of more shots within constrained memory limits.

![](_page_12_Figure_5.jpeg)

<span id="page-12-0"></span>Figure 5: Attention patterns of retrieval-augmented generation across layers in Mistral-7B-Instruct model [\(Jiang et al., 2023\)](#page-10-0)

![](_page_13_Figure_1.jpeg)

<span id="page-13-0"></span>Figure 6: Attention patterns of retrieval-augmented generation across layers in Mixtral-8x7B-Instruct Mixture-of-Experts model.

# **C Related Work**

**Interpretation of LLMs** Prior research has shown that attention matrices in LLMs are typically sparse [\(Chen et al., 2024a;](#page-9-10) [Xiao et al., 2023;](#page-11-6) [Zhang et al., 2024\)](#page-11-5), focusing disproportionately on a few tokens. For instance, [Xiao et al.](#page-11-6) [\(2023\)](#page-11-6) identified an "attention sink" phenomenon, where maintaining the Key and Value (KV) states of the first few tokens can substantially restore the performance of windowed attention, despite these tokens not being semantically crucial. Similarly, [Sun et al.](#page-11-7) [\(2024\)](#page-11-7) identified a "massive activations" pattern, where a minority of activations show significantly larger values than others within LLMs. Interestingly, these values remain relatively constant across different inputs and act as critical bias terms in the model.

Further explorations in this field reveal distinct patterns across various attention heads and layers. [Li et al.](#page-10-2) [\(2024\)](#page-10-2) observed that certain attention heads consistently target specific prompt attention features during decoding. Additionally, [Wang et al.](#page-11-10) [\(2023\)](#page-11-10) discovered that in In-Context Learning scenarios, label words in demonstration examples serve as semantic anchors. In the lower layers of an LLM, shallow semantic information coalesces around these label words, which subsequently guide the LLMs' final output predictions by serving as reference points. Recently, [Wu et al.](#page-11-4) [\(2024\)](#page-11-4) revealed that a special type of attention head, the so-called retrieval head, is largely responsible for retrieving information. Inspired by these findings that the attention mechanism exhibits varying behaviors across different layers, we discovered that "Massive Activation" does not consistently manifest across all layers in long context sequences; instead, it predominantly occurs in the upper layers. Additionally, we identified a novel trend of information aggregation specific to long-context inputs, which will be further explained in [§3.](#page-2-0)

**KV Cache Compression** There has been a growing interest in addressing LLMs' memory constraints on processing long context inputs. FastGen [\(Ge et al., 2023\)](#page-10-3) introduces an adaptive KV cache management strategy that optimizes memory use by tailoring retention tactics to the specific nature of attention heads. This method involves evicting long-range contexts from heads that prioritize local interactions, discarding non-special tokens from heads focused on special tokens, and maintaining a standard KV cache for heads that engage broadly across tokens. SnapKV [\(Li et al., 2024\)](#page-10-2) improves efficiency by compressing KV caches via selecting/clustering significant KV positions based on their attention scores. Heavy Hitter Oracle (H2O) [\(Zhang et al., 2024\)](#page-11-5) implements a dynamic eviction policy that effectively balances the retention of recent and historically significant tokens, optimizing memory usage while preserving essential information. StreamingLLM [\(Xiao et al., 2023\)](#page-11-6) enables LLMs trained on finite attention windows to handle infinite sequence lengths without fine-tuning, thus expanding the models' applicability to broader contexts. LM-Infinite [\(Han et al., 2023\)](#page-10-13) allows LLMs pre-trained with 2K or 4K-long segments to generalize to up to 200M length inputs while retaining perplexity without parameter updates.

While these approaches have significantly advanced the efficient management of memory for LLMs, they generally apply a fixed KV cache size across all layers. In contrast, our investigations into the attention mechanisms across different layers of LLMs reveal that the attention patterns vary from layer to layer, making a one-size-fits-all approach to KV cache management suboptimal. In response to this inefficiency, we propose a novel KV cache compression method, called PyramidKV that allocates different KV cache budgets across different layers, tailored to the unique demands and operational logic of each layer's attention mechanism. This layer-specific strategy takes a significant step toward balancing both memory efficiency and model performance, addressing a key limitation in existing methodologies.

# **D Pyramidal Information Funneling**

[Figure 5](#page-12-0) and [Figure 6](#page-13-0) shows the attention patterns of one QA example over six different layers (i.e., 0, 6, 12, 18, 24, and 30) for Mistral-7B-Instruct model and Mixtral-8x7B-Instruct Mixture-of-Experts model. [Figure 5](#page-12-0) and [Figure 6](#page-13-0) demonstrate that the Pyramidal Information Funneling phenomenon is also evident in both the Mistral model and Mixtral model . The results reveal that, akin to Llama-like models, Mistral exhibit a progressively narrowing attention focus across layers. This supports the universality of the Pyramidal Information Funneling phenomenon across diverse model families. We hope this addresses your concern and underscores the generalizability of our findings.

Our analysis uniquely examines attention metrics across all transformer layers, from 0 to 30, leading to the discovery of a key phenomenon we term Pyramidal Information Funneling.

[Lee et al.](#page-10-14) [\(2024\)](#page-10-14) conducted a limited investigation into attention patterns, focusing only on the lower layer (layer 0) and a single upper layer (layer 18). While [Lee et al.](#page-10-14) [\(2024\)](#page-10-14) noted that attention becomes more skewed in upper layers, it did not provide a fine-grained observation of attention patterns across all layers. In contrast, our study reveals several novel findings:

- **Localized Attention**: We observe that attention progressively narrows its focus, targeting specific components within the input sequence.
- **Massive Attention Mechanism**: In the upper layers, attention heavily concentrates on a small set of critical tokens. Notably, these tokens are not limited to the leading positions, as observed in [Lee et al.](#page-10-14) [\(2024\)](#page-10-14), but also appear at regular intervals across the sequence. The discrepancy arises from differences in input settings, with [Lee](#page-10-14) [et al.](#page-10-14) [\(2024\)](#page-10-14) identifying massive attention only at the initial tokens.

These insights motivated us to propose a token-selection method based on the highest attention scores in the upper layers, rather than solely relying on tokens from earlier positions.

To the best of our knowledge, [Chen et al.](#page-9-11) [\(2024b\)](#page-9-11) has not analyzed attention patterns across transformer layers.

Therefore, although [Lee et al.](#page-10-14) [\(2024\)](#page-10-14) and [Chen et al.](#page-9-11) [\(2024b\)](#page-9-11) are considered contemporaneous with our work, making a comparison unnecessary, the perspective of our observation is considered novel compared with [Lee et al.](#page-10-14) [\(2024\)](#page-10-14) and [Chen et al.](#page-9-11) [\(2024b\)](#page-9-11). Moreover, although [Lee et al.](#page-10-14) [\(2024\)](#page-10-14) also observed attention patterns, the method we proposed based

![](_page_15_Picture_1.jpeg)

Figure 7: Illustration of PyramidKV. At the lower level of the transformer, the PyramidKV selects more keys and values based on the exhibited average attention pattern. Fewer keys and values at the higher level are selected based on the massive activation pattern, where we observe that attention scores are concentrated over local regions.

<span id="page-15-0"></span>on our observations is significantly different from [Lee et al.](#page-10-14) [\(2024\)](#page-10-14), further highlighting the novelty of our work.

# **E Details of Proposed Method**

Based on the pyramidal information funneling observed across different layers, PyramidKV consists of two steps: (1) Dynamically allocating different KV cache sizes/budgets across different layers; and (2) Selecting important KV vectors in each attention head for caching as [Figure 7.](#page-15-0)

Our decision to use an arithmetic sequence is driven by three key factors:

- **Alignment with Pyramidal Information Funneling Pattern**: Empirical observations reveal a pyramidal information funneling pattern, where lower layers exhibit dispersed attention while higher layers concentrate on fewer tokens. Inspired by this, we adopt the arithmetic sequence design to align with this natural progression.
- **Superior Empirical Performance**: Through extensive experimentation across diverse datasets, we compared various methods, including the arithmetic sequence and adaptive approaches. Results consistently showed that the arithmetic sequence method outperformed others.
- **Computational Efficiency**: The arithmetic sequence method introduces minimal computational overhead compared to adaptive approaches, which require dynamically computing cache budgets across layers.

To perform KV cache eviction, we use torch.gather. Below, we outline the memory allocation and release process of torch.gather:

- **Index Selection**: Identify the positions of the elements to extract from the input tensor.
- **Memory Location Calculation**: Compute the specific memory locations of the elements to be extracted using the strides of the input tensor across each dimension.

- Output Tensor Creation: Allocate memory to create a new output tensor and copy the selected elements to their corresponding positions in the output tensor.
- **Memory Management**: Since torch gather is not an in-place operation, it creates a new tensor to store the results, while the memory of the original input tensor is released.

The speed-up offered by PyramidKV is complementary to that achieved through tensor parallelism and pipeline parallelism, as these approaches are not mutually exclusive. PyramidKV can be seamlessly integrated with both tensor parallelism and pipeline parallelism.

### <span id="page-16-0"></span>F Details of Evaluation

We use LongBench (Bai et al., 2023) to assess the performance of PyramidKV on tasks involving long-context inputs. LongBench is a meticulously designed benchmark suite that tests the capabilities of language models in handling extended documents and complex information sequences. This benchmark was created for multi-task evaluation of long context inputs.

We present the details of metrics, language and data for LongBench at Table 3. We run all the experiments on NVIDIA A100.

| Dataset                                                     | Source                                     | Avg len                  | Metric                          | Language                      | #data             |
|-------------------------------------------------------------|--------------------------------------------|--------------------------|---------------------------------|-------------------------------|-------------------|
| Single-Document QA NarrativeQA Qasper MultiFieldQA-en       | Literature, Film<br>Science<br>Multi-field | 18,409<br>3,619<br>4,559 | F1<br>F1<br>F1                  | English<br>English<br>English | 200<br>200<br>150 |
| Multi-Document QA<br>HotpotQA<br>2WikiMultihopQA<br>MuSiQue | Wikipedia<br>Wikipedia<br>Wikipedia        | 9,151<br>4,887<br>11,214 | F1<br>F1<br>F1                  | English<br>English<br>English | 200<br>200<br>200 |
| Summarization<br>GovReport<br>QMSum<br>MultiNews            | Government report<br>Meeting<br>News       | 8,734<br>10,614<br>2,113 | Rouge-L<br>Rouge-L<br>Rouge-L   | English<br>English<br>English | 200<br>200<br>200 |
| Few-shot Learning<br>TREC<br>TriviaQA<br>SAMSum             | Web question<br>Wikipedia, Web<br>Dialogue | 5,177<br>8,209<br>6,258  | Accuracy (CLS)<br>F1<br>Rouge-L | English<br>English<br>English | 200<br>200<br>200 |
| Synthetic Task<br>PassageCount<br>PassageRetrieval-en       | Wikipedia<br>Wikipedia                     | 11,141<br>9,289          | Accuracy (EM)<br>Accuracy (EM)  | English<br>English            | 200<br>200        |
| Code Completion<br>LCC<br>RepoBench-P                       | Github<br>Github repository                | 1,235<br>4,206           | Edit Sim<br>Edit Sim            | Python/C#/Java<br>Python/Java | 500<br>500        |

<span id="page-16-1"></span>Table 3: An overview of the dataset statistics in LongBench (Bai et al., 2023). 'Source' denotes the origin of the context. 'Accuracy (CLS)' refers to classification accuracy, while 'Accuracy (EM)' refers to exact match accuracy.

### G License

LongBench: MIT

### H Handle Rotary Embedding after Tokens are Removed in PyramidKV

We keep the rotary embedding unchanged after tokens are removed, so that LLMs can still capture the exact position information even if the tokens are removed. StreamingLLM (Xiao et al., 2023) shows that rolling ky cache with the correct relative position is crucial for maintaining performance. This is because StreamingLLM is designed to mainly handle unlimited context sizes, where contexts exceed the LLM's fixed context length. Without changing the rotary embedding after token removal, LLMs would receive rotary embedding of a non-monotonic position sequence. For example, after the first KV cache compression, LLMs might receive the input position embedding as  $[0, 1, 2, 3, 3096, 3097, \cdots, 4096]$ , and the position embedding of the generated sequences could be  $[1005, 1006, 1007, \cdots]$ . The position sequence of  $[0, 1, 2, 3, 3096, \dots, 4096, 1005, 1006, 1007, \dots]$  is a non-monotonic sequence, which may negatively hurts the performance. In contrast, our targeting settings will not process unlimited context size. For example, given a input sequence of 4012 length, after KV cache compression, the position sequence would be  $[0,4,6,16,\cdots,3927,3987,4012]$ , and the position sequence of the generated tokens would be [4013, 4014,  $\cdots$ ]. By keeping the rotary embedding unchanged after the tokens are removed, the LLM avoids non-monotonic position sequences, and the LLM can capture the exact position information even if the tokens are shifted. Our preliminary results show that rolling KV cache with the correct relative position will slightly decrease the performance.

# I Ablation Study

In this section, we present an ablation study for hyperparameters and allocation strategies.

Based on our observations of the attention pattern, we find that a relatively stable, linear arithmetic decrease aligns more closely with the underlying structure of the pattern. We conduct experiments comparing various allocation strategies.

We conducted hyperparameter testing on the original development sets of 16 datasets in LongBench. The parameter  $\beta$  demonstrated remarkable stability, showing minimal sensitivity to varying hyperparameter settings, which highlights its robustness. Conversely,  $\alpha$  consistently produced superior results when set to 8 or 16. Consequently, these values were adopted for subsequent experiments. In Appendix H.2 and H.3, we further analyzed the impact of hyperparameter selection on KV cache budget allocation across different layers. The experiments reaffirmed that  $\beta$  had negligible influence on the outcomes, underscoring its stability. Meanwhile,  $\alpha$  continued to deliver optimal results at values of 8 and 16.

### I.1 Allocation Srategies

Based on our observations of the attention pattern, we find that a relatively stable, linear arithmetic decrease aligns more closely with the underlying structure of the pattern.

We conduct experiments comparing various pyramidal allocation strategies (i.e., linear decay strategy, geometric decay strategy and exponential decay strategy) with a cache size of 64 as Table 4 to confirm that a linear strategy is indeed optimal or preferable.

We also propose three adaptive allocation baselines, which are based on the entropy, Gini coefficient, and sparsity of the attention values at each layer. The weight of each layer is calculated based on its corresponding metric (entropy, Gini coefficient, or sparsity), and the budget is allocated accordingly. Specifically:

- Entropy-based allocation: Layers with higher entropy receive higher weights. Each layer's entropy is calculated based on the the layer's attention.
- Gini coefficient-based allocation: Layers with higher Gini coefficients receive higher weights. Each layer's Gini coefficient is calculated based on the the layer's attention

The empirical results as Table 4 consistently showed that the linear strategy outperformed its counterparts, establishing it as the most effective approach for our use case. The experiment strengthens the rationale for choosing the specific allocation method.

|         | Single  | -Docum | ent QA | Multi-   | Document- | QA      | Sun       | nmariza | tion      | Few   | -shot Le | arning | Syntl  | netic   | Code       | Avg.     |
|---------|---------|--------|--------|----------|-----------|---------|-----------|---------|-----------|-------|----------|--------|--------|---------|------------|----------|
| Stra.   | NrtvQA  | Qasper | MF-en  | HotpotQA | 2WikiMQA  | Musique | GovReport | QMSum   | MultiNews | TREC  | TriviaQA | SAMSum | PCount | PRe     | LCC RB     | 8        |
| Geo.    | 20.51   | 15.04  | 29.4   | 34.93    | 26.41     | 16.6    | 18.32     | 21.68   | 18.81     | 52    | 87.51    | 36.15  | 5.18   | 69.17 5 | 3.11 44.9  | 1 34.36  |
| Exp.    | 20.58   | 14.82  | 28.74  | 34.34    | 26.24     | 16.11   | 18.41     | 21.63   | 18.75     | 52.00 | 87.94    | 36.26  | 5.19   | 69.17 5 | 54.34 43.2 | 21 34.23 |
| Lin.    | 21.13   | 14.18  | 30.26  | 35.12    | 23.76     | 16.17   | 18.33     | 21.65   | 19.23     | 58.00 | 88.31    | 37.07  | 5.23   | 69.50 5 | 2.61 45.7  | 4 34.76  |
| Entropy | . 18.12 | 14.12  | 27.22  | 33.21    | 21.16     | 15.16   | 17.76     | 19.87   | 17.09     | 51    | 87.31    | 34.29  | 5.09   | 68.91 5 | 0.12 42.9  | 8 32.71  |
| Gini.   | 17.92   | 14.61  | 28.21  | 32.67    | 19.98     | 15.98   | 16.20     | 19.29   | 18.21     | 51.00 | 86.21    | 34.97  | 5.11   | 65.51 5 | 51.98 43.3 | 37 32.58 |

<span id="page-18-0"></span>Table 4: Ablation study of allocation strategies.

#### **I.2** Hyper Parameter $\alpha$

We present the study of  $\alpha$  for LlaMa-3-8B-Instruct in 128 KV cache size budget at Table 5.We find that a small alpha value (i.e., 8, 16) leads to better performance than a larger alpha value (i.e., 24, 32, 40, 48).

|    | Single | -Docum | ent QA | Multi-   | Document | QA      |           | nmarizat |           |       | -shot Le |        | Synt  |                   | Avg.    |
|----|--------|--------|--------|----------|----------|---------|-----------|----------|-----------|-------|----------|--------|-------|-------------------|---------|
| α  | NrtvQA | Qasper | MF-en  | HotpotQA | WikiMQA  | Musique | GovRePort | OMSum.   | MultiNews | TREC  | iriviaQA | SAMSum | PCoun | PRE LCC RBS       | ?       |
| 8  | 21.40  | 16.92  | 31.62  | 38.45    | 28.72    | 18.59   | 19.96     | 22.49    | 20.96     | 66.50 | 89.35    | 38.43  | 5.92  | 69.00 57.86 51.8  | 0 37.37 |
| 16 | 23.37  | 16.21  | 33.93  | 38.24    | 27.28    | 20.57   | 19.71     | 21.93    | 20.86     | 60.00 | 88.75    | 38.34  | 5.48  | 69.12 57.84 53.4  | 2 37.19 |
| 24 | 22.85  | 14.51  | 32.26  | 38.38    | 28.36    | 20.33   | 19.55     | 21.72    | 20.72     | 54.50 | 88.71    | 38.46  | 5.48  | 69.50 56.83 53.6  | 5 36.61 |
| 32 | 23.01  | 14.54  | 31.68  | 38.86    | 29.90    | 19.16   | 19.20     | 21.83    | 20.52     | 49.50 | 87.01    | 38.01  | 5.75  | 69.50 57.02 54.5  | 4 36.25 |
| 40 | 21.70  | 13.06  | 30.14  | 36.78    | 27.34    | 18.88   | 18.72     | 21.37    | 19.79     | 44.00 | 87.74    | 38.43  | 6.08  | 69.25 56.11 53.89 | 9 35.21 |
| 48 | 21.51  | 12.30  | 29.77  | 39.04    | 26.76    | 17.97   | 18.65     | 21.20    | 20.29     | 44.50 | 87.73    | 38.44  | 5.51  | 69.25 56.73 53.8  | 8 35.22 |

<span id="page-18-1"></span>Table 5: Ablation on  $\alpha$ .

#### I.3 Hyper Parameter $\beta$

One topic we want to analyze for our ablation study is the selection of  $\beta$ , which can determine the staircase. The smaller  $\beta$  is, the gentler the staircase is; the larger  $\beta$  is, the steeper the staircase is. We want to investigate the effect of  $\beta$  step size on the final result. Results on 128 KV cache size and LlaMa-3-8B-Instruct are shown in Table 6. The results at Table 6 show that using a relatively small value of  $\beta$  yields better outcomes, and PyramidKV is generally robust to the selection of  $\beta$ .

|    | Single | -Docum | ent QA | Multi    | -Documen | t QA    | Su        | mmariza | tion      | Few   | -shot L  | earning | Synt  | hetic | Code       | Avg.    |
|----|--------|--------|--------|----------|----------|---------|-----------|---------|-----------|-------|----------|---------|-------|-------|------------|---------|
| β  | NrtvQA | Qasper | MF-en  | HotpotQA | 2WikiMQA | Musique | GovReport | QMSum   | MultiNews | TREC  | friviaQA | SAMSum  | PCoun | PRE   | LCC RB     | 2       |
| 20 | 21.40  | 16.92  | 33.79  | 39.73    | 28.72    | 18.59   | 19.86     | 22.48   | 20.95     | 66.50 | 89.35    | 38.39   | 5.92  | 69.00 | 56.49 47.9 | 5 37.25 |
| 18 | 21.71  | 16.24  | 33.59  | 39.89    | 27.94    | 18.38   | 19.76     | 22.32   | 21.20     | 66.50 | 88.98    | 38.93   | 5.46  | 69.50 | 56.47 49.2 | 3 37.25 |
| 16 | 21.74  | 14.86  | 33.64  | 39.18    | 28.17    | 18.77   | 19.57     | 22.25   | 21.48     | 66.50 | 89.69    | 38.87   | 5.82  | 69.50 | 57.02 50.1 | 1 37.32 |
| 14 | 22.53  | 16.31  | 33.50  | 40.50    | 28.15    | 19.26   | 19.66     | 22.39   | 21.38     | 65.50 | 90.02    | 38.56   | 5.75  | 69.50 | 57.51 49.7 | 1 37.51 |

<span id="page-18-2"></span>Table 6: Ablation on  $\beta$ .

# J Integation with MInference

We would like to clarify that PyramidKV and MInference Jiang et al. (2024) are complementary approaches addressing different aspects of KV cache optimization. Specifically:

 MInference focuses on accelerating the generation of KV caches during the prefilling stage of LLM inference. In contrast, PyramidKV targets efficient KV cache management during LLM decoding.

To evaluate their respective strengths, we compared PyramidKV and MInference on Longbench using a KV cache size of 128. The results demonstrated the superior performance of PyramidKV.

Furthermore, we demonstrate that MInference and PyramidKV can be seamlessly integrated to achieve highly efficient inference while maintaining performance comparable to full attention. The results of MInference combined with PyramidKV, evaluated on Longbench with a KV cache size of 128, as PyramidKV + MInference hybrid approach.

|                                    | Single | -Docum                  | ent QA                  | Multi                   | -Documen                | t QA                    | Su                      | mmariza                 | tion                    | Few                  | v-shot Le               | arning                  | Synth        | etic C                                 | Code              | Avg.             |
|------------------------------------|--------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|----------------------|-------------------------|-------------------------|--------------|----------------------------------------|-------------------|------------------|
| Stra.                              | NrtvQA | Qasper                  | MF-en                   | HotpotQA                | 2WikiMQA                | Musique                 | GovReport               | OMSum                   | MultiNew                | TREC                 | TriviaQA                | SAMSum                  | PCount       | PRE LCC                                | RB-P              |                  |
| PyramidKV<br>MInference<br>M. + P. |        | 20.61<br>30.63<br>31.74 | 38.28<br>40.41<br>39.98 | 43.23<br>44.28<br>43.10 | 31.62<br>35.22<br>35.21 | 20.94<br>20.65<br>21.60 | 21.27<br>28.43<br>27.41 | 22.69<br>23.35<br>23.06 | 22.83<br>26.75<br>26.76 | 71<br>72.00<br>73.00 | 90.48<br>87.90<br>88.03 | 39.86<br>42.78<br>43.36 | 5.83<br>6.30 | 69.25 56.9<br>64.00 58.7<br>64.00 58.5 | 4 50.16<br>6 5.06 | 5 39.31<br>38.86 |

Table 7: Comparison between PyramidKV, MInference and MInference-PyramidKV hybrid method.

In summary, we demonstrate that PyramidKV outperforms MInference on Longbench. Furthermore, when integrated with MInference, PyramidKV enhances its performance even further.

### K Comparison with PyramidInfer

Our work differs from PyramidInfer in two key aspects:

- **Decay Strategy**: While PyramidInfer Yang et al. (2024) employs a geometric decay strategy, our method adopts an arithmetic decay strategy. We argue that the relatively stable and linear nature of arithmetic decay better aligns with the behavior of the attention mechanism. This strategy is derived from empirically observed attention patterns, aiming to closely match them. Notably, our approach also achieves superior results, as demonstrated in the experimental results presented in the table
- Token Selection: PyramidInfer discards tokens in earlier layers, preventing them from being reconsidered in later layers. In contrast, our method allows previously discarded tokens to be re-evaluated in higher layers, recognizing that these tokens may still hold relevance at different stages of the model's processing.
- Pyramidal Information Funneling Pattern: A key contribution of our work lies in identifying and leveraging the pyramidal information funneling phenomenon within attention mechanisms. Through in-depth analysis, we observe that attention tends to disperse in earlier layers and progressively concentrates on crucial tokens in higher layers. This insight forms the foundation of our arithmetic decay strategy, ensuring that our method aligns more naturally with these intrinsic patterns.

Despite some similarities between the two approaches, these differences lead to significantly distinct outcomes. As shown in Table 8, our method consistently outperforms PyramidInfer, highlighting the effectiveness of our design choices.

|            | Single   | -Docum | ent QA | Mult   | i-Documer | ıt QA    | Su    | ımmariza | tion  | Few-  | shot Lear | ning  | Syntl | netic    | Code       | Avg.    |
|------------|----------|--------|--------|--------|-----------|----------|-------|----------|-------|-------|-----------|-------|-------|----------|------------|---------|
| Stra.      | 4Qvari   | asper  | MF-en  | rootQA | 2WikiMQA  | , usique | Repor | MSum     | HiNew | TREC  | AQaiv:    | MSum  | Count | PRe 1    | cc RB-1    | ?       |
|            | 17,      | Qr.    | 141    | Hou    | 51N11     | Mr       | Gov.  | Qr.      | Mur   | 1. 1  | rr. 5     | PC.   | γC    |          | <b>y</b> - |         |
| Pyramidinf | er 20.42 | 12.77  | 25.21  | 35.81  | 25.83     | 16.88    | 18.27 | 21.78    | 18.52 | 51.00 | 88.54     | 35.76 | 5.61  | 69.25 53 | .21 44.12  | 2 33.94 |
| PyramidKV  | 7 21.13  | 14.18  | 30.26  | 35.12  | 23.76     | 16.17    | 18.33 | 21.65    | 19.23 | 58.00 | 88.31     | 37.07 | 5.23  | 69.50 52 | .61 45.7   | 4 34.76 |

<span id="page-19-0"></span>Table 8: Comparison between PyramidKV and Pyramidinfer.

# <span id="page-20-1"></span>**L PyramidKV will cause minimal extra inference overhead.**

The allocation strategy and score-based selection add minimal complexity in the inference phase compared to the computation required for next-token predictions as [Table 9.](#page-20-2) Each row shows the setting of using a specific "[Prompt length, Generation length]" combination. We show the inference speed comparison between total inference time, time for allocation strategy and time for score-based selection on LlaMa-3-8B-Instruct. Each cell is the latency measured in seconds. Furthermore, our budget allocation can be calculated before inference, requiring only a one-time computation. Thus, PyramidKV will cause minimal extra inference overhead.

| Prompt Length | Generation Length | Inference Time | Allocation Time | Selection Time |
|---------------|-------------------|----------------|-----------------|----------------|
| 512           | 512               | 18.26          | 0.0000003       | 0.0194         |
| 512           | 1024              | 34.69          | 0.000002        | 0.0133         |
| 512           | 2048              | 70.69          | 0.000003        | 0.013          |
| 512           | 4096              | 138.62         | 0.000005        | 0.013          |
| 1024          | 512               | 17.32          | 0.000002        | 0.0131         |
| 1024          | 1024              | 34.67          | 0.000002        | 0.01288        |
| 1024          | 2048              | 70.21          | 0.000005        | 0.01296        |
| 1024          | 4096              | 138.61         | 0.000003        | 0.01297        |
| 2048          | 512               | 17.48          | 0.000004        | 0.0128         |
| 2048          | 1024              | 34.78          | 0.000006        | 0.0129         |
| 2048          | 2048              | 69.50          | 0.000003        | 0.01297        |
| 2048          | 4096              | 138.59         | 0.000003        | 0.013          |
| 4096          | 512               | 17.58          | 0.000002        | 0.013          |
| 4096          | 1024              | 34.93          | 0.000004        | 0.0129         |
| 4096          | 2048              | 69.65          | 0.000002        | 0.013          |
| 4096          | 4096              | 138.87         | 0.000002        | 0.013          |

<span id="page-20-2"></span>Table 9: Extra inference overhead of PyramidKV

# **M Inference Speed Comparison**

PyramidKV does not require extra computation time for budget allocation at inference by design. We show the inference speed comparison between PyramidKV and baselines on LlaMa-3-8B-Instruct as [Table 10.](#page-21-0) Each row shows the setting of using a specific "[Prompt length, Generation length]" combination. Each cell is the latency measured in seconds. PyramidKV does not sacrifice the speed. PyramidKV provides performance improvement and memory saving while runs at a comparable speed compared with baselines (i.e. SnapKV [\(Li et al.,](#page-10-2) [2024\)](#page-10-2), StreamingLLM [\(Xiao et al., 2023\)](#page-11-6) and H2O [\(Zhang et al., 2024\)](#page-11-5)). That's because the allocation strategy requires very limited additional complexity in the inference/generation phase compared with computation required for generation as [Appendix L.](#page-20-1)

# <span id="page-20-0"></span>**N PyramidKV Excels in all KV Cache Size Limitation**

The evaluation results from LongBench[\(Bai et al., 2023\)](#page-9-7) are shown in [Table 11,](#page-22-0) [Table 12,](#page-22-1) an[dTable 13.](#page-23-0) We report the results using LlaMa-3-8B-Instruct, LlaMa-3-70B-Instruct and Mistral-7B-Instruct[\(Jiang et al., 2023\)](#page-10-0) for different KV cache sizes.

Overall, PyramidKV consistently surpasses other method across a range of KV cache sizes and different backbone models, with its performance advantages becoming particularly pronounced in memory-constrained environments. Upon examining specific tasks, PyramidKV demonstrates a notably superior performance on the TREC task, a few-shot question answering challenge. This suggests that the model effectively aggregates information from the few-shot examples, highlighting the potential for further investigation into in-context learning tasks.

| Prompt Length | Generation Length | H2O    | SnapKV | StreamingLLM | PyramidKV |
|---------------|-------------------|--------|--------|--------------|-----------|
| 512           | 512               | 18.47  | 18.25  | 18.96        | 18.26     |
| 512           | 1024              | 35.10  | 34.76  | 36.20        | 34.69     |
| 512           | 2048              | 70.21  | 69.60  | 72.35        | 70.69     |
| 512           | 4096              | 140.80 | 139.42 | 146.37       | 138.62    |
| 1024          | 512               | 17.63  | 17.34  | 18.12        | 17.32     |
| 1024          | 1024              | 35.16  | 34.61  | 36.17        | 34.67     |
| 1024          | 2048              | 71.02  | 69.17  | 72.37        | 70.21     |
| 1024          | 4096              | 140.51 | 138.83 | 146.09       | 138.61    |
| 2048          | 512               | 17.64  | 19.54  | 18.22        | 17.48     |
| 2048          | 1024              | 35.09  | 34.76  | 36.29        | 34.78     |
| 2048          | 2048              | 70.84  | 69.56  | 72.46        | 69.50     |
| 2048          | 4096              | 140.16 | 139.55 | 145.22       | 138.59    |
| 4096          | 512               | 17.75  | 17.67  | 18.40        | 17.58     |
| 4096          | 1024              | 35.20  | 35.08  | 36.46        | 34.93     |
| 4096          | 2048              | 70.02  | 69.26  | 72.58        | 69.65     |
| 4096          | 4096              | 139.87 | 138.57 | 144.98       | 138.87    |

<span id="page-21-0"></span>Table 10: Performance comparison across different configurations and methods.

|        | Single | -Docum | ent QA |          | i-Documen | ~       | Sui         | nmariza   |             |       | -shot Le | arning | Synt  | hetic | Co    | ode   |       |
|--------|--------|--------|--------|----------|-----------|---------|-------------|-----------|-------------|-------|----------|--------|-------|-------|-------|-------|-------|
| Method | NrtvQA | Qasper | MF-en  | HotpotQA | 2WikiMOA  | Musique | GovReport   | QMSum     | MultiNews   | TREC  | TriviaQA | SAMSum | PCoun | PRe   | rec   | RB-P  | Avg.  |
|        | 18409  | 3619   | 4559   | 9151     | 4887      | 11214   | 8734        | 10614     | 2113        | 5177  | 8209     | 6258   | 11141 | 9289  | 1235  | 4206  |       |
|        |        |        |        |          |           | LlaMa   | -3-8B-Instr | uct, KV S | Size = Full |       |          |        |       |       |       |       |       |
| FKV    | 25.70  | 29.75  | 41.12  | 45.55    | 35.87     | 22.35   | 25.63       | 23.03     | 26.21       | 73.00 | 90.56    | 41.88  | 04.67 | 69.25 | 58.05 | 50.77 | 41.46 |
|        |        |        |        |          |           | LlaMa   | a-3-8B-Inst | ruct, KV  | Size = 64   |       |          |        |       |       |       |       |       |
| SKV    | 19.86  | 9.09   | 27.89  | 37.34    | 28.35     | 18.17   | 15.86       | 20.80     | 16.41       | 38.50 | 85.92    | 36.32  | 5.22  | 69.00 | 51.78 | 48.38 | 33.05 |
| H2O    | 20.80  | 11.34  | 27.03  | 37.25    | 30.01     | 17.94   | 18.29       | 21.49     | 19.13       | 38.00 | 84.70    | 37.76  | 5.63  | 69.33 | 53.44 | 50.15 | 33.89 |
| SLM    | 17.44  | 8.68   | 22.25  | 35.37    | 31.51     | 15.97   | 15.46       | 20.06     | 14.64       | 38.00 | 72.33    | 29.10  | 5.42  | 69.50 | 46.14 | 45.09 | 30.43 |
| Ours   | 21.13  | 14.18  | 30.26  | 35.12    | 23.76     | 16.17   | 18.33       | 21.65     | 19.23       | 58.00 | 88.31    | 37.07  | 5.23  | 69.50 | 52.61 | 45.74 | 34.76 |
|        |        |        |        |          |           | LlaMa   | a-3-8B-Inst | ruct, KV  | Size = 96   |       |          |        |       |       |       |       |       |
| SKV    | 20.45  | 10.34  | 31.84  | 37.85    | 28.65     | 18.52   | 17.90       | 21.26     | 19.07       | 41.50 | 86.95    | 37.82  | 5.08  | 69.12 | 54.69 | 51.31 | 34.51 |
| H2O    | 21.55  | 11.21  | 28.73  | 37.66    | 30.12     | 18.47   | 19.57       | 21.57     | 20.44       | 38.50 | 87.63    | 38.47  | 5.60  | 69.00 | 54.51 | 50.16 | 34.57 |
| SLM    | 18.67  | 8.43   | 24.98  | 38.35    | 30.59     | 16.37   | 17.33       | 19.84     | 18.41       | 41.00 | 73.92    | 29.38  | 5.80  | 69.50 | 47.15 | 45.61 | 31.58 |
| Ours   | 21.67  | 15.10  | 33.50  | 39.73    | 26.48     | 17.47   | 19.64       | 22.28     | 20.49       | 61.50 | 87.38    | 38.18  | 6.00  | 69.25 | 55.30 | 46.78 | 36.29 |
|        |        |        |        |          |           | LlaMa   | -3-8B-Instr | uct, KV   | Size = 128  |       |          |        |       |       |       |       |       |
| SKV    | 21.19  | 13.55  | 32.64  | 38.75    | 29.64     | 18.73   | 18.98       | 21.62     | 20.26       | 45.00 | 88.36    | 37.64  | 5.13  | 68.85 | 55.84 | 51.82 | 35.50 |
| H2O    | 22.12  | 13.20  | 31.61  | 37.79    | 32.71     | 18.45   | 20.32       | 22.02     | 21.10       | 38.50 | 87.75    | 39.14  | 5.83  | 69.50 | 55.06 | 50.97 | 35.37 |
| SLM    | 18.61  | 9.65   | 25.99  | 37.95    | 29.39     | 16.34   | 18.03       | 20.11     | 20.08       | 43.50 | 74.08    | 29.86  | 5.90  |       |       |       | 32.00 |
| Ours   | 21.40  | 16.92  | 33.79  | 39.73    | 28.72     | 18.59   | 19.86       | 22.48     | 20.95       | 66.50 | 89.35    | 38.39  | 5.92  | 69.00 | 56.49 | 47.95 | 37.25 |
|        |        |        |        |          |           | LlaMa-  | 3-8B-Instru | ıct, KV S | Size = 2048 |       |          |        |       |       |       |       |       |
| SKV    | 25.86  | 29.55  | 41.10  | 44.99    | 35.80     | 21.81   | 25.98       | 23.40     | 26.46       | 73.50 | 90.56    | 41.66  | 5.17  | 69.25 | 56.65 | 49.94 | 41.35 |
| SLM    | 21.71  | 25.78  | 38.13  | 40.12    | 32.01     | 16.86   | 23.14       | 22.64     | 26.48       | 70.00 | 83.22    | 31.75  | 5.74  |       |       |       | 37.82 |
| H2O    | 25.56  | 26.85  | 39.54  | 44.30    | 32.92     | 21.09   | 24.68       | 23.01     | 26.16       | 53.00 | 90.56    | 41.84  | 4.91  |       |       |       | 39.35 |
| Ours   | 25.40  | 29.71  | 40.25  | 44.76    | 35.32     | 21.98   | 26.83       | 23.30     | 26.19       | 73.00 | 90.56    | 42.14  | 5.22  | 69.25 | 58.76 | 51.18 | 41.49 |

<span id="page-22-0"></span>Table 11: Performance comparison of PyramidKV (Ours) with SnapKV (SKV), H2O, StreamingLLM (SLM) and FullKV (FKV) on LongBench for LlaMa-3-8B-Instruct. PyramidKV generally outperforms other KV Cache compression methods across various KV Cache sizes and LLMs. The performance strengths of PyramidKV are more evident in small KV Cache sizes. Bold text represents the best performance.

|                           | Single                                  | -Docum                                  | ent QA                                  |                                         | i-Documen                               | ~                                       | Sui                                     | nmariza                                 |                                         |                                         | -shot Le                                | arning                                  | Synt                                | netic                 | C              | ode                   |                                  |
|---------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|-------------------------------------|-----------------------|----------------|-----------------------|----------------------------------|
| Method                    | NrtvQA                                  | Qasper                                  | MF-en                                   | HotpotQA                                | 2WikiMQA                                | Musique                                 | GovReport                               | QMSum                                   | MultiNews                               | TREC                                    | friviaQA                                | SAMSum                                  | PCoun                               | PRe                   | Lec            | RB-P                  | Avg.                             |
|                           | 18409                                   | 3619                                    | 4559                                    | 9151                                    | 4887                                    | 11214                                   | 8734                                    | 10614                                   | 2113                                    | 5177                                    | 8209                                    | 6258                                    | 11141                               | 9289                  | 1235           | 4206                  |                                  |
|                           |                                         |                                         |                                         |                                         |                                         | Mistra                                  | al-7B-Instru                            | ict, KV S                               | ize = Full                              |                                         |                                         |                                         |                                     |                       |                |                       |                                  |
| FKV                       | 26.90                                   | 33.07                                   | 49.20                                   | 43.02                                   | 27.33                                   | 18.78                                   | 32.91                                   | 24.21                                   | 26.99                                   | 71.00                                   | 86.23                                   | 42.65                                   | 2.75                                | 86.98                 | 56.96          | 54.52                 | 42.71                            |
|                           |                                         |                                         |                                         |                                         |                                         | Mistr                                   | al-7B-Instr                             | uct, KV S                               | Size = 64                               |                                         |                                         |                                         |                                     |                       |                |                       |                                  |
| SKV<br>SLM<br>H2O<br>Ours | 16.94<br>15.01<br>18.19<br><b>20.91</b> | 17.17<br>13.84<br>19.04<br><b>20.21</b> | 39.51<br>28.74<br>37.40<br><b>39.94</b> | 36.87<br>30.97<br>30.18<br>33.57        | 22.26<br><b>24.50</b><br>22.22<br>22.87 | 15.18<br>13.42<br>13.77<br><b>15.70</b> | 14.75<br>13.25<br>16.60<br>17.31        | 20.35<br>19.46<br><b>21.52</b><br>21.23 | 21.45<br>19.17<br><b>21.98</b><br>21.41 | 37.50<br>35.50<br>37.00<br><b>54.00</b> | 84.16<br>76.91<br>81.02<br>81.98        | 37.28<br>29.61<br><b>38.62</b><br>36.96 | 4.50<br>4.67<br><b>5.00</b><br>3.58 | 27.33<br><b>66.03</b> | 38.71<br>43.54 | 35.29<br><b>40.46</b> | 30.72<br>25.60<br>30.88<br>32.19 |
|                           |                                         |                                         |                                         |                                         |                                         | Mistr                                   | al-7B-Instr                             | uct, KV S                               | Size = 96                               |                                         |                                         |                                         |                                     |                       |                |                       |                                  |
| SKV<br>SLM<br>H2O<br>Ours | 19.92<br>15.15<br>19.44<br><b>20.35</b> | 18.80<br>15.48<br>20.81<br><b>21.87</b> | 43.29<br>31.44<br>38.78<br>41.15        | 39.66<br>30.03<br>32.39<br>34.94        | 23.08<br>23.93<br>21.51<br>21.85        | 15.94<br>12.73<br>14.43<br>15.81        | 16.65<br>16.76<br>17.68<br><b>18.21</b> | 21.26<br>19.15<br><b>22.40</b><br>21.66 | 21.47<br>19.19<br><b>21.99</b><br>21.43 | 43.50<br>41.50<br>38.00<br><b>65.00</b> | 83.48<br>75.31<br>82.51<br><b>83.60</b> | 39.74<br>28.71<br><b>39.94</b><br>39.60 | 4.00<br>5.00<br><b>6.06</b><br>4.50 | 28.48<br>77.48        | 38.92<br>45.18 | 36.05<br><b>42.43</b> | 32.47<br>26.37<br>32.67<br>34.08 |
|                           |                                         |                                         |                                         |                                         |                                         | Mistra                                  | al-7B-Instru                            | ıct, KV S                               | ize = 128                               |                                         |                                         |                                         |                                     |                       |                |                       |                                  |
| SKV<br>SLM<br>H2O<br>Ours | 19.16<br>16.57<br>21.20<br><b>21.75</b> | 21.46<br>14.68<br>21.90<br>22.03        | 43.52<br>32.40<br>41.55<br><b>44.32</b> | 38.60<br>30.19<br>33.56<br>34.06        | 23.35<br>22.64<br>21.28<br>22.79        | 16.09<br>12.34<br>12.93<br>15.77        | 17.66<br>18.08<br><b>18.59</b><br>18.58 | 21.84<br>18.96<br><b>22.61</b><br>21.89 | 21.47<br>19.19<br><b>21.99</b><br>21.43 | 47.50<br>43.50<br>39.00<br><b>66.00</b> | 84.15<br>74.22<br>82.37<br>83.46        | 40.24<br>29.02<br><b>40.44</b><br>39.75 | 5.00<br>4.50<br><b>6.00</b><br>4.50 | 29.48<br><b>83.19</b> | 39.23<br>46.41 | 36.16<br>42.66        | 34.96<br>27.57<br>34.73<br>35.72 |
|                           |                                         |                                         |                                         |                                         |                                         | Mistra                                  | ıl-7B-Instru                            | ct, KV Si                               | ze = 2048                               |                                         |                                         |                                         |                                     |                       |                |                       |                                  |
| SKV<br>SLM<br>H2O<br>Ours | 25.89<br>20.31<br>25.76<br>25.53        | 32.93<br>26.64<br>31.10<br>32.21        | 48.56<br>45.72<br><b>49.03</b><br>48.97 | <b>42.96</b><br>35.25<br>40.76<br>42.26 | 27.42<br>24.31<br>26.52<br><b>27.50</b> | 19.02<br>12.20<br>17.07<br><b>19.36</b> | 26.56<br>27.47<br>24.81<br>26.60        | 24.47<br>21.57<br>23.64<br>23.97        | 26.69<br>24.51<br>26.60<br><b>26.73</b> | 70.00<br>68.50<br>55.00<br><b>71.00</b> | 86.27<br>71.95<br><b>86.35</b><br>86.25 | 42.57<br>31.19<br>42.48<br><b>42.94</b> | 5.50<br>5.00<br>5.50<br>4.50        | 22.56<br>88.15        | 43.38<br>49.93 | 37.08<br>46.57        | 41.56<br>32.35<br>39.95<br>41.63 |

<span id="page-22-1"></span>Table 12: Performance comparison of PyramidKV (Ours) with SnapKV (SKV), H2O, StreamingLLM (SLM) and FullKV (FKV) on LongBench for Mistral-7B-Instruct. PyramidKV generally outperforms other KV Cache compression methods across various KV Cache sizes and LLMs. The performance strengths of PyramidKV are more evident in small KV Cache sizes. Bold text represents the best performance.

|                                      | Single-Document QA             |                                | Multi-Document QA              |                          |                          | Summarization           |                         |                         | Few-shot Learning              |                         |                                | Synthetic                      |                | Code           |                |                |       |
|--------------------------------------|--------------------------------|--------------------------------|--------------------------------|--------------------------|--------------------------|-------------------------|-------------------------|-------------------------|--------------------------------|-------------------------|--------------------------------|--------------------------------|----------------|----------------|----------------|----------------|-------|
| Method                               | NrhvQA                         | Qasper                         | MEen                           | HotpotQA                 | 2WikiMQA                 | Musique                 | GovReport               | QMSum                   | MultiNews                      | TREC                    | TriviaQA                       | SAMSum                         | PCount         | PRe            | Lec            | RB-P           | Avg.  |
|                                      | 18409                          | 3619                           | 4559                           | 9151                     | 4887                     | 11214                   | 8734                    | 10614                   | 2113                           | 5177                    | 8209                           | 6258                           | 11141          | 9289           | 1235           | 4206           |       |
|                                      |                                |                                |                                |                          |                          | LlaMa-                  | 3-70B-Instr             | uct, KV                 | Size = Full                    |                         |                                |                                |                |                |                |                |       |
| FKV                                  | 27.75                          | 46.48                          | 49.45                          | 52.04                    | 54.9                     | 30.42                   | 32.37                   | 22.27                   | 27.58                          | 73.5                    | 92.46                          | 45.73                          | 12.5           | 72.5           | 40.96          | 63.91          | 46.55 |
|                                      |                                |                                |                                |                          |                          | LlaMa                   | -3-70B-Inst             | ruct, KV                | Size = 64                      |                         |                                |                                |                |                |                |                |       |
| SKV                                  | 23.92                          | 31.09                          | 36.54                          | 46.66                    | 50.40                    | 25.30                   | 18.05                   | 21.11                   | 19.79                          | 41.50                   | 91.06                          | 40.26                          | 12.00          |                |                |                |       |
| SLM<br>H2O                           | 22.07<br>25.45                 | 23.53<br>34.64                 | 27.31<br>33.23                 | 43.21<br>48.25           | <b>51.66</b> 50.30       | 23.85<br>24.88          | 16.62<br>20.03          | 19.74<br>21.50          | 15.20<br>21.39                 | 39.50<br>42.00          | 76.89<br>90.36                 | 33.06<br><b>41.58</b>          | 12.00<br>12.00 |                |                |                |       |
| Ours                                 | 25.47                          | 36.71                          | 42.29                          | 47.08                    | 46.21                    | 28.30                   | 20.60                   | 21.62                   | 21.62                          | 64.50                   | 89.61                          | 41.28                          | 12.50          |                |                |                |       |
| LlaMa-3-70B-Instruct, KV Size = 96   |                                |                                |                                |                          |                          |                         |                         |                         |                                |                         |                                |                                |                |                |                |                |       |
| SKV                                  | 25.78                          | 35.71                          | 42.13                          | 50.38                    | 51.46                    | 26.68                   | 19.61                   | 21.40                   | 21.98                          | 48.50                   | 92.11                          | 41.21                          | 12.00          | 72.00          | 44.85          | 59.05          | 41.55 |
| SLM                                  | 23.31                          | 29.46                          | 29.21                          | 41.85                    | 45.92                    | 23.00                   | 18.42                   | 19.71                   | 18.57                          | 45.00                   | 76.79                          | 33.54                          | 12.00          |                |                |                |       |
| H2O<br>Ours                          | 25.30<br>25.47                 | 35.13<br><b>37.61</b>          | 35.54<br>44.00                 | 47.39<br>47.33           | 50.61<br>45.36           | 26.20<br><b>27.91</b>   | 20.87<br><b>21.05</b>   | 21.80<br>21.60          | 22.93<br>22.31                 | 41.00<br>66.00          | 90.47<br>91.45                 | <b>43.42</b><br>42.36          | 12.00<br>12.00 |                |                |                |       |
| Ours                                 | 25.47                          | 37.61                          | 44.00                          | 47.33                    | 45.36                    |                         |                         |                         |                                | 66.00                   | 91.45                          | 42.36                          | 12.00          | 72.50          | 45.12          | 26.88          | 42.43 |
|                                      |                                |                                |                                |                          |                          |                         | -3-70B-Instr            |                         |                                |                         |                                |                                |                |                |                |                |       |
| SKV                                  | 26.22                          | 37.49                          | 45.70                          | 50.86                    | 52.82                    | 28.50                   | 20.38                   | 21.72                   | 22.56                          | 53.00                   | 91.61                          | 41.43                          |                |                |                |                | 42.58 |
| SLM<br>H2O                           | 24.25<br>25.61                 | 29.12<br>35.02                 | 29.24<br>37.74                 | 40.20<br>47.77           | 46.28<br>51.16           | 21.80<br>26.87          | 19.55<br>20.57          | 19.42<br>20.78          | 20.61<br>23.33                 | 48.00<br>42.00          | 76.60<br>91.65                 | 33.21<br>43.85                 | 12.00<br>12.00 |                |                |                |       |
| Ours                                 | 26.06                          | 40.35                          | 45.67                          | 50.20                    | 52.78                    | 29.36                   | 22.31                   | 22.02                   | 23.69                          | 71.00                   | 92.27                          | 44.33                          | 12.00          |                |                |                |       |
| LlaMa-3-70B-Instruct, KV Size = 2048 |                                |                                |                                |                          |                          |                         |                         |                         |                                |                         |                                |                                |                |                |                |                |       |
| SKV<br>SLM<br>H2O                    | 26.73<br>26.69<br><b>27.67</b> | 45.18<br>41.01<br><b>46.51</b> | 47.91<br>35.97<br><b>49.54</b> | <b>52.00</b> 46.55 51.49 | <b>55.24</b> 52.98 53.85 | 30.48<br>25.71<br>29.97 | 28.76<br>27.81<br>28.57 | 22.35<br>20.81<br>22.79 | 27.31<br>27.16<br><b>27.53</b> | 72.50<br>69.00<br>59.00 | 92.38<br>91.55<br><b>92.63</b> | 45.58<br>44.02<br><b>45.94</b> | 12.00<br>12.00 | 72.00<br>72.50 | 41.44<br>41.39 | 68.73<br>63.90 | 45.33 |
| Ours                                 | 27.22                          | 46.19                          | 48.72                          | 51.62                    | 54.56                    | 31.11                   | 29.76                   | 22.50                   | 27.27                          | 73.50                   | 91.88                          | 45.47                          | 12.00          | 72.50          | 41.36          | 69.12          | 46.55 |

<span id="page-23-0"></span>Table 13: Performance comparison of PyramidKV (Ours) with SnapKV (SKV), H2O, StreamingLLM (SLM) and FullKV (FKV) on LongBench for LlaMa-3-70B-Instruct. PyramidKV generally outperforms other KV Cache compression methods across various KV Cache sizes and LLMs. The performance strengths of PyramidKV are more evident in small KV Cache sizes. Bold text represents the best performance.

With a small budget, our proposed method enables more effective allocation, better preserving useful attention information. Second, with a large budget, such allocation becomes less critical, as it is sufficient to cover the necessary information. To further illustrate this phenomenon, we have included an ablation study titled "Attention Recall Rate Experiment" as Figure 8. The results show that with a small budget, PyramidKV improves the attention recall rate (the percentage of attention computed using the keys retrieved by the method and the query, relative to the attention computed using all keys and the query.). However, with a larger budget (i.e., 2k KV Cache Size), the improvement decreases. For 64, 128, 256, 512, 1024 and 2048 KV Cache sizes, PyramidKV's average attention recall rate improvements are 1.87%, 0.64%, 0.61%, 0.56%, 0.47% and 0.36%.

![](_page_24_Figure_2.jpeg)

<span id="page-24-1"></span>Figure 8: Attention recall rate (the percentage of attention computed using the keys retrieved by the method and the query, relative to the attention computed using all keys and the query.) comparison of PyramidKV and SnapKV.

### O LongBench results for 128 context length

We conducted additional experiments using Llama-3-8B-Instruct-Gradient-1048k with a sequence length of 128k as Table 14. The results, summarized in the table below, showcase the model's performance with extended context lengths. These findings provide further validation of the scalability and robustness of our approach.

| Method       | Single-Document QA |               |       | Multi-Document QA |          |         | Summarization |        |          | Few   | Few-shot Learning |        |          | ic Code        | Avg.     |
|--------------|--------------------|---------------|-------|-------------------|----------|---------|---------------|--------|----------|-------|-------------------|--------|----------|----------------|----------|
|              | NrtvQA             | Qasper Qasper | MF-en | HotpotQA          | 2WikiMQA | Musique | GovReport     | OMSum. | MultiNew | TREC  | friviaQA          | SAMSum | PCount P | Re Loc RB      | Я        |
| SnapKV       | 6.10               | 8.14          | 23.12 | 8.87              | 10.54    | 5.59    | 20.27         | 17.95  | 18.07    | 50.50 | 82.78             | 34.67  | 3.50 49  | .25 45.39 41.6 | 68 26.65 |
| H2Ô          | 3.47               | 7.49          | 14.17 | 7.30              | 8.74     | 4.55    | 24.13         | 17.83  | 21.91    | 61.50 | 81.45             | 23.60  | 3.55 41  | .80 43.25 38.5 | 51 25.20 |
| StreamingLLN | A 3.47             | 7.49          | 14.17 | 7.30              | 8.74     | 4.55    | 19.21         | 17.83  | 21.91    | 61.50 | 78.21             | 23.60  | 3.55 41  | .80 43.25 38.5 | 51 24.69 |
| PyramidKV    | 5.41               | 8.42          | 22.61 | 9.71              | 10.73    | 5.82    | 20.37         | 18.24  | 18.32    | 54.00 | 85.33             | 34.60  | 3.50 52  | .75 47.23 42.5 | 58 27.48 |

<span id="page-24-2"></span>Table 14: Comparison of PyramidKV with baselines at 128k context length.

# <span id="page-24-0"></span>P PyramidKV Preserves the Long-Context Understanding Ability

We perform Fact Retrieval Across Context Lengths ("Needle In A HayStack") (Liu et al., 2023a; Fu et al., 2024) to test the in-context retrieval ability of LLMs after leveraging different KV cache methods. We conducted the Needle-in-a-Haystack experiment using various LLMs

(i.e., Mistral-7B-Instruct-32k, LLaMA-3-8B-Instruct-8k, and LLaMA-3-70B-Instruct-8k), various KV cache sizes (i.e., 64, 96, and 128) and various methods (i.e., FullKV, PyramidKV, H2O and StreamingLLM). PyramidKV achieves Acc. performance closest to FullKV, while other methods show significant decreases. It is worth noting that PyramidKV with 128 KV cache size achieves the same 100.0 Acc. performance compared with FullKV with 8k context size for LLaMA-3-70B-Instruct.

[Figure 9,](#page-26-0) [Figure 10,](#page-27-0) [Figure 11](#page-28-0) show the results of **Mistral-7B-Instruct** [\(Jiang et al., 2023\)](#page-10-0) with different cache size (64, 96 and 128, respectively).

[Figure 12,](#page-29-0) [Figure 13,](#page-30-0) [Figure 14](#page-31-0) show the results of **LlaMa-3-8B-Instruct** with different cache size (64, 96 and 128, respectively).

[Figure 15,](#page-32-0) [Figure 16,](#page-33-0) [Figure 17](#page-34-0) show the results of **LlaMa-3-70B-Instruct** with different cache size (64, 96 and 128, respectively).

| Model       | Length | KV Cache | Full KV Acc. | PyramidKV Acc. | SnapKV Acc. | H2O Acc. |
|-------------|--------|----------|--------------|----------------|-------------|----------|
| Mistral-7B  | 32k    | 64       | 100.00       | 80.50          | 43.90       | 48.40    |
| Mistral-7B  | 32k    | 96       | 100.00       | 90.50          | 72.20       | 59.10    |
| Mistral-7B  | 32k    | 128      | 100.00       | 91.60          | 80.10       | 64.90    |
| LLaMa-3-8B  | 8k     | 64       | 100.00       | 92.90          | 62.00       | 31.90    |
| LLaMa-3-8B  | 8k     | 96       | 100.00       | 95.80          | 80.70       | 44.20    |
| LLaMa-3-8B  | 8k     | 128      | 100.00       | 97.40          | 87.40       | 49.10    |
| LLaMa-3-70B | 8k     | 64       | 100.00       | 99.60          | 76.20       | 47.30    |
| LLaMa-3-70B | 8k     | 96       | 100.00       | 98.60          | 94.40       | 69.90    |
| LLaMa-3-70B | 8k     | 128      | 100.00       | 100.00         | 98.60       | 82.30    |

Table 15: Recall Accuracy performance from Fact Retrieval Across Context Lengths ("Needle In A HayStack")

![](_page_26_Figure_1.jpeg)

<span id="page-26-0"></span>Figure 9: Results of the Fact Retrieval Across Context Lengths ("Needle In A HayStack") test in **Mistral-7B-Instruct** with **32k** context size in **64** KV cache size. The vertical axis of the table represents the depth percentage, and the horizontal axis represents the token length. PyramidKV mitigates the negative impact of KV cache compression on the long-context understanding capability of LLMs.

![](_page_27_Figure_1.jpeg)

<span id="page-27-0"></span>Figure 10: Results of the Fact Retrieval Across Context Lengths ("Needle In A HayStack") test in **Mistral-7B-Instruct** with **32k** context size in **96** KV cache size. The vertical axis of the table represents the depth percentage, and the horizontal axis represents the token length. PyramidKV mitigates the negative impact of KV cache compression on the long-context understanding capability of LLMs.

![](_page_28_Figure_1.jpeg)

<span id="page-28-0"></span>Figure 11: Results of the Fact Retrieval Across Context Lengths ("Needle In A HayStack") test in **Mistral-7B-Instruct** with **32k** context size in **128** KV cache size. The vertical axis of the table represents the depth percentage, and the horizontal axis represents the token length. PyramidKV mitigates the negative impact of KV cache compression on the long-context understanding capability of LLMs.

![](_page_29_Figure_1.jpeg)

<span id="page-29-0"></span>Figure 12: Results of the Fact Retrieval Across Context Lengths ("Needle In A HayStack") test in **LlaMa-3-8B-Instruct** with **8k** context size in **64** KV cache size. The vertical axis of the table represents the depth percentage, and the horizontal axis represents the token length. PyramidKV mitigates the negative impact of KV cache compression on the long-context understanding capability of LLMs.

![](_page_30_Figure_1.jpeg)

<span id="page-30-0"></span>Figure 13: Results of the Fact Retrieval Across Context Lengths ("Needle In A HayStack") test in **LlaMa-3-8B-Instruct** with **8k** context size in **96** KV cache size. The vertical axis of the table represents the depth percentage, and the horizontal axis represents the token length. PyramidKV mitigates the negative impact of KV cache compression on the long-context understanding capability of LLMs.

![](_page_31_Figure_1.jpeg)

<span id="page-31-0"></span>Figure 14: Results of the Fact Retrieval Across Context Lengths ("Needle In A HayStack") test in **LlaMa-3-8B-Instruct** with **8k** context size in **128** KV cache size. The vertical axis of the table represents the depth percentage, and the horizontal axis represents the token length. PyramidKV mitigates the negative impact of KV cache compression on the long-context understanding capability of LLMs.

![](_page_32_Figure_1.jpeg)

<span id="page-32-0"></span>Figure 15: Results of the Fact Retrieval Across Context Lengths ("Needle In A HayStack") test in **LlaMa-3-70B** with **8k** context size in **64** KV cache size. The vertical axis of the table represents the depth percentage, and the horizontal axis represents the token length. PyramidKV mitigates the negative impact of KV cache compression on the long-context understanding capability of LLMs.

![](_page_33_Figure_1.jpeg)

<span id="page-33-0"></span>Figure 16: Results of the Fact Retrieval Across Context Lengths ("Needle In A HayStack") test in **LlaMa-3-70B** with **8k** context size in **96** KV cache size. The vertical axis of the table represents the depth percentage, and the horizontal axis represents the token length. PyramidKV mitigates the negative impact of KV cache compression on the long-context understanding capability of LLMs.

![](_page_34_Figure_1.jpeg)

<span id="page-34-0"></span>Figure 17: Results of the Fact Retrieval Across Context Lengths ("Needle In A HayStack") test in **LlaMa-3-70B** with **8k** context size in **128** KV cache size. The vertical axis of the table represents the depth percentage, and the horizontal axis represents the token length. PyramidKV mitigates the negative impact of KV cache compression on the long-context understanding capability of LLMs.

# **Q Attention Patterns across heads in the Bottom Layer**

Retrieval heads are predominantly located in the higher layers. Notably, no retrieval heads are observed in bottom layers. To further investigate, we conducted additional experiments on the bottom layer to analyze the attention patterns of the heads as [Figure 18.](#page-35-0) Our findings indicate the absence of "massive attention" in any individual head.

![](_page_35_Figure_3.jpeg)

<span id="page-35-0"></span>Figure 18: Attention patterns of retrieval-augmented generation across heads in the bottom layer in LlaMa.

# **R PyramidKV Implementation at vLLM**

To help compare the vLLM implementation with the vanilla dense attention backend in terms of throughput, we perform the experiment. We present the throughput comparison between the PyramidKV vLLM implementation and the vanilla dense attention backend in a setting where the inputs have varying context lengths without shared prefixes.

In Figure [Figure 19,](#page-36-0) we plot the throughput of the LlaMa 8b model by varying length. We observe that relative throughput under compression decreases as the new input context length approaches the limit, causing new sequences to wait longer before being added to the decoding batch.

We find that allocating/releasing/moving/accessing very small chunks of memory may cause inefficiency and fragmentation in a naive implementation of PyramidKV at vLLM. As PyramidKV applies different allocation budgets for different layers. The top layers have less budget, while the bottom layers have more budget. The application of KV cache eviction with different budgets across layers at the standard paged attention frameworks (i.e., vLLM) is ineffective as it only reduces the cache size proportionally to the layer with the lowest compression rate, and all evictions beyond this rate merely increase cache fragmentation.

However, the problem could be solved by adapting paged attention to page out cache on a per-layer basis. We expand the block tables of each sequence to include block tables for each layer of the cache so that they can be retrieved for each layer's KV cache during attention without the use of fixed memory offsets.

![](_page_36_Figure_1.jpeg)

<span id="page-36-0"></span>Figure 19: Throughout performance of PyramidKV across different input context lengths using LlaMa-3-8b model.