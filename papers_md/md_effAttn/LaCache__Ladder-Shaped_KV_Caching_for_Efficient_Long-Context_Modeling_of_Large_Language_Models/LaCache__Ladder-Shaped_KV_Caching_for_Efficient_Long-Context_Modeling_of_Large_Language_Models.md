## LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models

Dachuan Shi <sup>1</sup> Yonggan Fu 1 2 Xiangchi Yuan <sup>1</sup> Zhongzhi Yu <sup>1</sup> Haoran You <sup>1</sup> Sixu Li <sup>1</sup> Xin Dong <sup>2</sup> Jan Kautz <sup>2</sup> Pavlo Molchanov <sup>2</sup> Yingyan (Celine) Lin 1 2

## Abstract

Recent advancements in Large Language Models (LLMs) have spurred interest in numerous applications requiring robust long-range capabilities, essential for processing extensive input contexts and continuously generating extended outputs. As sequence lengths increase, the number of Key-Value (KV) pairs in LLMs escalates, creating a significant efficiency bottleneck. In this paper, we propose a new KV cache optimization paradigm called LaCache, a training-free method for efficient and accurate generative inference of LLMs. LaCache enables LLMs to simultaneously address both of the critical challenges in longrange modeling: robust long-range capabilities and continuous generation without running outof-memory (OOM). Specifically, LaCache integrates two key innovations: (1) a ladder-shaped KV cache pattern that stores KV pairs not only sequentially (left-to-right within each layer) but also across layers (from shallow to deep), providing an extended span for capturing long-range dependencies under a fixed storage budget, thereby boosting long-range capabilities; and (2) an iterative compaction mechanism that progressively compresses older caches, freeing up space for new tokens within a fixed cache size. This token distance-based dynamic compression enables more effective continuous generation under constrained cache budgets. Experiments across various tasks, benchmarks, and LLM models consistently validate LaCache's effectiveness in enhancing LLMs' long-range capabilities. Our code is available at [https://github.com/GATECH-](https://github.com/GATECH-EIC/LaCache)[EIC/LaCache.](https://github.com/GATECH-EIC/LaCache)

*Proceedings of the* 42 nd *International Conference on Machine Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

## 1. Introduction

Large Language Models (LLMs) have significantly advanced natural language processing tasks [\(Achiam et al.,](#page-9-0) [2023;](#page-9-0) [Touvron et al.,](#page-10-0) [2023;](#page-10-0) [Jiang et al.,](#page-10-1) [2023;](#page-10-1) [Hurst et al.,](#page-10-2) [2024;](#page-10-2) [Team et al.,](#page-10-3) [2024;](#page-10-3) [OpenAI,](#page-10-4) [2025;](#page-10-4) [Guo et al.,](#page-9-1) [2025;](#page-9-1) [Gemini,](#page-9-2) [2025;](#page-9-2) [Anthropic,](#page-9-3) [2025\)](#page-9-3), but they also face major challenges due to their substantial computational costs. To mitigate this, key-value (KV) caching has been used to avoid recomputing attention keys and values during the auto-regressive decoding of LLMs. However, this approach introduces significant memory overhead that scales linearly with sequence length, leading to out-of-memory (OOM) issues on long sequences.

Existing KV cache eviction strategies attempt to address these challenges by pruning cached KV states to enhance memory efficiency. However, these strategies often struggle to balance two critical requirements for long-range LLMs in real-world applications: robust long-range capabilities and continuous generation without OOM. For example, StreamingLLM [\(Xiao et al.,](#page-11-0) [2023b\)](#page-11-0) prioritizes continuous generation but compromises accuracy on long-context tasks. Quest [\(Tang et al.,](#page-10-5) [2024\)](#page-10-5) maintains high accuracy but at the cost of substantial memory usage due to the need to cache the entire KV cache, eventually leading to OOM on long sequences. H2O [\(Zhang et al.,](#page-11-1) [2024\)](#page-11-1) reduces memory costs and achieves better accuracy than StreamingLLM, but its reliance on attention maps makes it incompatible with the efficient attention implementation FlashAttention [\(Dao,](#page-9-4) [2023\)](#page-9-4), leading to slow attention computation.

In response to the aforementioned limitations, we propose LaCache, a training-free KV cache optimization framework that employs a ladder-shaped storage pattern for accurate and efficient generative inference in LLMs. Our contributions are summarized as follows:

• We propose LaCache, which introduces a novel laddershaped KV cache pattern designed for accurate and costeffective long-context generation. This strategy stores KV pairs not only sequentially (left to right within each layer) but also across layers (from shallow to deep). This configuration extends the span for capturing long-range

<sup>1</sup>Georgia Tech <sup>2</sup>NVIDIA. Correspondence to: Yingyan (Celine) Lin <celine.lin@gatech.edu>.

dependencies under a constrained storage budget, thereby enhancing long-range capabilities. Specifically, it preserves the KV states of early tokens in earlier layers and progressively shifts the focus to later tokens in subsequent layers, forming a stepwise, ladder-like structure. As analyzed later, the ladder-shaped pattern improves the lower bound of overall information retention across all tokens.

- We further integrate LaCache with an iterative compaction mechanism to support continuous generation for infinitely long sequences without OOM. This approach periodically applies a ladder-based compression pattern to previously condensed KV states, freeing up space for new tokens. It compresses older tokens more aggressively while applying less compression to newer tokens, enabling the model to prioritize recent information while efficiently managing memory for incoming tokens.
- We evaluate and validate the effectiveness of LaCache through a series of experiments and ablation studies. Results across multiple benchmarks consistently demonstrate that LaCache enhances long-range capabilities and supports continuous generation. Additionally, due to its compatibility with FlashAttention, it outperforms importance-based methods such as H2O in terms of achievable accuracy-throughput trade-offs.

## 2. Related Work

Long-context LLM. The demand for long-context modeling has surged due to its ability to handle complex, multistep tasks and maintain coherent interactions. This has spurred extensive research on enhancing long-context generation [\(Li et al.,](#page-10-6) [2023;](#page-10-6) [Peng et al.,](#page-10-7) [2023;](#page-10-7) [Ye et al.,](#page-11-2) [2025\)](#page-11-2), enabling models to process more tokens per forward pass. While approximate attention mechanisms [\(Beltagy et al.,](#page-9-5) [2020;](#page-9-5) [Kitaev et al.,](#page-10-8) [2020;](#page-10-8) [Wang et al.,](#page-10-9) [2020\)](#page-10-9) have improved efficiency, they usually lead to degradation in task accuracy. Recent advances in positional embeddings, such as position interpolation and fine-tuning [\(Chen et al.,](#page-9-6) [2023;](#page-9-6) [Peng &](#page-10-10) [Quesnelle,](#page-10-10) [2023\)](#page-10-10), have further extended context windows. However, inference efficiency remains a bottleneck for long input sequences. In our proposed LaCache, we leverage token eviction to enhance efficiency in long-context generation, including continuous or infinite generation tasks.

KV cache eviction. KV cache eviction techniques mitigate excessive cache growth by removing non-essential tokens. Early methods [\(Xiao et al.,](#page-11-0) [2023b;](#page-11-0) [Han et al.,](#page-9-7) [2024\)](#page-9-7) rely on static, naive retention strategies that overlook model processing patterns and input context, leading to accuracy degradation. To improve long-context handling, dynamic approaches [\(Adnan et al.,](#page-9-8) [2024;](#page-9-8) [Liu et al.,](#page-10-11) [2024;](#page-10-11) [Zhang](#page-11-1) [et al.,](#page-11-1) [2024;](#page-11-1) [Wan et al.,](#page-10-12) [2024\)](#page-10-12) utilize attention weights to identify important tokens. For example, [Liu et al.](#page-10-11) [\(2024\)](#page-10-11)

detects repetitive patterns to estimate token significance, while [Wang et al.](#page-10-13) [\(2024\)](#page-10-13); [Shi et al.](#page-10-14) [\(2024\)](#page-10-14); [Yu et al.](#page-11-3) [\(2024\)](#page-11-3) merge similar tokens. However, these methods depend on full attention weights, making them incompatible with stateof-the-art (SOTA) efficient inference frameworks such as FlashAttention [\(Dao et al.,](#page-9-9) [2022\)](#page-9-9), which do not explicitly compute attention maps. This constraint limits their realdevice efficiency. To overcome this, LaCache introduces an attention-free KV cache eviction strategy, ensuring high accuracy while maintaining real-device efficiency.

Efficient LLM inference. While traditional efficiency techniques such as quantization, pruning, and distillation [\(Fran](#page-9-10)[tar et al.,](#page-9-10) [2022;](#page-9-10) [Lin et al.,](#page-10-15) [2024;](#page-10-15) [Fu et al.,](#page-9-11) [2024b;](#page-9-11) [Shi et al.,](#page-10-16) [2023;](#page-10-16) [Zhang et al.,](#page-11-4) [2023;](#page-11-4) [Yuan et al.,](#page-11-5) [2025\)](#page-11-5) remain valuable, system-level optimizations, such as FlashAttention [\(Dao et al.,](#page-9-9) [2022;](#page-9-9) [Dao,](#page-9-4) [2023;](#page-9-4) [Shah et al.,](#page-10-17) [2024\)](#page-10-17) and memory offloading [\(Sheng et al.,](#page-10-18) [2023\)](#page-10-18), have emerged as key enablers for large efficiency gains in LLM inference. Unlike most dynamic KV cache eviction methods, LaCache can be seamlessly integrated with these system-level approaches, enhancing real-device efficiency with SOTA inference frameworks.

## 3. The Proposed LaCache Framework

In this section, we present the proposed LaCache framework. We begin with an overview of LaCache in Sec. [3.1,](#page-1-0) followed by an introduction to the proposed ladder-shaped KV cache patterns in Sec. [3.2.](#page-2-0) Finally, to support efficient continuous generation without OOM, even for tasks involving infinitelength generation, we further enhance LaCache with an iterative compaction strategy, as described in Sec. [3.3.](#page-3-0)

## <span id="page-1-0"></span>3.1. LaCache: Motivation and Methodology Overview

Limitations of existing methods. Existing efficient generation methods for LLMs face limitations in achieving both generation accuracy and memory efficiency, which are crucial for continuous long-context generation without running out of memory. Specifically, as shown in Fig. [1](#page-2-1) (a), recency-based methods like StreamingLLM [\(Xiao et al.,](#page-11-0) [2023b\)](#page-11-0), which only maintain the KV cache of the latest tokens within a fixed-length sliding window with an O(1) memory complexity, can support infinite-length generation without OOM but may compromise generation accuracy. In contrast, retrieval-based methods like Quest [\(Tang et al.,](#page-10-5) [2024\)](#page-10-5), as shown in Fig. [1](#page-2-1) (b), store the full KV cache of all tokens and retrieve the most relevant ones for each new token on the fly to improve computational efficiency. This strategy can achieve high accuracy across tasks due to the maintenance of the entire KV cache, but suffers from massive cache storage with an O(T) memory complexity, leading to potential OOM issues when handling long contexts.

![](_page_2_Figure_1.jpeg)

Figure 1. Illustrative comparisons among (a) recency-based KV cache (Xiao et al., 2023b), (b) retrieval-based KV cache (Tang et al., 2024), and (c) our proposed LaCache featuring a ladder-shaped pattern. Previous KV cache storage strategies struggle to simultaneously balance the needs for both continuous generation without OOM and robust long-range capabilities. In contrast, our proposed LaCache allows LLMs to simultaneously satisfy the two requirements.

In light of the limitations of both approaches, as shown in Fig. 1 (c), we propose LaCache, a training-free KV cache optimization featuring a ladder-shaped pattern, designed to balance accuracy and storage cost, enabling accurate and continuous generation without suffering from OOM.

**Overview.** Motivated by the need for both accurate and efficient KV caching, our proposed LaCache features a ladder-shaped KV cache compression and storage pattern. We illustrate LaCache's pattern in Sec. 3.2. Specifically, to achieve effective KV compression while preserving important information of past tokens, rather than uniformly keeping the KV cache for the same set of tokens across all layers as in StreamingLLM (Xiao et al., 2023b), we preserve the KV states of early tokens in earlier layers and then progressively shifting the focus to later tokens in the subsequent layers, forming a stepwise, ladder-like structure.

Furthermore, to support continuous generation without suffering from OOM even for infinite-length generation, we augment LaCache with an iterative compaction strategy, as depicted in Sec. 3.3. Specifically, each time the KV cache reaches its capacity, we apply our LaCache with the ladder-shaped pattern to the already-compacted KV cache. This strategy ensures that older token information is progressively compressed further, while newer incoming tokens are compressed less. In the following subsections, we will elaborate on the ladder-shaped pattern and the iterative compaction strategy, which are the two key enablers of our LaCache framework.

#### <span id="page-2-0"></span>3.2. LaCache: Ladder-Shaped KV Cache Pattern

**The key insight.** Unlike StreamingLLM (Xiao et al., 2023b), which maintains the KV cache of the same set

<span id="page-2-1"></span>![](_page_2_Figure_8.jpeg)

<span id="page-2-2"></span>Figure 2. An illustration of LaCache's KV cache storage pattern. LaCache is leveraged to compact the original full KV cache into a compressed, ladder-shaped pattern, allowing for the storage of information from longer-range tokens compared to StreamingLLM (Xiao et al., 2023b) under the same KV cache budget, thereby providing stronger long-range sequence modeling.

of most recent tokens across all layers, our key insight is that while the information from recent tokens is critical for generation accuracy, their KV cache can be maintained and processed by fewer layers. In other words, different layers can maintain the KV cache corresponding to different sets of tokens. The key advantage of this approach is that, under the same KV cache budget, more tokens can be retained in the KV cache, effectively enlarging the context length and preserving more past information.

The proposed ladder-shaped pattern. The aforementioned insight inspires the design of our ladder-shaped KV cache pattern. As shown in Fig. 1 (c), our LaCache adopts a simple yet effective strategy to cache the KV states of varied tokens across different layers: it preserves the KV states of early tokens in earlier layers and then progressively shifts

![](_page_3_Figure_1.jpeg)

Figure 3. Visualize the trade-off between PPL and cache size for LaCache and over 1,500 randomly sampled KV cache patterns.

focus to later tokens in the subsequent layers, aligning with the temporal and sequential processing nature. This approach results in our ladder-shaped pattern, ensuring both storage efficiency and the retention of essential information across past tokens. More specifically, as shown in Fig. 2, to implement the ladder-shaped pattern denoted by the green box, we discard the KV states falling outside this pattern and then condense the original 2D KV cache into a more compact structure with reduced cache size.

**Further analysis.** We note that the ladder-shaped pattern can effectively cover potentially important tokens by improving the lower bound of information retention. Our work intentionally does not rely on attention maps to identify important tokens, thereby avoiding conflicts with existing optimizations designed for efficient attention calculation. Two rationales behind our ladder-shaped KV pattern are:

Firstly, continuously extending a repetitive pattern and assigning coverage as equally as possible to each layer, which is ensured by our ladder-shaped pattern, improves the lower bound of information retention. This is because, in the worst case, important token sets may appear in the layer with the least coverage, and an unequal coverage strategy would lead to an accuracy drop.

Secondly, since neighboring tokens in natural language typically have higher semantic relevance, our ladder-shaped pattern incorporates a smooth transition for each preserved cache segment. As a result, the ever-expanding ladder pattern with partial overlaps enables a smoother fade-out of older tokens, maintaining stable information retention.

To empirically verify the benefits of these rationales, we randomly generate over 1500 patterns under different KV cache sizes to explore various configurations and visualize the achieved trade-off between Perplexity (PPL) and cache size in Fig. 3. As shown, our ladder-shaped pattern lies on the Pareto optimality boundary.

![](_page_3_Figure_9.jpeg)

<span id="page-3-3"></span><span id="page-3-2"></span>Figure 4. An illustration of LaCache's iterative compaction. Iterative compaction is introduced to support continuous generation without running out of memory, even for infinite-length generation. Once the KV cache reaches the predefined size, LaCache's ladder-shaped pattern is applied to the already-compacted KV cache, freeing up space for new tokens.

**Implementation details.** To balance storage efficiency and generation accuracy, we need to ensure that our method properly eliminates redundancy in the stored KV states while accurately preserving past context information by retaining sufficient past KV states.

To satisfy these principles, two key design factors help balance both aspects and achieve an optimal trade-off: (1) the *span S* across consecutive layers, *i.e.*, the number of layers used to preserve the KV state corresponding to the same token, and (2) the *overlap O* of each layer, *i.e.*, the number of tokens with their KV states preserved in each layer. The larger the *span S*, the more layers are used to record the context for the same token, thereby improving context preservation with an increased storage cost. Similarly, the larger the *overlap O*, the more KV states are preserved by each layer, allowing for more accurate context recording at the cost of reduced storage efficiency. As demonstrated in Sec. 4, we calibrate these two design factors to minimize cache redundancy and maximize generation accuracy.

# <span id="page-3-0"></span>3.3. LaCache: Iterative Compaction for Continuous Infinite-Length Generation

To enable continuous generation in LLMs without encountering OOM issues, even with an infinite generation length, it is highly desirable to maintain a constant KV cache size. To achieve this, we need to augment our LaCache with an eviction mechanism that removes the KV states of past tokens when the predefined cache size is fully utilized.

Rationale and advantage of our method. In response to the aforementioned need, we propose an iterative compaction strategy. The rationale behind our method is simple: once the KV cache, already condensed using LaCache, is full, we apply LaCache again to further condense it.

The advantages of this approach include: (1) Thanks to La-

<span id="page-3-1"></span><sup>&</sup>lt;sup>1</sup>To avoid bubbles, slightly more positions are preserved for tokens located at the beginning and end of ladders.

<span id="page-4-1"></span>Table 1. Language modeling experiments are conducted on the concatenated Wikitext-2-raw-v1 dataset. We use decoding lengths ranging from 1K to 16K for each tested model and cache budget. When the decoding length exceeds the pre-training length, models using a full cache encounter a perplexity explosion issue. The numbers in brackets indicate the cache size.

|                                                                                        |                              |                              | Decoding Length              |                              |                              | Decoding Length                                                                        |                              |                              |                              |                              |                              |
|----------------------------------------------------------------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|----------------------------------------------------------------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|
| Model                                                                                  | 1K                           | 2K                           | 4K                           | 8K                           | 16K                          | Model                                                                                  | 1K                           | 2K                           | 4K                           | 8K                           | 16K                          |
| Llama2-7B (100%)                                                                       | 4.02                         | 4.18                         | 5.12                         | nan                          | nan                          | Llama2-7B-Chat (100%)                                                                  | 4.94                         | 5.32                         | 6.52                         | nan                          | nan                          |
| w/ StreamingLLM (512)<br>w/ LaCache (512)                                              | 5.54<br>4.53                 | 5.84<br>5.00                 | 6.32<br>5.81                 | 6.93<br>6.61                 | 5.36<br>5.19                 | w/ StreamingLLM (512)<br>w/ LaCache (512)                                              | 6.67<br>5.20                 | 7.41<br>6.01                 | 7.95<br>7.06                 | 8.97<br>8.35                 | 6.98<br>6.64                 |
| w/ StreamingLLM (256)<br>w/ LaCache (256)                                              | 6.08<br>5.57                 | 6.38<br>5.98                 | 6.90<br>6.60                 | 7.52<br>7.34                 | 5.92<br>5.77                 | w/ StreamingLLM (256)<br>w/ LaCache (256)                                              | 7.68<br>7.16                 | 8.45<br>7.91                 | 8.98<br>8.47                 | 9.89<br>9.57                 | 7.88<br>7.53                 |
| Llama2-13B (100%)                                                                      | 3.89                         | 3.70                         | 4.67                         | 134.25                       | nan                          | Llama3-8B (100%)                                                                       | 4.28                         | 4.39                         | 5.82                         | 6.16                         | 109.94                       |
| w/ StreamingLLM (512)<br>w/ LaCache (512)<br>w/ StreamingLLM (256)<br>w/ LaCache (256) | 4.92<br>4.40<br>5.64<br>5.22 | 5.02<br>4.68<br>5.52<br>5.17 | 5.66<br>5.42<br>6.17<br>5.88 | 6.28<br>6.09<br>6.79<br>6.50 | 4.82<br>4.69<br>5.29<br>5.08 | w/ StreamingLLM (512)<br>w/ LaCache (512)<br>w/ StreamingLLM (256)<br>w/ LaCache (256) | 5.46<br>4.61<br>6.39<br>5.71 | 5.33<br>4.89<br>5.97<br>5.61 | 6.73<br>6.40<br>7.40<br>7.11 | 6.99<br>6.78<br>7.66<br>7.38 | 5.52<br>5.40<br>6.06<br>5.86 |

Cache's ladder-shaped pattern design introduced in Sec. [3.2,](#page-2-0) early KV caches are discarded first when applying LaCache on the already-condensed KV cache, as shown in Fig. [4.](#page-3-3) This use of larger/smaller compression ratios on early/late KV caches aligns with the design principles of recencybased methods [\(Xiao et al.,](#page-11-0) [2023b\)](#page-11-0); (2) From a deployment perspective, iterative compaction using LaCache provides a unified solution and a clean interface, facilitating wider use.

Implementation details. We provide a more detailed illustration of how our iterative compaction works. As shown in Fig. [4,](#page-3-3) parts of the stored KV states are highlighted to demonstrate their changes after iterative compaction, while the non-highlighted parts are also occupied by other KV states. When the KV cache reaches its capacity, LaCache is applied to the stored KV states, which have already been condensed by LaCache when first enqueued into the KV cache. Consequently, KV states that fall outside the laddershaped pattern are discarded, and the freed space is allocated for new incoming tokens, thus enabling continuous infinitelength generation. In the second iteration of Fig. [4,](#page-3-3) older KV states are compressed more while newer ones are compressed less, thus better preserving recent information.

## <span id="page-4-0"></span>4. Experimental Results

#### 4.1. Experimental Settings

Models. To validate LaCache's general effectiveness, we apply it to LLMs with varying sizes and functionalities, including Llama2-7B/13B [\(Touvron et al.,](#page-10-0) [2023\)](#page-10-0), Llama3- 8B [\(Dubey et al.,](#page-9-12) [2024\)](#page-9-12), Llama2-7B/13B-Chat [\(Touvron](#page-10-0) [et al.,](#page-10-0) [2023\)](#page-10-0), Llama3.2-3B-Instruct [\(Dubey et al.,](#page-9-12) [2024\)](#page-9-12), SmolLM2-1.7B-Instruct [\(Allal et al.,](#page-9-13) [2024\)](#page-9-13), and LongChat-7b-v1.5 [\(Li et al.,](#page-10-6) [2023\)](#page-10-6).

Datasets. We evaluate LaCache on two tasks: long-context

modeling and long-context understanding. Specifically, for long-context modeling, we use Wikitext-2 [\(Merity,](#page-10-19) [2016\)](#page-10-19) and PG19 [\(Rae et al.,](#page-10-20) [2019\)](#page-10-20) datasets for evaluation. For long-context understanding, we employ LongBench [\(Bai](#page-9-14) [et al.,](#page-9-14) [2023\)](#page-9-14), Needle-In-A-Haystack [\(Fu et al.,](#page-9-15) [2024a\)](#page-9-15), and RULER [\(Hsieh et al.,](#page-9-16) [2024\)](#page-9-16) benchmarks to thoroughly assess LaCache's achievable performance.

Baselines. We benchmark LaCache against the standard full KV cache settings and prior KV cache compression methods, including StreamingLLM [\(Xiao et al.,](#page-11-6) [2023a\)](#page-11-6), H2O [\(Zhang et al.,](#page-11-1) [2024\)](#page-11-1), TOVA [\(Oren et al.,](#page-10-21) [2024\)](#page-10-21), PyramidInfer [\(Yang et al.,](#page-11-7) [2024\)](#page-11-7), and SnapKV [\(Li et al.,](#page-10-22) [2024\)](#page-10-22) under various cache budgets.

Implementation Details. We implement LaCache in Py-Torch [\(Paszke,](#page-10-23) [2019\)](#page-10-23). For all tasks, we use a batch size of 1 for evaluation. Specifically, for language modeling experiments, we follow the implementation in StreamingLLM [\(Xiao et al.,](#page-11-0) [2023b\)](#page-11-0) and H2O [\(Zhang et al.,](#page-11-1) [2024\)](#page-11-1), employing regular token-by-token generation for Wikitext-2 and a sliding window approach with a window length of 256 tokens for PG-19. For all 21 datasets within the LongBench benchmark, following LongBench's default setting, we retain the first 128 tokens unchanged for both LaCache and baseline methods, as these initial tokens primarily consist of system prompts and questions. For the Needle-In-A-Haystack [\(Fu et al.,](#page-9-15) [2024a\)](#page-9-15) benchmark, we adopt 50 repetitions for each test unit and a context length up to 128k. For the RULER [\(Hsieh et al.,](#page-9-16) [2024\)](#page-9-16) benchmark, we adopt 100 repetitions for each test unit and a context length of 16k.

#### 4.2. Long-Context Modeling Benchmarks

Benchmark on Wikitext-2. We first evaluate LaCache on the concatenated Wikitext-2-raw-v1 dataset in a standard token-by-token generation setting. Four models, Llama2- 7B, Llama2-7B-Chat, Llama2-13B, and Llama3-8B, are tested with KV cache budgets of 256 and 512.

As summarized in Tab. [1,](#page-4-1) our experimental results demonstrate that our LaCache consistently shows stronger capabilities in capturing long-range dependencies compared to the recency-based method StreamingLLM [\(Xiao et al.,](#page-11-0) [2023b\)](#page-11-0), under the same KV cache budget, across decoding lengths ranging from 1K to 16K. Specifically, with a KV cache budget of 512 and a 1K-length input, LaCache only degrades perplexity by (5.20 − 4.94)/4.94 ≈ 5% on Llama2-7B-Chat compared to using the full cache. In contrast, StreamingLLM [\(Xiao et al.,](#page-11-0) [2023b\)](#page-11-0), under the same cache budget, results in a (6.67 − 4.94)/4.94 ≈ 35% degradation in perplexity. This indicates that when targeting 2× KV cache compression, LaCache experiences significantly less degradation in language modeling performance (5% vs. 35%) compared to StreamingLLM [\(Xiao et al.,](#page-11-0) [2023b\)](#page-11-0).

Benchmark on PG19. To assess LaCache's capabilities on language modeling tasks with extremely long inputs, we compare it against full cache and StreamingLLM on the concatenated PG19 dataset, which comprises 100 books totaling 10 million tokens. We adopt a sliding window of 256 tokens for higher efficiency, following the settings in [\(Wolf,](#page-11-8) [2019\)](#page-11-8). The compacted KV cache output from each window is then passed to generate subsequent tokens, allowing the evaluation of the model's long-context capabilities.

As shown in Fig. [5,](#page-5-0) after an 8K input length, the perplexity of the Llama3-8B model using a full cache quickly escalates; after a 160K input length, an OOM issue arises on a single NVIDIA A100 GPU. In contrast, with our LaCache, the Llama3-8B model supports continuous generation with up to a 600K input length while maintaining reasonable perplexity.

Furthermore, Fig. [6](#page-5-1) summarizes the comparison between La-Cache and StreamingLLM [\(Xiao et al.,](#page-11-0) [2023b\)](#page-11-0) on the fully concatenated PG19 dataset. The consistent improvements achieved by LaCache further validate its strong long-range capabilities with inputs exceeding 10 million tokens.

Benchmark under an extremely small cache budget. To demonstrate LaCache's effectiveness under an extremely small cache budget, we further apply it to LLaMA3-8B-8K with a cache budget of 80. As shown in Tab. [2,](#page-5-2) LaCache

<span id="page-5-2"></span>Table 2. Evaluate LaCache on the Llama3-8B-8K model using a cache size equal to 1% of the pre-training sequence length (*i*.*e*., 80 tokens).

| Decoding Length                     | 1K | 2K                  | 4K | 8K | 16K                            |     | 32K 64K 128K    |
|-------------------------------------|----|---------------------|----|----|--------------------------------|-----|-----------------|
| Llama3-8B                           |    |                     |    |    | 4.28 4.39 5.82 6.16 109.94 nan | nan | nan             |
| w/ StreamingLLM 7.28 7.78 8.31 8.73 |    |                     |    |    | 8.88                           |     | 8.88 9.94 15.68 |
| w/ LaCache                          |    | 7.13 7.44 7.99 8.36 |    |    | 8.46                           |     | 8.43 9.53 15.08 |

![](_page_5_Figure_9.jpeg)

<span id="page-5-0"></span>Figure 5. Evaluate LaCache on the first ten books of the concatenated PG19 dataset, corresponding to a length of 600K tokens.

![](_page_5_Figure_11.jpeg)

<span id="page-5-1"></span>Figure 6. Evaluate LaCache on the entire concatenated PG19 dataset, corresponding to a length of 10 million tokens.

consistently achieves lower PPL than StreamingLLM and the full cache setting under the same decoding length.

#### 4.3. Long-Context Understanding Benchmarks

Benchmark on LongBench. The LongBench benchmark [\(Bai et al.,](#page-9-14) [2023\)](#page-9-14) evaluates LLMs' bilingual longcontext understanding capabilities, with most task lengths averaging between 5K and 15K tokens. Evaluation results on the 21 LongBench datasets for the LLaMA2-7B/13B-Chat models are presented in Tab. [3,](#page-6-0) and results for the SmolLM2-1.7B-Instruct model are shown in Tab. [4.](#page-7-0)

Our experimental results demonstrate that LaCache, requiring no additional computation or storage costs, outperforms StreamingLLM under the same KV cache budgets. For instance, with a 50% KV cache budget, LaCache reduces the average performance degradation from StreamingLLM's 2.4 to 1.5 on the LLaMA2-13B-Chat model, from 2.5 to 1.7 on the LLaMA2-7B-Chat model, and from 1.5 to 1.0 on the SmolLM2-1.7B-Instruct model.

Benchmark with more KV cache eviction methods. To further compare with attention-based KV cache eviction methods, we evaluate the trade-offs between task perfor-

<span id="page-6-0"></span>Table 3. Evaluate LaCache on Llama2-7B/13B-Chat models across 21 LongBench datasets under 50% and 25% KV cache budgets.

| Model               |       |       | Llama2-7B-Chat |         |       | Llama2-13B-Chat |              |       |         |       |
|---------------------|-------|-------|----------------|---------|-------|-----------------|--------------|-------|---------|-------|
|                     |       |       | StreamingLLM   | LaCache |       |                 | StreamingLLM |       | LaCache |       |
| Cache budget        | 100%  | 50%   | 25%            | 50%     | 25%   | 100%            | 50%          | 25%   | 50%     | 25%   |
| HotpotQA            | 33.84 | 29.98 | 30.74          | 32.62   | 30.60 | 38.86           | 37.09        | 37.16 | 38.08   | 36.49 |
| 2WikiMultihopQA     | 26.83 | 24.75 | 24.99          | 26.22   | 25.19 | 34.19           | 32.30        | 32.15 | 34.11   | 31.17 |
| MuSiQue             | 8.82  | 8.48  | 6.58           | 7.72    | 7.94  | 14.19           | 12.12        | 11.99 | 13.67   | 12.97 |
| DuReader            | 24.06 | 18.30 | 19.11          | 22.64   | 21.05 | 27.34           | 20.03        | 21.07 | 24.07   | 20.78 |
| MultiFieldQA-en     | 35.72 | 27.02 | 24.74          | 31.55   | 25.34 | 36.63           | 27.03        | 24.56 | 31.73   | 26.09 |
| MultiFieldQA-zh     | 33.32 | 23.84 | 20.13          | 25.69   | 22.55 | 34.13           | 24.90        | 23.11 | 27.49   | 25.44 |
| NarrativeQA         | 16.78 | 15.97 | 13.46          | 16.27   | 15.18 | 19.38           | 17.50        | 14.78 | 19.30   | 17.95 |
| Qasper              | 17.33 | 15.79 | 15.90          | 16.55   | 16.10 | 26.84           | 23.02        | 21.44 | 24.51   | 21.05 |
| GovReport           | 26.25 | 22.54 | 20.73          | 22.84   | 20.47 | 26.21           | 23.89        | 21.78 | 23.76   | 21.54 |
| QMSum               | 20.89 | 19.72 | 19.30          | 20.34   | 19.58 | 20.12           | 19.03        | 18.76 | 19.61   | 19.16 |
| MultiNews           | 25.83 | 25.07 | 23.32          | 25.16   | 23.27 | 26.06           | 25.38        | 23.71 | 25.33   | 23.79 |
| VCSUM               | 14.33 | 13.12 | 12.31          | 13.16   | 12.18 | 16.89           | 15.53        | 14.37 | 16.03   | 13.87 |
| TriviaQA            | 83.01 | 83.13 | 80.31          | 83.28   | 81.09 | 88.45           | 88.25        | 85.78 | 88.56   | 85.95 |
| SAMSum              | 41.28 | 39.97 | 38.46          | 39.47   | 38.86 | 36.77           | 36.50        | 36.01 | 36.99   | 35.85 |
| TREC                | 64.50 | 62.50 | 59.00          | 65.00   | 58.00 | 68.5            | 65.50        | 62.00 | 66.50   | 60.00 |
| LSHT                | 17.75 | 17.25 | 15.00          | 16.25   | 15.75 | 20.25           | 20.00        | 18.75 | 19.00   | 18.75 |
| PassageRetrieval-en | 11.50 | 6.50  | 6.50           | 5.00    | 6.50  | 9.00            | 8.50         | 7.50  | 10.00   | 7.00  |
| PassageCount        | 4.50  | 5.00  | 5.00           | 5.50    | 5.00  | 3.92            | 3.50         | 2.00  | 2.50    | 3.50  |
| PassageRetrieval-zh | 12.00 | 7.50  | 7.00           | 7.60    | 5.50  | 14.50           | 13.00        | 8.00  | 10.50   | 9.00  |
| LCC                 | 47.74 | 47.97 | 47.18          | 47.97   | 45.59 | 39.31           | 39.22        | 39.00 | 40.12   | 39.09 |
| RepoBench-P         | 44.35 | 43.44 | 43.82          | 43.41   | 43.54 | 42.98           | 41.14        | 39.46 | 41.70   | 38.33 |
| Average             | 29.08 | 26.56 | 25.41          | 27.34   | 25.68 | 30.69           | 28.30        | 26.82 | 29.22   | 27.04 |

![](_page_6_Figure_3.jpeg)

Figure 7. Evaluate the score-throughput trade-offs on a single H200 GPU with StreamingLLM [\(Xiao et al.,](#page-11-0) [2023b\)](#page-11-0), H2O [\(Zhang et al.,](#page-11-1) [2024\)](#page-11-1), TOVA [\(Oren et al.,](#page-10-21) [2024\)](#page-10-21), PyramidInfer [\(Yang et al.,](#page-11-7) [2024\)](#page-11-7), and SnapKV [\(Li et al.,](#page-10-22) [2024\)](#page-10-22) on LongBench [\(Bai et al.,](#page-9-14) [2023\)](#page-9-14). The top left subfigure presents the average performance across all 21 tasks, while the remaining subfigures demonstrate the sub-task performance on Question Answering, Summarization, Few-shot Learning, Synthetic Task, and Code Completion, respectively.

mance and throughput on LongBench using the LLaMA-2-7B-Chat model. As shown in Fig. [7,](#page-6-1) the importancebased KV cache eviction methods H2O [\(Zhang et al.,](#page-11-1) [2024\)](#page-11-1), TOVA [\(Oren et al.,](#page-10-21) [2024\)](#page-10-21), PyramidInfer [\(Yang et al.,](#page-11-7) [2024\)](#page-11-7),

<span id="page-6-1"></span>and SnapKV [\(Li et al.,](#page-10-22) [2024\)](#page-10-22) maintain good scores but suffers from low throughput, while the recency-based method StreamingLLM [\(Xiao et al.,](#page-11-0) [2023b\)](#page-11-0) experiences lower scores. Our method achieves a better trade-off between task

<span id="page-7-0"></span>Table 4. Evaluate LaCache on SmolLM2-1.7B-Instruct across 21 LongBench Datasets under 50% and 25% KV cache budgets.

|                     |       |       | StreamingLLM | LaCache |       |  |
|---------------------|-------|-------|--------------|---------|-------|--|
| Cache Budget        | 100%  | 50%   | 25%          | 50%     | 25%   |  |
| HotpotQA            | 24.08 | 24.22 | 22.21        | 24.32   | 22.81 |  |
| 2WikiMultihopQA     | 24.15 | 22.71 | 21.71        | 22.28   | 21.34 |  |
| MuSiQue             | 8.33  | 9.47  | 9.30         | 10.28   | 8.43  |  |
| DuReader            | 20.24 | 14.56 | 14.58        | 16.73   | 15.56 |  |
| MultiFieldQA-en     | 38.78 | 33.39 | 30.28        | 33.35   | 30.09 |  |
| MultiFieldQA-zh     | 16.82 | 14.63 | 12.12        | 13.94   | 12.27 |  |
| NarrativeQA         | 12.65 | 12.62 | 11.68        | 11.98   | 10.73 |  |
| Qasper              | 17.22 | 16.52 | 14.47        | 16.48   | 16.79 |  |
| GovReport           | 26.74 | 21.38 | 18.94        | 21.80   | 18.86 |  |
| QMSum               | 21.86 | 21.37 | 21.04        | 21.40   | 20.87 |  |
| MultiNews           | 25.67 | 25.42 | 24.75        | 25.54   | 24.88 |  |
| VCSUM               | 10.56 | 11.76 | 10.43        | 11.97   | 10.52 |  |
| TriviaQA            | 80.59 | 78.87 | 78.83        | 80.36   | 78.62 |  |
| SAMSum              | 22.05 | 24.15 | 26.12        | 24.63   | 27.80 |  |
| TREC                | 56.00 | 54.50 | 52.50        | 54.50   | 53.00 |  |
| LSHT                | 9.00  | 6.00  | 5.25         | 7.00    | 6.00  |  |
| PassageRetrieval-en | 9.50  | 5.50  | 5.00         | 6.50    | 5.50  |  |
| PassageCount        | 1.50  | 1.00  | 1.50         | 1.50    | 2.00  |  |
| PassageRetrieval-zh | 7.77  | 4.86  | 4.92         | 7.42    | 4.17  |  |
| LCC                 | 37.76 | 37.85 | 37.75        | 37.99   | 37.83 |  |
| RepoBench-P         | 34.37 | 33.67 | 34.14        | 34.12   | 33.82 |  |
| Average             | 24.07 | 22.59 | 21.79        | 23.05   | 22.00 |  |

performance and throughput compared to these baselines.

Extend to small LMs. To demonstrate LaCache's effectiveness across varying model sizes, we further apply it to the small LM SmolLM2-1.7B-Instruct. As shown in Tab. [4,](#page-7-0) LaCache consistently achieves higher accuracy than the baseline StreamingLLM under the same cache budgets.

Benchmark on Needle-In-A-Haystack. The Needle-In-A-Haystack benchmark [\(Fu et al.,](#page-9-15) [2024a\)](#page-9-15) assesses LLMs' capabilities of retrieving specific information ("the needle") embedded within extremely long text ("the haystack"), which is crucial for applications requiring precise information retrieval from long context. We benchmark La-Cache and StreamingLLM on Llama3.2-3B-Instruct-128k and LongChat-7b-v1.5-32k as shown in Fig. [8](#page-7-1) and Fig. [9.](#page-7-2)

The results demonstrate that LaCache nearly doubles the test accuracy compared to StreamingLLM under the same cache budget—for example, from 54.54% to 99.16% on the Llama3.2-3B-Instruct-128k model under a 50% cache budget and from 33.40% to 65.30% on the LongChat-7bv1.5-32k model under a 25% cache budget.

Benchmark on RULER. The RULER [\(Hsieh et al.,](#page-9-16) [2024\)](#page-9-16) benchmark utilizes synthetic examples to evaluate longcontext LLMs. It encompasses four task categories, including retrieval, multi-hop tracing, aggregation, and question answering. We benchmark LaCache and StreamingLLM on the LongChat-7b-v1.5-32k model under a 50% cache setting, as shown in Tab. [5.](#page-8-0)

![](_page_7_Figure_8.jpeg)

<span id="page-7-1"></span>Figure 8. Benchmark LaCache and StreamingLLM on Needle-In-A-Haystack [\(Fu et al.,](#page-9-15) [2024a\)](#page-9-15) using Llama3.2-3B-Instruct-128k [\(Dubey et al.,](#page-9-12) [2024\)](#page-9-12) under a 50% cache budget setting. Greener indicates better performance.

![](_page_7_Figure_10.jpeg)

<span id="page-7-2"></span>Figure 9. Benchmark LaCache and StreamingLLM on Needle-In-A-Haystack [\(Fu et al.,](#page-9-15) [2024a\)](#page-9-15) using LongChat-7b-v1.5-32k [\(Li](#page-10-6) [et al.,](#page-10-6) [2023\)](#page-10-6) under a 25% cache budget setting. Greener indicates better performance.

The experimental results further validate the consistently better performance of LaCache under the same KV cache conditions. Specifically, LaCache achieves a 5.06% higher average accuracy across 13 different tasks, with particularly large improvements on the *vt* and *cwe* tasks, where it significantly outperforms the baseline.

## 4.4. Ablation Studies

Hyperparameter *Span* S. In long-context understanding tasks such as LongBench, *Span* S is set as an integer approximately equal to the number of layers multiplied by the overall compression ratio, aiming for a uniform compression ratio distribution. For example, under a 50% cache budget, setting S equal to half the number of model layers results in a ∼50% compression ratio across different positions, avoiding situations where some locations are over-compressed while others are under-compressed. In language modeling tasks, S is set to 1/4 of the number of model layers, which was given by the empirical results from our ablation studies, as shown in Fig. [10,](#page-8-1) where Llama2-7B-Chat model under a 256 KV cache budget and the Wikitext-2 dataset are used

<span id="page-8-0"></span>Table 5. Evaluate LongChat-7b-v1.5-32k model [\(Li et al.,](#page-10-6) [2023\)](#page-10-6) on the RULER benchmark [\(Fu et al.,](#page-9-15) [2024a\)](#page-9-15) under a 50% cache budget setting. A higher number indicates better performance.

| Task               |       |       |       |       |       |       | single1 single2 single3 multikey1 multikey2 multikey3 multivalue multiquery |       | vt | cwe | fwe                                 | qa1 | qa2 | Avg. |
|--------------------|-------|-------|-------|-------|-------|-------|-----------------------------------------------------------------------------|-------|----|-----|-------------------------------------|-----|-----|------|
| StreamingLLM 45.00 |       | 49.00 | 45.00 | 53.00 | 50.00 | 45.00 | 47.00                                                                       | 42.00 |    |     | 29.40 17.20 42.00 75.00 43.00 44.82 |     |     |      |
| LaCache            | 57.00 | 43.00 | 26.00 | 52.00 | 64.00 | 31.00 | 62.75                                                                       | 50.25 |    |     | 60.80 61.00 45.67 67.00 41.00 50.88 |     |     |      |

![](_page_8_Figure_3.jpeg)

<span id="page-8-1"></span>Figure 10. Ablation studies on the hyperparameters for language modeling. Perplexity (lower is better) is reported in the figure.

<span id="page-8-2"></span>Table 6. Ablation studies on the hyperparameters for long-context understanding. Scores (higher is better) are reported in the table.

| Setting         |       |       |       | O = 0 O = S/4 O = S/2 ∆(S/4 - 0) ∆(S/2 - 0) |       |
|-----------------|-------|-------|-------|---------------------------------------------|-------|
| QA tasks        | 19.48 | 18.94 | 18.48 | -0.54                                       | -1.00 |
| Synthetic tasks | 5.17  | 5.67  | 6.17  | +0.50                                       | +1.00 |

for examining the impact of hyperparameters.

Hyperparameter *Overlap* O. The choice of *Overlap* O depends on the task type. Specifically, a larger O allows the information of a single token to be distributed across more positions, which is better suited for tasks requiring complex semantic understanding and greater global context. In contrast, a small overlap concentrates the information in fewer positions, which is more appropriate for tasks where the answers appear in a very narrow window. For language modeling tasks, O is set to 1/2 of S for achieving better semantic continuity. For long-context understanding tasks, as shown in Tab. [6,](#page-8-2) a larger overlap consistently improves performance on tasks that require more global information, such as synthetic tasks (PassageCount, PassageRetrieval-en, and PassageRetrieval-zh) while reducing performance on tasks

that rely more on local information, such as QA tasks (NarrativeQA, Qasper, MultiFieldQA-en, and MultiFieldQA-zh).

## 5. Limitations and Future Work

While LaCache demonstrates advantages for accurate and efficient long-context generation in LLMs, it also has limitations that present opportunities for future exploration: Although the ladder-shaped KV cache pattern is effective, it may not be optimal for every scenario. Alternative patterns could further enhance memory efficiency and performance. Future work can explore diverse KV storage configurations based on our core insight: recent tokens are crucial for generation accuracy, but fewer layers might suffice for effectively processing and storing their KV caches. Additionally, LaCache is implemented in a training-free setting to ensure efficiency and ease of deployment. Incorporating fine-tuning could further improve performance by adapting the KV cache pattern to specific tasks. Future work can extend LaCache to support fine-tuning and benchmark its performance against training-dependent methods.

## 6. Conclusion

In this work, we introduce LaCache, a novel, training-free, and easy-to-deploy KV cache optimization framework designed to enhance both the efficiency and effectiveness of LLMs in long-context generation tasks. LaCache addresses the limitations of existing KV caching methods through a ladder-shaped KV cache storage pattern and an iterative compaction mechanism. These innovations enable LLMs to better capture long-range dependencies, optimize memory usage, and sustain continuous generation even under fixed storage constraints. Specifically, by sequentially storing KV pairs both within and across layers, the ladder-shaped structure allows the model to preserve crucial information at different levels of context. Additionally, the iterative compaction mechanism dynamically manages memory, ensuring that essential information is prioritized. Our results show that LaCache significantly improves memory efficiency while maintaining high generation quality, outperforming baseline methods across various benchmarks. By offering a scalable and storage-efficient solution, LaCache enhances the capacity of LLMs to manage extended contexts and continuous generation, paving the way for further innovations in long-range LLM optimization.

## Acknowledgment

This work was partially supported by the National Science Foundation (NSF) through the Computing and Communication Foundations (CCF) program (Award ID: 2400511), the Division of Information & Intelligent Systems (IIS) program (Award ID: 2403297), and CoCoSys, one of the seven centers in JUMP 2.0, a Semiconductor Research Corporation (SRC) program sponsored by DARPA. It was also supported by the Department of Health and Human Services Advanced Research Projects Agency for Health (ARPA-H) under Agreement Number 140D042490003. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of the Advanced Research Projects Agency for Health or the U.S. Government.

## Impact Statement

This work aims to enhance the generation efficiency of LLMs on long-context tasks, achieving continuous generation without OOM while maintaining long-context accuracy. As such, it can facilitate wider use of LLMs and does not suffer from additional societal consequences if LLMs are properly used.

## References

- <span id="page-9-0"></span>Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023.
- <span id="page-9-8"></span>Adnan, M., Arunkumar, A., Jain, G., Nair, P., Soloveychik, I., and Kamath, P. Keyformer: Kv cache reduction through key tokens selection for efficient generative inference. *Proceedings of Machine Learning and Systems*, 6:114–127, 2024.
- <span id="page-9-13"></span>Allal, L. B., Lozhkov, A., Bakouch, E., Blazquez, G. M., ´ Tunstall, L., Piqueres, A., Marafioti, A., Zakka, C., von Werra, L., and Wolf, T. Smollm2 - with great data, comes great performance, 2024.
- <span id="page-9-3"></span>Anthropic. Introducing claude 4, 2025. URL [https:](https://www.anthropic.com/news/claude-4) [//www.anthropic.com/news/claude-4](https://www.anthropic.com/news/claude-4).
- <span id="page-9-14"></span>Bai, Y., Lv, X., Zhang, J., Lyu, H., Tang, J., Huang, Z., Du, Z., Liu, X., Zeng, A., Hou, L., Dong, Y., Tang, J., and Li, J. Longbench: A bilingual, multitask benchmark for long context understanding. *arXiv preprint arXiv:2308.14508*, 2023.
- <span id="page-9-5"></span>Beltagy, I., Peters, M. E., and Cohan, A. Longformer: The long-document transformer. *arXiv preprint arXiv:2004.05150*, 2020.

- <span id="page-9-6"></span>Chen, S., Wong, S., Chen, L., and Tian, Y. Extending context window of large language models via positional interpolation. *arXiv preprint arXiv:2306.15595*, 2023.
- <span id="page-9-4"></span>Dao, T. Flashattention-2: Faster attention with better parallelism and work partitioning. *arXiv preprint arXiv:2307.08691*, 2023.
- <span id="page-9-9"></span>Dao, T., Fu, D., Ermon, S., Rudra, A., and Re, C. Flashat- ´ tention: Fast and memory-efficient exact attention with io-awareness. *Advances in Neural Information Processing Systems*, 35:16344–16359, 2022.
- <span id="page-9-12"></span>Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024.
- <span id="page-9-10"></span>Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. Gptq: Accurate post-training quantization for generative pretrained transformers. *arXiv preprint arXiv:2210.17323*, 2022.
- <span id="page-9-15"></span>Fu, Y., Panda, R., Niu, X., Yue, X., Hajishirzi, H., Kim, Y., and Peng, H. Data engineering for scaling language models to 128k context. *arXiv preprint arXiv:2402.10171*, 2024a.
- <span id="page-9-11"></span>Fu, Y., Yu, Z., Li, J., Qian, J., Zhang, Y., Yuan, X., Shi, D., Yakunin, R., and Lin, Y. C. Amoeballm: Constructing any-shape large language models for efficient and instant deployment. *Advances in Neural Information Processing Systems*, 2024b.
- <span id="page-9-2"></span>Gemini. Gemini 2.5: Our most intelligent ai model, 2025. URL [https://blog.](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/#gemini-2-5-thinking) [google/technology/google-deepmind/](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/#gemini-2-5-thinking) [gemini-model-thinking-updates-march-20](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/#gemini-2-5-thinking)25/ [#gemini-2-5-thinking](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/#gemini-2-5-thinking).
- <span id="page-9-1"></span>Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.
- <span id="page-9-7"></span>Han, C., Wang, Q., Peng, H., Xiong, W., Chen, Y., Ji, H., and Wang, S. Lm-infinite: Zero-shot extreme length generalization for large language models. In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, pp. 3991–4008, 2024.
- <span id="page-9-16"></span>Hsieh, C.-P., Sun, S., Kriman, S., Acharya, S., Rekesh, D., Jia, F., Zhang, Y., and Ginsburg, B. Ruler: What's the real context size of your long-context language models? *arXiv preprint arXiv:2404.06654*, 2024.

- <span id="page-10-2"></span>Hurst, A., Lerer, A., Goucher, A. P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A., Radford, A., et al. Gpt-4o system card. *arXiv preprint arXiv:2410.21276*, 2024.
- <span id="page-10-1"></span>Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. d. l., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al. Mistral 7b. *arXiv preprint arXiv:2310.06825*, 2023.
- <span id="page-10-8"></span>Kitaev, N., Kaiser, Ł., and Levskaya, A. Reformer: The efficient transformer. *arXiv preprint arXiv:2001.04451*, 2020.
- <span id="page-10-6"></span>Li, D., Shao, R., Xie, A., Sheng, Y., Zheng, L., Gonzalez, J., Stoica, I., Ma, X., and Zhang, H. How long can context length of open-source llms truly promise? In *NeurIPS 2023 Workshop on Instruction Tuning and Instruction Following*, 2023.
- <span id="page-10-22"></span>Li, Y., Huang, Y., Yang, B., Venkitesh, B., Locatelli, A., Ye, H., Cai, T., Lewis, P., and Chen, D. Snapkv: Llm knows what you are looking for before generation. *Advances in Neural Information Processing Systems*, 37:22947– 22970, 2024.
- <span id="page-10-15"></span>Lin, J., Tang, J., Tang, H., Yang, S., Chen, W.-M., Wang, W.-C., Xiao, G., Dang, X., Gan, C., and Han, S. Awq: Activation-aware weight quantization for on-device llm compression and acceleration. *Proceedings of Machine Learning and Systems*, 6:87–100, 2024.
- <span id="page-10-11"></span>Liu, Z., Desai, A., Liao, F., Wang, W., Xie, V., Xu, Z., Kyrillidis, A., and Shrivastava, A. Scissorhands: Exploiting the persistence of importance hypothesis for llm kv cache compression at test time. *Advances in Neural Information Processing Systems*, 36, 2024.
- <span id="page-10-19"></span>Merity, S. The wikitext long term dependency language modeling dataset. *Salesforce Metamind*, 9, 2016.
- <span id="page-10-4"></span>OpenAI. Openai gpt-4.5 system card, 2025. URL [https://cdn.openai.com/](https://cdn.openai.com/gpt-4-5-system-card-2272025.pdf) [gpt-4-5-system-card-2272025.pdf](https://cdn.openai.com/gpt-4-5-system-card-2272025.pdf).
- <span id="page-10-21"></span>Oren, M., Hassid, M., Adi, Y., and Schwartz, R. Transformers are multi-state rnns. *arXiv preprint arXiv:2401.06104*, 2024.
- <span id="page-10-23"></span>Paszke, A. Pytorch: An imperative style, high-performance deep learning library. *arXiv preprint arXiv:1912.01703*, 2019.
- <span id="page-10-10"></span>Peng, B. and Quesnelle, J. Ntk-aware scaled rope allows llama models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation, 2023.

- <span id="page-10-7"></span>Peng, B., Quesnelle, J., Fan, H., and Shippole, E. Yarn: Efficient context window extension of large language models. *arXiv preprint arXiv:2309.00071*, 2023.
- <span id="page-10-20"></span>Rae, J. W., Potapenko, A., Jayakumar, S. M., Hillier, C., and Lillicrap, T. P. Compressive transformers for longrange sequence modelling. *arXiv preprint*, 2019. URL <https://arxiv.org/abs/1911.05507>.
- <span id="page-10-17"></span>Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., and Dao, T. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. *Advances in Neural Information Processing Systems*, 37:68658–68685, 2024.
- <span id="page-10-18"></span>Sheng, Y., Zheng, L., Yuan, B., Li, Z., Ryabinin, M., Chen, B., Liang, P., Re, C., Stoica, I., and Zhang, C. Flexgen: ´ High-throughput generative inference of large language models with a single gpu. In *International Conference on Machine Learning*, pp. 31094–31116. PMLR, 2023.
- <span id="page-10-16"></span>Shi, D., Tao, C., Jin, Y., Yang, Z., Yuan, C., and Wang, J. Upop: Unified and progressive pruning for compressing vision-language transformers. In *International Conference on Machine Learning*, pp. 31292–31311. PMLR, 2023.
- <span id="page-10-14"></span>Shi, D., Tao, C., Rao, A., Yang, Z., Yuan, C., and Wang, J. Crossget: Cross-guided ensemble of tokens for accelerating vision-language transformers. In *Forty-first International Conference on Machine Learning*, 2024.
- <span id="page-10-5"></span>Tang, J., Zhao, Y., Zhu, K., Xiao, G., Kasikci, B., and Han, S. Quest: Query-aware sparsity for efficient long-context llm inference. *arXiv preprint arXiv:2406.10774*, 2024.
- <span id="page-10-3"></span>Team, G., Mesnard, T., Hardin, C., Dadashi, R., Bhupatiraju, S., Pathak, S., Sifre, L., Riviere, M., Kale, M. S., Love, ` J., et al. Gemma: Open models based on gemini research and technology. *arXiv preprint arXiv:2403.08295*, 2024.
- <span id="page-10-0"></span>Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E., ` Azhar, F., et al. Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023.
- <span id="page-10-12"></span>Wan, Z., Wu, X., Zhang, Y., Xin, Y., Tao, C., Zhu, Z., Wang, X., Luo, S., Xiong, J., and Zhang, M. D2o: Dynamic discriminative operations for efficient generative inference of large language models. *arXiv preprint arXiv:2406.13035*, 2024.
- <span id="page-10-9"></span>Wang, S., Li, B. Z., Khabsa, M., Fang, H., and Ma, H. Linformer: Self-attention with linear complexity. *arXiv preprint arXiv:2006.04768*, 2020.
- <span id="page-10-13"></span>Wang, Z., Jin, B., Yu, Z., and Zhang, M. Model tells you where to merge: Adaptive kv cache merging for llms on long-context tasks. *arXiv preprint arXiv:2407.08454*, 2024.

- <span id="page-11-8"></span>Wolf, T. Huggingface's transformers: State-of-theart natural language processing. *arXiv preprint arXiv:1910.03771*, 2019.
- <span id="page-11-6"></span>Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., and Han, S. Smoothquant: Accurate and efficient post-training quantization for large language models. In *International Conference on Machine Learning*, pp. 38087–38099. PMLR, 2023a.
- <span id="page-11-0"></span>Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. Efficient streaming language models with attention sinks. *arXiv preprint arXiv:2309.17453*, 2023b.
- <span id="page-11-7"></span>Yang, D., Han, X., Gao, Y., Hu, Y., Zhang, S., and Zhao, H. Pyramidinfer: Pyramid kv cache compression for high-throughput llm inference. *arXiv preprint arXiv:2405.12532*, 2024.
- <span id="page-11-2"></span>Ye, Z., Xia, K., Fu, Y., Dong, X., Hong, J., Yuan, X., Diao, S., Kautz, J., Molchanov, P., and Lin, Y. C. Longmamba: Enhancing mamba's long-context capabilities via training-free receptive field enlargement. In *The Thirteenth International Conference on Learning Representations*, 2025.
- <span id="page-11-3"></span>Yu, H., Yang, Z., Li, S., Li, Y., and Wu, J. Effectively compress kv heads for llm. *arXiv preprint arXiv:2406.07056*, 2024.
- <span id="page-11-5"></span>Yuan, X., Zhang, C., Liu, Z., Shi, D., Vosoughi, S., and Lee, W. Superficial self-improved reasoners benefit from model merging. *arXiv preprint arXiv:2503.02103*, 2025.
- <span id="page-11-4"></span>Zhang, C., Song, D., Ye, Z., and Gao, Y. Towards the law of capacity gap in distilling language models. *arXiv preprint arXiv:2311.07052*, 2023.
- <span id="page-11-1"></span>Zhang, Z., Sheng, Y., Zhou, T., Chen, T., Zheng, L., Cai, R., Song, Z., Tian, Y., Re, C., Barrett, C., et al. H2o: ´ Heavy-hitter oracle for efficient generative inference of large language models. *Advances in Neural Information Processing Systems*, 36, 2024.