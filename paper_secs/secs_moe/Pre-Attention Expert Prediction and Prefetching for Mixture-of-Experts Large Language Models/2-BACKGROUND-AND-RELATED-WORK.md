# 2 BACKGROUND AND RELATED WORK

#### 2.1 MoE Architecture and Inference Challenges

Modern MoE architectures replace traditional FFNs in Transformers with collections of expert modules managed by learned routing functions [\(Shazeer et al.,](#page-11-0) [2017;](#page-11-0) [Fedus](#page-10-0) [et al.,](#page-10-0) [2022\)](#page-10-0). This paradigm has evolved from foundational models like GShard [\(Lepikhin et al.,](#page-10-0) [2020\)](#page-10-0) and Switch Transformer [\(Fedus et al.,](#page-10-0) [2022\)](#page-10-0) to recent large-scale implementations including DeepSeek-V2 [\(DeepSeek-AI,](#page-10-0) [2024\)](#page-10-0), DeepSeek-V3 [\(DeepSeek-AI et al.,](#page-10-0) [2024\)](#page-10-0), Qwen3 [\(Team,](#page-11-0) [2025\)](#page-11-0), Phi-3 [\(Abdin et al.,](#page-9-0) [2024\)](#page-9-0), Mixtral 8x7B and 8x22B [\(Jiang et al.,](#page-10-0) [2024\)](#page-10-0), DBRX [\(Team et al.,](#page-11-0) [2024a\)](#page-11-0), Hunyuan-Large [\(Team et al.,](#page-11-0) [2024b\)](#page-11-0), and many other models. These models demonstrate the scalability of MoE architectures, with total parameters ranging from Mixtral 8x7B's

![](_page_1_Figure_9.jpeg)

Figure 2. (a) Aggregated expert invocation heatmap and (b) Distribution of expert activation frequencies on DeepSeek-V2-Lite across the first 9 layers when generating 300 tokens.

47 billion to DeepSeek-V3's 671 billion parameters, while maintaining sparse activation patterns that significantly reduce computational requirements during inference.

During inference, each input token is dynamically assigned to a subset of experts based on the router's decisions, typically selecting the top-k experts with highest routing scores. For instance, Hunyuan-Large activates 52 billion out of 389 billion parameters [\(Team et al.,](#page-11-0) [2024b\)](#page-11-0). Although this sparse activation pattern enables significant computational savings compared to dense alternatives such as GPT-3 [\(Du et al.,](#page-10-0) [2021\)](#page-10-0), it introduces substantial challenges for practical deployment in resource-constrained environments. As Figure 1 shows, fetching the experts becomes the bottleneck. The issue arises from the unpredictable nature of expert routing decisions. Unlike dense models where all parameters are accessed, MoE models dynamically select experts depending on the current input. This unpredictability is evident in our analysis of DeepSeek-V2-Lite (Figure 2). The expert activation patterns across the first 9 layers during 300 tokens of inference exhibit a high Shannon entropy of 0.976, confirming the near-uniform expert distribution and the inherent challenge of anticipating which experts will be activated for any given token.

This unpredictability creates I/O bottlenecks when expert

must be loaded into GPU memory during inference. Critical challenges include: dynamic load imbalance where expert utilization becomes severely skewed [\(Li et al.,](#page-11-0) [2024a;](#page-11-0) [Zadeh](#page-11-0) [et al.,](#page-11-0) [2024\)](#page-11-0), communication overhead from inter-GPU allto-all patterns required for distributed expert routing [\(Li](#page-11-0) [et al.,](#page-11-0) [2022;](#page-11-0) [Wang et al.,](#page-11-0) [2024\)](#page-11-0), and memory efficiency bottlenecks during expert caching and dynamic loading [\(Xue](#page-11-0) [et al.,](#page-11-0) [2024;](#page-11-0) [Li et al.,](#page-11-0) [2024b\)](#page-11-0). In addition, recent MoE architectures have introduced additional complexity through shared and fine-grained expert designs [\(Sun et al.,](#page-11-0) [2024\)](#page-11-0). DeepSeekMoE subdivides experts into smaller, specialized units while maintaining shared experts that are always activated. This design improves expert specialization but further complicates prediction due to the increased number of routing decisions per layer. Similarly, auxiliary-loss-free load balancing strategies [\(Li et al.,](#page-11-0) [2024a\)](#page-11-0) and dynamic expert selection mechanisms [\(Zuo et al.,](#page-11-0) [2024\)](#page-11-0) aim to address routing instabilities but create more complex prediction scenarios for inference optimization systems.

#### 2.2 Expert Prefetching Methods

#### *2.2.1 Caching-Based Methods*

Expert caches are specialized data caches combining traditional software cache-related techniques and optimizations for MoE LLM expert serving. MoE-Infinity [\(Xue et al.,](#page-11-0) [2025\)](#page-11-0) is a sparsity-aware cache exploiting the skewed reuse patterns of MoE LLM experts. Their method traces the sparse set of activated experts to guide expert prediction, improving inference efficiency on personal machines. HOB-BIT [\(Tang et al.,](#page-11-0) [2024\)](#page-11-0) is a caching-based method that replaces less critical cache-miss experts with low-precision ones to reduce the loading latency. They efficiently manage the expert cache and achieve good speedup in decoding inference, but the cache hit rate is only around 55% on Mixtral-8x7B and Phi-MoE.

### *2.2.2 Prediction-Based Methods*

Existing works mainly focus on cross-layer expert prediction and prefetching, as illustrated in Fig.3 (b). PopFetcher [\(Zhang et al.,](#page-11-0) [2025a\)](#page-11-0) aims to accelerate MoE LLM training by utilizing the communication bandwidth when computing the attention blocks to hide the expert loading latency. They prefetch the experts for the next layer based on the expert popularity by a heuristic approach considering the skewed and correlated expert selection patterns. DuoServe-MoE [\(Zhang et al.,](#page-11-0) [2025b\)](#page-11-0) applies two CUDA streams to overlap the expert loading and computation in the prefilling stage. Their lightweight layer-level expert predictor achieves 54- 67% of top-2 accuracy and 90.3-95.5% hit-1 accuracy on Mixtral-8x7B and Mixtral-8x22B. SP-MoE [\(Chen et al.,](#page-10-0) [2025\)](#page-10-0) optimizes the expert prefetching for speculative decoding that drafts multiple tokens per step. They achieve

![](_page_2_Figure_7.jpeg)

Figure 3. Example MoE layers with and without expert prediction.

about 70-90% of prediction accuracy in most layers thanks to the similarity of draft and target methods and about 44.3% of hit rate on the drafted tokens. Note that SP-MoE targets a different workload from this paper.

#### *2.2.3 Hybird Caching- and Prediction-Based Methods*

Combining the caching with prediction may further boost the expert prediction and serving accuracy. FATE [\(Fang](#page-10-0) [et al.,](#page-10-0) [2025\)](#page-10-0) applies prediction, caching, and mixedprecision experts in the perfetching system. It achieves 78.8% accuracy by prediction and 97.2% accuracy by loading all experts above a confidence score threshold. Pre-Gated MoE [\(Hwang et al.,](#page-10-0) [2024\)](#page-10-0) modifies the expert router to select the experts for the next layer instead of the current layer. They also enhance the inference system with caching to efficiently prefetch experts, reducing GPU memory usage and hiding expert-loading latency.

