# PRE-ATTENTION EXPERT PREDICTION AND PREFETCHING FOR MIXTURE-OF-EXPERTS LARGE LANGUAGE MODELS

Shien Zhu <sup>1</sup> Samuel Bohl <sup>1</sup> Robin Oester <sup>1</sup> Gustavo Alonso <sup>1</sup>

# ABSTRACT

Mixture-of-Experts (MoE) Large Language Models (LLMs) efficiently scale-up the model while keeping relatively low inference cost. As MoE models only activate part of the experts, related work has proposed expert prediction and caching methods to prefetch the experts for faster inference. However, existing approaches utilize the activations from the previous layer for prediction, incurring low accuracy and leave the first layer unoptimized. Applying complex layers or even training standalone networks for better prediction introduces high computation overhead. In this paper, we propose pre-attention expert prediction to achieve accurate and lightweight expert prefetching. The key insight is that some functions in LLMs are ranking-preserving, indicating that matching the ranking of selected experts using simple linear functions is possible. Therefore, we utilize the activations before the attention block in the same layer with 2 linear functions and ranking-aware loss to achieve accurate prediction, which also supports prefetching in the first layer. Our lightweight, pre-attention expert routers achieve 93.03% accuracy on DeepSeek V2 Lite, 94.69% on Qwen3-30B, and 97.62% on Phi-mini-MoE, showing about 15% improvement on absolute accuracy over the state-of-the-art FATE results.

# 1 INTRODUCTION

The Mixture-of-Experts (MoE) architecture is widely adopted by the latest Large Language Models (LLMs) such as Llama 4 [\(Meta,](#page-11-0) [2025\)](#page-11-0), DeepSeek-V3 [\(DeepSeek-AI et al.,](#page-10-0) [2025\)](#page-10-0), and QWen-3 [\(Team,](#page-11-0) [2025\)](#page-11-0). MoE LLMs can achieve high cognition and generation performance while keeping inference cost relatively low. The key reason is that MoE LLMs only activate part of the experts in the Feed-Forward Networks (FFNs), which avoids the high computation cost of using all experts. For example, DeepSeek-V3 only activates 1 shared expert and 8 of 256 routed experts (activates 37 out of 671 billion parameters), significantly reducing runtime memory and computation overhead. As the MoE routers are placed inside the FFNs, the expert selectionloading-computation pipeline suffers from the long expert loading time, especially when the MoE models are too large to fit in the GPU memory.

To solve the expert-loading bottleneck in MoE LLMs, various prediction and caching methods have been proposed to efficiently prefetch and serve the experts. MoE-Infinity [\(Xue et al.,](#page-11-0) [2025\)](#page-11-0) designs a sparsity-aware expert cache to trace the activated experts to reduce the Time-Per-Output-

Under review by a journal. Copyright 2025 by the author(s). Token (TPOT). PopFetcher [\(Zhang et al.,](#page-11-0) [2025a\)](#page-11-0) prefetches the experts of the next layer based on their popularity. HOB-BIT [\(Tang et al.,](#page-11-0) [2024\)](#page-11-0) proposes a multi-dimensional cache manager and dynamic expert loader to accelerate expert loading. Pre-Gated MoE [\(Hwang et al.,](#page-10-0) [2024\)](#page-10-0) modifies the gate function to select the experts of the next layer and applies caching methods for faster inference.

However, we observe three problems in these prediction systems. First, it is very hard to achieve high prediction accuracy by predicting from the previous layer. SP-MoE [\(Chen](#page-10-0) [et al.,](#page-10-0) [2025\)](#page-10-0) prefetches the experts for speculative decoding that drafts multiple tokens per step and achieves more than 70% of prediction accuracy in most layers. DuoServe-MoE [\(Zhang et al.,](#page-11-0) [2025b\)](#page-11-0) accelerates the inference by duo CUDA streams and a layer-level expert predictor with 54- 67% of top-2 accuracy. FATE [\(Fang et al.,](#page-10-0) [2025\)](#page-10-0) uses the activations from the previous layer for prediction, and the prediction part contributes 78.8% to accuracy. In addition, many proposals report only the performance gain, without the prediction accuracy. Second, predicting from the previous layer has a limitation on prefetching for the first layer. Although they still predict the experts for the first layer, the actual prediction accuracy for the first and second layers is significantly lower than that of the other layers [\(Fang](#page-10-0) [et al.,](#page-10-0) [2025\)](#page-10-0). For this reason, AdapMoE [\(Zhong et al.,](#page-11-0) [2024\)](#page-11-0) tries to mitigate the prediction accuracy gap of the first few layers by integrating prefetching and cache management techniques. Third, increasing the complexity of the predic-

<sup>1</sup> Systems Group, D-INFK, ETH Zurich, Switzerland. Correspondence to: Shien Zhu <shien.zhu@inf.ethz.ch>, Gustavo Alonso <alonso@inf.ethz.ch>.

<span id="page-1-0"></span>![](_page_1_Figure_1.jpeg)

Figure 1. Token generation pipeline in typical MoE architectures (profiled on DeepSeek-V2-Lite on a Nvidia V100 GPU).

tor can bring higher prediction accuracy, but also incur high computational overhead. The prediction latency has to be as short as possible to leave enough time to fetch the experts, otherwise the prefetching benefits will diminish.

In this paper, we propose pre-attention expert prediction to achieve accurate and lightweight expert prefetching for MoE LLM inference. Our key insight is that the softmax and layer normalization functions are ranking-preserving, allowing ranking-based expert selection to be approximated by simple linear functions, as in the original expert router. First, in contrast to related work predicting based on the previous layer, we propose to predict experts within the same layer before the attention block to achieve a more accurate prediction. This also solves the problem of expert prefetching in the first layer. Second, we design lightweight pre-attention expert routers with 2 linear layers with the intermediate size being 2048, which has significantly lower computation cost than methods with standalone networks. Third, as we infer that matching the ranking of selected experts is possible, we propose ranking-aware loss functions to train the expert prediction routers.

Our pre-attention expert routers achieve 93.03% accuracy on DeepSeek V2 Lite, 94.69% on Qwen3-30B, and 97.62% on Phi-mini-MoE, showing 15% improvement on absolute accuracy over the results in FATE [\(Fang et al.,](#page-10-0) [2025\)](#page-10-0).

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

# 3 MOTIVATION

While existing cross-layer prediction approaches have demonstrated considerable success [\(Fang et al.,](#page-10-0) [2025;](#page-10-0) [Xue](#page-11-0) [et al.,](#page-11-0) [2025\)](#page-11-0), they face inherent architectural complexity and cross-layer dependency management challenges. As Fig.3 (c) shows, same-layer prediction using pre-attention normalized weights offers a fundamentally different approach that leverages temporal proximity between token representation formation and expert selection decisions.

#### 3.1 Hypothesis for Same-Layer Prediction

We hypothesize that same-layer expert prediction may be feasible based on two key observations. First, pre-attention weights contain more recent information than previous layer outputs, capturing token representation at the critical juncture immediately before expert routing decisions. This temporal proximity suggests potential for more accurate routing decisions compared to cross-layer extrapolation.

<span id="page-3-0"></span>Second, expert selection fundamentally requires ranking rather than exact continuous value prediction [\(Shazeer et al.,](#page-11-0) [2017;](#page-11-0) [Fedus et al.,](#page-10-0) [2022\)](#page-10-0). As softmax and layer normalization are ranking-preserving, recent work suggests that attention mechanisms may exhibit approximately rankingpreserving properties [\(Nguyen et al.,](#page-11-0) [2025;](#page-11-0) [Yang et al.,](#page-11-0) [2025a\)](#page-11-0). If this property holds, linear functions with a simple non-linear activation function may potentially capture the ranking relationships necessary for expert selection.

#### 3.2 Benefits of Same-Layer Prediction

Same-layer prediction has several practical advantages over cross-layer approaches. First, it eliminates the bootstrap problem for initial MoE layers where no previous gate input exists, as every layer has its predictor. Second, same-layer prediction eliminates architectural complexity by removing cross-layer communication overhead. Unlike methods that require coordinating information across transformer layers, same-layer approaches operate independently using readily available pre-attention weights, eliminating the need for additional memory buffers, state management across layers, or inter-layer communication protocols. Third, same-layer prediction offers improved interpretability through direct relationships between token characteristics and expert selection within the same computational context. This enables more straightforward analysis of prediction behavior compared to cross-layer approaches that must account for complex inter-layer dependencies. Fourth, pre-attention prediction enables parallel execution with self-attention processing, which requires 0.74-1.13 milliseconds across different hardware configurations. This parallel execution window provides sufficient time for both prediction computation and expert prefetching without introducing additional latency to the inference pipeline.

# 4 PRE-ATTENTION EXPERT PREDICTION

### 4.1 Problem Formulation

We formulate the general expert prediction problem for different MoE configurations and deployment scenarios. For example, Qwen3-30B selects 8 experts, but Phi-mini selects 2 experts in one FFN. Thus, we predict the top-k expert indices that will be selected by the original routers.

Let X ∈ R<sup>d</sup> denote the pre-attention weights for a given token at layer l, where d represents the hidden dimension. The standard MoE routing mechanism computes expert selection through equations (1)-(2), where W<sup>g</sup> ∈ R<sup>E</sup>×<sup>d</sup> are the gating parameters for E experts, and TopK(·, k) selects the indices of the k highest-scoring experts.

$$\mathbf{g} = \text{Softmax}(\mathbf{W}_g \cdot \mathbf{X}) \tag{1}$$

$$\hat{\mathbf{Y}}_{\text{true}} = \text{TopK}(\mathbf{g}, k) \tag{2}$$

Our objective is to learn a mapping f(X; θ) → Yˆ that accurately predicts the expert selection Yˆ = {y1, y2, . . . , yk} using only the pre-attention information available within the current layer. As equations (3)-(4) show, s ∈ R<sup>E</sup> represents the predicted expert selection scores, and we select the top-k experts by ranking these scores directly.

$$\mathbf{s} = f(\mathbf{X}; \ \theta) \tag{3}$$

$$\hat{\mathbf{Y}} = \text{TopK}(\mathbf{s}, k) \tag{4}$$

We address three distinct deployment scenarios through different formulations. For standard deployment scenarios, we predict the precise set of k experts that will be selected. For over-provisioning scenarios, we predict a larger set of experts (e.g., 10 instead of 6 for DeepSeek V2 Lite) to achieve higher hit rates at the cost of increased I/O overhead. For I/O bandwidth-constrained edge scenarios where only one expert can be loaded in parallel with attention computation, we evaluate top-1 accuracy, which measures whether the single highest-scoring predicted expert is among the k experts that will actually be selected by the routing function.

#### 4.2 Pre-Attention Prediction Workflow

Our approach exploits the natural pipeline timing of transformer inference to perform expert prediction with minimal overhead. During the standard transformer forward pass, pre-attention normalization produces weights that capture the token representation immediately before expert routing.

| Hardware Configuration | Timing (ms)      |
|------------------------|------------------|
| Pre-Attention Norm     |                  |
| Tesla V100-SXM2-32GB   | 0.1292 ± 0.0120  |
| NVIDIA A100-PCIE-40GB  | 0.0771 ± 0.0007  |
| NVIDIA A100 80GB PCIe  | 0.0750 ± 0.0015  |
| Self-Attention         |                  |
| Tesla V100-SXM2-32GB   | 1.1279 ± 0.0388  |
| NVIDIA A100-PCIE-40GB  | 0.7607 ± 0.0069  |
| NVIDIA A100 80GB PCIe  | 0.7385 ± 0.0074  |
| Post-Attention Norm    |                  |
| Tesla V100-SXM2-32GB   | 0.1292 ± 0.0120  |
| NVIDIA A100-PCIE-40GB  | 0.0823 ± 0.0012  |
| NVIDIA A100 80GB PCIe  | 0.0797 ± 0.0040  |
| Expert Selection       |                  |
| Tesla V100-SXM2-32GB   | 0.1432 ± 0.0138  |
| NVIDIA A100-PCIE-40GB  | 0.0972 ± 0.0025  |
| NVIDIA A100 80GB PCIe  | 0.1018 ± 0.0039  |
| Expert Computation     |                  |
| Tesla V100-SXM2-32GB   | 10.3075 ± 1.7038 |
| NVIDIA A100-PCIE-40GB  | 6.1970 ± 1.0668  |
| NVIDIA A100 80GB PCIe  | 6.8111 ± 1.1864  |

Table 1. Timing comparison of key transformer operations in DeepSeek-V2-Lite across different GPU configurations. All measurements represent mean ± standard deviation over 50 samples.

![](_page_4_Figure_1.jpeg)

Figure 4. Expert Selector Architecture Comparison

We clone these weights to the CPU for parallel prediction computation while the GPU continues with self-attention.

The prediction process operates in three stages. First, we extract the pre-attention weights X immediately after layer normalization and before self-attention computation. Second, we feed these weights through our trained prediction model fl(X; θl) to generate expert probability scores, where we maintain a separate predictor for each layer l with layerspecific parameters θ<sup>l</sup> . This layer-wise approach allows each predictor to specialize in the unique expert selection patterns characteristic of its corresponding transformer layer. Third, we select the top-k experts based on these scores for prefetching or caching decisions. The critical advantage of this approach is timing. The prediction computation occurs in parallel with self-attention, which typically requires 0.73-1.13 milliseconds across different hardware configurations (Table [1\)](#page-3-0). This parallel execution window provides sufficient time for both prediction computation (0.075-0.129 ms) and an early start for expert prefetching operations without introducing additional latency to the inference pipeline.

# 4.3 Predictor Architectures

We formulate expert prediction as a multi-label classification problem for training purposes, where we create binary labels indicating whether each expert was among the top-k selected experts in the ground truth. Afterwards, during inference, we select experts by ranking the predicted confidence score of all experts and choosing the top-k highest scoring experts, similar to the original MoE gating mechanism.

We evaluated two different architectures for expert selection, as illustrated in Figure 4. Architecture 1 employs linear layers and additional regularization components: a linear layer mapping the input dimension to a hidden dimension (2048), followed by BatchNorm1d, GELU activation, dropout (p=0.1), and a final linear layer that outputs logits

| Parameter        | DeepSeek-V2-Lite | Qwen3-30B | Phi-mini-MoE |
|------------------|------------------|-----------|--------------|
| Total Params     | 16.4B            | 30.5B     | 7.6B         |
| Active Params    | 2.4B             | 3.3B      | 2.4B         |
| Layers           | 27               | 48        | 32           |
| Experts/Layer    | 64               | 128       | 16           |
| Active Experts   | 6 (+ 2 shared)   | 8         | 2            |
| Activation Ratio | 9.4%             | 6.3%      | 12.5%        |

Table 2. Representative MoE models used in evaluation.

for each expert. Architecture 2 uses a streamlined design that contains two linear layers and a SiLU activation in between. This configuration can evaluate the impact of different regularization techniques and activation functions on expert selection accuracy across different MoE models.

### 4.4 Training Dataset and Representative MoE Models

We obtain the training datasets for the predictors from MMLU [\(Hendrycks et al.,](#page-10-0) [2021\)](#page-10-0) to ensure various realistic expert usage patterns. We collect 10M training samples comprising pre-attention activations and corresponding ground-truth affinity scores and expert selections from three actual MoE models.

We selected three representative MoE models that span different scales and architectural configurations while fitting within our hardware constraints. DeepSeek V2 Lite [\(DeepSeek-AI,](#page-10-0) [2024\)](#page-10-0) enables direct comparison with existing research including MoE-Infinity [\(Xue et al.,](#page-11-0) [2025\)](#page-11-0) and FATE [\(Fang et al.,](#page-10-0) [2025\)](#page-10-0), Qwen3-30B-A3B [\(Team,](#page-11-0) [2025\)](#page-11-0) tests scalability to larger architectures, and Phi-mini-MoE [\(Abdin et al.,](#page-9-0) [2024\)](#page-9-0) validates effectiveness in resourceconstrained scenarios. Table 2 summarizes the key architectural parameters of these models. The varying expert counts (16, 64, 128), activation patterns (2, 6, 8), and model scales (7.6B, 16.4B, 30.5B) collectively demonstrate the generalizability of our approach across the spectrum of practical MoE deployments. Their activation ratios (6.3-12.5%) provide a challenging prediction scenario that forecasts expert selection from a large candidate pool.

### 4.5 Loss Function Design

The loss function is critical for expert prediction, as the problem exhibits unique characteristics from standard multilabel classification tasks. We systematically evaluate multiple loss functions to identify an effective training objective for our pre-attention prediction approach. We find that the loss function with multi-label classification principles yields superior results than regression-based approaches. Rather than regression-based approximating the continuous routing affinity scores directly, we find that the classificationoriented loss function based on discrete expert selections is more effective, achieving 90.72% vs 86.61% accuracy in comparative experiments.

<span id="page-5-0"></span>![](_page_5_Figure_1.jpeg)

Figure 5. Sample of sorted true affinity scores.

Therefore, we incorporate several optimizations inspired by recent advances in multi-label learning. We use focal and weighted binary cross-entropy loss to address class imbalance inherent in the multi-label setting, where only top-k experts are positive for each sample. In addition, we incorporate ranking-aware loss components to preserve the relative order of experts.

#### *4.5.1 Affinity Score Distribution*

We observe an interesting pattern when inspecting the affinity scores produced by the MoE gating mechanism. As Fig. 5 sampled on a DeepSeek-V2-Lite layer shows, the distribution of expert scores exhibits a distinct three-tier structure: the top-ranked experts (typically the first few ones) show significantly high affinity scores, followed by a middle tier where scores remain relatively flat across a large number of experts, and finally a sharp drop-off for the lowest-ranked experts. As only a few experts receive high routing confidence, the expert prediction in the moderate region (experts 5 and 6 in Fig. 5) is very challenging. Leveraging this observation, we design the loss function to reflect the inherent ranking structure of expert selection. Rather than treating all experts equally, we assign higher importance to top-ranked experts, moderate importance to middle-tier experts, and lower weight to remaining experts.

#### *4.5.2 Loss Function Formulations*

We evaluated three primary categories of loss functions through systematic single batch experimentation in our training dataset of 10M MMLU-derived samples.

Regression-based approaches treat expert prediction as a continuous score prediction problem. We implement a Mean Squared Error (MSE) loss that directly predicts the routing scores produced by the actual MoE gating mechanism:

$$\mathcal{L}_{MSE}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{E} (s_{ij} - \hat{s}_{ij})^2$$
 (5)

where sij represents the ground truth routing score for expert j on sample i, and sˆij denotes the predicted score. While conceptually straightforward, this approach achieves only 86.61% top-k accuracy under standardized evaluation conditions, indicating that direct score regression fails to

| Loss Function     | Architecture 1 | Architecture 2 |
|-------------------|----------------|----------------|
| MSE Regression    | 86.61%         | 86.61%         |
| Weighted BCE      | 89.00%         | 90.19%         |
| Focal Loss        | 86.40%         | 87.64%         |
| Ranking-aware BCE | 89.19%         | 89.95%         |

Table 3. Single-Epoch Loss Function Performance Comparison

capture the discrete nature of expert selection decisions.

Standard multi-label classification formulates the problem as binary expert selection prediction using weighted binary cross-entropy loss that prioritizes top-ranked experts. As equations (6)-(8) show, the loss function depends on two conditions: if expert j is among the top-k selected experts for sample i and if expert j is a top-10 expert in groundtruth for sample i. zˆij represents the predicted logits and σ(·) is the sigmoid function. This weighted formulation achieves 90.19% accuracy with Architecture 2, outperforming the regression-based approach for expert selection.

$$\mathcal{L}_{\text{WBCE}}(\theta) = -\frac{1}{N \cdot E} \sum_{i=1}^{N} \sum_{j=1}^{E} w_{ij} l_{ij}$$
 (6)

$$l_{ij} = \begin{cases} \log(\sigma(\hat{z}_{ij})) & \text{if expert } j \text{ is top-k} \\ \log(1 - \sigma(\hat{z}_{ij})) & \text{otherwise} \end{cases}$$
 (7)

$$w_{ij} = \begin{cases} 3.0 & \text{if expert } j \text{ is real top-10} \\ 0.5 & \text{otherwise} \end{cases}$$
 (8)

Ranking-aware multi-label classification incorporates the inherent ranking structure of expert selection through weighted and ranking-aware loss components. As equations (9)-(11) show, the LWBCE(θ) part is almost the same as the previous loss function, except that the weighted binary cross-entropy component assigns greater importance to top-ranked experts (wij = 1.5 for experts ranking 11-30).

$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{WBCE}}(\theta) + \lambda \mathcal{L}_{\text{ranking}}(\theta)$$
(9)
$$w_{ij} = \begin{cases} 3.0 & \text{if expert } j \text{ is real top-10} \\ 1.5 & \text{if expert } j \text{ is real top-11 to 30} \\ 0.5 & \text{otherwise} \end{cases}$$

$$\mathcal{L}_{\text{ranking}}(\theta) = \sum_{i=1}^{N} \sum_{j,k \in \text{top-10}}^{s_{ij} > s_{ik}} \text{ReLU}(m - (s_{ij}^{raw} - s_{ik}^{raw}))$$

$$\tag{11}$$

The pairwise ranking loss Lranking(θ) ensures relative ordering preservation within highly-ranked experts, where only valid expert pairs (sij > sik) contribute to the loss. The margin m is set to [0.1] according to the validation performance. Our formulation combines weighted binary cross-entropy with pairwise ranking constraints, achieving

![](_page_6_Figure_1.jpeg)

Figure 6. Expert Prediction Accuracy Architecture 1

89.19% accuracy with Architecture 1 and representing the best-performing approach for that architecture. Though the ranking-aware loss function does not achieve the highest accuracy on single-epoch experiment for architecture 2, it shows obvious better accuracy compared with focal loss [\(Lin et al.,](#page-11-0) [2018\)](#page-11-0) and MSE regression.

#### *4.5.3 Experimental Loss Function Comparison*

We evaluate the loss function variants using standardized single-epoch training on 10M samples to ensure fair comparison. Table [3](#page-5-0) presents the comparative results across different formulations and architectures. The ranking-aware BCE and weighted BCE formulation achieve the highest accuracy of 90.19% and 89.19% respectively. The weighted component addresses the class imbalance, while the ranking component ensures that high-importance expert relationships are preserved during training.

Focal loss [\(Lin et al.,](#page-11-0) [2018\)](#page-11-0), designed for extreme class imbalance, achieves only 86.40% accuracy in our experiments. This lower performance suggests that the class imbalance in expert prediction is better addressed through explicit weighting rather than dynamic loss scaling, likely due to the structured nature of expert selection patterns.

Based on the loss function analysis presented in Table [3,](#page-5-0) we employ ranking-aware BCE loss for Architecture 1 and weighted BCE loss for Architecture 2, selecting the best formulation for each architecture for higher accuracy.

#### *4.5.4 Hyperparameter Selection*

We determine near-optimal hyper-parameters for our ranking-aware loss through grid search on validation data. The ranking loss weight λ = 0.3 in equation [\(9\)](#page-5-0) provides an excellent balance between classification accuracy and ranking preservation. Lower values (λ < 0.1) reduce ranking quality while higher values (λ > 0.5) hurt overall classification performance.

The margin parameter m in the ranking loss is set to 0.1 based on the typical separation between expert routing scores in our training data. Larger margins (> 0.2) make the ranking constraints too strict, while smaller margins (< 0.05) provide insufficient ranking signal during training.

![](_page_6_Figure_11.jpeg)

Figure 7. Expert Prediction Accuracy Architecture 2

# 5 EXPERIMENTAL RESULTS

### 5.1 Experimental Setup

We evaluate our pre-attention expert prediction across three representative MoE models: DeepSeek V2 Lite [\(DeepSeek-](#page-10-0)[AI,](#page-10-0) [2024\)](#page-10-0), Qwen3 [\(Team,](#page-11-0) [2025\)](#page-11-0), and Phi-mini [\(Abdin et al.,](#page-9-0) [2024\)](#page-9-0), spanning different architectural configurations and expert selection strategies. Our training regimen employs 10M samples in 30 epochs to achieve high accuracy across these models. We conducted all experiments on a system equipped with an NVIDIA TITAN RTX 24GB GPU, 128GB of system memory, and 8 CPU cores. Training the predictors for each Transformer layer using GPU is necessary, given the large sample volume. Nevertheless, GPU is not a strict requirement for inference of the predictors. CPU-only inference of the predictors is sufficient for deployment scenarios where the predictor overhead must be minimized.

We evaluate three metrics. First, the exact match accuracy, where the prediction must precisely identify the selected experts. Exact match accuracy measures the percentage of predictions that correctly identify all k selected experts. Second, the over-provisioning accuracy, where loading additional experts can achieve higher hit rates at an increased I/O cost. Third, the top-1 accuracy of the predictor, which evaluates the percentage of cases where the single highestscoring predicted expert is among the k experts that will actually be selected. This is critical for edge deployment scenarios where I/O bandwidth constraints limit parallel expert loading to a single expert during attention computation.

### 5.2 Prediction Accuracy Results

Our experimental results demonstrate substantial improvements over existing approaches across all evaluated models. As Table [4](#page-7-0) shows, we achieve 93.03% exact-match accuracy on DeepSeek-V2-Lite, representing a 15% improvement on absolute accuracy over Fate's [\(Fang et al.,](#page-10-0) [2025\)](#page-10-0) 78.79% decoding accuracy. Qwen3-30B achieves approximately 94.69% exact-match accuracy, while Phi-mini reaches 97.62% accuracy, demonstrating that simpler MoE configurations benefit more from our approach.

The superior performance on Phi-mini could highlight an important characteristic of our method: prediction ac-

<span id="page-7-0"></span>

| Arch | Accuracy     | DeepSeek-V2-Lite | Qwen3-30B | Phi-mini-MoE |
|------|--------------|------------------|-----------|--------------|
|      | Exact-match  | 93.03%           | 94.69%    | 97.62%       |
| 1    | Over-provis. | 98.65%           | 98.81%    | 99.05%       |
|      | Top-1        | 98.85%           | 99.55%    | 98.95%       |
|      | Exact-match  | 92.31%           | 91.82%    | 96.63%       |
| 2    | Over-provis. | 98.15%           | 97.21%    | 98.34%       |
|      | Top-1        | 98.64%           | 98.07%    | 98.02%       |

Table 4. Prediction accuracy of the first layer on three models. Note that the first layer is the hardest one in related works. Other layers have similar or better accuracy than the first layer.

curacy might correlate with model complexity. Simpler routing decisions are easier to predict using pre-attention weights, while more complex routing patterns in larger models present greater challenges but still achieve substantial improvements over existing methods.

The over-provisioning results reveal the practical trade-offs available for different deployment scenarios. Loading 10 experts instead of the required 6 for DeepSeek achieves 98.65% hit rate, representing around 67% additional I/O overhead for a 4.5% improvement in hit rate. Similarly, over-provisioning Qwen3 with 12 experts loaded instead of 8 achieves 98.81% accuracy, while Phi-mini with 3 experts instead of the required 2 leads to 99.05% accuracy. These trade-offs prove attractive for cloud deployments where I/O bandwidth is abundant, but prediction accuracy is critical for performance.

Top-1 accuracy results demonstrate the effectiveness of our approach for I/O bandwidth-constrained edge scenarios. We achieve 98.85% top-1 accuracy on DeepSeek V2 Lite, 99.55% on Qwen3-30B, and 98.95% on Phi-mini-MoE. These results indicate that our prediction method can reliably identify at least one correct expert for parallel loading during attention computation, providing substantial benefits for edge deployments where I/O bandwidth constraints limit the number of experts that can be loaded simultaneously with layer execution.

#### 5.3 Expert Loading Performance Analysis

To establish the practical requirements for expert prediction accuracy, we conducted comprehensive timing benchmarks on representative hardware configurations using DeepSeek-V2-Lite. Our experimental setup measured expert loading latencies across three scenarios: Tesla V100-32GB, A100- 40GB, and A100-80GB systems, representing typical deployment environments for MoE inference. Each expert contains 16.5MB of parameters (5.78M parameters × 2 bytes per bfloat16 value), and standard token processing requires loading 6 experts. We measured both disk-to-GPU and memory-to-GPU transfer times using optimized implementations with pinned memory, contiguous tensor layouts, and non-blocking CUDA streams.

| Hardware        | Disk→GPU | Memory→GPU |
|-----------------|----------|------------|
| Tesla V100-32GB | 48.1 ms  | 9.5 ms     |
| A100-40GB       | 49.8 ms  | 8.5 ms     |
| A100-80GB       | 33.5 ms  | 4.0 ms     |

Table 5. Expert Loading Performance (6 experts, 99MB total)

Storage-to-GPU transfers represent the critical bottleneck in MoE inference. As the profiling results in Table 5 shows, loading 6 experts requires 33.5-49.8ms across the tested hardware, with per-expert costs ranging from 5.6-8.3ms. The high latency is due to storage bandwidth limitations and the inherent serialization of disk I/O operations. Pre-cached experts in system memory achieve dramatically improved transfer rates. Loading 6 experts from memory requires 4.0- 9.5ms depending on hardware generation, representing 5- 8.4× speedup over disk access. Per-expert memory transfer costs range from 0.7-1.6ms, approaching the theoretical limits imposed by PCIe bandwidth and memory hierarchy overhead.

Analysis of the inference pipeline reveals the computational context for expert prediction and the opportunities for parallel execution. The transformer layer processes tokens through the sequence: pre-attention norm → self-attention → post-attention norm → expert selection + gating → expert computation, illustrated by Figure [1.](#page-1-0) Our prediction system utilizes the weights available immediately after preattention norm (0.075-0.129ms) to predict expert requirements in 0.15ms, then executes this prediction pipeline in parallel with the subsequent self-attention (0.738-1.128ms) and post-attention norm (0.080-0.129ms) computations.

Parallel execution of expert predictor and attention blocks is necessary to leave enough time to prefetch the experts. Compared with the naive parallel execution in Fig. [8a,](#page-8-0) we can apply fine-grained expert computation like Fig. [8b](#page-8-0) to hide all expert loading latency. This parallel execution strategy provides 0.818-1.257ms of available time for expert prefetching operations before the pipeline reaches the standard expert gating step (0.097-0.143ms). During this window, correctly predicted experts can be loaded from memory (0.7-1.6ms per expert) and made ready for immediate use. When the pipeline reaches expert selection, our predictions are validated against the standard gating decisions. Correctly predicted experts proceed immediately to computation, while mispredicted experts trigger emergency loading (5.6-8.3ms per expert) during the expert computation phase (6.197-10.308ms total). The parallel execution model ensures that prediction accuracy directly impacts overall throughput by enabling immediate expert access, depicted in Figure [8b,](#page-8-0) or not requiring additional latency in the event of a misprediction, as shown in Figure [8c.](#page-8-0)

<span id="page-8-0"></span>![](_page_8_Figure_1.jpeg)

(a) A Naive Expert Prefetching Pipeline

![](_page_8_Figure_3.jpeg)

(b) Best-Case Scenario with accurate expert prediction

![](_page_8_Figure_5.jpeg)

(c) Worse-Case Scenario with incorrect expert prediction

Figure 8. Execution pipelines of a Transformer layer.

# 6 PERFORMANCE ANALYSIS

#### 6.1 Accuracy vs. System Complexity

Our same-layer prediction approach achieves substantial accuracy improvements while reducing system complexity compared to cross-layer methods. The 15-19 percentage point improvement over Fate [\(Fang et al.,](#page-10-0) [2025\)](#page-10-0) (93.03% vs 78.79% for DeepSeek-V2) demonstrates that leveraging same-layer information provides superior prediction signals compared to cross-layer approaches. Compared with DuoServe-MoE [\(Zhang et al.,](#page-11-0) [2025b\)](#page-11-0) that achieves 54- 67% top-2 accuracy on Mixtral-8x7B and Mixtral-8x22B, our method generalizes to three MoE models with a much higher 93-98% (+30%) exact-match accuracy. Compared with caching based method HOBBIT [\(Tang et al.,](#page-11-0) [2024\)](#page-11-0) with around 55% hit rate on Mixtral-8x7B and Phi-MoE, we achieve about 40% improvement in absolute accuracy.

The elimination of cross-layer communication overhead represents a significant system simplification. While Fate requires coordinating information across transformer layers and managing temporal dependencies between layer executions, our approach operates independently within each layer using readily available pre-attention activation tensors.

Prediction latency analysis shows minimal overhead with 0.15 ms for expert prediction across all model configurations. This overhead represents less than 10% of the

pre-MLP computation pipeline (pre-attention norm + selfattention + post-attention norm), ensuring that the prediction computation remains sufficiently fast to enable parallel fetching of experts while the current layer executes, which preserves the parallelization benefits of expert prediction.

#### 6.2 I/O Savings Analysis

The substantial accuracy improvements translate directly into quantifiable I/O performance benefits. Our improved prediction accuracy directly reduces the frequency of expert loading operations during inference. When predictions are correct (93.03% of tokens), experts are prefetched during self-attention computation, achieving zero loading latency when the pipeline reaches expert selection. When predictions fail (6.97% of tokens), the system loads experts normally without additional penalty.

The expected expert loading time per token is calculated as the misprediction rate multiplied by the expert loading time. Our approach achieves expected loading times of 0.27-0.64 ms/token compared to Fate's 0.85-2.01 ms/token across different hardware configurations. On Tesla V100 systems, our expected time is (1 − 0.9303) × 9.5 = 0.66 ms/token compared to Fate's (1 − 0.7879) × 9.5 = 2.01 ms/token, providing 1.37 ms savings per token.

The frequency-based improvements are substantial: 93.03% of tokens experience zero expert loading latency compared to 78.79% with Fate's approach, representing 14.24 percentage points more tokens with immediate expert access. Over extended inference sessions of 1000 tokens, this translates to 569-1352 ms total latency savings across different hardware configurations, excluding the reloading time on wrong predictions.

Overprovisioning strategies further improve performance by loading 10 experts instead of the required 6, achieving 98.65% prediction accuracy. This reduces expected loading time to (1−0.9865)×loading time = 1.35%×loading time, making 98.65% of tokens achieve zero loading latency at the cost of 67% additional I/O overhead during prefetching.

### 6.3 Deployment Strategy Recommendations

Cloud environment with enough resources should adopt overprovisioning strategies that load 10 experts instead of required 6 to achieve 98 + % hit rates. The 67% I/O overhead increase might be accommodated by cloud-scale I/O bandwidth [\(Bodner et al.,](#page-10-0) [2025\)](#page-10-0) while the < 2% miss rate minimizes performance-critical cache miss penalties.

Edge devices with I/O bandwidth constraints should utilize strategies based on their parallel loading capacity. For devices that can load multiple experts in parallel with attention computation, precise prediction with 93.03% accuracy provides optimal resource utilization. However, for edge

<span id="page-9-0"></span>scenarios where I/O bandwidth limits parallel loading to a single expert during the attention computation window, top-1 prediction becomes critical. Our top-1 prediction achieves 98.6-99.1% hit rates across the different models, ensuring that loading the highest-confidence predicted expert in parallel with attention processing maximizes the utility of the limited parallel execution window.

# 7 DISCUSSION

### 7.1 Why the Method Works

The effectiveness of pre-attention prediction stems from the fundamental information flow within transformer architectures [\(Abnar & Zuidema,](#page-10-0) [2020\)](#page-10-0). Pre-attention activation tensors represent the token state immediately before expert routing decisions, providing the more relevant and temporally accurate information for predicting routing outcomes.

Unlike cross-layer approaches that attempt to extrapolate routing decisions from previous layer states, our method leverages the input closer to the original routing function. Using the tensor in the same layer eliminates the uncertainty inherent in temporal prediction across layer boundaries and provides access to the complete information set used by the routing mechanism.

The superior performance on simpler models like Phi-mini (97.62% accuracy) compared to more complex models (93.03-94.69% accuracy) might reflect the relationship between routing complexity and prediction difficulty. Models with more specialized expert functions and complex routing patterns present greater prediction challenges, but still benefit substantially from the pre-attention information.

Analysis of model-specific patterns reveals that pre-attention activation tensors capture both semantic content and structural patterns that determine expert routing. The activation tensors encode information about token types, positional patterns, and contextual relationships that routing functions use to make expert selection decisions, consistent with recent studies of attention mechanisms in MoE models [\(Piekos](#page-11-0) [et al.,](#page-11-0) [2025;](#page-11-0) [Yang et al.,](#page-11-0) [2025b\)](#page-11-0).

#### 7.2 Future Work

End-to-end system integration building on recent advances in MoE system design represents the most immediate opportunity for extending this work. Combining our prediction approach with optimized caching strategies and dynamic overprovisioning based on inference patterns could further improve performance across diverse deployment scenarios.

Extension to larger MoE models with more experts and more complex routing patterns would validate the approach's scalability. While current results demonstrate effectiveness across models with 2-8 selected experts, larger models with 16+ expert selections might present additional challenges and optimization opportunities.

Dynamic over-provisioning strategies that adjust expert loading based on prediction confidence scores may further optimize the trade-off between I/O overhead and hit rates. Predictions with high confidence scores could operate in exact match mode while less confident predictions could increase over-provisioning to maintain hit rate targets.

# 8 CONCLUSION

In this paper, we target the expert prediction and prefetching problem in MoE LLMs, especially the limitation of crosslayer prediction on the first few layers. We propose a novel same-layer expert prediction using pre-attention activation tensors to solve this problem. Our key observation is that pre-attention activation tensors contains more recent information than the tensors from the previous layer, and matching the ranking of expert selection scores using simple linear functions is possible. Therefore, we propose two lightweight expert predictors with ranking-aware loss functions, eliminating the architectural complexity and cross-layer communication overhead that limits current approaches.

Our proposed pre-attention expert prediction achieves 93.03% exact-match accuracy on DeepSeek V2 Lite, 94.69% on Qwen3, and 97.62% on Phi-mini, showing substantial 15 − 19% absolute improvements over existing cross-layer prediction method FATE [\(Fang et al.,](#page-10-0) [2025\)](#page-10-0) and 30 − 40% improvement compared other prediction and caching methods [\(Zhang et al.,](#page-11-0) [2025b;](#page-11-0) [Tang et al.,](#page-11-0) [2024\)](#page-11-0).

Future MoE architectures can build upon these findings to incorporate native expert prefetching capabilities, while system designers can leverage our parallel execution strategy to optimize inference pipelines across different hardware configurations. Combining high prediction accuracy and reduced system complexity can create more opportunities for MoE system design and deployment strategies.

# REFERENCES

Abdin, M., Aneja, J., Awadalla, H., Awadallah, A., Awan, A. A., Bach, N., Bahree, A., Bakhtiari, A., Bao, J., Behl, H., Benhaim, A., Bilenko, M., Bjorck, J., Bubeck, S., Cai, M., Cai, Q., Chaudhary, V., Chen, D., Chen, D., Chen, W., Chen, Y.-C., Chen, Y.-L., Cheng, H., Chopra, P., Dai, X., Dixon, M., Eldan, R., Fragoso, V., Gao, J., Gao, M., Gao, M., Garg, A., Giorno, A. D., Goswami, A., Gunasekar, S., Haider, E., Hao, J., Hewett, R. J., Hu, W., Huynh, J., Iter, D., Jacobs, S. A., Javaheripi, M., Jin, X., Karampatziakis, N., Kauffmann, P., Khademi, M., Kim, D., Kim, Y. J., Kurilenko, L., Lee, J. R., Lee, Y. T., Li, Y., Li, Y., Liang, C., Liden, L., Lin, X., Lin, Z., Liu, C., Liu, L., Liu, M.,

- <span id="page-10-0"></span>Liu, W., Liu, X., Luo, C., Madan, P., Mahmoudzadeh, A., Majercak, D., Mazzola, M., Mendes, C. C. T., Mitra, A., Modi, H., Nguyen, A., Norick, B., Patra, B., Perez-Becker, D., Portet, T., Pryzant, R., Qin, H., Radmilac, M., Ren, L., de Rosa, G., Rosset, C., Roy, S., Ruwase, O., Saarikivi, O., Saied, A., Salim, A., Santacroce, M., Shah, S., Shang, N., Sharma, H., Shen, Y., Shukla, S., Song, X., Tanaka, M., Tupini, A., Vaddamanu, P., Wang, C., Wang, G., Wang, L., Wang, S., Wang, X., Wang, Y., Ward, R., Wen, W., Witte, P., Wu, H., Wu, X., Wyatt, M., Xiao, B., Xu, C., Xu, J., Xu, W., Xue, J., Yadav, S., Yang, F., Yang, J., Yang, Y., Yang, Z., Yu, D., Yuan, L., Zhang, C., Zhang, C., Zhang, J., Zhang, L. L., Zhang, Y., Zhang, Y., Zhang, Y., and Zhou, X. Phi-3 technical report: A highly capable language model locally on your phone, 2024. URL <https://arxiv.org/abs/2404.14219>.
- Abnar, S. and Zuidema, W. Quantifying attention flow in transformers. In Jurafsky, D., Chai, J., Schluter, N., and Tetreault, J. (eds.), *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pp. 4190–4197, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main. 385. URL [https://aclanthology.org/2020.](https://aclanthology.org/2020.acl-main.385/) [acl-main.385/](https://aclanthology.org/2020.acl-main.385/).
- Bodner, T., Radig, T., Justen, D., Ritter, D., and Rabl, T. An empirical evaluation of serverless cloud infrastructure for large-scale data processing, 2025. URL [https://](https://arxiv.org/abs/2501.07771) [arxiv.org/abs/2501.07771](https://arxiv.org/abs/2501.07771).
- Chen, L., Wen, Z., Wu, T., Zhang, X., and Wu, C. Spmoe: Speculative decoding and prefetching for accelerating moe-based model inference. *arXiv preprint arXiv:2510.10302*, 2025.
- DeepSeek-AI. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model, 2024.
- DeepSeek-AI, Guo, X., Bi, X., Chen, D., Dai, D., Deng, C., Ding, H., Dong, K., Du, Q., Fan, S., et al. Deepseek-v3 technical report. *arXiv preprint arXiv:2412.19437*, 2024.
- DeepSeek-AI, Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., Dai, D., Guo, D., Yang, D., Chen, D., Ji, D., Li, E., Lin, F., Dai, F., Luo, F., Hao, G., Chen, G., Li, G., Zhang, H., Bao, H., Xu, H., Wang, H., Zhang, H., Ding, H., Xin, H., Gao, H., Li, H., Qu, H., Cai, J. L., Liang, J., Guo, J., Ni, J., Li, J., Wang, J., Chen, J., Chen, J., Yuan, J., Qiu, J., Li, J., Song, J., Dong, K., Hu, K., Gao, K., Guan, K., Huang, K., Yu, K., Wang, L., Zhang, L., Xu, L., Xia, L., Zhao, L., Wang, L., Zhang, L., Li, M., Wang, M., Zhang, M., Zhang, M., Tang, M., Li, M., Tian, N., Huang, P., Wang, P., Zhang, P., Wang, Q., Zhu, Q., Chen, Q., Du, Q., Chen, R. J., Jin, R. L., Ge, R., Zhang, R., Pan, R., Wang, R.,

- Xu, R., Zhang, R., Chen, R., Li, S. S., Lu, S., Zhou, S., Chen, S., Wu, S., Ye, S., Ye, S., Ma, S., Wang, S., Zhou, S., Yu, S., Zhou, S., Pan, S., Wang, T., Yun, T., Pei, T., Sun, T., Xiao, W. L., Zeng, W., Zhao, W., An, W., Liu, W., Liang, W., Gao, W., Yu, W., Zhang, W., Li, X. Q., Jin, X., Wang, X., Bi, X., Liu, X., Wang, X., Shen, X., Chen, X., Zhang, X., Chen, X., Nie, X., Sun, X., Wang, X., Cheng, X., Liu, X., Xie, X., Liu, X., Yu, X., Song, X., Shan, X., Zhou, X., Yang, X., Li, X., Su, X., Lin, X., Li, Y. K., Wang, Y. Q., Wei, Y. X., Zhu, Y. X., Zhang, Y., Xu, Y., Xu, Y., Huang, Y., Li, Y., Zhao, Y., Sun, Y., Li, Y., Wang, Y., Yu, Y., Zheng, Y., Zhang, Y., Shi, Y., Xiong, Y., He, Y., Tang, Y., Piao, Y., Wang, Y., Tan, Y., Ma, Y., Liu, Y., Guo, Y., Wu, Y., Ou, Y., Zhu, Y., Wang, Y., Gong, Y., Zou, Y., He, Y., Zha, Y., Xiong, Y., Ma, Y., Yan, Y., Luo, Y., You, Y., Liu, Y., Zhou, Y., Wu, Z. F., Ren, Z. Z., Ren, Z., Sha, Z., Fu, Z., Xu, Z., Huang, Z., Zhang, Z., Xie, Z., Zhang, Z., Hao, Z., Gou, Z., Ma, Z., Yan, Z., Shao, Z., Xu, Z., Wu, Z., Zhang, Z., Li, Z., Gu, Z., Zhu, Z., Liu, Z., Li, Z., Xie, Z., Song, Z., Gao, Z., and Pan, Z. Deepseek-v3 technical report, 2025. URL <https://arxiv.org/abs/2412.19437>.
- Du, N., Huang, Y., Dai, A. M., Tong, S., Lepikhin, D., Xu, Y., Krikun, M., Zhou, Y., Yu, A. W., Firat, O., et al. Glam: Efficient scaling of language models with mixtureof-experts. *arXiv preprint arXiv:2112.06905*, 2021.
- Fang, Z., Hong, Z., Huang, Y., Lyu, Y., Chen, W., Yu, Y., Yu, F., and Zheng, Z. Fate: Fast edge inference of mixture-of-experts models via cross-layer gate, 2025. URL <https://arxiv.org/abs/2502.12224>.
- Fedus, W., Zoph, B., and Shazeer, N. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23:1–39, 2022.
- Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. *Proceedings of the International Conference on Learning Representations (ICLR)*, 2021.
- Hwang, R., Wei, J., Cao, S., Hwang, C., Tang, X., Cao, T., and Yang, M. Pre-gated moe: An algorithm-system codesign for fast and scalable mixture-of-expert inference. In *2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)*, pp. 1018–1031. IEEE, 2024.
- Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D. S., Casas, D. d. l., Hanna, E. B., Bressand, F., et al. Mixtral of experts. *arXiv preprint arXiv:2401.04088*, 2024.
- Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., Krikun, M., Shazeer, N., and Chen, Z. Gshard: Scaling

- <span id="page-11-0"></span>giant models with conditional computation and automatic sharding. *arXiv preprint arXiv:2006.16668*, 2020.
- Li, J., Jiang, Y., Zhu, Y., Wang, C., and Xu, H. Accelerating distributed moe training and inference with lina. *arXiv preprint arXiv:2210.17223*, 2022.
- Li, L., Xiong, J., Chen, Z., Zhang, X., Li, L., Wang, F., Li, X., Zhang, W., and Wang, J. Auxiliary-loss-free load balancing strategy for mixture-of-experts. *arXiv preprint arXiv:2408.15664*, 2024a.
- Li, Z., Xu, Y., Wang, Z., Yang, F., Liu, S., Chen, H., and Wang, P. Moetuner: Optimized mixture of expert serving with balanced expert placement and token routing. *arXiv preprint arXiv:2502.06643*, 2024b.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Dollar, P. ´ Focal loss for dense object detection, 2018. URL [https:](https://arxiv.org/abs/1708.02002) [//arxiv.org/abs/1708.02002](https://arxiv.org/abs/1708.02002).
- Meta. Llama models, 2025. URL [https:](https://github.com/meta-llama/llama-models/tree/main) [//github.com/meta-llama/llama-models/](https://github.com/meta-llama/llama-models/tree/main) [tree/main](https://github.com/meta-llama/llama-models/tree/main).
- Nguyen, T., Tran, N. N., Nguyen, K., and Baraniuk, R. G. Improving routing in sparse mixture of experts with graph of tokens, 2025. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2505.00792) [2505.00792](https://arxiv.org/abs/2505.00792).
- Piekos, P., Csordas, R., and Schmidhuber, J. Mixture of sparse attention: Content-based learnable sparse attention via expert-choice routing, 2025. URL [https:](https://arxiv.org/abs/2505.00315) [//arxiv.org/abs/2505.00315](https://arxiv.org/abs/2505.00315).
- Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., and Dean, J. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer, 2017. URL <https://arxiv.org/abs/1701.06538>.
- Sun, D., Bi, X., Chen, D., Chen, G., Dai, D., Deng, C., Ding, H., Dong, K., Du, Q., Fan, S., et al. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. *arXiv preprint arXiv:2401.06066*, 2024.
- Tang, P., Liu, J., Hou, X., Pu, Y., Wang, J., Heng, P.-A., Li, C., and Guo, M. Hobbit: A mixed precision expert offloading system for fast moe inference. *CoRR*, abs/2411.01433, 2024. URL [https://doi.org/10.](https://doi.org/10.48550/arXiv.2411.01433) [48550/arXiv.2411.01433](https://doi.org/10.48550/arXiv.2411.01433).
- Team, D. M. R., Portes, J., Chintala, R., Karkhanis, A., Norris, S. O., Patel, S., Reddy, R., Tahvildari, N., Zha, S., Zhou, M., et al. Dbrx: A new state-of-the-art open llm. *arXiv preprint arXiv:2403.16261*, 2024a.
- Team, Q. Qwen3 technical report, 2025. URL [https:](https://arxiv.org/abs/2505.09388) [//arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388).

- Team, T. H., Sun, X., Qin, Y., Zhang, D., Xie, T., Zhang, Y., Xu, W., Zheng, Z., Huang, Y., Guo, Y., et al. Hunyuan-large: An open-source moe model with 52 billion activated parameters by tencent. *arXiv preprint arXiv:2411.02265*, 2024b.
- Wang, Z., Yang, Z., Liu, S., Zhang, Q., Chen, H., Li, T., Chen, Y., Wang, P., and Yang, F. Advancing moe efficiency: A collaboration-constrained routing (c2r) strategy for better expert parallelism design. *arXiv preprint arXiv:2504.01337*, 2024.
- Xue, L., Fu, Y., Lu, Z., Mai, L., and Marina, M. Moeinfinity: Efficient moe inference on personal machines with sparsity-aware expert cache, 2025. URL [https:](https://arxiv.org/abs/2401.14361) [//arxiv.org/abs/2401.14361](https://arxiv.org/abs/2401.14361).
- Xue, X., Sun, X., Dobre, O. A., and Chen, M. Expertflow: Optimized expert activation and token allocation for efficient mixture-of-experts inference. *arXiv preprint arXiv:2410.17954*, 2024.
- Yang, Q., Wang, J., Li, X., Wang, Z., Chen, C., Chen, L., Yu, X., Liu, W., Hao, J., Yuan, M., and Li, B. Attentionpredictor: Temporal pattern matters for efficient llm inference, 2025a. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2502.04077) [2502.04077](https://arxiv.org/abs/2502.04077).
- Yang, Y., Wang, C., and Li, J. Umoe: Unifying attention and ffn with shared experts, 2025b. URL [https://](https://arxiv.org/abs/2505.07260) [arxiv.org/abs/2505.07260](https://arxiv.org/abs/2505.07260).
- Zadeh, M., Ahmed, R., Shwartz-Ziv, R., Rudolph, A., Siddiqi, S., and Hooker, S. Latent prototype routing: Achieving near-perfect load balancing in mixture-of-experts. *arXiv preprint arXiv:2506.21328*, 2024.
- Zhang, J., Ma, C., Wang, X., Nie, Y., Li, Y., Xu, Y., Liao, X., Li, B., and Jin, H. {PopFetcher}: Towards accelerated {Mixture-of-Experts} training via popularity based {Expert-Wise} prefetch. In *2025 USENIX Annual Technical Conference (USENIX ATC 25)*, pp. 1053–1069, 2025a.
- Zhang, Y., Pinkert, G., Yang, N., Li, Y., and Yuan, D. Duoserve-moe: Dual-phase expert prefetch and cache scheduling for efficient moe llm inference. *arXiv preprint arXiv:2509.07379*, 2025b.
- Zhong, S., Liang, L., Wang, Y., Wang, R., Huang, R., and Li, M. Adapmoe: Adaptive sensitivity-based expert gating and management for efficient moe inference. In *Proceedings of the 43rd IEEE/ACM International Conference on Computer-Aided Design*, pp. 1–9, 2024.
- Zuo, Q., Zhao, Z., Xu, Y., Liu, Y., and Zhou, M. Harder tasks need more experts: Dynamic routing in moe models. *arXiv preprint arXiv:2403.07652*, 2024.