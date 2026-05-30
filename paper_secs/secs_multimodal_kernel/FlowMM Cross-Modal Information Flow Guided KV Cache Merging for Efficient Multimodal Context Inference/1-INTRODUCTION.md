# 1 INTRODUCTION

Multimodal large language models (MLLMs) based on transformer architecture [\(Wang et al., 2024a;](#page-11-0) [Liu et al., 2023;](#page-10-0) [Chen et al., 2024b;](#page-9-0) [OpenAI, 2024\)](#page-11-1) have revolutionized the integration of visual and textual understanding, enabling sophisticated cross-modal reasoning across tasks such as visual question answering, image captioning, and multimodal dialogue. Unlike traditional language models that process sequential text tokens, MLLMs face unique computational challenges due to their heterogeneous input modalities. Visual inputs, typically encoded as high-dimensional feature maps or patch embeddings [\(Dosovitskiy et al., 2021;](#page-10-1) [Liu et al., 2023\)](#page-10-0), generate substantially longer token sequences than their textual counterparts. This multimodal architecture significantly amplifies the memory burden of key-value (KV) cache. Addressing these multimodal-specific memory efficiency challenges has become paramount for scaling MLLMs to real-world applications with limited computational resources. A promising approach involves selectively retaining only the most critical tokens while evicting others [\(Zhang et al., 2023a;](#page-12-0) [Li et al., 2024;](#page-10-2) [Xiao et al., 2023b\)](#page-11-2). Though effective for memory compression, such eviction-based approaches rely heavily on current token importance assessments. This risks unintentionally and permanently discarding tokens essential for subsequent decoding steps, leading to contextual degradation.

Recently, KV cache merging techniques [\(Zhang et al., 2024;](#page-12-1) [Wang et al., 2024b\)](#page-11-3) have gained attention as an alternative strategy. By consolidating eviction-targeted states into compact representations, these methods preserve richer contextual information. However, existing merging solutions primarily target unimodal LLMs and exhibit suboptimal performance when naively applied to multimodal scenarios. Specifically, multimodal tokens exhibit significant distributional divergence [\(Li](#page-10-3) [et al., 2023\)](#page-10-3), and indiscriminate merging risks information confusion or semantic distortion. Furthermore, intricate cross-modal interactions within MLLMs [\(Alayrac et al., 2022\)](#page-9-1) necessitate care-

<sup>∗</sup>Corresponding author.

ful consideration of attention patterns and dependencies during merging. These challenges render traditional unimodal approaches inadequately equipped to accurately identify mergeable state sets without critical information loss.

To address these challenges, we investigate the impact of KV cache merging in multimodal settings. We empirically find that merging performance varies dramatically across different model layers, revealing significant differences in how layers process heterogeneous modalities. This observation aligns with established principles of attention information flow in prior works (Zhang et al., 2025; Ye et al., 2025). Building on this insight, we introduce FlowMM, an adaptive framework for cross-modal information flow-guided multimodal KV cache merging. FlowMM proactively captures cross-modal interaction patterns across transformer layers by analyzing multimodal attention flow, then dynamically applies layer-specific merging strategies to consolidate critical contextual information.

Further, we identify highly sensitive tokens whose merging substantially degrades model performance. We posit these tokens carry task-critical information vulnerable to corruption during merging. To mitigate this, FlowMM incorporates a sensitivity-

![](_page_1_Figure_4.jpeg)

Figure 1: Comparison between Eviction-Based Compression (a), Traditional Merging-Based Compression (b), and our Cross-Modal Information Flow Guided Merging (c).

adaptive token matching strategy that jointly evaluates token similarity and sensitivity, prioritizing low-sensitivity tokens for merging while preserving high-sensitivity, information-rich tokens. Notably, FlowMM operates without fine-tuning and serves as a plug-and-play solution, delivering adaptive KV cache compression optimized for multimodal contexts.

We conduct extensive experiments with leading MLLMs, including Qwen2.5-VL (Bai et al., 2025), InternVL-2.5 (Chen et al., 2024b), and MobileVLM-V2 (Chu et al., 2024). Their performance is evaluated on MileBench (Song et al., 2024), a comprehensive benchmark encompassing diverse multimodal long-context tasks: temporal multi-image reasoning, semantic multi-image understanding, needle-in-a-haystack retrieval, and complex image search. FlowMM consistently outperforms strong baselines at equivalent KV cache sparsity levels. Specifically, FlowMM achieves a 1.3x to 1.8x reduction in decoding latency while simultaneously reducing the KV cache memory footprint by 80% to 95%. Crucially, these significant efficiency gains are attained while maintaining competitive performance across all evaluated multimodal context tasks.

Overall, our contributions can be summarized as follows:

- We introduce FlowMM, an adaptive framework for cross-modal information flow-guided multimodal KV cache merging. FlowMM dynamically analyzes cross-modal attention flow patterns across layers and employs layer-specific merging strategies, effectively consolidating critical multimodal context.
- To prevent corrupting task-critical information, FlowMM incorporates a sensitivity-adaptive token matching strategy. This jointly evaluates token similarity and sensitivity, merging low-sensitivity tokens while preserving high-sensitivity, information-rich ones.
- We validate FlowMM through extensive experiments across various multimodal context tasks. The results demonstrate that FlowMM reduces KV cache memory usage by up to 80% while consistently surpassing the performance of existing compression methods.

### 2 RELATED WORK

### 2.1 EFFICIENT INFERENCE FOR LARGE LANGUAGE MODELS

Achieving efficient inference in large-scale models requires optimizing three critical resources: model parameters, activation memory, and KV cache size. For parameter reduction, post-training

quantization techniques including GPTQ (Frantar et al., 2022), AWQ (Lin et al., 2024), and SmoothQuant (Xiao et al., 2023a) significantly compress weight bitwidth with minimal accuracy loss, while pruning methods such as SparseGPT (Frantar & Alistarh, 2023) and Wanda (Sun et al., 2023) remove redundant weights or channels. Activation optimizations similarly employ quantization exemplified by ZeroQuant (Yao et al., 2022) alongside dynamic sparsity strategies.

However, the memory footprint of KV cache during autoregressive decoding escalates dramatically with sequence length and model scale, emerging as a dominant bottleneck. This challenge intensifies in MLLMs, where vision encoders generate extensive visual tokens, significantly exacerbating KV cache pressure in subsequent LLM decoding stages (Zhou et al., 2025b).

To alleviate MLLM input burdens, prominent strategies focus on reducing visual tokens fed to the LLM. MobileVLM (Chu et al., 2023) employs aggressive compression via lightweight projections and pooling; LLaVA-PruMerge (Shang et al., 2024) and MADTP (Cao et al., 2024) introduce adaptive token pruning/merging mechanisms; FastV (Chen et al., 2024a) combines early-layer attention with late-layer pruning. These approaches effectively shorten input sequences, indirectly mitigating downstream KV cache demands. Critically, existing methods primarily target visual token reduction before LLM processing and often require task-specific fine-tuning. They address KV cache efficiency only as a secondary effect, lacking direct optimization mechanisms. Specialized techniques for compressing KV caches in MLLMs remain an underexplored research frontier.

#### 2.2 KV CACHE COMPRESSION

To address the critical bottleneck of KV cache memory overhead in MLLM inference, we systematically examine three dominant compression paradigms: eviction, quantization, and merging.

Eviction methods aggressively prune KV states by retaining only salient tokens while discarding others. Representative approaches like H2O (Zhang et al., 2023a) and SnapKV (Li et al., 2024) leverage attention scores to prioritize token retention. However, this irreversible information loss frequently induces context fragmentation and hallucinations, severely compromising long-context modeling (Jiang et al., 2025). Quantization techniques preserve full context coverage while reducing bit precision. MiKV (Yang et al., 2024) retains low-precision representations of evicted states, while KIVI (Liu et al., 2024b) and GEAR (Kang et al., 2024) employ channel-wise key and tokenwise value quantization (Zhou et al., 2025c;d). Although memory-efficient, these methods typically require specialized retraining or calibration, hindering seamless integration with existing LLM infrastructures.

Merging strategies condense multiple KV states into compact representations, minimizing performance degradation under memory constraints. MiniCache (Liu et al., 2024a) exploits inter-layer similarity for intra-layer compression, and CaM (Zhang et al., 2024) aggregates eviction candidates into preserved states (Zhou et al., 2025a). Crucially, these single-modal optimizations exhibit limited efficacy in MLLMs due to cross-modal distribution shifts and attention pattern divergence, failing to preserve modality-specific information fidelity.

### 3 Method

### 3.1 PRELIMINARY

MLLMs follow an autoregressive inference paradigm similar to text-only LLMs during the reasoning process, but they need to process heterogeneous input sequences containing both textual and visual tokens. Considering a multimodal input prompt consisting of interleaved text and image tokens, we can represent the concatenated prompt embeddings as:

$$\mathbf{X} = {\mathbf{X}_1^{\mathrm{T}}, \mathbf{X}_1^{\mathrm{I}}, \dots, \mathbf{X}_N^{\mathrm{T}}, \mathbf{X}_M^{\mathrm{I}}} \in \mathbb{R}^{L_{\mathrm{p}} \times d}$$

$$\tag{1}$$

where  $X^T$  and  $X^I$  denote textual and visual embeddings respectively,  $L_p$  is the total prompt length, and d is the hidden dimension. In the prompt encoding phase, the key and value tensors for each transformer layer are computed as:

$$\mathbf{K}_0 = \mathbf{X}\mathbf{W}^K, \quad \mathbf{V}_0 = \mathbf{X}\mathbf{W}^V \tag{2}$$

![](_page_3_Figure_1.jpeg)

Figure 2: **Overview of FlowMM.** (a) Cross-modal information flow analysis determines whether each layer exhibits predominantly cross-modal or intra-modal interactions, enabling layer-specific merging strategies. (b) Sensitivity-adaptive token matching jointly considers token similarity and sensitivity scores, preserving high-sensitivity tokens while merging similar low-sensitivity tokens to maintain critical contextual information.

where  $\mathbf{W}^K, \mathbf{W}^V \in \mathbb{R}^{d \times d}$  are the projection matrices. In the generation phase, the model sequentially produces tokens while maintaining and updating the KV cache. At generation step t, the KV cache is updated by concatenating new key-value pairs:

$$\mathbf{K}_t = [\mathbf{K}_{t-1}, \mathbf{k}_t], \quad \mathbf{V}_t = [\mathbf{V}_{t-1}, \mathbf{v}_t]$$
(3)

where  $\mathbf{k}_t = \mathbf{x}_t \mathbf{W}^K$  and  $\mathbf{v}_t = \mathbf{x}_t \mathbf{W}^V$  represent the key and value projections of the new token embedding  $\mathbf{x}_t$ . Finally, the attention output for the current step is computed as:

$$\mathbf{o}_t = \operatorname{Softmax}\left(\frac{\mathbf{q}_t \mathbf{K}_t^{\top}}{\sqrt{d}}\right) \mathbf{V}_t \tag{4}$$

While KV cache eliminates redundant computations in autoregressive generation, it creates substantial memory overhead as the sequence length grows. This challenge is especially severe in multimodal settings due to long visual token sequences from image encoders. KV cache Merge addresses this by compressing the cache through merging semantically similar key-value pairs, preserving vital attention information. The core principle of KV cache merge involves identifying tokens with high semantic similarity and consolidating their representations. This process can be formulated as:

$$\mathbf{K}^{\text{merged}} = f_{\text{merge}}(\mathbf{K}_t, \mathbf{S}), \quad \mathbf{V}^{\text{merged}} = g_{\text{merge}}(\mathbf{V}_t, \mathbf{S})$$
 (5)

where  $\mathbf{S} \in \mathbb{R}^{L_t \times L_t}$  represents a similarity matrix that captures pairwise relationships between tokens, and  $f_{\text{merge}}$ ,  $g_{\text{merge}}$  are merging functions that aggregate similar representations. This compression strategy effectively reduces memory complexity from  $O(L_{\text{p}}+t)$  to  $O(L_{\text{compressed}})$  where  $L_{\text{compressed}} \ll L_{\text{p}}+t$ , enabling efficient processing of long-context multimodal sequences. However, the success of KV cache merge critically depends on maintaining the integrity of multimodal information while achieving significant compression ratios, which presents unique challenges in the context of heterogeneous token representations.

#### 3.2 Observation

In this section, we explore how attention flow patterns influence KV cache merging of MLLMs in multimodal scenarios, presenting experimental findings. The study is conducted on the Qwen2.5-VL-7B (Bai et al., 2025).

### 3.2.1 Cross-modal Information Patterns.

Unlike traditional unimodal LLMs, MLLMs jointly process encoded visual and textual tokens to solve multimodal tasks, where cross-modal interactions generate responses. We first analyze patterns in cross-modal information transfer. Specifically, we conduct zero-shot inference on three datasets: ALFRED, MMCoQA, and Text Needle In A Haystack, measuring the proportion of attention scores allocated to tokens originating from the heterogeneous modality. All attention scores are aggregated through head-wise averaging.

As illustrated in Figure 3(a), our analysis reveals a striking divergence in crossmodal information flow patterns across the layers of MLLMs. This pattern exhibits consistent trends across diverse tasks. In the shallower layers, token interactions are predominantly intra-modal, characterized by significantly lower cross-modal attention scores. Conversely, deeper layers undergo a distinct shift, where intermodal interactions become dominant, corresponding to a substantial increase in

![](_page_4_Figure_4.jpeg)

<span id="page-4-0"></span>Figure 3: (a) Layer-wise divergence in cross-modal attention. (b) The performance comparison between full cache and information-flow merging.

cross-modal attention scores. We posit that shallow layers primarily facilitate low-level, unimodal feature extraction, while deeper layers progressively specialize in cross-modal fusion and higher-level semantic abstraction. This inherent functional disparity renders prior KV cache compression methods that apply uniform merging strategies across all layers inherently suboptimal. Consequently, our findings motivate the development of layer-specific merging schemes explicitly designed to align with these distinct cross-modal dynamics.

### 3.2.2 Cross-Modal Information Flow Merging.

To investigate the significance of cross-modal information patterns for multimodal KV cache merging, we empirically design merging strategies across diverse tasks. Specifically, we implement aligned information flow merging, performing intra-modal merging in layers with low cross-modal interaction and inter-modal merging in layers with high interaction. We contrast this with misaligned merging (applying intra-modal merging at high-interaction layers and inter-modal merging at low-interaction layers) and compare both against full cache.

As illustrated in Figure 3(b), aligned merging achieves performance comparable to full caching, while misaligned merging causes significant degradation. For instance, in the ALFRED task, misaligned merging only attain approximately 50% of the accuracy of full cache. We posit that reverse information flow merging may cause modal information confusion or semantic distortion. For example, prematurely merging across modalities without sufficient interaction between heterogeneous modality tokens at the shallow layers could disrupt the original modality representation of the tokens. This insight indicates that effective multimodal KV cache merging requires alignment with the inherent cross-modal information flow.

### 3.3 FLOWMM

### 3.3.1 Information Flow Guided Merging.

Cross-modal information flow characterizes the interaction intensity between heterogeneous modality tokens across different layers within MLLMs. Neglecting this flow significantly impairs KV cache merging performance. To address this, we introduce a Multimodal Information Flow-Guided KV Cache Merging strategy. This approach dynamically adjusts layer-specific merging strategy by quantifying cross-modal interaction intensity at each layer. Specifically, we define the cross-modal interaction ratio for a layer as the proportion of attention scores allocated to heterogeneous modality tokens:

$$\rho^{l} = \frac{1}{H} \sum_{h=1}^{H} \frac{A_{v \to t}^{l,h} + A_{t \to v}^{l,h}}{A^{l,h}},\tag{6}$$

where H denotes the number of attention heads, and  $A^{l,h}$  represents the sum of attention scores for the h-th attention head in the l-th layer. We define the cross-modal attention scores  $A^{l,h}_{v\to t}$  and  $A^{l,h}_{t\to v}$  as follows:

$$A_{v \to t}^{l,h} = \sum_{v \in V} \sum_{t \in T} \alpha_{v \to t}^{l,h}, \quad A_{t \to v}^{l,h} = \sum_{t \in T} \sum_{v \in V} \alpha_{t \to v}^{l,h}, \tag{7}$$

where V and T denote the sets of visual tokens and text tokens, respectively, and  $\alpha_{i \to j}$  represents the attention score from the i-th token to the j-th token. Then, we introduce a cross-modal merging threshold  $\theta$  to dynamically guide merging strategies. When the cross-modal attention interaction ratio  $\rho^l$  at layer l exceeds  $\theta$ , significant cross-modal interactions exist, warranting cross-modal merging. Conversely, if  $\rho^l$  falls below  $\theta$ , the layer predominantly processes intra-modal information, and we advocate more conservative intra-modal merging.

A crucial step in performing KV cache merging is identifying the sets that will be merged. Directly clustering and merging the KV cache sets is computationally expensive and may fail to leverage task-specific information, potentially leading to the disruption of context that is relevant to the task. Therefore, in this work, we first evaluate token importance. Previous studies have shown that using cumulative attention to evaluate token importance can be biased. To address this, we opt to use proxy tokens to provide a more equitable assessment of token importance:

$$\mathcal{I}^{l,h}(i) = \sum_{j \in \mathcal{P}} \alpha_{j \to i}^{l,h},\tag{8}$$

where  $\mathcal{P}$  denotes the set of proxy tokens. We designate a small subset of tokens near the end of the prompt as proxies, as these tokens typically capture task-specific contextual information. We select the top-B KV pairs with highest token importance to form a pivot set  $K^p$  capturing the most critical task information. The non-pivotal set  $K^n$  are then merged into the pivot set to minimize excessive loss of contextual information.

### 3.3.2 Sensitivity-Adaptive Token Matching.

Building upon the flow-guided multimodal KV cache merging, we introduce a critical component to address the risk of semantic corruption during state consolidation: Sensitivity-Adaptive Token Matching. This method specifically targets the identification and preservation of highly sensitive tokens crucial for maintaining model performance.

We define the sensitivity of a token within the current context as its contribution to preserving the model's output fidelity. A token is deemed highly sensitive if merging its KV state with others results in a substantial negative impact on the accuracy or relevance of subsequent model generations. Sensitivity is thus intrinsically linked to the token's unique informational value and its role in the multimodal reasoning chain. However, directly measuring the impact of merging each token through repeated perturbation tests during inference, incurs prohibitive computational costs for real-time scenarios. To address this, we propose attention scores as an efficient sensitivity metric. Attention scores directly quantify a token's influence on the current generation step, offering a near-zero-overhead approximation of sensitivity.

We assess the similarity between  $K^p$  and  $K^n$  by employing cosine similarity:

$$u_{i,j} = \frac{k_i^T k_j}{\|k_i\| \|k_j\|},\tag{9}$$

where  $u_{i,j}$  represents the cosine similarity between token i and token j, and  $\|\cdot\|$  is the norm. We then identify the nearest token in  $K^p$  for each token in  $K^n$ , as formulated below:

$$k_*^{\text{nearest}} = \underset{\substack{j \in K^p \\ \mathcal{I}_j \le \tau}}{\operatorname{Argmax}}(u_{i,j}), \tag{10}$$

where  $\tau$  denotes the sensitivity threshold. Tokens exceeding  $\tau$  are categorized as highly sensitive and thus prioritized for maximal information preservation, minimizing disruption during processing.

<span id="page-6-0"></span>Table 1: Performance of KV cache compression methods under 20% cache budget. The best results are highlighted in bold. ∆ denotes the difference to the Full Cache baseline. Note that for MobileVLM-V2-3B on the ImageNeedle task, we don't report its performance because even with full cache, its accuracy is nearly zero. This indicates that the model itself may not be well-suited for this particular task, and thus, the evaluation of effectiveness in this context would not be meaningful.

| Method          | ALFRED IEdit |      |             |       |       |       | STD MMCoQA CLEVR-C TextNeedle ImageNeedle Average |       | ∆     |
|-----------------|--------------|------|-------------|-------|-------|-------|---------------------------------------------------|-------|-------|
| Qwen2.5-VL-7B   |              |      |             |       |       |       |                                                   |       |       |
| Full Cache      | 36.92        |      | 30.16 28.13 | 50.50 | 45.45 | 11.56 | 24.38                                             | 32.44 | -     |
| StreamingLLM    | 27.61        |      | 30.43 26.85 | 46.00 | 37.00 | 4.38  | 1.88                                              | 24.88 | -7.57 |
| H2O             | 34.31        |      | 30.91 26.63 | 45.50 | 42.49 | 4.69  | 5.00                                              | 27.08 | -5.36 |
| D2O             | 33.59        |      | 30.56 26.16 | 39.50 | 41.58 | 4.69  | 8.75                                              | 26.40 | -6.04 |
| KVMerge         | 27.94        |      | 31.16 27.83 | 44.50 | 37.95 | 9.69  | 15.00                                             | 27.72 | -4.72 |
| LOOK-M          | 34.76        |      | 30.58 25.37 | 39.50 | 40.41 | 2.50  | 3.13                                              | 25.18 | -7.26 |
| FlowMM          | 35.43        |      | 31.67 28.08 | 48.50 | 41.79 | 10.00 | 17.13                                             | 30.37 | -2.07 |
| InternVL2.5-8B  |              |      |             |       |       |       |                                                   |       |       |
| Full Cache      | 35.34        | 9.12 | 26.37       | 52.50 | 22.76 | 25.00 | 24.69                                             | 27.97 | -     |
| StreamingLLM    | 23.36        |      | 10.71 25.33 | 51.50 | 19.39 | 10.00 | 2.50                                              | 20.40 | -7.57 |
| H2O             | 33.05        |      | 11.31 26.21 | 51.50 | 21.01 | 10.93 | 21.25                                             | 25.04 | -2.93 |
| D2O             | 32.63        |      | 11.08 26.64 | 50.00 | 20.82 | 11.25 | 21.25                                             | 24.81 | -3.16 |
| KVMerge         | 33.61        |      | 11.42 26.33 | 51.50 | 20.47 | 17.63 | 21.45                                             | 26.06 | -1.91 |
| LOOK-M          | 31.81        |      | 11.18 26.82 | 51.00 | 21.21 | 8.25  | 17.15                                             | 23.92 | -4.05 |
| FlowMM          | 34.77        |      | 11.93 26.59 | 52.00 | 22.58 | 23.12 | 23.93                                             | 27.85 | -0.12 |
| MobileVLM-V2-3B |              |      |             |       |       |       |                                                   |       |       |
| Full Cache      | 25.18        | 6.55 | 13.57       | 7.00  | 15.97 | 9.68  | -                                                 | 12.99 | -     |
| StreamingLLM    | 13.73        | 5.82 | 7.65        | 2.50  | 6.55  | 3.12  | -                                                 | 6.56  | -6.43 |
| H2O             | 24.86        | 6.22 | 10.27       | 3.00  | 14.06 | 2.50  | -                                                 | 10.15 | -2.84 |
| D2O             | 23.96        | 6.48 | 11.59       | 4.50  | 13.85 | 4.34  | -                                                 | 10.79 | -2.20 |
| KVMerge         | 24.47        | 6.39 | 11.51       | 4.00  | 14.67 | 5.38  | -                                                 | 11.07 | -1.92 |
| LOOK-M          | 24.40        | 6.12 | 10.86       | 4.00  | 13.04 | 2.87  | -                                                 | 10.22 | -2.78 |
| FlowMM          | 25.06        | 6.57 | 12.73       | 5.50  | 15.39 | 8.67  | -                                                 | 12.32 | -0.67 |

