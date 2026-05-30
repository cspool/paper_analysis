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

