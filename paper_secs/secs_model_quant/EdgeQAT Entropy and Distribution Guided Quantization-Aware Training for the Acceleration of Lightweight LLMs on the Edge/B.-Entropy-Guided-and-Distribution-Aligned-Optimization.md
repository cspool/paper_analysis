# B. Entropy-Guided and Distribution-Aligned Optimization

Based on the analysis in Section III, the performance loss is primarily attributed to the quantized attention module (especially the query and key) with deteriorated representation capability. To address this issue, we propose the entropy-guided and distribution-aligned optimization method, which statistically maximizes the entropy of representations and restores the capability of the quantized self-attention module. According to the work [22], for Gaussian distributions, quantizers with maximum output entropy (MOE) and minimum average error (MAE) are approximately equivalent, up to a multiplicative constant. In essence, minimizing the error caused by quantization is equivalent to maximizing the information entropy of quantized values. As observed in Figure 2, the distributions of the query  ${\bf q}$  and the key  ${\bf k}$  in the self-attention modules follow the Gaussian distribution as below,

$$\mathbf{q} \sim \mathcal{N}(\mu_{\mathbf{q}}, \sigma_{\mathbf{q}}),$$
 (8)

$$\mathbf{k} \sim \mathcal{N}(\mu_{\mathbf{k}}, \sigma_{\mathbf{k}}).$$
 (9)

The entropy can be represented as follows,

$$\mathcal{H}(\mathbf{q}) = -\sum_{i} p(\mathbf{q}_i) \log p(\mathbf{q}_i) = \frac{1}{2} \log 2\pi e \sigma_{\mathbf{q}}^2, \tag{10}$$

$$\mathcal{H}(\mathbf{k}) = -\sum_{i} p(\mathbf{k}_{i}) \log p(\mathbf{k}_{i}) = \frac{1}{2} \log 2\pi e \sigma_{\mathbf{k}}^{2}.$$
 (11)

To maximize the entropy  $\mathcal{H}(\mathbf{q}) \propto \sigma_{\mathbf{q}}^2$  and  $\mathcal{H}(\mathbf{k}) \propto \sigma_{\mathbf{k}}^2$  during the training process, we incorporate the entropy loss  $\mathcal{L}_E$  to optimize the total entropy of query and key for all layers and heads. Specifically, we re-scale the entropy loss as follows:

$$\mathcal{L}_E = -\log\left(\sum_{l=1}^{L} \sum_{h=1}^{H} \log\left(1 + \sigma_{\mathbf{q}}^2 \sigma_{\mathbf{k}}^2\right)\right), \quad (12)$$

where L and H denote the number of layers and heads, respectively. To prevent the occurrence of NaNs when scaling loss with log operation, we increment deviation product by 1.

Next, we focus on fixing the distribution pattern issue in the attention map. As shown in Figure 3, the column distribution pattern with the initial tokens from the FP16 counterpart disappears after quantization in the quantized attention map. To minimize the difference between the quantized attention map and the FP16 counterpart, a distribution loss  $\mathcal{L}_D$  is introduced based on the cosine similarity between the FP16 attention map  $attn_f$  and quantized one  $attn_q$  in each layer as follows:

$$\mathcal{L}_D = \log \left( \sum_{l=1}^{L} \sum_{h=1}^{H} \frac{attn_q \cdot attn_f}{\|attn_q\|_2 \cdot \|attn_f\|_2} \right).$$
 (13)

We re-scale the loss with the logarithmic operation to match the scale of the original loss.

#### C. Token Adaptive Quantization

Similar to token pruning [9], [42], two features of the transformer structure: token-level redundancy and sequential computing, open up possibilities for token adaptive mixed-precision quantization. Based on the analysis section, we assess the token importance with the averaged attentivity to the initial token through all transformer heads, denoted by the first column of the attention map (i.e., attn[:,0]). Considering the trade-off between task performance and practical hardware efficiency, we assign 8 bits for more important tokens and 4 bits for less attentive ones. Specifically, we adopt adaptive quantization for  $\forall i \in [0, N-1]$  as follows,

$$\beta(\mathbf{x}_i \mid attn, \rho) = \{ \begin{array}{ll} 8, & attn[i, 0] \geq \mathrm{TopK}(attn[:, 0], \mathrm{Int}(\rho * N)), \\ 4, & \mathrm{others}, \end{array}$$

where  $\mathbf{x}_i$  denotes  $i^{th}$  token during training and generation processes,  $\rho$  represents important token ratio, N denotes

number of tokens, function β(x<sup>i</sup> | attn, ρ) returns bit width for the i th token given attention map attn and ρ, and TopK(·, k) denotes top-k function that returns kth largest element.

We design a Token Control Logic Module (TCLM) for adaptive quantization as shown in Figure 5. First, β(x<sup>i</sup> | attn[: , 0], ρ) evaluates the importance of the i th input token according to the averaged attentivity. When x<sup>i</sup> is informative, they are concatenated together for the following 8-bit layerwise integer quantization; Otherwise, if x<sup>i</sup> is less informative, they are concatenated for the following 4-bit quantization. After the layer-wise integer quantization, our proposed MKMP multiplier is called to execute mixed integer MAC. For the TopK(·, k) implementation, the fast top-k sorting operator, Heapsort, is leveraged, to support the ρ important token selection. Heapsort and Concatenation are existing operands with marginal overhead.

