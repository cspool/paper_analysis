# Squat: Quant Small Language Models on the Edge

Xuan Shen 1 , Peiyan Dong 2 , Zhenglun Kong 3 , Yifan Gong 1 , Changdi Yang 1 , Zhaoyang Han 1 , Yanyue Xie 1 , Lei Lu 1 , Cheng Lyu 1 , Chao Wu 1 , Yanzhi Wang 1 and Pu Zhao 1 <sup>1</sup>Northeastern University, <sup>2</sup>MIT, <sup>3</sup>Harvard University

*Abstract*—A growing trend has emerged in designing highquality Small Language Models (SLMs) with a few million parameters. This trend is driven by the increasing concerns over cloud costs, privacy, and latency. Considering that full parameter training is feasible for SLMs on mobile devices, Quantization-Aware Training (QAT) is employed to improve efficiency by reducing computational overhead and memory footprint. However, previous QAT works adopt fine-grained quantization methods to compress models with billions of parameters on GPUs, incompatible with current commodity hardware, such as mobile and edge devices, which relies on Single Instruction Multiple Data (SIMD) instructions. Thus, the generalization of these methods to SLMs on mobile devices is limited. In this paper, we propose Squat method, an effective QAT framework with deployable quantization for SLMs on mobile devices. Specifically, we propose entropy-guided and distribution-aligned distillation to mitigate the distortion of attention information from quantization. Besides, we employ sub-8-bit token adaptive quantization, assigning varying bit widths to different tokens based on their importance. Furthermore, we develop a SIMD-based Multi-Kernel Mixed-Precision (MKMP) multiplier to support sub-8-bit mixedprecision MAC on mobile devices. Our extensive experiments verify the substantial improvements of our method compared to other QAT methods across various datasets. Furthermore, we achieve an on-device speedup of up to 2.37 × compared with its FP16 counterparts, signaling a great advancement. Code: https://github.com/shawnricecake/squant

*Index Terms*—Quantization-Aware Training, Mobile, Small Language Models

# I. INTRODUCTION

Large Language Models (LLMs) [4], [5], [26], [39], [48] have emerged as the dominant force in the field of Natural Language Processing (NLP). These models are increasingly integrated into various applications on daily life devices to enhance task performance and user experience. With growing concerns about the cost and latency of deploying LLMs in the cloud, the trend is increasingly shifting towards deploying Small Language Models (SLMs) on edge platforms such as mobile devices. Recently, MobileLLM [21], SmolLM2 [1], BabyLlama [37], [38] and other works [33], [40], [49] have prioritized the design of high-quality SLMs with fewer than one billion parameters, making such lightweight models a viable option for mobile deployment. Meanwhile, given the widespread support for quantized computation by Single Instruction Multiple Data (SIMD) mechanisms on mobile and edge devices [2], [28]–[32], [34], [35], model quantization emerges as a promising approach to accelerate SLMs on these platforms. Existing methods using Post-Training Quantization (PTQ) often suffer significant accuracy drops in sub-8-bit settings. Some PTQ methods like GPTQ [11], SqueezeLLM [17], and AWQ [19] only adopt the weight-only quantization, which leave the activations in float16 and can not fully exploit efficient integer computation on edge devices [11], [19], [27], [45]. Other PTQ methods, including Agile-Quant [27], SmoothQuant [45], and ZeroQuant [44], [47], quantize both weights and activations. However, their performance drops significantly when the activation bit width is reduced to 4 bit. In contrast, Quantization-Aware Training (QAT) allows fine-tuning to reduce quantization error, and models with both weights and activations quantized can take full advantage of fast integer matrix multiplication on edge hardware. Meanwhile, for SLMs, full parameter training is feasible, making QAT a practical and effective option. This enables us to jointly optimize both weights and activations during training, leading to better performance at lower bit widths—especially when targeting 4-bit deployment.

Recent QAT works [6], [16], [20] employ channel-wise or token-wise (fine-grained) quantization for weights and activations with sub-8-bit precision, resulting in multiple scaling factors for a single matrix. However, such fine-grained quantization methods are incompatible with computation kernels in general mobile devices and edge processors, leading to significant computation overhead. Specifically, standard SIMD-based libraries [10], [14] do not support running quantized networks with sub-8-bit precision, and the general matrix multiply (GeMM) kernel in SIMD cannot handle matrix multiplications (MAC) with multiple scaling factors for integer operations [2]. In conclusion, to effectively harness the power of SLMs on mobile devices, it is necessary to address the challenges of quantization algorithm design and corresponding efficient hardware implementation.

Therefore, to build efficient and accurate SLMs on mobile devices, we present a novel framework called Squat in this paper. Specifically, we first identify that, for the self-attention module, the information distortion brought by quantizing queries and keys is the most critical factor leading to information loss. We then introduce an entropy-guided optimization method to mitigate the loss by maximizing the information entropy. Simultaneously, we align the distributions of the quantized attention maps to the FP16 one to minimize the difference. Furthermore, the sub-8-bit token adaptive quantization is implemented, which assigns varying bit widths to different tokens based on their informativeness, thereby further reducing redundancy. Moreover, we recompile the existing INT8 multiplier and develop a SIMD-based Multi-Kernel Mixed-Precision (MKMP) multiplier for the proposed token adaptive quantization method. Finally, Squat framework can accelerate the inference of mixed-precision SLMs with sub-8-bit quantization on mobile devices and edge processors. In our experiments, we uniformly quantize the weights with 4 bits or 8 bits, and adaptively quantize the activations with 4 bits or 8 bits. The results show that our method can achieve better task performance than other coarse-grained QAT methods. Moreover, our results demonstrate that our proposed adaptive quantization approach yields superior performance with a mixed strategy compared to a uniform strategy. For instance, a combination of half 4-bit and half 8-bit quantization outperforms uniform 6-bit quantization. Furthermore, our quantized models with the proposed MKMP multiplier can achieve a practical speedup of up to 2.37× on mobile devices. Our contributions are summarized below:

- 1. We design the entropy-guided and distribution-aligned QAT method to mitigate information distortion brought by quantization.
- 2. We design the token importance-aware adaptive quantization method for activations (i.e., tokens), further reducing model redundancy and outperforming bit-equivalent uniform quantization.
- 3. We develop a SIMD-based MKMP multiplier for the acceleration of sub-8-bit mixed-precision LLM inference on mobile devices.
- 4. We achieve better task performance than other QAT methods with an on-device speedup of up to 2.37×.

#### II. BACKGROUND AND RELATED WORKS

#### *A. Efficient Design of SLMs*

Recently, a growing trend [40] is the use of SLMs for specialized tasks on mobile devices. These models, such as MiniCPM [13], Octopus [7], SmolLM2 [1], and MobileLLM [21], have shown that they can perform effectively on general tasks while being more suitable for deployment on mobile platforms. To facilitate SLM developments, quantization has emerged as a powerful technique to reduce the computational and memory demands with low-bit format. Current quantization methods can be broadly categorized into PTQ and QAT. Previous weight-only quantization methods, such as GPTQ [11], SqueezeLLM [17], and AWQ [19], focus on optimizing quantized weights while leaving activations unquantized. This approach fails to exploit efficient integer computation capabilities on edge devices. PTQ works, such as SmoothQuant [45], ZeroQuant [47], ZeroQuant-FP [44], and Agile-Quant [27] can achieve multiple different quantization configurations for both weights and activations. On the other hand, QAT methods, like LLM-QAT [20], TSLD [16], EfficientQAT [6], and others [36] use fine-tuning with fine-grained quantization (e.g., channel-wise and token-wise ) to recover task performance. However, these fine-grained quantization techniques are difficult to be deployed on mainstream edge processors, making them less practical for mobile devices.

#### *B. Hardware Implementation for Quantization*

Low-precision linear algebra kernels are designed to maximize computing throughput by adapting existing wider bit-

![](_page_1_Figure_10.jpeg)

Figure 1: Accuracy analysis of LLaMA-58M on the Anaphor Agr. subdataset of BLiMP with different quantized modules.

width kernels to handle lower-precision operands. This approach improves performance based on lower-precision SIMD instructions (e.g., vmlaq s8() in ARMv8 ISA), which process more elements in parallel than higher-precision instructions (e.g., vmlaq f32()). State-of-the-art (SOTA) low-precision linear algebra kernels, such as Google's GEMMLOWP [14] and Meta's QNNPACK [10], can greatly enhance model efficiency with 8-bit quantization (W8A8) scenarios, e.g., on a 64 bit ARM Cortex-A72 CPU. However, two critical challenges have yet to be addressed to achieve SOTA complex quantization frameworks. (i) More aggressive sub-8-bit quantization does not offer additional performance benefits, as commercial CPUs/GPUs only support SIMD operations with 8-bit or higher precision. Thus, low-precision kernels simply zeroextend sub-8-bit operands to byte boundaries, treating them as 8-bit operands. (ii) Quantization process is executed with a SIMD-based GeMM engine that supports layer-wise (coarsegrained) quantization with a single scaling factor per matrix, which means the fine-grained quantization methods adopted by SOTA methods are limited on the edge.

#### III. ANALYSIS

To identify the bottlenecks of layer-wise QAT, we analyze performance deterioration when quantizing each part of the model. Additionally, we assess token importance using the attention map, specifically examining its relationship with the initial token of each head.

## *A. Quantized Self-Attention Module*

We first analyze the degradation on task performance by applying layer-wise quantization (i.e., coarse-grained quantization) to various components of the model without fine-tuning. In Figure 1, the MLP module, the entire self-attention module, and specific parts of the self-attention module (query and key) are quantized, respectively. Other parts of the model remain as FP16. As shown in Figure 1, we identify that the quantized query and key are the main factors that lead to the substantial performance drop, as the performance drop caused by query and key nearly mirrors the loss from quantizing the entire self-attention module. We further visualize the distributions of quantized queries and keys, which approximate Gaussian

![](_page_2_Figure_0.jpeg)

Figure 2: Distributions of query and key at the last layer of FP16 and quantized LLaMA-58M. The main difference is from the variance of the distribution.

![](_page_2_Figure_2.jpeg)

Figure 3: Averaged attention maps through all heads at the last layer of the FP16 and quantized models. The main difference is from the first column of the heatmap.

distributions [24], [27]. As shown in Figure 2, there is a significant difference in distributions between the quantized and FP16 versions for either the query or key (e.g., different variance). The distribution difference results in an entropy discrepancy, inevitably leading to a deterioration in the attention module's representational capability.

#### B. Token Importance

In LLMs, a unique initial token is placed at the start of the input sequence to define token positions, visible to all subsequent tokens due to autoregressive language modeling. Removing interactions between the initial token and other ones can fundamentally alter the model's output. However, as shown in Figure 3, a distinct column pattern associated with the initial token in the FP16 version disappears in the quantized version. Besides, we highlight that the initial token remains critical for assessing token importance, which is

essential for further optimizations like token pruning [9], [15], [18], [27]. In generative models, the self-attention mechanism limits each token's interactions to those preceding it, and the initial token encapsulates the informational content of each generated token. Thus, we evaluate token importance based on each token's average attentivity to the initial token with all heads (i.e., the first column of attention map).

#### IV. METHODOLOGY

In this section, we first provide the preliminary of QAT. Then, we propose the design of entropy loss and distribution loss. We further introduce token adaptive quantization method based on token importance. We also design the MKMP multiplier for adaptive quantization deployment on mobile devices.

#### A. Preliminary

For layer-wise QAT, we adopt the symmetric quantization for both weights  $\mathbf{w}$  and activations  $\mathbf{x}$  as follows,

$$Q(\mathbf{w}) = \lfloor \text{CLIP}(\frac{\mathbf{w}}{\alpha_{\mathbf{w}}}, -2^{b_{\mathbf{w}}-1}, 2^{b_{\mathbf{w}}-1} - 1) \rceil; \tag{1}$$

$$\hat{\mathbf{w}} = \mathcal{Q}(\mathbf{w}) \cdot \alpha_{\mathbf{w}}; \tag{2}$$

$$Q(\mathbf{x}) = \lfloor \text{CLIP}(\frac{\mathbf{x}}{\alpha_{\mathbf{x}}}, -2^{b_{\mathbf{x}}-1}, 2^{b_{\mathbf{x}}-1} - 1) \rceil; \tag{3}$$

$$\hat{\mathbf{x}} = \mathcal{Q}(\mathbf{x}) \cdot \alpha_{\mathbf{x}},\tag{4}$$

where  $\mathcal{Q}(\cdot)$  denotes the quantization function. CLIP $(x, r_1, r_2)$  returns x with values in the range from  $r_1$  to  $r_2$  through clipping with the lower bound  $r_1$  and upper bound  $r_2$ .  $\lfloor \cdot \rfloor$  represents rounding to the nearest integer.  $\mathbf{x}$  is the activations and  $\mathbf{w}$  means the weights.  $\hat{\mathbf{x}}$  and  $\hat{\mathbf{w}}$  denote the dequantized activations and weights with scaling factor  $\alpha$ .  $b_{\mathbf{x}}$  and  $b_{\mathbf{w}}$  denote bit width for activations and weights.

For the forward propagation, the linear projection can be calculated as follows:

$$\mathcal{F}_{Linear}(\mathbf{x}, \mathbf{w}) = \hat{\mathbf{x}} \times \hat{\mathbf{w}}$$

$$= \alpha_{\mathbf{x}} \alpha_{\mathbf{w}} \left[ \mathcal{Q}(\mathbf{x}) \times \mathcal{Q}(\mathbf{w}) \right], \tag{5}$$

where the  $\mathcal{F}_{Linear}$  denotes the normal matrix multiplication. For backward propagation, gradients are computed as follows:

$$\begin{split} \frac{\partial \mathcal{J}}{\partial \mathbf{x}} &= \frac{\partial \mathcal{J}}{\partial \hat{\mathbf{x}}} \frac{\partial \hat{\mathbf{x}}}{\partial \mathbf{x}} \\ &= \left\{ \begin{array}{ll} \frac{\partial \mathcal{J}}{\partial \hat{\mathbf{x}}}, & \mathbf{x} \in \left[-2^{b_{\mathbf{x}}-1}, 2^{b_{\mathbf{x}}-1} - 1\right], \\ 0, & \text{otherwise.} \end{array} \right. \end{split}$$
(6)

$$\frac{\partial \mathcal{J}}{\partial \mathbf{w}} = \frac{\partial \mathcal{J}}{\partial \mathbf{x}} \frac{\partial \mathbf{x}}{\partial \hat{\mathbf{w}}} \frac{\partial \hat{\mathbf{w}}}{\partial \mathbf{w}} 
= \begin{cases} \frac{\partial \mathcal{J}}{\partial \mathbf{x}} \frac{\partial \mathbf{x}}{\partial \hat{\mathbf{w}}}, & \mathbf{w} \in [-2^{b_{\mathbf{w}}-1}, 2^{b_{\mathbf{w}}-1} - 1], \\ 0, & \text{otherwise.} \end{cases} (7)$$

where  $\mathcal{J}$  denotes the loss function, and straight-through estimator (STE) [3] is adopted to retain derivation of gradients.

![](_page_3_Figure_0.jpeg)

Figure 4: Adaptive distillation pipeline based on token importance score (colored in red), with maximum entropy loss and attention map cosine similarity loss (both colored in green).

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

# *D. Adaptive Training Pipeline*

We visualize our training pipeline in Figure 4. We use the FP16 model (colored in yellow) to distill the quantized model (colored in blue) during QAT. We apply soft distillation, which trains a student model to mimic a teacher model by minimizing the KL divergence between their softmax outputs [12]. The distillation loss is defined as:

$$\mathcal{L}_{distill} = (1 - \gamma) \cdot \mathcal{L}_{CE} + \gamma \tau^2 \cdot \mathcal{L}_{KL}, \tag{15}$$

where τ is the temperature for the distillation, and γ is the coefficient balancing the KL divergence loss LKL and the cross-entropy loss LCE. In the quantization modules, the tokens are adaptively quantized with either 8 bits or 4 bits based on their scores (colored in red in Figure 4) generated from the most recent attention map.

The entropy loss L<sup>E</sup> and the distribution loss L<sup>D</sup> (both colored in green) are added to the total loss for optimization during training as follows,

$$\mathcal{L}_{total} = \mathcal{L}_{distill} + r_E \cdot \mathcal{L}_E + r_D \cdot \mathcal{L}_D, \tag{16}$$

where the ratios r<sup>E</sup> and r<sup>D</sup> are used to scale the entropy and distribution losses, respectively. In our experiments, we set r<sup>E</sup> = 0.5 and r<sup>D</sup> = 1 to facilitate better optimization.

# *E. Multi-Kernel Mixed-Precision Multiplier*

The standard SIMD-based INT8 multipliers do not support mixed-precision integer MAC operations and typically zeroextend sub-8-bit operands to byte boundaries as 8-bit operands. To implement the proposed layer-wise token-adaptive quantization, we develop a SIMD-based MKMP multiplier to enable this mixed-precision quantization on devices as shown in Figure 5. After token adaptive quantization, we use existing INT8 multipliers for the 8-bit concatenated tokens and implement the INT4 multipliers. The INT4 multiplier is built on the existing INT8 multiplier, which concatenates weights from adjacent rows and multiplies them with a shared activation value in the SIMD kernel.

By concatenating weight matrices within GeMM, our approach significantly reduces the number of mathematical computation instructions required on processors compared to traditional byte-level quantized implementation kernels. This reduction is proportional to the concatenation density for the same workload. As shown in Figure 6, two 4-bit operands are concatenated into a single 16-bit register unit. This choice aligns with current practices in 8-bit quantized implementation kernels, where 8-bit data is often extended to 16 bits during product computations. This enables the use of efficient instructions like *mla* in Arm ISAs, which utilize a 32-bit destination register (INT32 datatype) to perform multiplication and accumulation in a single instruction. The 16-bit intermediate registers facilitate concatenation while offering redundancy beyond the actual sub-byte values. A low-bit priority strategy ensures that the bit width is utilized evenly, minimizing redundant zeros for subsequent computations. The 16-bit-wide multiplication operation is then performed, with the results internally split to maintain mathematical accuracy for the subsequent addition steps. Theoretically, this design reduces the computational burden—both in terms of multiplications and additions—for 4-bit GeMM by half compared to the conventional approach, which expands 4-bit data to 8-bit for use in byte-level quantization kernels.

Notably, while this methodology originates from weightmatrix concatenation, it is equally applicable to activationactivation matrix multiplications in transformer models. Similar to weight matrices, one of the activation matrices can be concatenated while preserving the logic and characteristics of design. This versatility makes it a broadly adaptable low-bit acceleration strategy. Using the SIMD-based memory mechanism, the INT4 multiplier employs bit-shift and row-by-row summation to add up intermediate values. INT4 multiplier can save 50% hardware resources of INT8 multiplier. By integrating quantization operator, we streamline entire MKMP multiplier within the GeMM kernel. Due to LLMs' huge memory readout, we optimize and assign computing threads for different operations and overlap memory readout time from the compiler level.

#### V. EXPERIMENTAL SETUP

## *A. Quantization Setup*

For the verification and deployment of our proposed methods, we experiment with lightweight LLMs, including LLaMA-58M [38], [39] and GPT2-97M [25]. We adopt the pretrain datasets from the work [46] and then perform regexbased cleaning on them. The cleaned datasets are tokenized using BytePair Encoding (BPE) with a vocabulary size of 16000. The models are then evaluated on BLiMP [43] for the zero-shot test, and (Super) GLUE [41] for the fine-tuning test. In the absence of prior coarse-grained QAT studies for LLMs, we compare with well-known static quantization methods as baselines, including NIPQ [23], PACT [8], and LLM-QAT [20]. The same fine-tuning recipe with distillation based on the FP16 pretrained model is adopted for all experiments.

![](_page_5_Figure_0.jpeg)

Figure 5: Comparison of INT8 multiplier and SIMD-based MKMP multiplier to support mixed-precision MAC for adaptive token quantization.

Table I: LLaMA-58M quantization results on the BLiMP dataset, including the BLiMP Supplement.

| # Bits     | # Bits   FP16   W8A8 |      | W4A8 |         |      | W4A4 |      |         |      |      |      |         |      |
|------------|----------------------|------|------|---------|------|------|------|---------|------|------|------|---------|------|
| Method     | /                    | NIPQ | PACT | LLM-QAT | Ours | NIPQ | PACT | LLM-QAT | Ours | NIPQ | PACT | LLM-QAT | Ours |
| BLiMP Main |                      |      |      |         |      |      |      |         |      |      |      |         |      |
| AA         | 89.8                 | 85.5 | 86.4 | 88.0    | 88.1 | 58.1 | 86.6 | 87.1    | 87.6 | 66.2 | 85.8 | 85.9    | 85.7 |
| AS         | 73.1                 | 70.9 | 70.7 | 72.4    | 72.2 | 55.5 | 70.3 | 72.4    | 72.3 | 54.4 | 69.6 | 72.0    | 71.3 |
| Bind.      | 72.7                 | 71.1 | 71.0 | 71.9    | 72.3 | 61.7 | 70.6 | 71.8    | 72.2 | 51.5 | 68.2 | 71.5    | 72.4 |
| C/R        | 67.5                 | 65.5 | 64.6 | 66.6    | 66.7 | 54.7 | 64.0 | 65.8    | 66.7 | 53.6 | 63.6 | 65.4    | 66.3 |
| D-NA       | 90.8                 | 86.9 | 86.3 | 89.0    | 89.2 | 54.2 | 86.6 | 90.1    | 89.1 | 53.4 | 84.8 | 87.1    | 87.5 |
| Ell.       | 73.3                 | 60.4 | 59.7 | 68.4    | 69.4 | 29.9 | 59.7 | 67.2    | 69.8 | 33.8 | 56.8 | 63.2    | 65.1 |
| F-G        | 71.8                 | 70.2 | 69.0 | 71.8    | 72.1 | 66.7 | 69.3 | 71.7    | 72.0 | 61.1 | 66.8 | 70.2    | 70.4 |
| IF         | 93.1                 | 94.6 | 94.8 | 95.1    | 95.0 | 45.8 | 95.2 | 93.3    | 94.9 | 52.2 | 93.7 | 94.1    | 94.9 |
| ΙE         | 51.2                 | 48.2 | 49.2 | 51.3    | 51.7 | 43.6 | 50.0 | 51.9    | 52.1 | 48.5 | 43.3 | 48.2    | 51.3 |
| NPI-L      | 56.5                 | 50.0 | 52.1 | 57.9    | 58.3 | 26.8 | 52.2 | 57.3    | 57.7 | 36.6 | 48.2 | 45.9    | 44.5 |
| Quan.      | 73.3                 | 73.7 | 75.8 | 81.0    | 79.0 | 57.2 | 78.2 | 79.4    | 79.3 | 42.7 | 78.0 | 78.2    | 80.0 |
| S-VA       | 75.4                 | 68.4 | 67.8 | 73.1    | 73.2 | 46.3 | 67.7 | 73.0    | 74.0 | 48.6 | 64.5 | 68.0    | 70.3 |
| Avg.       | 74.0                 | 70.5 | 70.6 | 73.8    | 73.9 | 50.0 | 70.9 | 73.4    | 74.0 | 50.2 | 68.6 | 71.0    | 71.8 |
|            | BLiMP Supplement     |      |      |         |      |      |      |         |      |      |      |         |      |
| Hyper.     | 49.3                 | 48.0 | 49.0 | 49.6    | 48.9 | 49.5 | 48.7 | 48.7    | 49.6 | 50.9 | 50.3 | 49.3    | 50.5 |
| QAC-E      | 51.6                 | 48.4 | 51.5 | 49.1    | 50.1 | 35.9 | 50.0 | 49.8    | 50.1 | 37.5 | 48.4 | 49.3    | 50.1 |
| QAC-t      | 41.8                 | 40.6 | 40.0 | 41.6    | 41.3 | 34.5 | 40.6 | 40.6    | 41.3 | 33.9 | 39.3 | 40.6    | 41.9 |
| S-AI       | 88.5                 | 89.1 | 87.9 | 88.6    | 88.5 | 67.8 | 89.8 | 89.1    | 89.2 | 54.6 | 87.3 | 87.3    | 89.0 |
| TT         | 66.1                 | 58.2 | 57.1 | 62.0    | 61.5 | 43.2 | 57.5 | 60.3    | 61.8 | 51.4 | 55.7 | 59.2    | 60.1 |
| All Avg.   | 69.7                 | 66.5 | 66.6 | 69.2    | 69.3 | 48.9 | 66.9 | 68.8    | 69.4 | 48.9 | 64.9 | 66.9    | 67.8 |

![](_page_5_Figure_4.jpeg)

Figure 6: SIMD-based INT4-concatenated multiplier design.

## B. Hardware Deployment

We use the OnePlus 11 smartphone, powered by the Snapdragon 8 Gen 2, as our mobile platform, utilizing all available cores for multi-threaded computation. Similarly, on the Raspberry Pi 5 with its BCM2712 quad-core Arm Cortex A76 processor, we deploy our quantized model and distribute

the computations across all four cores. Latency is reported based on 1000 iterations for each test.

#### VI. EXPERIMENTAL RESULTS

## A. Zero-Shot Evaluation

We first verify the effectiveness of our proposed QAT framework on the BLiMP [43] dataset with zero-shot (i.e., no fine-tuning) evaluations, and the results are shown in Table I. We compare our method with the other three QAT works, including NIPQ, PACT, and LLM-QAT, under different bitwidth settings including W8A8 (meaning 8-bit weight and 8-bit activation quantization), W4A8, and W4A4. As observed, our approach achieves better performance than all other three works in terms of the average accuracy of all subdatasets on the BLiMP dataset. Our method performs the best on most of the subdatasets across three bit-width configurations. Especially for the W4A8 setting, which is the most practical in wide applications, our method achieves an average accuracy of 69.4%, which is close to that of the FP16 model (only

0.3% drop) and even surpasses the W8A8 setting (69.3%). For the W4A4 setting, our method maintains an average accuracy of 67.8%, showcasing a clear advantage over other methods. Only our method can achieve a competitive average accuracy close to that of the FP16 model, while the baselines usually suffer from substantial accuracy drops. NIPQ fails to restore the accuracy when the model weights are quantized to 4 bits. For PACT, it is sensitive to the bit width of the activations, as evidenced by the poor results under the W4A4 setting. The LLM-QAT method consistently produces models with an lower average accuracy than our method.

## *B. Generalization Verification*

Additionally, we deliver the evaluation results of the GPT2- 97M model with the W4A4 setting to verify the generalization of our method in Table II. We conduct the experiments on the BLiMP main dataset. Our method can achieve the highest average accuracy with the best performance on most of the subdatasets, demonstrating our clear advantages over QAT baselines. Among the baselines struggling to restore the accuracy, the NIPQ and PACT perform much worse with large margins. Thus, the clear advantages, achieved by our method compared to other QAT methods, validates the generalization of our proposed Squant method for the small language models.

## VII. FINE-TUNING EVALUATION

To further demonstrate the effectiveness of the proposed Squant framework, we finetune the quantized models from different QAT frameworks on the (Super) GLUE dataset and show the evaluation results in Table III. To make a fair comparison, we use the same finetuning recipe for all methods. As observed, the proposed Squat method can restore the performance on all subdatasets and demonstrate a clear advantage in average accuracy compared to all the other three methods. In detail, other methods struggle to optimize the quantized model. The NIPQ can only restore the model performance on the WSC subdataset, and fail to the average accuracy. The PACT and LLM-QAT methods yield poor results on some subdataset. For instance, PACT exhibits bad results on the RTE and MultiRC subdatasets, while LLM-QAT experiences significant performance losses on the WSC subdataset. Therefore, the effectiveness of our proposed Squant framework on the downstreaming tasks is verified by the clear accuracy advantages.

# VIII. HARDWARE EFFICIENCY

Our MKMP multiplier is compatible with mainstream processors on edge platforms, such as mobile phones and Raspberry Pi IoT processors, which typically face challenges when processing low-bit data due to their SIMD instructions supporting only 8-bit or larger data granularity. We deliver the latency profiling results with model size and accuracy in Table IV, and we can draw the following conclusions: 8-bit quantization provides more than 1.4× acceleration on smartphones and over 1.6× acceleration on Raspberry Pi. As the high-end CPUs on smartphones can afford more robust floating-point processing capabilities, the acceleration attained

Table II: GPT2-97M with W4A4 on BLiMP Main dataset.

| Method | FP16 | NIPQ | PACT | LLM-QAT | Ours |
|--------|------|------|------|---------|------|
| AA     | 87.0 | 38.1 | 69.8 | 84.3    | 84.5 |
| AS     | 71.3 | 57.4 | 63.7 | 70.5    | 71.7 |
| Bind.  | 70.2 | 49.8 | 64.4 | 69.7    | 69.8 |
| C/R    | 66.1 | 54.2 | 62.6 | 65.1    | 65.3 |
| D-NA   | 87.4 | 51.4 | 72.3 | 86.9    | 86.0 |
| Ell.   | 62.1 | 39.6 | 39.2 | 59.8    | 59.9 |
| F-G    | 70.7 | 43.3 | 63.2 | 70.5    | 70.4 |
| IF     | 94.1 | 52.3 | 90.0 | 94.3    | 95.4 |
| IE     | 47.2 | 59.7 | 44.9 | 46.5    | 46.8 |
| NPI-L  | 48.5 | 71.3 | 44.4 | 47.5    | 44.8 |
| Quan.  | 68.0 | 27.5 | 46.7 | 69.5    | 69.4 |
| S-VA.  | 66.2 | 48.1 | 55.5 | 65.1    | 66.0 |
| Avg.   | 69.9 | 49.4 | 59.7 | 69.1    | 69.2 |

Table III: LLaMA-58M with W4A4 on (Super)GLUE.

| Method | FP16 | NIPQ | PACT | LLM-QAT | Ours |
|--------|------|------|------|---------|------|
| CoLA   | 69.5 | 33.3 | 69.3 | 68.5    | 68.4 |
| SST-2  | 87.2 | 49.4 | 85.4 | 85.0    | 84.1 |
| MRPC   | 63.2 | 32.2 | 69.4 | 69.3    | 69.5 |
| QQP    | 84.3 | 42.4 | 82.5 | 83.7    | 84.1 |
| MNLI   | 72.9 | 35.4 | 67.5 | 70.8    | 70.8 |
| MNLIm  | 73.7 | 35.8 | 69.1 | 71.5    | 71.1 |
| QNLI   | 81.1 | 47.2 | 74.4 | 78.2    | 79.4 |
| RTE    | 61.6 | 50.5 | 48.5 | 54.6    | 53.5 |
| BoolQ  | 67.2 | 58.4 | 60.3 | 62.4    | 62.9 |
| MulRC  | 58.9 | 53.2 | 46.1 | 53.7    | 54.1 |
| WSC    | 61.4 | 61.4 | 53.0 | 52.9    | 56.6 |
| Avg.   | 71.0 | 45.4 | 65.9 | 68.2    | 68.6 |

through quantization on smartphones is not as significant as the improvements observed on the Raspberry Pi 5. Meanwhile, for the W4A4 configuration, we achieve more than 2.2× acceleration on smartphones and 2.3× acceleration on Raspberry Pi, separately. Overall, the GPT2-97M model achieves greater acceleration in our framework compared to the LLaMA-58M model. This is largely due to its higher parameter amount, which enables more efficiency improvement through memory access reduction on edge devices. Additionally, the 4-bit compression and concatenation technique amplifies this advantage, delivering a 2.26× acceleration compared to the 1.43× speedup achieved with 8-bit quantization on smartphones for GPT2-97M model.

Also, the introduction of mixed precision is essential as it bridges the gap between the latency of 4-bit and 8-bit configurations. While the theoretical, computational workload is halved, some overhead is introduced due to internal shifts of concatenated weights and the recovery of stored results in INT8 format. However, using W4A4 precision can lead to a noticeable performance drop in LLM tasks. To address this, we employ our MKMP multiplier for the mixed W4A4 and W4A8 configurations. This approach not only achieves further acceleration compared to 8-bit quantization but also maintains the model performance as shown in Figure 8. For Raspberry Pi, the additional acceleration achieved through the mixed configuration becomes over 40%.

## IX. ABLATION STUDY

#### *A. Loss Ablation*

As shown in Figure 7, we adopt ablation study for proposed entropy loss L<sup>E</sup> and distribution loss LD. The results in blue denote the LLaMA-58M and the results in red denote GPT2- 97M. The results are obtained with the W4A4 configuration. We can identify that, compared to entropy loss, distribution loss more effectively improves the model performance. Besides, we make the observation that combining the two losses generates better results than using either single one. Both loss types are verified to be effective when used for both LLaMA-58M and GPT2-97M, which validates the generalization of the proposed loss optimization method.

#### *B. Mixed Strategy*

Ablation for quantization with mixed or uniform strategy is included in Figure 8. The results in blue denote the quantization with mixed strategy while the results in grey denote the quantization with uniform strategy. Results are evaluated with LLaMA-58M model on BLiMP Main dataset using a Raspberry Pi 5. Results show that mixed strategy yields superior quantization performance (higher accuracy and lower latency in ms/Token) compared to uniform strategy. The inference acceleration using a mixed strategy verifies superior performance compared to uniform quantization at any bit level (5, 6, 7, or 8 bits). Specifically, for 6-bit activation quantization strategy, the mixed-precision strategy with half 4-bit and half 8-bit quantization shows impressive better accuracy than the uniform strategy.

#### X. CONCLUSION AND LIMITATION

In this paper, we introduce the Squant method, an entropy-guided and distribution-aligned token adaptive mixedprecision QAT framework, designed to accelerate small language models on mobile devices. Besides, we adaptively quantize tokens with different bit widths based on their importance, which further accelerates the inference and maintains performance. Meanwhile, we implement the corresponding multiplier, which helps the mobile devices benefit from our proposed 4-bit quantization optimization algorithm. We effectively restore the model performance to that of FP16 counterparts and achieve up to 2.37× speedup on mobile devices. We will verify our method on larger models with hundreds of millions of parameters in our further work.

## REFERENCES

[1] L. B. Allal, A. Lozhkov, E. Bakouch, G. M. Blazquez, G. Penedo, ´ L. Tunstall, A. Marafioti, H. Kydl´ıcek, A. P. Lajar ˇ ´ın, V. Srivastav, et al. Smollm2: When smol goes big–data-centric training of a small language model. *arXiv preprint arXiv:2502.02737*, 2025.

![](_page_7_Figure_10.jpeg)

Figure 7: Loss ablation with average accuracy of LLaMA-58M and GPT2-97M on BLiMP Main dataset in W4A4.

![](_page_7_Figure_12.jpeg)

Figure 8: Mixed and uniform quantization results of LLaMA-58M on the BLiMP Main dataset with Rapberry Pi 5.

- [2] S. Ashfaq, M. AskariHemmat, S. Sah, E. Saboori, O. Mastropietro, and A. Hoffman. Accelerating deep learning model inference on arm cpus with ultra-low bit quantization and runtime. *arXiv preprint arXiv:2207.08820*, 2022.
- [3] Y. Bengio, N. Leonard, and A. Courville. Estimating or propagating ´ gradients through stochastic neurons for conditional computation. *arXiv preprint arXiv:1308.3432*, 2013.
- [4] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. *NeurIPS*, 33:1877–1901, 2020.
- [5] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei. Language models are few-shot learners. 2020.
- [6] M. Chen, W. Shao, P. Xu, J. Wang, P. Gao, K. Zhang, and P. Luo. Efficientqat: Efficient quantization-aware training for large language models. *arXiv preprint arXiv:2407.11062*, 2024.

Table IV: Latency results (ms/Token) of LLaMA-58M and GPT2-97M with 128 input sequence length on mobile (Onepluss 11) and edge (Raspberry Pi 5) devices.

| W                    | FP16  | INT8          | INT4          | INT4          | INT4          | INT4          | INT4         |  |  |
|----------------------|-------|---------------|---------------|---------------|---------------|---------------|--------------|--|--|
| A                    | FP16  | INT8          | INT8          | 4:8 (1:3)     | 4:8 (1:1)     | 4:8 (3:1)     | INT4         |  |  |
| LLaMA-58M (ms/Token) |       |               |               |               |               |               |              |  |  |
| MB                   | 110.6 | 55.3          | 27.7          | 27.7          | 27.7          | 27.7          | 27.7         |  |  |
| Mobile               | 4.54  | 3.22 (1.41×)  | 2.56 (1.77×)  | 2.39 (1.90×)  | 2.23 (2.04×)  | 2.10 (2.16×)  | 2.02 (2.24×) |  |  |
| Raspberry Pi         | 15.63 | 9.40 (1.66×)  | 7.50 (2.08×)  | 7.30 (2.14×)  | 7.08 (2.21×)  | 6.89 (2.27×)  | 6.78 (2.31×) |  |  |
| GPT2-97M (ms/Token)  |       |               |               |               |               |               |              |  |  |
| MB                   | 185.5 | 92.7          | 46.3          | 46.3          | 46.3          | 46.3          | 46.3         |  |  |
| Mobile               | 6.22  | 4.35 (1.43×)  | 3.42 (1.82×)  | 3.06 (2.06×)  | 3.02 (2.03×)  | 2.86 (2.17×)  | 2.75 (2.26×) |  |  |
| Raspberry Pi         | 23.04 | 13.75 (1.68×) | 12.45 (1.85×) | 11.24 (2.05×) | 10.98 (2.10×) | 10.01 (2.30×) | 9.74 (2.37×) |  |  |

- [7] W. Chen and Z. Li. Octopus v2: On-device language model for super agent, 2024.
- [8] J. Choi, Z. Wang, S. Venkataramani, P. I.-J. Chuang, V. Srinivasan, and K. Gopalakrishnan. Pact: Parameterized clipping activation for quantized neural networks. *arXiv preprint arXiv:1805.06085*, 2018.
- [9] P. Dong, M. Sun, A. Lu, Y. Xie, K. Liu, Z. Kong, X. Meng, Z. Li, X. Lin, Z. Fang, et al. Heatvit: Hardware-efficient adaptive token pruning for vision transformers. In *2023 IEEE International Symposium on High-Performance Computer Architecture (HPCA)*, pages 442–455. IEEE, 2023.
- [10] M. Dukhan, Y. Wu, and H. Lu. Qnnpack: Open source library for optimized mobile deep learning, 2018.
- [11] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh. Gptq: Accurate post-training quantization for generative pre-trained transformers. *arXiv preprint arXiv:2210.17323*, 2022.
- [12] G. Hinton, O. Vinyals, and J. Dean. Distilling the knowledge in a neural network. *arXiv preprint arXiv:1503.02531*, 2015.
- [13] S. Hu, Y. Tu, X. Han, C. He, G. Cui, X. Long, Z. Zheng, Y. Fang, Y. Huang, W. Zhao, X. Zhang, Z. L. Thai, K. Zhang, C. Wang, Y. Yao, C. Zhao, J. Zhou, J. Cai, Z. Zhai, N. Ding, C. Jia, G. Zeng, D. Li, Z. Liu, and M. Sun. Minicpm: Unveiling the potential of small language models with scalable training strategies, 2024.
- [14] B. Jacob and P. Warden. gemmlowp: A small self-contained lowprecision gemm library. *Retrieved June*, 14:2018, 2017.
- [15] M. Kim, S. Gao, Y.-C. Hsu, Y. Shen, and H. Jin. Token fusion: Bridging the gap between token pruning and token merging. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, pages 1383–1392, 2024.
- [16] M. Kim, S. Lee, J. Lee, S. Hong, D.-S. Chang, W. Sung, and J. Choi. Token-scaled logit distillation for ternary weight generative language models. *Advances in Neural Information Processing Systems*, 36, 2024.
- [17] S. Kim, C. Hooper, A. Gholami, Z. Dong, X. Li, S. Shen, M. W. Mahoney, and K. Keutzer. Squeezellm: Dense-and-sparse quantization. *arXiv preprint arXiv:2306.07629*, 2023.
- [18] Z. Kong, P. Dong, X. Ma, X. Meng, W. Niu, M. Sun, X. Shen, G. Yuan, B. Ren, H. Tang, et al. Spvit: Enabling faster vision transformers via latency-aware soft token pruning. In *European Conference on Computer Vision*, pages 620–640. Springer, 2022.
- [19] J. Lin, J. Tang, H. Tang, S. Yang, X. Dang, and S. Han. Awq: Activationaware weight quantization for llm compression and acceleration. *arXiv preprint arXiv:2306.00978*, 2023.
- [20] Z. Liu, B. Oguz, C. Zhao, E. Chang, P. Stock, Y. Mehdad, Y. Shi, R. Krishnamoorthi, and V. Chandra. Llm-qat: Data-free quantization aware training for large language models. *arXiv preprint arXiv:2305.17888*, 2023.
- [21] Z. Liu, C. Zhao, F. Iandola, C. Lai, Y. Tian, I. Fedorov, Y. Xiong, E. Chang, Y. Shi, R. Krishnamoorthi, et al. Mobilellm: Optimizing sub-billion parameter language models for on-device use cases. *arXiv preprint arXiv:2402.14905*, 2024.
- [22] D. Messerschmitt. Quantizing for maximum output entropy (corresp.). *IEEE Transactions on Information Theory*, 17(5):612–612, 1971.
- [23] S. Park, J. So, J. Shin, and E. Park. Nipq: Noise injection

- pseudo quantization for automated dnn optimization. *arXiv preprint arXiv:2206.00820*, 2022.
- [24] H. Qin, Y. Ding, M. Zhang, Q. Yan, A. Liu, Q. Dang, Z. Liu, and X. Liu. Bibert: Accurate fully binarized bert. *The International Conference on Learning Representations (ICLR)*, 2022.
- [25] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever. Language models are unsupervised multitask learners. 2019.
- [26] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, et al. Language models are unsupervised multitask learners. *OpenAI blog*, 1(8):9, 2019.
- [27] X. Shen, P. Dong, L. Lu, Z. Kong, Z. Li, M. Lin, C. Wu, and Y. Wang. Agile-quant: Activation-guided quantization for faster inference of llms on the edge. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 38, pages 18944–18951, 2024.
- [28] X. Shen, C. Han, Y. Zhou, Y. Xie, Y. Gong, Q. Wang, Y. Wang, Y. Wang, P. Zhao, and J. Gu. Draftattention: Fast video diffusion via lowresolution attention guidance. *arXiv preprint arXiv:2505.14708*, 2025.
- [29] X. Shen, Z. Han, L. Lu, Z. Kong, P. Dong, Z. Li, Y. Xie, C. Wu, M. Leeser, P. Zhao, X. Lin, and Y. Wang. Hotaq: Hardware oriented token adaptive quantization for large language models. *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems*, pages 1–1, 2024.
- [30] X. Shen, W. Ma, J. Liu, C. Yang, R. Ding, Q. Wang, H. Ding, W. Niu, Y. Wang, P. Zhao, J. Lin, and J. Gu. Quartdepth: Post-training quantization for real-time depth estimation on the edge. In *Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)*, pages 11448–11460, June 2025.
- [31] X. Shen, W. Ma, Y. Zhou, E. Tang, Y. Xie, Z. Li, Y. Gong, Q. Wang, H. Ding, Y. Wang, Y. Wang, P. Zhao, J. Lin, and J. Gu. Fastcar: Cache attentive replay for fast auto-regressive video generation on the edge. *arXiv preprint arXiv:2505.14709*, 2025.
- [32] X. Shen, Z. Song, Y. Zhou, B. Chen, Y. Li, Y. Gong, K. Zhang, H. Tan, J. Kuen, H. Ding, Z. Shu, W. Niu, P. Zhao, Y. Wang, and J. Gu. Lazydit: Lazy learning for the acceleration of diffusion transformers. *Proceedings of the AAAI Conference on Artificial Intelligence*, 39(19):20409–20417, Apr. 2025.
- [33] X. Shen, Z. Song, Y. Zhou, B. Chen, J. Liu, R. Zhang, R. A. Rossi, H. Tan, T. Yu, X. Chen, Y. Zhou, T. Sun, P. Zhao, Y. Wang, and J. Gu. Numerical pruning for efficient autoregressive models. *Proceedings of the AAAI Conference on Artificial Intelligence*, 39(19):20418–20426, Apr. 2025.
- [34] X. Shen, P. Zhao, Y. Gong, Z. Kong, Z. Zhan, Y. Wu, M. Lin, C. Wu, X. Lin, and Y. Wang. Search for efficient large language models. In *NeurIPS*, 2024.
- [35] X. Shen, H. Zheng, Y. Gong, Z. Kong, C. Yang, Z. Zhan, Y. Wu, X. Lin, Y. Wang, P. Zhao, and W. Niu. Sparse learning for state space models on mobile. In *The Thirteenth International Conference on Learning Representations*, 2025.
- [36] C. Tao, L. Hou, W. Zhang, L. Shang, X. Jiang, Q. Liu, P. Luo, and N. Wong. Compression of generative pre-trained language models via quantization. *arXiv preprint arXiv:2203.10705*, 2022.
- [37] J.-L. Tastet and I. Timiryasov. Babyllama-2: Ensemble-distilled models

- consistently outperform teachers with limited data. *arXiv preprint arXiv:2409.17312*, 2024.
- [38] I. Timiryasov and J.-L. Tastet. Baby llama: knowledge distillation from an ensemble of teachers trained on a small dataset with no performance penalty. *arXiv preprint arXiv:2308.02019*, 2023.
- [39] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Roziere, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, ` A. Joulin, E. Grave, and G. Lample. Llama: Open and efficient foundation language models. *arXiv*, 2023.
- [40] C. Van Nguyen, X. Shen, R. Aponte, Y. Xia, S. Basu, Z. Hu, J. Chen, M. Parmar, S. Kunapuli, J. Barrow, et al. A survey of small language models. *arXiv preprint arXiv:2410.20011*, 2024.
- [41] A. Wang, Y. Pruksachatkun, N. Nangia, A. Singh, J. Michael, F. Hill, O. Levy, and S. Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. *Advances in neural information processing systems*, 32, 2019.
- [42] H. Wang, Z. Zhang, and S. Han. Spatten: Efficient sparse attention architecture with cascade token and head pruning. In *2021 IEEE International Symposium on High-Performance Computer Architecture (HPCA)*, pages 97–110. IEEE, 2021.
- [43] A. Warstadt, A. Parrish, H. Liu, A. Mohananey, W. Peng, S.-F. Wang, and S. R. Bowman. BLiMP: The Benchmark of Linguistic Minimal Pairs for English. *Transactions of the Association for Computational Linguistics*, 8:377–392, 07 2020.
- [44] X. Wu, Z. Yao, and Y. He. Zeroquant-fp: A leap forward in llms posttraining w4a8 quantization using floating-point formats. *arXiv preprint arXiv:2307.09782*, 2023.
- [45] G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han. Smoothquant: Accurate and efficient post-training quantization for large language models. In *International Conference on Machine Learning*, pages 38087–38099. PMLR, 2023.
- [46] Y. Yang, E. Sulem, I. Lee, and D. Roth. Penn & BGU BabyBERTa+ for Strict-Small BabyLM Challenge. Technical report, 2023.
- [47] Z. Yao, R. Yazdani Aminabadi, M. Zhang, X. Wu, C. Li, and Y. He. Zeroquant: Efficient and affordable post-training quantization for largescale transformers. *Advances in Neural Information Processing Systems*, 35:27168–27183, 2022.
- [48] S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. Diab, X. Li, X. V. Lin, et al. Opt: Open pre-trained transformer language models. *arXiv*, 2022.
- [49] P. Zhao, X. Shen, Z. Kong, Y. Shen, S.-E. Chang, T. Rupprecht, L. Lu, E. Nan, C. Yang, Y. He, et al. Fully open source moxin-7b technical report. *arXiv preprint arXiv:2412.06845*, 2024.