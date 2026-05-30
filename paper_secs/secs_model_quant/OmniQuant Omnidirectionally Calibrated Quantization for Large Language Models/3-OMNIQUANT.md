# 3 OMNIQUANT

Challenge of LLM quantization. Two main difficulties lie in quantizing an LLM. First, the activation is hard to quantize due to the existence of outlier channels. Considering that weight distribution is flat and uniform, SmoothQuant [\(Xiao et al.,](#page-11-6) [2023\)](#page-11-6) and Outlier Suppression+ [\(Wei et al.,](#page-11-9) [2023\)](#page-11-9) tackle this issue by migrating the quantization difficulty from activations to weights with a pre-defined migration strength or grid-searching based optimization. Second, the quantization error of weights also plays a pivotal role in the final performance due to the importance of weights corresponding to activations. SqQR [\(Dettmers et al.,](#page-9-3) [2023b\)](#page-9-3) and OWQ [\(Lee et al.,](#page-10-6) [2023\)](#page-10-6) propose to retain crucial weights in full-precision, while AWQ [\(Lin et al.,](#page-10-4) [2023\)](#page-10-4) safeguards these weights using grid-searched channel-wise scaling. Although these methods have achieved certain success in compressing various LLMs, they often lead to suboptimal performance and fail to deal with extremely low-bit quantization due to the crude design of hand-crafted quantization parameters such as migration strength and scaling factors.

In this section, we introduce a differentiable quantization technique for LLM called OmniQuant where quantization parameters are learned with better flexibility. Towards this goal, OmniQuant is implemented with a block-wise quantization error minimization framework as presented in Sec[.3.1.](#page-3-0) To tackle the aforementioned challenges of LLM quantization, we devise two novel strategies for additional learnable quantization parameters including a learnable weight clipping (LWC) to mitigate the difficulty in quantizing weights and a learnable equivalent transformation (LET) to further shift the challenge of quantization from activations to weights. We introduce LWC and LCT in Sec. [3.2](#page-4-0) and Sec. [3.3,](#page-4-1) respectively.

## <span id="page-3-0"></span>3.1 BLOCK-WISE QUANTIZATION ERROR MINIMIZATION

Previous PTQ methods with gradient optimization, such as AdaRound [\(Nagel et al.,](#page-11-10) [2020\)](#page-11-10), BRECQ [\(Li et al.,](#page-10-8) [2021\)](#page-10-8) cannot be applied in models with billions of parameters because they are hard to optimize due to the huge solution space. Instead of turning the whole model, we propose a new optimization pipeline with block-wise quantization error minimization where the additional quantization parameters can be optimized in a differentiable manner. We formulate the optimization goal as follows:

<span id="page-4-2"></span>
$$\arg\min_{\Theta_1,\Theta_2} ||\mathcal{F}(\mathbf{W}, \mathbf{X}) - \mathcal{F}(Q_w(\mathbf{W}; \Theta_1, \Theta_2), Q_a(\mathbf{X}, \Theta_2))||, \tag{1}$$

where F represents the mapping function for a transformer block in the LLM, W and X are fullprecision weight and activation, Qw(·) and Qa(·) represent weight and activation quantizer, respectively, Θ<sup>1</sup> and Θ<sup>2</sup> are quantization parameters in learnable weight clipping (LWC) and learnable equivalent transformation (LET), respectively. The Block-wise quantization in Eqn.[\(1\)](#page-4-2) sequentially quantizes the parameters of one transformer block before moving on to the next.

Block-wise minimization in Eqn.[\(1\)](#page-4-2) has two advantages. First, equipped with block-wise minimization in Eqn.[\(1\)](#page-4-2), OmniQuant can optimize quantization parameters in LWC and LET jointly, making it capable enough to encompass both weight-only and weight-activation quantization. Second, block-wise minimization is easy to optimize with minimal resource requirements. OmniQuant only determines a few quantization parameters with optimality, which is easier than optimizing the whole weights in previous PTQ-based methods [\(Nagel et al.,](#page-11-10) [2020;](#page-11-10) [Li et al.,](#page-10-8) [2021\)](#page-10-8). Empirically, we find that all models from the LLaMA-2 family [\(Touvron et al.,](#page-11-11) [2023b\)](#page-11-11) can be quantized on a single A100-40G GPU utilizing only 128 training samples.

## <span id="page-4-0"></span>3.2 LEARNABLE WEIGHT CLIPPING

OmniQuant employs a module of learnable weight clipping (LWC) to reduce the difficulty of quantizing the weights in an LLM. Similar to previous methods with learnable clipping threshold [\(Esser](#page-9-7) [et al.,](#page-9-7) [2019;](#page-9-7) [Liu et al.,](#page-10-9) [2022;](#page-10-9) [Choi et al.,](#page-9-8) [2018\)](#page-9-8), LWC also determines the optimal dynamic range of the weights by optimizing a clipping threshold. However, we find that directly employing prior arts such as PACT [\(Choi et al.,](#page-9-8) [2018\)](#page-9-8) and LSQ [\(Esser et al.,](#page-9-7) [2019\)](#page-9-7) in quantization would produce unsatisfactory performance, as demonstrated in Table [A14](#page-19-0) in the Appendix.

Instead of directly learning a clipping threshold as in previous methods [\(Esser et al.,](#page-9-7) [2019;](#page-9-7) [Choi](#page-9-8) [et al.,](#page-9-8) [2018\)](#page-9-8), LWC optimizes a clipping strength as formulated by

<span id="page-4-3"></span>
$$\mathbf{W_q} = \operatorname{clamp}(\lfloor \frac{\mathbf{W}}{h} \rceil + z, 0, 2^N - 1), \text{ where } h = \frac{\gamma \max(\mathbf{W}) - \beta \min(\mathbf{W})}{2^N - 1}, z = -\lfloor \frac{\beta \min(\mathbf{W})}{h} \rceil$$
(2)

where ⌊·⌉ indicates round operation. N is the target bit number. W<sup>q</sup> and W denote the quantized and full-precision weights, respectively. h is the normalization factor for weights and z is the zeropoint value. The clamp operation constrains the value within the range of N-bit integer, specifically [0, 2 <sup>N</sup> − 1]. In Eqn.[\(2\)](#page-4-3), γ ∈ [0, 1] and β ∈ [0, 1] are learnable clipping strengths for the upper and the lower bound of weights, respectively. We instantiate γ and β by the sigmoid function[\\*](#page-0-0). Hence, Θ<sup>1</sup> = {γ, β} in Eqn.[\(1\)](#page-4-2).

Note that LWC degrades into a vanilla MinMax quantization scheme used in existing works [\(Xiao](#page-11-6) [et al.,](#page-11-6) [2023\)](#page-11-6)[,Frantar et al.](#page-9-2) [\(2022\)](#page-9-2) when γ = 1 and β = 1. By inheriting the benefits of Min-Max quantization, LWC only needs to adjust the clipping strengths to determine an optimal clipping threshold, which would reduce the optimization difficulty. Clipped by an optimal threshold, the original weights would be easy to quantize. As indicated by the experiments in Table [1,](#page-6-0) our proposed learnable weight clipping method significantly outperforms previous weight-only quantization techniques [\(Frantar et al.,](#page-9-2) [2022;](#page-9-2) [Lin et al.,](#page-10-4) [2023\)](#page-10-4)).

## <span id="page-4-1"></span>3.3 LEARNABLE EQUIVALENT TRANSFORMATION

Other than LWC which enables quantization-friendly weights by optimizing the clipping threshold, we further reduce the difficulty of weight-activation quantization by a learnable equivalent transformation (LET). Considering that outliers in the activation map are systematic and unique to specific channels, previous methods such as SmoothQuant [\(Xiao et al.,](#page-11-6) [2023\)](#page-11-6) migrate the difficulty of quantization from activations to weights with a mathematically equivalent transformation. However, they hand-craft the equivalent parameters, leading to suboptimal results.

<sup>\*</sup>Sigmoid(t) = 1/(1 + exp<sup>−</sup><sup>t</sup> )

Thanks to the inclusion of block-wise quantization error minimization, our LET can determine the optimal equivalent parameters in a differentiable way. Inspired by SmoothQuant (Xiao et al., 2023) and Outlier Suppression+ (Wei et al., 2023), we adopt channel-wise scaling and channel-wise shifting to manipulate the activation distribution, providing an effective solution for the outlier issue. Specifically, we investigate the equivalent transformation across both the linear layer and attention operation, as illustrated in Figure 3.

**Linear layer.** The linear layer takes an input token sequence  $\mathbf{X} \in \mathbb{R}^{T \times C_{in}}$  where T is the token length and is the multiplication of the weight matrix  $\mathbf{W} \in \mathbb{R}^{C_{in} \times C_{out}}$  and bias vector  $\mathbf{B} \in \mathbb{R}^{1 \times C_{out}}$ . A mathematically equivalent linear layer is expressed as:

<span id="page-5-0"></span>
$$\mathbf{Y} = \mathbf{X}\mathbf{W} + \mathbf{B} = \underbrace{\left[ (\mathbf{X} - \delta) \oslash s \right]}_{\tilde{\mathbf{X}}} \cdot \underbrace{\left[ s \odot \mathbf{W} \right]}_{\tilde{\mathbf{W}}} + \underbrace{\left[ \mathbf{B} + \delta \mathbf{W} \right]}_{\tilde{\mathbf{B}}}$$
(3)

where  $\mathbf{Y}$  represents the output,  $\mathbf{s} \in \mathbb{R}^{1 \times C_{in}}$  and  $\delta \in \mathbb{R}^{1 \times C_{in}}$  are channel-wise scaling and shifting parameters, respectively,  $\tilde{\mathbf{X}}$ ,  $\tilde{\mathbf{W}}$  and  $\tilde{\mathbf{B}}$  are equivalent activation, weight and bias, respectively, ' $\bigcirc$ ' and ' $\bigcirc$ ' are elementwise division and multiplication. By Eqn.(3), the activations are transformed to be quantization-friendly at a cost of increased quantization difficulty in weights. In this sense, LWC in Sec. 3.2 can improve the performance of weight-activation quantization achieved by LET because it renders weights quantization-friendly. Finally, we perform quantization on transformed activations and weights, as given by

<span id="page-5-1"></span>
$$\mathbf{Y} = Q_a(\tilde{\mathbf{X}})Q_w(\tilde{\mathbf{W}}) + \tilde{\mathbf{B}},\tag{4}$$

where  $Q_a$  is the vanilla MinMax quantizer and  $Q_w$  is the MinMax quantizer with learnable weight clipping (i.e. our LWC).

Note that the scaling and shifting parameters in  $\hat{\mathbf{X}}$  can be absorbed into the previous normalization or linear layer and the the scaling factors in  $\hat{\mathbf{W}}$  can be fused into the original linear weight  $\mathbf{W}$ . Therefore, the equivalent transformation in Eqn.(3) can effectively reduce quantization errors without introducing additional parameters or costs. We employ this equivalent transformation in all linear layers of the LLM except for the second linear layer of FFN as shown in Figure 3. This may be because the high sparsity of features after the non-linear layer (Liu et al., 2023c) leads to unstable gradients when applying learnable equivalent transformations.

**Attention operation.** Beyond the linear layer, the attention operation also accounts for a significant proportion of the computation. Additionally, the auto-regressive pattern of LLM necessitates storing the key-value(KV) cache for each token, which results in substantial memory demands for long sequences. Therefore, we also quantize  $\mathbf{Q}/\mathbf{K}/\mathbf{V}$  matrixes into low-bit in the weight-activation quantization setting. Specifically, the learnable equivalent transform of the self-attention affinity matrix can be written as:

<span id="page-5-2"></span>
$$\mathbf{P} = \operatorname{Softmax}(\mathbf{Q}\mathbf{K}^T) = \operatorname{Softmax}((\underbrace{\mathbf{Q} \otimes s_a}_{\hat{\mathbf{Q}}})(\underbrace{s_a \odot \mathbf{K}^T}_{\hat{\mathbf{K}}^T})). \tag{5}$$

where  $s_a \in \mathbb{R}^{1 \times C_{out}}$  is the scaling factor in the affinity matrix. Similar to Eqn.(4), the quantized affinity matrix calculation is expressed as  $\mathbf{P} = \operatorname{Softmax}(Q_a(\widetilde{\mathbf{Q}})Q_a(\widetilde{\mathbf{K}}^T))$ . Here we also use Min-Max quantization scheme as  $Q_a$  to quantize  $\widetilde{\mathbf{Q}}/\widetilde{\mathbf{K}}$  matrixes. From Eqn.(4) and Eqn.(5) we know that  $\Theta_2 = \{\delta, s, s_a\}$  in Eqn.(1).

The channel-wise scaling factors in  $\tilde{\mathbf{Q}}$  and  $\tilde{\mathbf{K}}$ , as seen in Eq.(5), can be absorbed into linear weights of the query and key projection, respectively. It is worth mentioning that the explicit transformation of  $\mathbf{V}$  is omitted as its distribution has already been channel-wise altered by the inverse transformation associated with the output projection linear layer.

#### 4 EXPERIMENTS

#### 4.1 SETTINGS

**Quantization.** We experiment with both weight-only and weight-activation quantization. For the former, default settings are INT4/INT3/INT2 per-channel weight quantization. Group-wise weight

<span id="page-6-0"></span>

| Table 1: Weight-only quantization Results of LLaMA-1 and LLaMA-2 Models.                | We report |
|-----------------------------------------------------------------------------------------|-----------|
| WikiText2 perplexity in this table, C4 perplexity can be found in Table A19 in Appendix |           |

|                                         | erpiexity in this |             |        |        |       |        |        |             |
|-----------------------------------------|-------------------|-------------|--------|--------|-------|--------|--------|-------------|
| LLaMA1                                  | &2 / PPL↓         | 1-7B        | 1-13B  | 1-30B  | 1-65B | 2-7B   | 2-13B  | 2-70B       |
| FP16                                    | -                 | 5.68        | 5.09   | 4.10   | 3.53  | 5.47   | 4.88   | 3.31        |
|                                         | RTN               | 1.1e5       | 6.8e4  | 2.4e4  | 2.2e4 | 3.8e4  | 5.6e4  | 2.0e4       |
| W2A16                                   | GPTQ              | 2.1e3       | 5.5e3  | 499.75 | 55.91 | 7.7e3  | 2.1e3  | 77.95       |
|                                         | OmniQuant         | 15.47       | 13.21  | 8.71   | 7.58  | 37.37  | 17.21  | <b>7.81</b> |
| WO 4 1 6                                | RTN               | 1.9e3       | 781.20 | 68.04  | 15.08 | 4.2e3  | 122.08 | 27.27       |
| W2A16                                   | GPTQ              | 44.01       | 15.60  | 10.92  | 9.51  | 36.77  | 28.14  | NAN         |
| g128                                    | AWQ               | 2.6e5       | 2.8e5  | 2.4e5  | 7.4e4 | 2.2e5  | 1.2e5  | -           |
|                                         | OmniQuant         | 9.72        | 7.93   | 7.12   | 5.95  | 11.06  | 8.26   | 6.55        |
| WO 4 1 6                                | RTN               | 188.32      | 101.87 | 19.20  | 9.39  | 431.97 | 26.22  | 10.31       |
| W2A16                                   | GPTQ              | 22.10       | 10.06  | 8.54   | 8.31  | 20.85  | 22.44  | NAN         |
| g64                                     | AWQ               | 2.5e5       | 2.7e5  | 2.3e5  | 7.4e4 | 2.1e5  | 1.2e5  | -           |
|                                         | OmniQuant         | 8.90        | 7.34   | 6.59   | 5.65  | 9.62   | 7.56   | 6.11        |
|                                         | RTN               | 25.73       | 11.39  | 14.95  | 10.68 | 539.48 | 10.68  | 7.52        |
| W3A16                                   | GPTQ              | 8.06        | 6.76   | 5.84   | 5.06  | 8.37   | 6.44   | 4.82        |
| *************************************** | AWQ               | 11.88       | 7.45   | 10.07  | 5.21  | 24.00  | 10.45  | -           |
|                                         | OmniQuant         | 6.49        | 5.68   | 4.74   | 4.04  | 6.58   | 5.58   | 3.92        |
| XX/2 A 1.6                              | RTN               | 7.01        | 5.88   | 4.87   | 4.24  | 6.66   | 5.51   | 3.97        |
| W3A16                                   | GPTQ              | 6.55        | 5.62   | 4.80   | 4.17  | 6.29   | 5.42   | 3.85        |
| g128                                    | AWQ               | 6.46        | 5.51   | 4.63   | 3.99  | 6.24   | 5.32   | -           |
|                                         | OmniQuant         | 6.15        | 5.44   | 4.56   | 3.94  | 6.03   | 5.28   | 3.78        |
| ·                                       | RTN               | 6.43        | 5.55   | 4.57   | 3.87  | 6.11   | 5.20   | 3.67        |
| W4A16                                   | GPTQ              | 6.13        | 5.40   | 4.48   | 3.83  | 5.83   | 5.13   | 3.58        |
| ** 17110                                | AWQ               | 6.08        | 5.34   | 4.39   | 3.76  | 6.15   | 5.12   | -           |
|                                         | OmniQuant         | 5.86        | 5.21   | 4.25   | 3.71  | 5.74   | 5.02   | 3.47        |
| WAAIC                                   | RTN               | 5.96        | 5.25   | 4.23   | 3.67  | 5.72   | 4.98   | 3.46        |
| W4A16                                   | GPTQ              | 5.85        | 5.20   | 4.23   | 3.65  | 5.61   | 4.98   | 3.42        |
| g128                                    | AWQ               | 5.81        | 5.20   | 4.21   | 3.62  | 5.62   | 4.97   | -           |
|                                         | OmniQuant         | <b>5.77</b> | 5.17   | 4.19   | 3.62  | 5.58   | 4.95   | 3.40        |

quantization is represented by 'g', e.g., W3A16g128 means 3-bit weight-only quantization with a 128-group size. In weight-activation quantization, defaults are INT6/INT4 per-channel weight and per-token activation quantization (Dettmers et al., 2022). All intermediate activations are quantized into low-bit, excluding the SoftMax output, kept at full precision due to its long-tail distribution making it unsuitable for uniform quantization.

**Training** The channel-wise scaling factor is initialized with SmoothQuant (Xiao et al., 2023), and the channel-wise shifting factor is initialized using Outlier Suppression+ (Wei et al., 2023). To optimize the learnable parameters, we utilize the AdamW optimizer with zero weight decay. The learning rate for learnable weight clipping and equivalent transformation is set as 5e-3 and 1e-2, respectively. We employ a calibration dataset consisting of 128 randomly selected 2048-token segments from WikiText2 (Merity et al., 2016). The entire training process is facilitated on a single Nvidia A100 GPU, using a batch size of 1 over 20 epochs, except for W2A16 quantization that leverages 40 epochs. For weight-activation quantization, both learnable weight clipping and equivalent transformation are activated. For weight-only, both are used for OPT, but only the clipping is for LLaMA, as Table A3 shows negligible benefits from the equivalent transformation for LLaMA.

**Models.** We test on OPT(125M-66B)(Zhang et al., 2022)), LLaMA(7B-65B) (Touvron et al., 2023a), LLaMA-2(7B-70B) (Touvron et al., 2023b), Falcon-180B (Penedo et al., 2023), and instruction-tuned LLaMA-2-chat (Touvron et al., 2023b) for generalizability. While the main paper highlights the LLaMA results, comprehensive details for other models are available in Sec. A8 of the Appendix.

**Evaluation.** Following the previous work (Lin et al., 2023; Frantar et al., 2022), we evaluate quantized models by reporting the perplexity of language generation experiments, specifically on Wiki-Text2 (Merity et al., 2016), PTB (Marcus et al., 1994)), C4 (Raffel et al., 2020). Moreover, accuracy is evaluated in zero-shot tasks including PIQA (Bisk et al., 2020), ARC (Clark et al., 2018), BoolQ (Clark et al., 2019), and HellaSwag (Clark et al., 2018). We adhere to the GPTQ (Frantar et al., 2022) settings for language generation experiments, and implement the lm-eval-harness (Gao et al., 2021) for the execution of all zero-shot tasks.

**Baselines.** For weight-only quantization, we compare with vanilla round-to-nearest quantization (RTN), GPTQ (Frantar et al., 2022), and AWQ (Lin et al., 2023). For weight-activation quantization, we compare our method with SmoothQuant (Xiao et al., 2023), Outlier Supression + (Wei et al., 2023), RPTQ (Yuan et al., 2023), and the recent QAT method LLM-QAT (Liu et al., 2023b). Note

<span id="page-7-0"></span>Table 2: **Weight-activation quantization results of LLaMA Models.** This table reports the accuracy of 6 zero-shot tasks. Perplexity results can be found in Table A23 & A24 at Appendix.

| LLaMA / Acc  | #Bits | Method      | PIQA  | ARC-e | Arc-c | BoolQ | HellaSwag | Winogrande | Avg.  |
|--------------|-------|-------------|-------|-------|-------|-------|-----------|------------|-------|
|              | FP16  | -           | 77.47 | 52.48 | 41.46 | 73.08 | 73.00     | 67.07      | 64.09 |
|              | W6A6  | SmoothQuant | 76.75 | 51.64 | 39.88 | 71.75 | 71.67     | 65.03      | 62.81 |
| II -MA 1 7D  | W6A6  | OS+         | 76.82 | 51.35 | 41.13 | 72.08 | 71.42     | 65.98      | 61.13 |
| LLaMA-1-7B   | W6A6  | OmniQuant   | 77.09 | 51.89 | 40.87 | 72.53 | 71.61     | 65.03      | 63.17 |
|              | W4A4  | SmoothQuant | 49.80 | 30.40 | 25.80 | 49.10 | 27.40     | 48.00      | 38.41 |
|              | W4A4  | LLM-QAT     | 51.50 | 27.90 | 23.90 | 61.30 | 31.10     | 51.90      | 41.27 |
|              | W4A4  | LLM-QAT+SQ  | 55.90 | 35.50 | 26.40 | 62.40 | 47.80     | 50.60      | 46.43 |
|              | W4A4  | OS+         | 62.73 | 39.98 | 30.29 | 60.21 | 44.39     | 52.96      | 48.43 |
|              | W4A4  | OmniQuant   | 66.15 | 45.20 | 31.14 | 63.51 | 56.44     | 53.43      | 52.65 |
|              | FP16  | -           | 79.10 | 59.89 | 44.45 | 68.01 | 76.21     | 70.31      | 66.33 |
| TT 354 1 10D | W6A6  | SmoothQuant | 77.91 | 56.60 | 42.40 | 64.95 | 75.36     | 69.36      | 64.43 |
| LLaMA-1-13B  | W6A6  | OS+         | 78.29 | 56.90 | 43.09 | 66.98 | 75.09     | 69.22      | 64.92 |
|              | W6A6  | OmniQuant   | 78.40 | 57.28 | 42.91 | 67.00 | 75.82     | 68.27      | 64.95 |
|              | W4A4  | SmoothQuant | 61.04 | 39.18 | 30.80 | 61.80 | 52.29     | 51.06      | 49.36 |
|              | W4A4  | OS+         | 63.00 | 40.32 | 30.38 | 60.34 | 53.61     | 51.54      | 49.86 |
|              | W4A4  | OmniQuant   | 69.69 | 47.39 | 33.10 | 62.84 | 58.96     | 55.80      | 54.37 |
|              | FP16  | -           | 80.08 | 58.92 | 45.47 | 68.44 | 79.21     | 72.53      | 67.44 |
| II MA 1 20D  | W6A6  | SmoothQuant | 77.14 | 57.61 | 42.91 | 65.56 | 78.07     | 69.92      | 65.20 |
| LLaMA-1-30B  | W6A6  | OS+         | 80.14 | 58.92 | 45.05 | 68.02 | 77.96     | 71.98      | 67.01 |
|              | W6A6  | OmniQuant   | 79.81 | 58.79 | 45.22 | 68.38 | 78.95     | 72.21      | 67.23 |
|              | W4A4  | SmoothQuant | 58.65 | 35.53 | 27.73 | 60.42 | 35.56     | 48.06      | 44.83 |
|              | W4A4  | OS+         | 67.63 | 46.17 | 34.40 | 60.70 | 54.32     | 52.64      | 52.62 |
|              | W4A4  | OmniQuant   | 71.21 | 49.45 | 34.47 | 65.33 | 64.65     | 59.19      | 56.63 |
|              | FP16  | -           | 80.79 | 58.71 | 46.24 | 82.29 | 80.72     | 77.50      | 71.04 |
| II MA 1 (5D  | W6A6  | SmoothQuant | 80.25 | 57.92 | 45.50 | 80.22 | 80.18     | 74.76      | 69.80 |
| LLaMA-1-65B  | W6A6  | OS+         | 79.67 | 55.68 | 45.22 | 80.02 | 78.03     | 73.95      | 68.76 |
|              | W6A6  | OmniQuant   | 81.01 | 58.12 | 46.33 | 80.64 | 79.91     | 75.69      | 70.28 |
|              | W4A4  | SmoothQuant | 64.47 | 40.44 | 29.82 | 59.38 | 39.90     | 52.24      | 47.71 |
|              | W4A4  | OS+         | 68.06 | 43.98 | 35.32 | 62.75 | 50.73     | 54.30      | 52.52 |
|              | W4A4  | OmniQuant   | 71.81 | 48.02 | 35.92 | 73.27 | 66.81     | 59.51      | 59.22 |

that we reproduce SmoothQuant and Outlier Suppression+ with per-channel weight quantization and per-token activation quantization for fair comparisons.

## 4.2 Weight-only Quantization Results

The results of the LLaMA family can be found in Table 1, while the results for OPT are presented in the Sec. A8 of Appendix. As illustrated by the tables, OmniQuant consistently outperforms the prior LLM weight-only quantization method across various LLM families (OPT, LLaMA-1, LLaMA-2) and diverse quantization configurations, including W2A16, W2A16g128, W2A16g64, W3A16, W3A16g128, W4A16, and W4A16g128. These findings suggest OmniQuant's versatility, being adaptable to a multitude of quantization configurations. For instance, while AWQ (Lin et al., 2023) is particularly effective with group-wise quantization, OmniQuant demonstrates superior performance across both channel-wise and group-wise quantization. Furthermore, the performance benefits of OmniQuant become more pronounced as the quantization bit size decreases.

## 4.3 WEIGHT-ACTIVATION QUANTIZATION RESULTS

In weight-activation quantization, our main focus lies on W6A6 and W4A4 quantization. We exclude W8A8 quantization as SmoothQuant can nearly achieve lossless W8A8 quantized models when compared with full-precision counterparts. The results of the LLaMA family can be found in Table 2, while the results for OPT are presented in Table A25 of Appendix. Table 2 illustrates the zero-shot task accuracy of LLaMA weight-activation quantization. Notably, OmniQuant markedly enhances the average accuracy by  $+4.99\% \sim +11.80\%$  across various models at W4A4 quantization. Remarkably, in the LLaMA-7B, OmniQuant even surpasses the recent QAT method, LLM-QAT (Liu et al., 2023b), by an impressive margin of +6.22%. This improvement demonstrates the efficacy of incorporating additional learnable parameters, which proves to be more beneficial than the global weight tuning utilized by QAT.

<span id="page-8-1"></span>![](_page_8_Figure_1.jpeg)

Figure 4: Comparing W3A16g128 quantization among RTN, AWQ (Lin et al., 2023), and Omni-Quant under Vicuna-Bench (Chiang et al., 2023). Win rates are calculated without considering tie samples. A higher win rate indicates the better performance of the former of vs. pairs.

<span id="page-8-0"></span>Table 3: Deployment of weight-only quantization through MLC-LLM. We report the memory size of quantized weights (denoted as 'WM') and the running memory (denoted as 'RM') and speed in NVIDIA A100-80G.

| LLaMA     |      | 7B   |         |      | 13B   |         |       | 30B   |         |       | 65B   |         |
|-----------|------|------|---------|------|-------|---------|-------|-------|---------|-------|-------|---------|
|           | WM   | RM   | token/s | WM   | RM    | token/s | WM    | RM    | token/s | WM    | RM    | token/s |
|           |      |      |         |      |       |         |       |       | 23.9    |       |       |         |
| W4A16g128 | 3.8G | 5.7G | 134.2   | 7.0G | 10.0G | 91.3    | 16.7G | 21.7G | 43.6    | 33.0G | 41.0G | 24.3    |
| W3A16g128 | 3.2G | 5.1G | 83.4    | 5.8G | 8.7G  | 57.6    | 13.7G | 18.7G | 29.0    | 27.0G | 35.1G | 15.2    |
| W2A16g128 | 2.2G | 4.1G | 83.9    | 4.0G | 7.5G  | 92.6    | 9.2G  | 14.1G | 36.7    | 18.0G | 25.6G | 24.8    |

#### 4.4 QUANTIZATION OF INSTRUCTION-TUNED MODELS

To validate the generalization capability of our method, we test the quantization on LLaMA-2-chat (Touvron et al., 2023b), an instruction-tuned model for chatbots. Using the GPT-4 evaluation protocol (Chiang et al., 2023), performance is assessed on the Vicuna benchmark (Chiang et al., 2023) comprising 80 questions. To negate position bias (Zheng et al., 2023), each pair is compared in both sequences, totaling 160 trials per comparison. Figure 4 compares RTN, AWQ (Lin et al., 2023), and OmniQuant. In LLaMA-2-7b-chat, OmniQuant matches AWQ with a 50% win rate but surpasses RTN more (80.3% vs. 69.4%). In LLaMA-2-13b-chat, while AWQ lags behind RTN, OmniQuant consistently improves quantization model performance.

#### 4.5 ACCELERATION ON REAL DEVICE

MLC-LLM† provides a versatile deployment solution for diverse language models across various hardwares. It particularly excels in deploying quantized models on CUDA. One of OmniQuant's strengths lies in its ability to avoid extra operations for quantized models, allowing MLC-LLM to seamlessly run models created with OmniQuant. Table,3 shows memory requirements and inference speeds of the LLaMA family on an NVIDIA A100-80G. 'Weights Memory (WM)' represents quantized weight storage, and 'Running Memory (RM)' indicates the memory for inference, with the latter being higher due to certain retained activations. Inference speed is gauged by generating 512 tokens. It is evident that quantized models significantly reduce memory usage compared to 16-bit full-precision models. For instance, models with W4A16g128 and W2A16g128 quantization almost double the inference speed. However, MLC-LLM's support for INT3/INT2 is currently suboptimal, particularly for INT3. Enhancements to INT3/INT2 quantization speed are in our future roadmap. Additionally, we only explore the deployment of weight-only quantization in this study due to that W4A4 and W6A6 quantization methods lack out-of-the-box hardware support.

## 5 Conclusion

We present OmniQuant, a method advancing weight-only and weight-activation quantization to low-bit formats. OmniQuant's core principle is to retain original full-precision weights while adding learnable parameters. It uses learnable weight clipping and learnable equivalent transformation to optimize weight and activation for quantization. While incorporating gradient updates, OmniQuant maintains training efficiency comparable to existing PTQ methods. It outperforms current methods in language generation and zero-shot tasks and is suited for instruction-tuned LLMs. In addition, OmniQuant also ensures hardware compatibility as its added parameters can be absorbed.

<sup>†</sup>https://github.com/mlc-ai/mlc-llm

## ACKNOWLEDGMENTS

This paper is partially supported by the National Key R&D Program of China No.2022ZD0161000 and the General Research Fund of Hong Kong No.17200622. We thank Wentao Liu from SenseTime for his valuable insights and discussions regarding LLM deployment. We also acknowledge Siyuan Feng from Apache TVM for assisting in deploying our OmniQuant in the MLC LLM project.

