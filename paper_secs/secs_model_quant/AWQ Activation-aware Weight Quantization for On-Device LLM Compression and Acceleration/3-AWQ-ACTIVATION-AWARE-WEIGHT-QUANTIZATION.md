# 3 AWQ: ACTIVATION-AWARE WEIGHT QUANTIZATION

*Quantization* maps a floating-point number into lower-bit integers. It is an effective method to reduce the model size and inference costs of LLMs [\(Dettmers et al.,](#page-11-0) [2022;](#page-11-0) [Frantar et al.,](#page-11-0) [2022;](#page-11-0) [Yao et al.,](#page-13-0) [2022;](#page-13-0) [Xiao et al.,](#page-13-0) [2022\)](#page-13-0). In this section, we first propose a weight-only quantization method to improve accuracy *without training/regression* by protecting more "important" weights. And then develop a data-driven method to search for the optimal scaling that reduces quantization errors (Figure 2).

## 3.1 Improving LLM Quantization by Preserving 1% Salient Weights

We observe that the weights of LLMs are *not equally important*: there is a small fraction of *salient* weights that are much more important for LLMs' performance compared to others. Skipping the quantization of these salient weights can help bridge the performance degradation due

to the quantization loss *without* any training or regression (Figure 2(b)). To verify the idea, we benchmark the performance of quantized LLMs when skipping part of the weight channels in Table [1.](#page-3-0) We measured the performance of INT3 quantized models while keeping some ratios of weight channels in FP16. A widely used method to determine the importance of weights is to look at its magnitude or L2-norm [\(Han et al.,](#page-12-0) [2015;](#page-12-0) [Frankle & Carbin,](#page-11-0) [2018\)](#page-11-0). But we find skipping the weight channels with large norm (*i.e*., FP16% (based on W)) does not significantly improve the quantized performance, leading to a similar marginal improvement as random selection. Interestingly, selecting weights based on *activation magnitude* can significantly improve the performance despite keeping only 0.1%-1% of channels in FP16. We hypothesize that the input features with larger magnitudes are generally more important. Keeping the corresponding weights in FP16 can preserve those features, which contributes to better model performance.

Limitations: Despite keeping 0.1% of weights in FP16 can improve the quantized performance without a noticeable increase in model size (measured in total bits), such a mixedprecision data type will make the system implementation difficult. We need to come up with a method to protect the important weights without actually keeping them as FP16.

### 3.2 Protecting Salient Weights by Activation-aware Scaling

We propose an alternative method to reduce the quantization error of the salient weight by *per-channel scaling*, which does not suffer from the hardware inefficiency issue.

#### Analyzing the quantization error.

We start by analyzing the error from weight-only quantization. Consider a group/block of weight w; the linear operation can be written as y = wx, and the quantized counterpart is y = Q(w)x. Specifically, the quantization

<sup>\*</sup>https://github.com/ggerganov/llama.cpp

<sup>†</sup> https://github.com/turboderp/exllama

<span id="page-3-0"></span>

| PPL J    | FP16  | RTN       | FP16% | (based | on act.) | FP16%  | (based | on W) | FP10   | 5% (rando | om)   |
|----------|-------|-----------|-------|--------|----------|--------|--------|-------|--------|-----------|-------|
| <b>v</b> | Ψ     | (w3-g128) | 0.1%  | 1%     | 3%       | 0.1%   | 1%     | 3%    | 0.1%   | 1%        | 3%    |
| OPT-1.3B | 14.62 | 119.00    | 25.03 | 16.91  | 16.68    | 108.71 | 98.55  | 98.08 | 119.76 | 109.38    | 61.49 |
| OPT-6.7B | 10.86 | 23.54     | 11.58 | 11.39  | 11.36    | 23.41  | 22.37  | 22.45 | 23.54  | 24.23     | 24.22 |
| OPT-13B  | 10.13 | 46.04     | 10.51 | 10.43  | 10.42    | 46.07  | 48.96  | 54.49 | 44.87  | 42.00     | 39.71 |

**Table 1.** Keeping a small fraction of weights (0.1%-1%) in FP16 significantly improves the performance of the quantized models over round-to-nearest (RTN). It is only effective when we select the important weights in FP16 by looking at *activation* distribution instead of *weight* distribution. We highlight results with a decent perplexity in green. We used INT3 quantization with a group size of 128 and measured the WikiText perplexity  $(\downarrow)$ .

OPT (PPL↓)

FP16

RTN

s = 2

AWQ

1% FP16

1.3B

14.62

119.47

16.91

18.63

16.32

| OPT-6.7B                                           | s = 1 | s = 1.25 | s = 1.5 | s = 2 | s = 4 |
|----------------------------------------------------|-------|----------|---------|-------|-------|
| proportion of $\Delta^{'} \neq \Delta$             | 0%    | 2.8%     | 4.4%    | 8.2%  | 21.2% |
| average $\Delta'/\Delta$                           | 1     | 1.005    | 1.013   | 1.038 | 1.213 |
| average $\frac{\Delta'}{\Delta} \cdot \frac{1}{s}$ | 1     | 0.804    | 0.676   | 0.519 | 0.303 |
| Wiki-2 PPL                                         | 23.54 | 12.87    | 12.48   | 11.92 | 12.36 |

**Table 2.** Statistics when multiplying the 1% salient channels by s>1. Scaling up the salient channels significantly improves the perplexity (23.54 to 11.92). As s goes larger, the percentage of changed  $\Delta$  increases, and the error reduction rate for salient channels also increases. However, the best perplexity is achieved at s=2, since further increasing s will increase the quantization error for *non-salient* channels.

**Table 3.** AWQ protects salient weights and reduces quantization error by using a scaling-based method. It consistently outperforms Round-to-nearest quantization (RTN) and achieves comparable performance as mixed-precision (1% FP16) while being more hardware-friendly. We use 3-bit quantization with group size 128.

2.7B

12.47

298.00

13.69

14.94

13.58

6.7B

10.86

23.54

11.39

11.92

11.39

13B

10.13

46.04

10.43

10.80

10.56

30B

9.56

18.80

9.85

10.32

9.77

function is defined as:

$$Q(\mathbf{w}) = \Delta \cdot \text{Round}(\frac{\mathbf{w}}{\Delta}), \quad \Delta = \frac{\max(|\mathbf{w}|)}{2^{N-1}}, \quad (1)$$

where N is the number of quantization bits, and  $\Delta$  is the quantization scaler determined by the absolute maximum value. Now consider a weight element  $w \in \mathbf{w}$ , if we multiply w with s > 1 and the inversely scale x, we will have  $Q(w \cdot s)(x/s)$ , which is:

$$Q(w \cdot s) \cdot \frac{x}{s} = \Delta' \cdot \text{Round}(\frac{ws}{\Delta'}) \cdot x \cdot \frac{1}{s}, \tag{2}$$

where  $\Delta'$  is the new quantization scaler after applying s. We empirically find that: (1) The expected error from Round( $\cdot$ ) (denoted as RoundErr( $\cdot$ )) does not change: since the round function maps a floating-point number to an integer, the error is roughly uniformly distributed from [0,0.5], resulting in an average error of 0.25; i.e., RoundErr( $\cdot$ )  $\sim$  0.25. (2) Scaling up a single element w usually does not change the maximum value from the group  $\mathbf{w}$ . Therefore we have  $\Delta' \approx \Delta$ ; (3) As  $\Delta$  and x are represented in FP16, they have no quantization error. Consequently, the quantization error from equation 1 and 2 can be expressed as

$$\begin{aligned} & \operatorname{Err}(Q(w)x) = \Delta \cdot \operatorname{RoundErr}(\frac{w}{\Delta}) \cdot x \\ & \operatorname{Err}(Q(w \cdot s)(\frac{x}{s})) = \Delta^{'} \cdot \operatorname{RoundErr}(\frac{ws}{\Delta^{'}}) \cdot x \cdot \frac{1}{s} \end{aligned} \tag{3}$$

The ratio of the new error to the original error is  $\frac{\Delta^{'}}{\Delta} \cdot \frac{1}{s}$ . Given  $\Delta^{'} \approx \Delta$  and s>1, the relative error is smaller for the salient weight w.

To verify the idea, we multiply the 1% salient channels with s > 1 for the OPT-6.7B model, and measure the change in  $\Delta$  for each group in Table 2. We find that scaling up the salient channels is quite effective: the perplexity improves from 23.54 for s = 1 (simply RTN) to 11.92 for s = 2. As s goes larger, the percentage of changed  $\Delta$  generally gets larger, but the percentage is still quite small for s < 2(less than 5%); the relative error for the salient channels continues to go smaller as s increases. Nonetheless, the best PPL actually appears at s=2. This is because if we use a very large s, it will increase the relative error for the nonsalient channels when  $\Delta$  increases (the error of non-salient channels will be amplified by  $\frac{\Delta'}{\Delta}$ , and the ratio is larger than 1 for 21.2% of the channels under s=4), which can damage the model's overall accuracy. Therefore, we need to also consider the error from non-salient channels when protecting salient ones.

**Searching to scale.** To consider both salient and non-salient weights, we choose to automatically search for an optimal (per input channel) scaling factor that minimizes the output difference after quantization for a certain layer.

<span id="page-4-0"></span>![](_page_4_Figure_1.jpeg)

![](_page_4_Figure_2.jpeg)

![](_page_4_Figure_3.jpeg)

- (a) Generation stage is slower
- (b) Generation stage is bounded by memory bandwidth
- (c) Weight loading is more expensive

Figure 3. Bottleneck analysis for Llama-2-7B on NVIDIA RTX 4090. Left: In on-device LLM applications, generation stage is much slower than the context stage. Middle: The generation stage is memory bound and has low arithmetic intensity. W4A16 quantization can effectively improve the arithmetic intensity by 4×. Right: The amount of weight access is orders of magnitude larger than the amount of activation access. Thus, weight-only quantization is more effective for on-device LLMs.

Formally, we want to optimize the following objective:

$$\mathbf{s}^* = \operatorname*{arg\,min}_{\mathbf{s}} \mathcal{L}(\mathbf{s})$$

$$\mathcal{L}(\mathbf{s}) = \|Q(\mathbf{W} \cdot \operatorname{diag}(\mathbf{s}))(\operatorname{diag}(\mathbf{s})^{-1} \cdot \mathbf{X}) - \mathbf{W}\mathbf{X}\|$$
(4)

Here Q means the weight quantization function (*e.g*., INT3/INT4 quantization with group size 128), W is the original weights in FP16, and X is the input features cached from a small calibration set (we take a small calibration set from he pre-training dataset in order not to overfit to a specific task). s is a per-(input) channel scaling factor; for s −1 · X, it can usually be fused into the previous operator [\(Wei et al.,](#page-13-0) [2022b;](#page-13-0) [Xiao et al.,](#page-13-0) [2022\)](#page-13-0). Since the quantization function is not differentiable, we are not able to directly optimize the problem with vanilla backpropagation. There are some techniques relying on approximated gradients [\(Bengio et al.,](#page-11-0) [2013;](#page-11-0) [Esser et al.,](#page-11-0) [2019\)](#page-11-0), which we found still suffers from unstable convergence.

To make the process more stable, we define a *search space* for the optimal scale by analyzing the factors that will affect the choice of scaling factor. As shown in the last section, the saliency of weight channels is actually determined by the activation scale (thus "activation-awareness"). Therefore, we simply use a very simple search space:

$$\mathbf{s} = \mathbf{s_X}^{\alpha}, \quad \alpha^* = \operatorname*{arg\,min}_{\alpha} \mathcal{L}(\mathbf{s_X}^{\alpha})$$
 (5)

s<sup>X</sup> is the average magnitude of activation (per-channel), and we use a single hyper-parameter α to balance between the protection of salient and non-salient channels. We can find the best α by a fast grid search over the interval of [0, 1] (0 means we do not scale; 1 corresponds to the most aggressive scaling in our search space). We further apply weight clipping to minimize the MSE error of quantization. We provide an ablation study on OPT models under INT3-g128 quantization in Table [5;](#page-6-0) AWQ consistently outperforms round-to-nearest quantization (RTN) and achieves comparable performance as mixed-precision (1% FP16) while being more hardware-friendly.

Advantages. Our method does not rely on any regression [\(Frantar et al.,](#page-11-0) [2022\)](#page-11-0) or backpropagation, which is required by many quantization-aware training methods. It has minimal reliance on the calibration set since we only measure the average magnitude per channel, thus preventing over-fitting (Figure [8\)](#page-9-0). Therefore, our method requires fewer data for the quantization process and can preserve LLMs' knowledge outside of the calibration set's distribution. See Section [5.3](#page-9-0) for more details.

