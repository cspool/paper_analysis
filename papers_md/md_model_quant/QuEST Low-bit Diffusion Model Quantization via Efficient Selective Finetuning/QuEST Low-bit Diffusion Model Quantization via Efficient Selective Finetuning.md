# <span id="page-0-1"></span>QuEST: Low-bit Diffusion Model Quantization via Efficient Selective Finetuning

Haoxuan Wang<sup>1</sup> Yuzhang Shang<sup>2</sup> Zhihang Yuan<sup>4</sup> Junyi Wu<sup>1</sup> Junchi Yan<sup>3</sup> Yan Yan<sup>1†</sup>

# **Abstract**

The practical deployment of diffusion models is still hindered by the high memory and computational overhead. Although quantization paves a way for model compression and acceleration, existing methods face challenges in achieving low-bit quantization efficiently. In this paper, we identify imbalanced activation distributions as a primary source of quantization difficulty, and propose to adjust these distributions through weight finetuning to be more quantization-friendly. We provide both theoretical and empirical evidence supporting finetuning as a practical and reliable solution. Building on this approach, we further distinguish two critical types of quantized layers: those responsible for retaining essential temporal information and those particularly sensitive to bit-width reduction. By selectively finetuning these layers under both local and global supervision, we mitigate performance degradation while enhancing quantization efficiency. Our method demonstrates its efficacy across three high-resolution image generation tasks, obtaining state-of-the-art performance across multiple bit-width settings. Code is available at https: //github.com/hatchetProject/QuEST.

#### 1. Introduction

Diffusion models [6, 14, 30, 36] have recently achieved remarkable success in image generation. However, this success comes at the cost of two major obstacles that limit their efficiency [3]. The first obstacle is the denoising process which requires hundreds to thousands of inference time steps, slowing down the generation speed drastically. The other is the increasing model size, driven by demands for better image fidelity and higher image resolutions. Both factors contribute to considerable latency and increased computational requirements, impeding the application of diffusion models to real-world settings where both time and computational power are carefully restricted.

Neural network quantization offers a feasible solution

<span id="page-0-0"></span>

| Метнор           | Data<br>Free | TIME & MEMORY<br>EFFICIENT | LOW-BIT<br>COMPATIBLE | FULLY<br>QUANTIZED |
|------------------|--------------|----------------------------|-----------------------|--------------------|
| PTQ [19]         | ~            | <b>v</b>                   | ×                     | ~                  |
| QAT [22]         | X            | ×                          | ~                     | ~                  |
| EFFICIENTDM [10] | ~            | <b>✓</b>                   | <b>~</b>              | ×                  |
| OURS             | ~            | <b>✓</b>                   | ~                     | ~                  |

Table 1. **Comparison with different frameworks.** Our method is both efficient and effective for low-bit diffusion model quantization, also achieving a reduced overall bit-width.

for accelerating inference speed and reducing memory consumption simultaneously [8], making it a natural solution for deploying diffusion models efficiently. It aims to compress high-bit model parameters into low-bit approximations with negligible performance degradation. For example, 4-bit weight and 4-bit activation quantization can achieve up to 8× inference time speedup and memory reduction theoretically [23]. Hence, low-bit quantization of diffusion models emerges as a viable approach for efficiency enhancement. Unfortunately, existing diffusion model quantization methods that perform well at higher bitwidths face significant limitations in low-bit settings: some only adjust the quantization parameters and fail under lowbit conditions [11, 19, 32], while others succeed but require substantial computational resources comparable to training a diffusion model from scratch [22, 34]. In this work, we aim for efficient low-bit quantization, thereby circumventing the latter choice of resource-intensive training.

We first reveal the current challenge within diffusion models that impede the effectiveness of current efficient low-bit quantization methods [16, 19, 33]. As illustrated in Fig. 1(a): activation distributions tend to be imbalanced, with most values clustering near zero, while essential high-magnitude values are sparse and inconsistently distributed. Existing quantization methods [19, 20, 32] either approximate large and sparse values, inadequately estimating numerous small values, or focus on small values while overlooking the large ones, thereby impeding the reduction of quantization error. To overcome this challenge, we propose to adjust the activation distributions via weight finetuning, where its feasibility is justified both theoretically and empirically. Nevertheless, finetuning the entire diffusion model is a highly computationally-expensive and time-consuming

<sup>&</sup>lt;sup>1</sup>University of Illinois Chicago <sup>2</sup>University of Central Florida <sup>3</sup>Shanghai Jiao Tong University <sup>4</sup>Houmo AI

<sup>†</sup>Corresponding author

<span id="page-1-0"></span>process, requiring over 80GB memory and numerous hours [5, 22]. Thus, developing an efficient finetuning strategy tailored for diffusion model quantization is important.

To facilitate efficient quantization, we further identify two key properties of quantized diffusion models that unlock new opportunities: ① diffusion models exhibit varying functions at distinct time steps [2], therefore preserving accurate temporal information is important during quantization; and ② diffusion models possess complex network architectures, incorporating various types of modules. Whereas previous works consider each module as equally important and apply quantization uniformly, we reveal that certain modules are particularly sensitive to perturbations from quantization, while others are more resilient.

Based on the above findings, we propose a novel quantization approach for diffusion models, termed QuEST (Quantization via Efficient Selective FineTuning). Confronting the revealed quantization challenge, we first theoretically justify that weight finetuning can enhance model robustness toward large activation perturbations in low-bit settings, thereby reducing quantization error. In contrast, previous methods have struggled to properly balance clipping error and rounding error. Then we empirically demonstrate that by finetuning the model weights, the activation distributions are modified to be more amenable to quantization. As shown in Fig. 1(a), the activation distribution is adjusted by reducing the amount of large, sparse values and enhancing the compactness of the value distribution.

Following the idea of weight finetuning, we compare the effects of quantizing different modules of diffusion models (Fig. 1(b)) and identify two types of layers as primary culprits to performance degradation: time embedding layers exhibiting property ① and attention-related layers associated with property ②. Consequently, we selectively and progressively finetune the small subsets of identified layers in conjunction with all activation quantization parameters, as illustrated in Fig. 1(c). The learning objective is crafted to align the quantized model with its full-precision counterpart at both local and global levels. Involving less than 7% of the total parameters, QuEST not only substantially enhances low-bit quantized model performance, but is also notably time-efficient and can be conducted in a data-free manner. Our contributions are summarized as follows:

- We identify the current challenge in low-bit diffusion model quantization that hinders effective low-bit quantization, and propose to adjust the activation distributions via weight finetuning for easier quantization. Both theoretical and empirical discussions are provided.
- We uncover and validate two properties in quantized diffusion models as the main factors for degraded performance. Motivated by the identified properties, we introduce QuEST, a parameter-efficient finetuning strategy that trains the diffusion model selectively and pro-

- gressively, achieving low-bit quantization capability with time and memory efficiency.
- Experiments on three high-resolution image generation tasks over four models demonstrate the superiority of our method, achieving state-of-the-art performance under various bit-width settings.

#### 2. Related Works

#### 2.1. Diffusion Model Inference

Diffusion models [14, 27, 30] generate samples via an iterative denoising process. During inference, the initial input is sampled from a Gaussian distribution:  $x_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ , and the final output  $x_0$  is obtained through a denoising process:

$$p_{\theta}(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \tilde{\boldsymbol{\mu}}_{\theta,t}(x_t), \tilde{\beta}_t \mathbf{I}), \tag{1}$$

where  $\tilde{\mu}_{\theta,t}$  and  $\tilde{\beta}_t$  are calculated from the model's output. This denoising process in a typical diffusion model requires tens to thousands of iterations, making efficient inference extremely challenging. Practically, diffusion models typically adopt a UNet architecture [31], incorporating an encoder and a decoder. Usually, encoders and decoders are lightweight and computationally inexpensive, so our focus is on quantizing the UNet structures in latent diffusion models, in alignment with the other works.

#### 2.2. Diffusion Model Quantization

Model quantization is a dominant technique for optimizing the inference memory and speed of deep learning models by reducing the precision of the tensors used in computation. The researches for diffusion model quantization fall into three categories: Quantization-Aware Training (QAT) [17, 21, 22], Post-Training Quantization (PTQ) [19, 20, 25, 32, 37, 38], and Parameter-Efficient Fine-Tuning methods [10]. QAT methods [22] train all parameters from scratch, being effective for low-bit quantization but are extremely resource-intensive. PTQ methods [11, 16, 33, 35, 39] calculate the quantization parameters based on a small calibration set, offering better efficiency. However, PTQ methods often rely on complex designs and fail at lower bit-widths. To achieve low-bit compatibility with high efficiency, Parameter-Efficient Fine-Tuning methods were proposed. The representative work EfficientDM [10] trains a low-rank adapter (LoRA) [15] for each layer to reduce training costs, and successfully scales to W4A4.

Our proposed method also adopts a parameter-efficient finetuning strategy, and differs from EfficientDM in the following aspects: Firstly, EfficientDM introduces extra weight parameters, requiring substantial training iterations on the LoRA weights. Our method instead does not include additional parameters. Secondly, EfficientDM does not quantize the matrix multiplications in the attention mechanism, as well as certain linear layers. Our method quantizes

<span id="page-2-3"></span><span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Figure 1. Overview of our observations and method. (a) Illustration of the challenge we identified in low-bit diffusion model quantization and a potential solution. We propose to ease the quantization difficulty by refining the activation distribution to make it more quantizationfriendly. (b) Overview of property ❶ and property ❷, whose importance are identified based on their impact on the model's generation performance, making them suitable module candidates for efficient finetuning. (c) Framework of the proposed method. WTE, WA, W<sup>F</sup> are the weights of the time embeddings layers, attention-related layers, and other frozen layers, respectively. s is the quantization parameter. To train with efficiency, we adopt a selective and progressive finetuning strategy, incorporating temporal layer alignment (TLA) and critical module alignment (CMA). A global loss is also used for network-level guidance, improving the generated image quality.

all layers, and is more time-efficient. Tab. [1](#page-0-0) summarizes the differences between our method with the other works.

# 3. Methodology

## 3.1. Preliminaries

The quantization process for a single value x in a vector can be formulated as:

$$\hat{x} = \text{clamp}(\text{round}(\frac{x}{s}) + Z; q_{min}, q_{max}),$$
 (2)

where xˆ is the quantized integer result, round(·) represents rounding algorithms such as the round-to-nearest operator [\[20\]](#page-8-11) and AdaRound [\[28\]](#page-8-19), s is referred to as the scaling factor and Z is the zero-point. clamp is the function that clamps values into the range of [qmin, qmax], which is determined by the bit-width. Reversely, transforming the quantized values back into the full-precision form is:

$$\tilde{x} = (\hat{x} - Z) * s. \tag{3}$$

This is denoted as the dequantization process. The quantization and dequantization processes are performed on both model weights and layer outputs (also termed as 'activation'). Eq. [\(2\)](#page-2-1) indicates that the quantization error is composed of two factors: the clipping error produced by range clamping and the rounding error caused by the rounding function, where they exhibit a trade-off relationship [\[20\]](#page-8-11). While previous approaches strive for an optimal balance between the two errors, they neglect the intrinsic characteristics in quantized diffusion models. In the following sections, we first examine the current challenges in diffusion model quantization and outline our finetuning motivation, providing theoretical justification. We then identify

<span id="page-2-2"></span>![](_page_2_Figure_10.jpeg)

<span id="page-2-1"></span>Figure 2. Illustration of the imbalanced activation distributions. In the full-precision model, the majority of values cluster near zero with sporadic large values, presenting challenges for low-bit quantization. Our method refines the activation distributions by eliminating the large and sparse values, enabling easier quantization.

two properties that enable efficient finetuning, forming the basis of our proposed method.

## 3.2. The Challenge in Low-bit Quantization

Previous works [\[16,](#page-8-10) [19\]](#page-8-4) primarily address the varying activation distributions across different time steps, facilitating diffusion model quantization at higher bit-widths. However, these methods experience failure in low-bit settings. To investigate the potential reason, we focus on the activation distribution itself and conduct a layer-wise analysis, revealing the following challenge in full-precision diffusion models that impedes effective quantization:

Challenge: Despite the majority of values being close to zero in the activation outputs, there exist numerically large and sparse values holding significant importance.

Fig. [2](#page-2-2) provides a detailed analysis, where activation values are clustered into uniformly distributed bins. We find that in some layers, though the majority of values are close <span id="page-3-2"></span>to zero, there exist values that are relatively large and diverse (circled in green). Take the bin plot on the left as an example, the original activation values (blue line) range from [-10, 34] but with most values between [-0.6, 1.7]. Visualizations for more layers can be found in Appendix [9.](#page-11-0) This phenomenon poses difficulties in minimizing the clipping error and is unfriendly for effective quantization.

Moreover, these large and sparse values are important for generation performance preservation. We find that when replacing the few tokens with maximum values by random noises, the generated images' quality is critically degraded (as shown in Appendix [10\)](#page-11-1). With these large values being important and the small values appearing frequently, neither of them is negligible and needs to be carefully quantized at the same time. Unfortunately, typical quantization methods fall short of this ability under low-bit settings, where the rounding error often outweighs the clipping error during optimization and results in over-clipped values, generating corrupted images. This inspires us to refine the activation distributions to attain more quantization-friendly distributions, as depicted in Fig. [1\(](#page-2-0)a).

However, the activation distribution cannot be directly manipulated. To address this, we instead finetune the model weights under quantization constraints, producing a *new yet similar* full-precision model whose quantized counterpart maintains performance comparable to the original fullprecision model. Our experiments interestingly reveal that the proposed finetuning strategy effectively eliminates the large and sparse values (Fig. [2\)](#page-2-2), reducing quantization difficulty. We detail our approach in Sec. [3.3](#page-3-0) and provide further theoretical analysis in Sec. [3.4.](#page-4-0)

# <span id="page-3-0"></span>3.3. Quantization via Efficient Selective Finetuning

In this section, we introduce QuEST, an efficient finetuning method for diffusion models that can significantly boost low-bit performance with less time and memory usage. We also present the two unique properties in quantized diffusion models, which serve as the foundation for the design of our method. Fig. [1\(](#page-2-0)c) illustrates our approach.

#### 3.3.1. Data-free Efficient Network-wise Training.

We first present the general training pipeline of our method. To alleviate the need for substantial training data, we construct the calibration set in a data-free manner. By feeding random Gaussian noises x<sup>T</sup> into the full-precision model and sampling over different time steps, we can obtain the calibration data needed for finetuning the quantized model. In practice, we only have to infer the full-precision model a few times to gather the needed number of calibration samples, totaling 128 or 256 samples per time step.

As depicted in Fig. [1\(](#page-2-0)c), to overcome the quantization challenge efficiently, we update partial model weights (WTE and WA) that only account for a small subset of parameters related to the time step t. The remaining weight parameters W<sup>F</sup> are kept frozen during optimization. We also fix the weight quantization parameters during training, reducing the amount of parameters that need to be optimized. For instance, in LDM-4 [\[30\]](#page-8-2), no more than 7% of the parameters are adjusted. The choices for the weights to be finetuned will be discussed in the following sections.

The activation quantization parameters can be viewed as additional model parameters. Therefore, we further propose a network-wise training strategy. Different from quantization methods using layer-wise or block-wise reconstruction [\[19,](#page-8-4) [32\]](#page-9-1) that bind quantization parameters with their corresponding layers or blocks, we optimize all activation scaling factors together with the partial weight parameters. Additionally, while layer/block-wise optimization methods can only reconstruct sequentially, we update the required parameters at once. In this way, we significantly save the time and memory needed for quantization.

# 3.3.2. Temporal Layer Alignment

The inference process of diffusion models is highly dependent on the temporal information. Specifically, integer time steps are transformed into time embeddings through one or two linear layers, then added to the intermediate model features. Motivated by this observation, we make the following analysis that is consistent with previous works [\[16,](#page-8-10) [33\]](#page-9-3):

*Property* ❶*: Although time embeddings depend solely on time steps and are easily obtainable, precise temporal information is crucial for optimal quantization.*

<span id="page-3-1"></span>

| TE Setting<br>(W8A8 Model) | FID ↓ | sFID ↓ | TE Setting<br>(W4A8 Model) | FID ↓ | sFID ↓ |
|----------------------------|-------|--------|----------------------------|-------|--------|
| PTQ                        | 7.58  | 22.07  | PTQ                        | 8.59  | 22.74  |
| FP                         | 6.77  | 22.03  | FP                         | 7.55  | 21.69  |
| Ours                       | 5.61  | 21.22  | Ours                       | 6.95  | 23.17  |

Table 2. Ablations on time embedding (TE) settings. Finetuning the TE layers with our method surpasses full-precision embeddings, while the latter outperforms standard quantized ones.

Tab. [2](#page-3-1) provides an empirical justification, where we quantitatively show the performance drop when quantizing time embeddings to different bit-width. Under W8A8 and W4A8 bit-width settings, solely quantizing the time embeddings can lead to an increase of 0.81 and 1.04 (relatively 15%) in FID, respectively. We infer the reason is that inaccurate time embeddings can cause mismatched input and model functionality, resulting in possible oscillations in the sequence of noise removal. Previous works either propose to learn dynamic quantization parameters across different time steps through a simple network [\[33\]](#page-9-3), or calibrate the time embedding layers and projection layers across all time steps [\[16\]](#page-8-10). We instead focus on finetuning the time embedding layers, adjusting fewer modules without introducing additional parameters. The results in Tab. [2](#page-3-1) also suggest that our method can improve the quantization performance, even surpassing the full-precision baseline.

<span id="page-4-5"></span>Concretely, in a single forward process, identical time embeddings are injected into different parts of the model, passed through projection layers, and merged with the latent image representations. This implies that the time information operates independently from the primary network flow. Thus, we refine the time embedding layer l's weight w<sup>l</sup> along with its activation quantization parameters s<sup>l</sup> :

$$\mathcal{L}_{\text{TLA}} = \sum_{l \in \mathbb{C}_{\text{TE}}} \mathbb{E}_t[||O(t; \mathbf{w}_l) - \tilde{O}(t; \mathbf{w}_l, \mathbf{s}_l)||^2], \quad (4)$$

where CTE represents the set of time embedding layers. O(t; wl) is the intermediate activation of the full-precision model representing the ground truth, and O˜(t; w<sup>l</sup> , sl) is the quantized activation. This objective function indicates that the chosen weight parameters are consistently updated across different time steps, so as to ensure robustness to diverse temporal inputs. Different from other methods [\[10,](#page-8-6) [16\]](#page-8-10) that obtain different sets of quantization parameters for each time step, we only use a single set for varying time steps, improving time efficiency and memory storage.

#### 3.3.3. Critical Module Alignment

While inaccurate time embedding quantization reduces performance under low-bit settings, it does not cause the complete generation failure observed in fully quantized models. Through careful layer-wise empirical study, we make the following observation:

*Property* ❷*: Not all activations respond equally to reduced bit-width, as different activations exhibit varying levels of sensitivity, with certain critical layers being especially sensitive to quantization.*

<span id="page-4-1"></span>![](_page_4_Figure_6.jpeg)

Figure 3. Effect of decreasing different activations' bit-width on the model performance. The generation failure of FeedForward layers emerges at 6 bits, while all other linear layers barely fail at 4 bits and all convolution layers only fail at 4 bits.

Fig. [3](#page-4-1) illustrates the sensitivity of different activations to quantization. Specifically, we quantize three different types of activations to lower bits while maintaining the others' bit-width to 8-bit, and observe how the decreasing bit-width affects generation performance. Compared to weights that only fall into linear and convolutional layers, activations are more diverse and complex, making their effective quantization more challenging. Surprisingly, we observe that the FeedForward layer [\[7\]](#page-8-20) activations cause generation failure at as early as 6 bits, whereas the activations of all other linear layers (containing 5 times more layers) barely fail at 4 bits and all convolution layers (containing 3 times more layers) only fail at 4 bits. This indicates that these activations are especially sensitive to low-bit quantization, making them essential to be specially dealt with.

<span id="page-4-2"></span>Denote C<sup>A</sup> as the set containing all attention-related [\[9\]](#page-8-21) layers and given their image calibration inputs zt,l, we optimize the corresponding weights and all quantization parameters s, except for the ones already updated:

<span id="page-4-3"></span>
$$\mathcal{L}_{\text{CMA}} = \sum_{l \in \mathbb{C}_A} \mathbb{E}_t[||O(z_{t,l}; \mathbf{w}_l) - \tilde{O}(\tilde{z}_{t,l}; \mathbf{w}_l, \hat{\mathbf{s}})||^2], \quad (5)$$

where w<sup>l</sup> are the weight parameters of the lth layer, z˜t,l is the quantized layer input, ˆs = s \ s<sup>l</sup> , l ∈ CTE, which represents all the quantization parameters without the ones already finetuned in Eq. [\(4\)](#page-4-2). Note that we use different inputs to optimize each module, so as to enhance the robustness of the modules to the input perturbations.

#### 3.3.4. Progressive Alignment with Global Loss

As investigated in the previous sections, two crucial types of layers are identified and selected for weight finetuning to enable quantization efficiency: time embedding layers and attention-related layers. We progressively align these components with the full-precision model due to their distinct, non-overlapping functionalities. Since temporal information is independent of the image input and determined early in the model, we first finetune the time embedding layers to provide accurate time step guidance for each subsequent module. Then we optimize the attention-related modules with the refined time embeddings.

However, the above selective finetuning strategy only aligns the local information in the model, but is unaware of the global error reduction of the quantized model and the quantization parameters of the unselected layers. To improve the final generated images' quality, we further aim to minimize the target task loss to provide global supervision:

<span id="page-4-4"></span>
$$\mathcal{L}_{G} = \mathbb{E}_{t}[||O(x_{t}; \mathbf{w}) - \tilde{O}(x_{t}; \mathbf{w}, \mathbf{s})||^{2}], \tag{6}$$

where w represents all the model weights, O(xt; w) represents the final output of the full-precision counterpart and O˜(xt; w, s) is the final output of the quantized model.

By integrating Eq. [\(4\)](#page-4-2), Eq. [\(5\)](#page-4-3) and Eq. [\(6\)](#page-4-4), the final objective is formulated as:

$$\arg\min_{\mathbf{w}_l} (\mathcal{L}_{TLA} + \mathcal{L}_{CMA} + 2\mathcal{L}_G), \ l \in \mathbb{C}_{TE} \cup \mathbb{C}_A.$$
 (7)

# <span id="page-4-0"></span>3.4. Finetuning from a Theoretical Perspective

The above proposed method is motivated by the intuition that finetuning the model weights can adjust the activa<span id="page-5-5"></span>tion distribution such that the imbalanced activation phenomenon can be alleviated. In this part, we *attempt to explain why finetuning may be a feasible solution*, offering additional insights for readers. However, we note that this is not a theoretical guarantee of the proposed method.

We first review the underlying theory underpinning conventional post-training-quantization methods, which typically employ the reconstruction-based approach. Denote the full-precision diffusion model's activations at time t as  $\mathbf{z}_t = [z_{1,t}, z_{2,t}, ..., z_{n,t}]$ , the final loss as  $L(\mathbf{z}_t; \mathbf{w})$ , where n is the number of layers. L can be any loss function and here we use the mean squared error (MSE). We treat quantization as a type of perturbation and formulate the influence of activation quantization using Taylor expansion, assuming model weight  $\mathbf{w}$  is frozen:

<span id="page-5-0"></span>
$$\mathbb{E}[L(z_{n,t} + \Delta; \mathbf{w})] - \mathbb{E}[L(z_{n,t}; \mathbf{w})]$$

$$\approx \Delta^{\mathrm{T}} \overline{\mathbf{g}}^{(z_{n,t})} + \frac{1}{2} \Delta^{\mathrm{T}} \overline{\mathbf{H}}^{(z_{n,t})} \Delta, \qquad (8)$$

where  $\Delta$  is the activation perturbation,  $\overline{\mathbf{g}}^{(\mathbf{z})}$  is the gradient and  $\overline{\mathbf{H}}^{(z_{n,t})}$  is the Hessian matrix. According to [20, 41], for a well-trained model,  $\overline{\mathbf{g}}^{(z_{n,t})} = \nabla_{z_{n,t}} L$  approaches 0. Thus the above equation can be simplified to:

$$\frac{1}{2}\Delta^{\mathrm{T}}\overline{\mathbf{H}}^{(z_{n,t})}\Delta = \frac{1}{2}(\tilde{z}_{n,t} - z_{n,t})^{\mathrm{T}}\overline{\mathbf{H}}^{(z_{n,t})}(\tilde{z}_{n,t} - z_{n,t}). (9)$$

However, under low-bit settings, the reasoning from Eq. (8) to Eq. (9) is inaccurate, where the activation perturbation  $\Delta$  is too large for a meaningful Taylor expansion. Thus we have the following proposition:

**Proposition 3.1.** Reconstruction-based post-training quantization methods may lose their theoretical guarantee due to the large value perturbations under low-bit quantization.

Since the inaccuracy arises from the large activation perturbation  $\Delta$ , we transform  $\Delta$  into a smaller perturbation  $\epsilon$  and derive the following theorem:

<span id="page-5-2"></span>**Theorem 3.2.** Given an n layer diffusion model at time t with quantized activations as  $\tilde{\mathbf{z}}_t = [\tilde{z}_{1,t}, \tilde{z}_{2,t}, ..., \tilde{z}_{n,t}]$  and  $\tilde{z}_{n,t} = z_{n,t} + \Delta$ , where  $z_{n,t}$  is the ground truth and  $\Delta$  is the large perturbation caused by low-bit quantization. Denote the target task MSE loss as  $L(\mathbf{z}_t; \mathbf{w})$ , the quantization error can be transformed into:

$$\mathbb{E}[L(z_{n,t} + \Delta; \mathbf{w})] - \mathbb{E}[L(z_{n,t}; \mathbf{w})]$$

$$\approx 2\epsilon^{\mathrm{T}} \sum_{i=1}^{K} (\tilde{z}_{n-1,t}^{i} \cdot \mathbf{w}_{n} - z_{n,t})$$

$$+ \frac{1}{2} \sum_{i=1}^{K} (\tilde{z}_{n,t}^{i} - z_{n,t})^{\mathrm{T}} \overline{\mathbf{H}}^{(z_{n,t}+(i-1)\epsilon)} (\tilde{z}_{n,t}^{i} - z_{n,t}) \quad (10)$$

where  $\mathbf{w}_n$  is the weight for layer n and  $\tilde{z}_{n,t}^i = \tilde{z}_{n-1,t}^i \cdot \mathbf{w}_n$ , K is a large constant and  $\Delta = K\epsilon$ .

<span id="page-5-4"></span>

| Dataset  | Method       | Bit-width (W/A) | Size<br>(MB) | FID↓  |
|----------|--------------|-----------------|--------------|-------|
|          | FP           | 32/32           | 1045.6       | 2.95  |
|          | PTQ4DM       | 8/8             | 279.1        | 4.75  |
|          | Q-Diffusion  | 8/8             | 279.1        | 4.53  |
|          | PTQ-D        | 8/8             | 279.1        | 3.75  |
|          | EfficientDM* | 8/8             | 279.1        | N/A   |
|          | Ours         | 8/8             | 279.1        | 3.03  |
| LSUN-    | PTQ4DM       | 4/8             | 148.4        | N/A   |
| Bedrooms | Q-Diffusion  | 4/8             | 148.4        | 5.37  |
| (LDM-4)  | PTQ-D        | 4/8             | 148.4        | 5.94  |
|          | EfficientDM* | 4/8             | 148.4        | 15.15 |
|          | Ours         | 4/8             | 148.4        | 3.26  |
|          | PTQ4DM       | 4/4             | 148.4        | N/A   |
|          | Q-Diffusion  | 4/4             | 148.4        | N/A   |
|          | PTQ-D        | 4/4             | 148.4        | N/A   |
|          | EfficientDM* | 4/4             | 148.4        | 10.60 |
|          | Ours         | 4/4             | 148.4        | 5.64  |
|          | FP           | 32/32           | 1125.4       | 4.02  |
|          | PTQ4DM*      | 8/8             | 330.6        | 63.93 |
|          | Q-Diffusion  | 8/8             | 330.6        | 6.94  |
|          | PTQ-D*       | 8/8             | 330.6        | 10.76 |
|          | EfficientDM* | 8/8             | 330.6        | N/A   |
|          | Ours         | 8/8             | 330.6        | 6.55  |
| LSUN-    | PTQ4DM*      | 4/8             | 189.9        | N/A   |
| Churches | Q-Diffusion  | 4/8             | 189.9        | 7.80  |
| (LDM-8)  | PTQ-D*       | 4/8             | 189.9        | 7.33  |
|          | EfficientDM* | 4/8             | 189.9        | 9.29  |
|          | Ours         | 4/8             | 189.9        | 7.33  |
|          | PTQ4DM*      | 4/4             | 189.9        | N/A   |
|          | Q-Diffusion  | 4/4             | 189.9        | N/A   |
|          | PTQ-D*       | 4/4             | 189.9        | N/A   |
|          | EfficientDM* | 4/4             | 189.9        | 14.34 |
|          | Ours         | 4/4             | 189.9        | 11.76 |

<span id="page-5-1"></span>Table 3. Quantization performance on LSUN-Bedrooms/Churches 256×256. "N/A" denotes generation failure. "\*" denotes the results obtained by re-implementing the open-source code. More baseline and metric comparisons are included in the Appendix.

<span id="page-5-3"></span>Theorem 3.2 indicates that, to minimize quantization error,  $\mathbf{w}_n$  should ideally be fine-tuned so that, for any i, the weights fit the corresponding input  $\tilde{z}_{n-1,t}^i + (i-1)\epsilon$ . This adjustment captures variations that the full-precision model may overlook. In other words, fine-tuning optimizes model weights for better robustness towards large input activation perturbations, facilitating easier quantization. Moreover, since the finetuned and quantized model is aligned with the original full-precision model, the potential impact on generation performance can be avoided. Note that the second term in Eq. (10) can be ignored within an acceptable upper bound, as it is of second order and shares a common zero-loss solution with the first term.

<span id="page-6-2"></span><span id="page-6-0"></span>

| Bit-width (W/A) | Method                                       | Size<br>(MB)                     | FID↓                              | sFID↓                             | IS↑                                   |
|-----------------|----------------------------------------------|----------------------------------|-----------------------------------|-----------------------------------|---------------------------------------|
| 32/32           | FP                                           | 1529.7                           | 11.28                             | 7.70                              | 364.73                                |
| 8/8             | Q-Diffusion<br>PTQ-D<br>EfficientDM*         | 428.7<br>428.7<br>435.0          | 10.60<br><b>10.05</b><br>11.38    | 9.29<br>9.01<br>8.04              | 350.93<br>359.78<br>362.34            |
|                 | Ours                                         | 428.7                            | 10.43                             | 6.07                              | 365.12                                |
| 4/8             | Q-Diffusion<br>PTQ-D<br>EfficientDM*         | 237.5<br>237.5<br>243.8          | 9.29<br>8.74<br>9.93              | 9.29<br>7.98<br>7.34              | 336.80<br>344.72<br>353.83            |
|                 | Ours                                         | 237.5                            | 8.48                              | 6.55                              | 354.97                                |
| 4/4             | Q-Diffusion<br>PTQ-D<br>EfficientDM*<br>Ours | 237.5<br>237.5<br>243.8<br>237.5 | N/A<br>N/A<br>6.97<br><b>5.98</b> | N/A<br>N/A<br>9.28<br><b>7.93</b> | N/A<br>N/A<br>199.96<br><b>202.45</b> |

Table 4. Quantization performance on ImageNet 256×256. "\*" denotes the results obtained by re-running the open-source code.

## 4. Experiments

#### 4.1. Experiment Settings

To verify the effectiveness of our proposed method, we conduct experiments on three types of generation tasks: Unconditional image generation on LSUN-Bedrooms and LSUN-Churches datasets [40], class-conditional image generation on ImageNet [4], and text-to-image generation. The model architectures we quantize include LDMs and Stable Diffusion [30], and use "WnAm" to represent the quantization setting: n-bit weight quantization and m-bit activation quantization. DDIM samplers [14] are adopted for LDMs and the PLMS sampler [26] is used for Stable Diffusion. We generate 256 samples per time step for constructing the calibration set. The Adam optimizer [18] is adopted and the learning rate for weight finetuning and scaling factor finetuning is set as  $1e^{-5}$  and  $1e^{-4}$  respectively.

We compare with popular PTQ methods including PTQ4DM [32], Q-Diffusion [19] and PTQ-D [11], as well as the state-of-the-art efficient finetuning method EfficientDM [10]. The performance of different quantized LDMs is evaluated using the Fréchet Inception Distance (FID) [13], spatial FID (sFID) [29] and Inception Score (IS) [1]. Unless specified, quantitative results are obtained by sampling 50,000 images and evaluated using the official evaluation scripts [6]. For Stable Diffusion, we use the CLIP Score [12] for evaluation. All experiments are conducted on A6000 GPUs.

#### 4.2. Experiment Results and Analysis

**Unconditional Generation:** We evaluate the performance of our method over LDM-4 (LSUN-Bedrooms  $256 \times 256$ ) and LDM-8 (LSUN-Churches  $256 \times 256$ ) using the DDIM sampler with 200 and 500 time steps, respectively. Results are shown in Tab. 3 using FID, where our method outperforms the other baselines by a good margin. Note that the

<span id="page-6-1"></span>

| Bit-width (W/A) | Method      | Size (MB) | CLIP Score↑  |
|-----------------|-------------|-----------|--------------|
| 32/32           | FP          | 3279.1    | 31.50        |
| 8/8             | Q-Diffusion | 949.0     | 31.43        |
|                 | Ours        | 949.0     | <b>31.47</b> |
| 4/8             | Q-Diffusion | 539.1     | 31.39        |
|                 | Ours        | 539.1     | <b>31.50</b> |
| 4/4             | Q-Diffusion | 539.1     | N/A          |
|                 | Ours        | 539.1     | 28.85        |

Table 5. Quantization performance on Stable Diffusion v1.4  $(512\times512)$  using COCO2014 prompts.

Inception Score is not a reasonable metric for datasets that have significantly different domains and categories from ImageNet [19], thus not included. We further provide comparison with TFMQ-DM [16] in Appendix 6.

Class-conditional Generation: We evaluate the performance using LDM-4 on ImageNet 256×256 using the DDIM sampler (20 steps). As shown in Tab. 4, three metrics are used for evaluation. Note that sFID uses additional intermediate spatial features for calculation compared with FID. We can also see that FID is not a valid metric for ImageNet LDM-4 evaluation: All methods have lower FID when quantized to lower bits, conflicting with human perception. We show that our method not only succeeds in W4A4 quantization, but also improves the generation quality under higher bit settings. Under all three kinds of bitwidth settings, our method is able to outperform the SOTA PTQ methods and EfficientDM in both sFID and IS. Examples of our generated images are included in Appendix 11. **Text-to-image Generation:** We use Stable Diffusion v1.4 as the model for quantization with the PLMS sampler sampling 50 time steps. Tab. 5 shows the results. Images are generated based on the 10,000 prompts sampled from the COCO2014 [24] validation set, and CLIP Score is calculated based on the ViT-B/16 backbone. Given the limited works done on Stable Diffusion, we can only compare with O-Diffusion and the full-precision baseline.

![](_page_6_Figure_12.jpeg)

Figure 4. Visual comparison with Q-Diffusion and EfficientDM. QuEST outperforms the baselines with better visual quality.

#### 4.3. Ablations and Discussions

Efficiency comparison with PTQ methods and the impact of individual components. Tab. 6 compares the efficiency and performance against the post-training quantization (PTQ) approach on the LSUN-Bedrooms dataset. Although our method uses the same amount of calibration data

<span id="page-7-2"></span><span id="page-7-0"></span>

| Method      | Bit-width (W/A) | Calibration data size | Time cost<br>(GPU hours) | Memory cost<br>(MB) | Model size<br>(MB) | FID↓ |
|-------------|-----------------|-----------------------|--------------------------|---------------------|--------------------|------|
| FP          | 32/32           | -                     | -                        | -                   | 1045.6             | 2.95 |
| PTQ [19]    | 4/8             | 5120                  | 23.08                    | 10334               | 148.4              | 5.37 |
| Baseline    | 4/8             | 5120                  | 11.52                    | 9822                | 148.4              | 6.95 |
| + TLA       | 4/8             | 5120                  | 13.13                    | 11862               | 148.4              | 4.41 |
| + TLA + CMA | 4/8             | 5120                  | 15.25                    | 12178               | 148.4              | 3.26 |

| TLA           | w/o $\mathcal{L}_G$   | w/ $\mathcal{L}_G$   |
|---------------|-----------------------|----------------------|
| FID↓<br>sFID↓ | 8.99<br>15.23         | 6.41<br>11.18        |
|               |                       |                      |
| CMA           | w/o $\mathcal{L}_{G}$ | w/ $\mathcal{L}_{G}$ |

Table 6. Component and efficiency comparisons on LDM-4 (LSUN-Bedrooms 256  $\times$  256). The baseline method is direct quantization with the Adaptive Rounding [28] strategy.

Table 7. Influence of global loss supervision on performance.

as the PTQ approach, it achieves better time efficiency with only a 20% increase in GPU memory usage. We also illustrate the contribution of each component to generation performance. The results indicate that sequentially finetuning the time embedding layers, followed by attention-related layers, yields consistent performance improvements.

Tab. 7 presents a comparison of performance with and without the global loss  $\mathcal{L}_G$ . The results indicate that supervising the quantized model using the output difference from the full-precision counterpart is essential for performance improvement, enhancing the FID by 2.58 and 5.21 for TLA and CMA, respectively. However, when the learning process is only supervised by the global loss, we find that the performance degrades by 7.13 FID and 9.39 sFID for TLA, suggesting that the global loss alone is insufficient for optimal performance.

<span id="page-7-1"></span>

| Bit-width | Method                               | Time (h)             | Memory<br>(MB)          | Iters               | FID↓                         |
|-----------|--------------------------------------|----------------------|-------------------------|---------------------|------------------------------|
| W4A8      | EfficientDM<br>Full-finetune<br>Ours | 2.60<br>0.85<br>0.45 | 12004<br>15076<br>12178 | 32k<br>2.2k<br>2.2k | 15.15<br>5.38<br><b>3.82</b> |
| W4A4      | EfficientDM Full-finetune Ours       | 2.60<br>0.85<br>0.45 | 12004<br>15076<br>12178 | 32k<br>2.2k<br>2.2k | 10.60<br>6.36<br><b>6.12</b> |

Table 8. Efficiency comparison with other finetuning methods.

How QuEST adjusts the activation distribution. Our approach is motivated by the imbalanced activation distribution in diffusion models, hence we aim to analyze how our fine-tuning strategy addresses this challenge. As shown in Fig. 2, our method refines the activation distribution, making it more conducive to quantization. Specifically, the activation value ranges shrink from [-10, 34] to [-4, 14] and from [-11, 20] to [-4, 4]. Additionally, the standard deviations decrease from 0.171 to 0.157 and from 0.073 to 0.071, while the mean remains consistent. This results in a more compact activation distribution, effectively reducing both rounding and clipping errors during quantization.

Comparison with precomputed time embeddings. In diffusion models, time embeddings are independent of input conditions and noise. A potential approach is to precompute these embeddings and reuse them directly. However, this strategy overlooks the compatibility between different mod-

ules in a quantized model. We take this into consideration and optimize the time embeddings with  $\arg\min_{\mathbf{w}_l}(\mathcal{L}_{TLA} + \mathcal{L}_G), \quad l \in \mathbb{C}_{TE}$  so that the time embedding layers are also trained to minimize the final prediction error. As shown in Tab. 2, adding this optimization objective enhances quantization performance, even surpassing the full-precision baseline (which uses precomputed features).

Integration with LoRA finetuning. Different ways exist for finetuning quantized models. We further employ QALoRA [10] to finetune on the ImageNet 256×256 dataset. A rank of 32 is used for the LoRA weights, and the parameters are trained over 100 time steps for 160 epochs. We find that integrating the QALoRA technique leads to a 5.62 increase in FID, indicating that finetuning the original layers is a better solution for performance preservation.

Efficiency comparison with other finetuning methods. We compare with EfficientDM and full-finetuning in terms of actual training costs on LDM-4 in Tab. 8. The setting of full-finetuning is aligned with our method. We observe that: compared with EfficientDM, our method requires fewer training iterations and time to obtain better performance with comparable GPU memory cost. Compared with full-finetuning, our method costs less time and memory, as well as achieving better performance. The bottleneck in computational costs becomes more severe when scaled to larger models such as Stable Diffusion. We find that while full-finetuning quickly encounters OOM, our method is able to finetune SD on a single GPU with 48GB memory.

#### 5. Conclusion

We have proposed QuEST, an efficient data-free finetuning framework for low-bit diffusion model quantization. Our method is motivated by the current challenge in low-bit diffusion model quantization and guided by the two underlying properties found in quantized diffusion models. To alleviate the performance degradation, we propose to finetune the time embedding layers and the attention-related layers under the supervision of the full-precision counterpart. Experimental results on three high-resolution image generation tasks (including Stable Diffusion) demonstrate the effectiveness and efficiency of QuEST, achieving low-bit compatibility with less time and memory cost.

Acknowledgments: This research is supported by NSF IIS-2525840, CNS-2432534, ECCS-2514574, NIH 1RF1MH133764-01 and Cisco Research unrestricted gift. This article solely reflects opinions and conclusions of authors and not funding agencies.

# References

- <span id="page-8-27"></span>[1] Shane Barratt and Rishi Sharma. A note on the inception score, 2018. [7](#page-6-2)
- <span id="page-8-13"></span>[2] Jooyoung Choi, Jungbeom Lee, Chaehun Shin, Sungwon Kim, Hyunwoo Kim, and Sungroh Yoon. Perception prioritized training of diffusion models, 2022. [2](#page-1-0)
- <span id="page-8-3"></span>[3] Florinel-Alin Croitoru, Vlad Hondru, Radu Tudor Ionescu, and Mubarak Shah. Diffusion models in vision: A survey. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 45(9):10850–10869, 2023. [1](#page-0-1)
- <span id="page-8-22"></span>[4] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *2009 IEEE conference on computer vision and pattern recognition*, pages 248–255. IEEE, 2009. [7](#page-6-2)
- <span id="page-8-12"></span>[5] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms, 2023. [2](#page-1-0)
- <span id="page-8-0"></span>[6] Prafulla Dhariwal and Alex Nichol. Diffusion models beat gans on image synthesis, 2021. [1,](#page-0-1) [7](#page-6-2)
- <span id="page-8-20"></span>[7] Zhida Feng, Zhenyu Zhang, Xintong Yu, Yewei Fang, Lanxin Li, Xuyi Chen, Yuxiang Lu, Jiaxiang Liu, Weichong Yin, Shikun Feng, Yu Sun, Li Chen, Hao Tian, Hua Wu, and Haifeng Wang. Ernie-vilg 2.0: Improving text-toimage diffusion model with knowledge-enhanced mixtureof-denoising-experts. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 10135–10145, 2023. [5](#page-4-5)
- <span id="page-8-7"></span>[8] Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W Mahoney, and Kurt Keutzer. A survey of quantization methods for efficient neural network inference. In *Low-Power Computer Vision*, pages 291–326. Chapman and Hall/CRC, 2022. [1](#page-0-1)
- <span id="page-8-21"></span>[9] Meng-Hao Guo, Cheng-Ze Lu, Zheng-Ning Liu, Ming-Ming Cheng, and Shi-Min Hu. Visual attention network. *Computational Visual Media*, 9(4):733–752, 2023. [5](#page-4-5)
- <span id="page-8-6"></span>[10] Yefei He, Jing Liu, Weijia Wu, Hong Zhou, and Bohan Zhuang. Efficientdm: Efficient quantization-aware finetuning of low-bit diffusion models, 2023. [1,](#page-0-1) [2,](#page-1-0) [5,](#page-4-5) [7,](#page-6-2) [8](#page-7-2)
- <span id="page-8-9"></span>[11] Yefei He, Luping Liu, Jing Liu, Weijia Wu, Hong Zhou, and Bohan Zhuang. Ptqd: Accurate post-training quantization for diffusion models, 2023. [1,](#page-0-1) [2,](#page-1-0) [7](#page-6-2)
- <span id="page-8-28"></span>[12] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning, 2022. [7](#page-6-2)
- <span id="page-8-25"></span>[13] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium, 2018. [7](#page-6-2)
- <span id="page-8-1"></span>[14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020. [1,](#page-0-1) [2,](#page-1-0) [7](#page-6-2)

- <span id="page-8-18"></span>[15] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021. [2](#page-1-0)
- <span id="page-8-10"></span>[16] Yushi Huang, Ruihao Gong, Jing Liu, Tianlong Chen, and Xianglong Liu. Tfmq-dm: Temporal feature maintenance quantization for diffusion models, 2024. [1,](#page-0-1) [2,](#page-1-0) [3,](#page-2-3) [4,](#page-3-2) [5,](#page-4-5) [7](#page-6-2)
- <span id="page-8-15"></span>[17] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. Quantization and training of neural networks for efficient integer-arithmetic-only inference, 2017. [2](#page-1-0)
- <span id="page-8-24"></span>[18] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2017. [7](#page-6-2)
- <span id="page-8-4"></span>[19] Xiuyu Li, Yijiang Liu, Long Lian, Huanrui Yang, Zhen Dong, Daniel Kang, Shanghang Zhang, and Kurt Keutzer. Q-diffusion: Quantizing diffusion models. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 17535–17545, 2023. [1,](#page-0-1) [2,](#page-1-0) [3,](#page-2-3) [4,](#page-3-2) [7,](#page-6-2) [8](#page-7-2)
- <span id="page-8-11"></span>[20] Yuhang Li, Ruihao Gong, Xu Tan, Yang Yang, Peng Hu, Qi Zhang, Fengwei Yu, Wei Wang, and Shi Gu. Brecq: Pushing the limit of post-training quantization by block reconstruction. *arXiv preprint arXiv:2102.05426*, 2021. [1,](#page-0-1) [2,](#page-1-0) [3,](#page-2-3) [6](#page-5-5)
- <span id="page-8-16"></span>[21] Yanjing Li, Sheng Xu, Baochang Zhang, Xianbin Cao, Peng Gao, and Guodong Guo. Q-vit: Accurate and fully quantized low-bit vision transformer, 2022. [2](#page-1-0)
- <span id="page-8-5"></span>[22] Yanjing Li, Sheng Xu, Xianbin Cao, Baochang Zhang, and Xiao Sun. Q-dm: An efficient low-bit quantized diffusion model. In *NeurIPS 2023*, 2023. [1,](#page-0-1) [2](#page-1-0)
- <span id="page-8-8"></span>[23] Tailin Liang, John Glossner, Lei Wang, Shaobo Shi, and Xiaotong Zhang. Pruning and quantization for deep neural network acceleration: A survey. *Neurocomputing*, 461:370– 403, 2021. [1](#page-0-1)
- <span id="page-8-29"></span>[24] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollar. Microsoft ´ coco: Common objects in context, 2015. [7](#page-6-2)
- <span id="page-8-17"></span>[25] Jiawei Liu, Lin Niu, Zhihang Yuan, Dawei Yang, Xinggang Wang, and Wenyu Liu. Pd-quant: Post-training quantization based on prediction difference metric, 2023. [2](#page-1-0)
- <span id="page-8-23"></span>[26] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on manifolds, 2022. [7](#page-6-2)
- <span id="page-8-14"></span>[27] Hao Ma, Jingyuan Yang, and Hui Huang. Taming diffusion model for exemplar-based image translation. *Computational Visual Media*, 10(6):1031–1043, 2024. [2](#page-1-0)
- <span id="page-8-19"></span>[28] Markus Nagel, Rana Ali Amjad, Mart Van Baalen, Christos Louizos, and Tijmen Blankevoort. Up or down? adaptive rounding for post-training quantization. In *International Conference on Machine Learning*, pages 7197–7206. PMLR, 2020. [3,](#page-2-3) [8](#page-7-2)
- <span id="page-8-26"></span>[29] Charlie Nash, Jacob Menick, Sander Dieleman, and Peter W. Battaglia. Generating images with sparse representations, 2021. [7](#page-6-2)
- <span id="page-8-2"></span>[30] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image syn- ¨ thesis with latent diffusion models, 2021. [1,](#page-0-1) [2,](#page-1-0) [4,](#page-3-2) [7](#page-6-2)

- <span id="page-9-4"></span>[31] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation, 2015. [2](#page-1-0)
- <span id="page-9-1"></span>[32] Yuzhang Shang, Zhihang Yuan, Bin Xie, Bingzhe Wu, and Yan Yan. Post-training quantization on diffusion models. In *CVPR*, 2023. [1,](#page-0-1) [2,](#page-1-0) [4,](#page-3-2) [7](#page-6-2)
- <span id="page-9-3"></span>[33] Junhyuk So, Jungwon Lee, Daehyun Ahn, Hyungjun Kim, and Eunhyeok Park. Temporal dynamic quantization for diffusion models, 2023. [1,](#page-0-1) [2,](#page-1-0) [4](#page-3-2)
- <span id="page-9-2"></span>[34] Yang Sui, Yanyu Li, Anil Kag, Yerlan Idelbayev, Junli Cao, Ju Hu, Dhritiman Sagar, Bo Yuan, Sergey Tulyakov, and Jian Ren. Bitsfusion: 1.99 bits weight quantization of diffusion model, 2024. [1](#page-0-1)
- <span id="page-9-7"></span>[35] Changyuan Wang, Ziwei Wang, Xiuwei Xu, Yansong Tang, Jie Zhou, and Jiwen Lu. Towards accurate data-free quantization for diffusion models, 2023. [2](#page-1-0)
- <span id="page-9-0"></span>[36] Chen Wang, Hao-Yang Peng, Ying-Tian Liu, Jiatao Gu, and Shi-Min Hu. Diffusion models for 3d generation: A survey. *Computational Visual Media*, 11(1):1–28, 2025. [1](#page-0-1)
- <span id="page-9-5"></span>[37] Xiuying Wei, Ruihao Gong, Yuhang Li, Xianglong Liu, and Fengwei Yu. Qdrop: Randomly dropping quantization for extremely low-bit post-training quantization, 2023. [2](#page-1-0)
- <span id="page-9-6"></span>[38] Junyi Wu, Haoxuan Wang, Yuzhang Shang, Mubarak Shah, and Yan Yan. Ptq4dit: Post-training quantization for diffusion transformers, 2024. [2](#page-1-0)
- <span id="page-9-8"></span>[39] Yuewei Yang, Xiaoliang Dai, Jialiang Wang, Peizhao Zhang, and Hongbo Zhang. Efficient quantization strategies for latent diffusion models, 2023. [2](#page-1-0)
- <span id="page-9-10"></span>[40] Fisher Yu, Yinda Zhang, Shuran Song, Ari Seff, and Jianxiong Xiao. Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop. *arXiv preprint arXiv:1506.03365*, 2015. [7](#page-6-2)
- <span id="page-9-9"></span>[41] Zhihang Yuan, Chenhao Xue, Yiqi Chen, Qiang Wu, and Guangyu Sun. Ptq4vit: Post-training quantization framework for vision transformers with twin uniform quantization, 2022. [6](#page-5-5)

# **QuEST: Low-bit Diffusion Model Quantization via Efficient Selective Finetuning**

# Supplementary Material

The supplementary material is organized as follows: Sec. 6 provides comparison with TFMQ-DM; Sec. 7 provides comparison on the low-resolution dataset; Sec. 8 provides the proof and detailed analysis for Theorem 3.2; Sec. 9 presents additional examples of the imbalanced distributions across different models; Sec. 10 highlights the importance of the large values in activations; Sec. 11 offers further generated examples from our method across varying bit-widths; and Sec. 12 discusses limitations and broader considerations.

# <span id="page-10-0"></span>6. More Baseline Comparisons

We further compare with TFMQ [16] below:

| Bedroom  | W8A8         | W4A8        |
|----------|--------------|-------------|
| TFMQ-DM  | 3.14         | 3.68        |
| QuEST    | <b>3.03</b>  | <b>3.26</b> |
| ImageNet | W8A8         | W4A8        |
| TFMQ-DM  | 10.79        | 10.29       |
| QuEST    | <b>10.43</b> | <b>8.48</b> |

Table 9. Comparing TFMQ.

We also supplement the metrics for Table 3:

| W8A8        | sFID ↓ | IS↑  |
|-------------|--------|------|
| QDiffusion  | 8.19   | 2.25 |
| PTQD        | 9.89   | 2.25 |
| EfficientDM | N/A    | N/A  |
| Ours        | 6.86   | 2.27 |
| W4A4        | sFID ↓ | IS↑  |
| QDiffusion  | N/A    | N/A  |
| DTOD        | NT/A   | N/A  |
| PTQD        | N/A    | IN/A |
| EfficientDM | 15.15  | 2.27 |

Table 10. Additional metrics on LSUN-Bedrooms. "N/A" represents generation failure.

# <span id="page-10-1"></span>7. Low-resolution dataset comparison

<span id="page-10-3"></span>We further include experiments on CIFAR10 in Tab. 11.

|             | W8A8 | W4A4  |
|-------------|------|-------|
| Q-Diffusion | 3.75 | N/A   |
| EfficientDM | 3.75 | 10.48 |
| QuEST       | 3.71 | 9.37  |

Table 11. FID comparison on CIFAR10.

<span id="page-10-5"></span>![](_page_10_Figure_14.jpeg)

(a) Activation Distribution on Conditional LDM4 (ImageNet 256 × 256)

![](_page_10_Figure_16.jpeg)

(b) Activation Distribution on Unconditional LDM4 (LSUN-Bedrooms  $256 \times 256$ )

Figure 5. Illustrations of imbalanced activation distributions on conditional LDM4 (ImageNet  $256 \times 256$ ) and unconditional LDM4 (LSUN-Bedrooms  $256 \times 256$ ).

#### <span id="page-10-2"></span>8. Proof for Theorem 3.2

We provide the detailed proof for Theorem 3.2 here. The notations are consistent with the ones in the main paper.

Since the perturbation  $\Delta$  is too large for accurate Taylor expansion, we can resolve it by introducing a new perturbation  $\epsilon = \Delta/K$ , where we divide  $\Delta$  by a constant K so that  $\epsilon$  is small enough for approximation. Then, Eq. (8) is rewritten as follows:

$$\mathbb{E}[L(z_{n,t} + \Delta; \mathbf{w})] - \mathbb{E}[L(z_{n,t}; \mathbf{w})]$$

$$= \sum_{i=1}^{K} \left( \mathbb{E}[L(z_{n,t} + \frac{i}{K}\Delta; \mathbf{w})] - \mathbb{E}[L(z_{n,t} + \frac{i-1}{K}\Delta; \mathbf{w})] \right)$$

$$\approx \sum_{i=1}^{K} \left( \epsilon^{T} \overline{\mathbf{g}}^{(z_{n,t} + (i-1)\epsilon)} + \frac{1}{2} \epsilon^{T} \overline{\mathbf{H}}^{(z_{n,t} + (i-1)\epsilon)} \epsilon \right), \quad (11)$$

<span id="page-10-4"></span>where the approximation step follows Taylor expansion and only the first two main components are kept. The first term in Eq. (11) cannot be ignored because samples such as  $z_{n,t}+(i-1)\epsilon$  may not be included in the learned distribution of the model. The second term can still be minimized by reconstruction since only the difference between quantized model output and ground-truth matters. In the following, we temporarily exclude the second term for simplicity since it can always be minimized through aligning the activation outputs.

<span id="page-11-4"></span>![](_page_11_Picture_0.jpeg)

Randomly corrupt 1 token

![](_page_11_Picture_2.jpeg)

Corrupt 1 maximum value token

![](_page_11_Picture_4.jpeg)

Randomly corrupt 3 tokens

![](_page_11_Picture_6.jpeg)

Corrupt 3 maximum value tokens

Figure 6. Comparison of different corruptions made on different tokens.

Given the objective function (MSE loss) of diffusion models, we analyze that:

 $\sum_{i=1}^{K} \epsilon^{T} \overline{\mathbf{g}}^{(z_{n,t}+(i-1)\epsilon)} = 2\epsilon^{T} \sum_{i=1}^{K} (\tilde{z}_{n-1,t}^{i} \cdot \mathbf{w}_{n} - \overline{z}_{n,t})$   $\approx 2\epsilon^{T} \sum_{i=1}^{K} (\tilde{z}_{n-1,t}^{i} \cdot \mathbf{w}_{n} - z_{\text{FP}}), \quad (12)$ 

where  $\mathbf{w}_n$  is the weight for layer n,  $\tilde{z}_{n-1,t}^i$  is the activation of the (n-1)th layer in a quantized model to get  $z_{n,t}+(i-1)\epsilon$ . Ground-truth  $\overline{z}_{n,t}$  can be approximated by the full-precision output  $z_{\mathrm{FP}}$ . We see that  $\tilde{z}_{n-1,t}^i$  and  $z_{\mathrm{FP}}$  cannot be changed, thus to minimize Eq. (12), we need to finetune  $\mathbf{w}_n$ . From a general perspective, Eq. (12) also indicates that the model has not converged well to a local minimum given the perturbed inputs, thus when we finetune the model layers given the quantized inputs, we are actually training the model towards convergence over new samples and increasing its robustness.

# <span id="page-11-0"></span>9. Examples of Imbalanced Activation Distributions

Apart from Fig. 2, we show that the imbalance in the activation distribution is a common phenomenon in different model structures and datasets. In Fig. 5, we show more re-

sults of activation distributions of latent diffusion models on ImageNet 256  $\times$  256 and LSUN-Bedrooms 256  $\times$  256.

### <span id="page-11-1"></span>10. Importance of large values in activations

<span id="page-11-3"></span>As shown in Fig. 2, quite a few values are rather large and diversely distributed. These values pose difficulties on activation quantization, and being rather important and not negligible. To demonstrate this, we corrupt certain tokens in the activation outputs of the diffusion model and check the corresponding generated images. The corruption is done by setting the token values as all zeros. As shown in Fig. 6, we compare two settings: (1) corrupt a certain number of tokens randomly; (2) corrupt the same number of the tokens with the largest values.

We see that when corrupting randomly, generation performance is hardly effected. However, corrupting the same amount of tokens (even only one token) with the largest values leads to significantly degenerated images.

#### <span id="page-11-2"></span>11. More generated image examples

#### 11.1. Unconditional Image Generation

The generated images for LSUN-Bedrooms  $256 \times 256$  under different bit-widths are shown in Fig. 7. Images for LSUN-Churches  $256 \times 256$  are shown in Fig. 9.

<span id="page-12-1"></span>![](_page_12_Picture_0.jpeg)

(a) Full Precision

![](_page_12_Picture_2.jpeg)

(b) W8A8

![](_page_12_Picture_4.jpeg)

(c) W4A8

![](_page_12_Picture_6.jpeg)

Figure 7. Unconditional image generation examples for LSUN-Bedrooms 256×256.

# 11.2. Class-conditional image generation

Fig. [10](#page-13-1) shows the generated images for 3 different classes.

# 11.3. Text-to-image generation

Fig. [8](#page-12-2) shows the generated images using Stable Diffusion v1.4 under different bit-width.

<span id="page-12-2"></span>![](_page_12_Figure_12.jpeg)

Figure 8. Text-to-image generation results on Stable Diffusion.

# <span id="page-12-0"></span>12. Limitations and Broader Impacts

The primary objective of this paper is to further the research in enhancing the efficiency of diffusion models. While it confronts societal consequences akin to those faced by research on generative models, it is important to recognize the potential impacts that quantized models could have on current techniques, including watermarking and safety checking. Inappropriate integration of current methodologies may result in unforeseen performance issues, a factor that deserves attention and awareness.

<span id="page-13-1"></span><span id="page-13-0"></span>![](_page_13_Figure_0.jpeg)

Figure 9. Unconditional image generation examples for LSUN-Churches 256×256.

Figure 10. Conditional image generation results for ImageNet 256×256.