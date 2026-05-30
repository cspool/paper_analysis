# <span id="page-0-1"></span>*PassionSR*: Post-Training Quantization with Adaptive Scale in One-Step Diffusion based Image Super-Resolution

Libo Zhu<sup>1</sup> , Jianze Li<sup>1</sup> , Haotong Qin2\*, Wenbo Li<sup>3</sup> , Yulun Zhang1\*, Yong Guo<sup>4</sup> , Xiaokang Yang<sup>1</sup>

<sup>1</sup>Shanghai Jiao Tong University, <sup>2</sup>ETH Zurich, ¨ <sup>3</sup>Chinese University of Hong Kong, <sup>4</sup>Max Planck Institute for Informatics

# Abstract

*Diffusion-based image super-resolution (SR) models have shown superior performance at the cost of multiple denoising steps. However, even though the denoising step has been reduced to one, they require high computational costs and storage requirements, making it difficult for deployment on hardware devices. To address these issues, we propose a novel post-training quantization approach with adaptive scale in one-step diffusion (OSD) image SR, PassionSR. First, we simplify OSD model to two core components, UNet and Variational Autoencoder (VAE) by removing the CLIPEncoder. Secondly, we propose Learnable Boundary Quantizer (LBQ) and Learnable Equivalent Transformation (LET) to optimize the quantization process and manipulate activation distributions for better quantization. Finally, we design a Distributed Quantization Calibration (DQC) strategy that stabilizes the training of quantized parameters for rapid convergence. Comprehensive experiments demonstrate that PassionSR with 8-bit and 6-bit obtains comparable visual results with full-precision model. Moreover, PassionSR achieves significant advantages over recent leading low-bit quantization SR methods. Code will be at* <https://github.com/libozhu03/PassionSR>*.*

# 1. Introduction

Image super-resolution (SR) is a fundamental and challenging task in computer vision, aiming to reconstruct highresolution (HR) images from low-resolution (LR) inputs by recovering lost structures and details. Over time, various image SR models have been developed to tackle this challenge. Early image SR methods [\[2,](#page-8-0) [4,](#page-8-1) [7,](#page-8-2) [53\]](#page-9-0) focus on simple synthetic degradations, such as bicubic downsampling, to create LR-HR pairs. GAN approaches [\[21,](#page-8-3) [37,](#page-9-1) [49\]](#page-9-2) introduce more complex degradation processes. Although these methods have advanced image quality significantly, when applied to real-world datasets, they encounter some issues, such as training stability and performance drop, showing the limitations of early image SR methods.

<span id="page-0-0"></span>![](_page_0_Picture_9.jpeg)

Figure 1. Visual comparison (×4) between full-precision (FP) multi-step and one-step diffusion SR models and our 8-bit quantized PassionSR. Compared to FP models, PassionSR achieves about 81.77% params reduction and 4× speedup.

Recently, diffusion-based image SR models have been attracting researchers' attention. Diffusion models exhibit strong performance across various tasks, including image SR (see Fig. [1\)](#page-0-0), due to their excellent generation ability and robust training stability. Diffusion-based SR models [\[23,](#page-8-4) [36,](#page-9-4) [43\]](#page-9-5) leverage their capability to capture complex data distributions, achieving superior perceptual quality.

However, achieving high-quality results with diffusion models comes at the expense of substantial computational demands, latency, and storage requirements. It poses significant challenges for deployment on hardware devices. Several strategies have been developed to improve the efficiency of diffusion models. For instance, fast diffusion samplers [\[31,](#page-9-6) [54\]](#page-9-7) and distillation techniques [\[32\]](#page-9-8) have reduced the number of denoising steps. With the advance of score distillation-based methods [\[40,](#page-9-9) [47\]](#page-9-10), one-step diffusion (OSD) image super-resolution (OSDSR) models have become feasible, such as SinSR [\[38\]](#page-9-11), OSEDiff [\[42\]](#page-9-3), and DFOSD [\[17\]](#page-8-5). Despite reducing denoising steps to one, OS-DSR models still face high computational costs and storage requirements. For example, in OSEDiff [\[42\]](#page-9-3), the parameters and Ops are 1,303M and 4,523G, respectively. The high computational complexity causes a significant burden for mobile devices like smartphones. This problem hinders the widespread use of diffusion models, especially in scenarios with limited storage and computational resources.

<sup>\*</sup>Corresponding authors: Haotong Qin and Yulun Zhang

<span id="page-1-2"></span><span id="page-1-0"></span>

| Components | UNet    | VAE       | DAPE    | ClipEncoder | Total     |
|------------|---------|-----------|---------|-------------|-----------|
| Params (M) | 865,785 | 83,614    | 160,335 | 193,055     | 1,302,789 |
| MACs (G)   | 339.241 | 1,781.123 | 126.591 | 14.856      | 2,261.811 |

Table 1. Params and FLOPs statistics in OSEDiff [\[42\]](#page-9-3).

To compress the OSDSR model, quantization stands out as an effective approach. Model quantization [\[6,](#page-8-6) [11,](#page-8-7) [16\]](#page-8-8) is a powerful compression technique that reduces weights and activations from full-precision (FP) to low-bit precision. Thereby, it significantly lowers storing and computational demands by substituting floating-point operations with integer operations. However, performance gap between quantized and FP versions is inevitable. Minimizing this gap is crucial for the successful application of quantization techniques to OSDSR models.

Although current low-bit quantization strategies [\[10,](#page-8-9) [18,](#page-8-10) [20,](#page-8-11) [28\]](#page-9-12) have achieved promising results for multi-step diffusion quantization, significant performance drops still occur when applied to OSDSR models. Additionally, existing strategies are often hard to achieve optimal compression rates. We encounter three primary challenges:

*I. Complex Model Structure.* Unlike many multi-step diffusion models, the OSDSR model includes numerous submodules. *(i)* In the previous multi-step diffusion quantization methods, most attention has been paid to UNet quantization while Variational Autoencoder (VAE) keeps FP. However, in OSDSR models, because UNet inference steps are reduced to one, it is essential to quantize VAE, which accounts for over 80% of the computational load, as shown in Tab. [1.](#page-1-0) *(ii)* We need to design special calibration strategies for branch modules (*e.g.*, DAPE, CLIPEncoder), which brings lots of difficulties for quantization.

*II. Transition from Multi-Step to One-Step.* Most existing quantization techniques are designed for multi-step diffusion models and incorporate special techniques tailored to their multi-step nature. These techniques do not perform as effectively on OSDSR as on multi-step models. Many of them are even infeasible on OSDSR (see Fig. [2\)](#page-1-1). It means we need to design a new calibration strategy tailored for OSDSR, taking better advantage of its features rather than applying the previous methods on OSDSR directly.

*III. Imbalanced Activation Distribution.* Imbalanced activation distribution is a prevalent issue in model quantization. The presence of excessive outliers complicates the determination of optimal quantized parameters in traditional post-training quantization (PTQ) methods. And quantization-aware training (QAT) often encounters convergence challenges due to the quantization function and high demands in training time and memory usage.

Based on the above analyses, we propose a novel *P*osttraining quantization method with *A*daptive *S*cale in one-*S*tep diffus*ION* based image *S*uper-*R*esolution, named *PassionSR*. In this work, we select OSEDiff [\[42\]](#page-9-3) as our quantization backbone due to its excellent performance and high

<span id="page-1-1"></span>![](_page_1_Picture_8.jpeg)

LSQ [\[8\]](#page-8-13) Q-Diffusion [\[18\]](#page-8-10) EfficientDM [\[9\]](#page-8-14) PassionSR (ours) Figure 2. Visual comparison (×4) of one-step diffusion SR models. We use OSEDiff as a 32-bit full-precision (FP) reference and provide 6-bit quantized version with different methods.

inference speed. *Firstly*, we perform a pruning operation to simplify the model to its two core components, the UNet and VAE, with minimal or even no performance drop. We name the FP model structure after pruning as PassionSR-FP. It is easier for us to design calibration strategies. *Secondly*, we propose a *Learnable Boundary Quantizer* (LBQ) and *Learnable Equivalent Transformation* (LET) to the quantization process. LBQ allows for training-based optimization. While, LET enables control over the distribution of activations without any additional computational expense, which is achieved through an adaptable scale parameter. The training process renders this scale parameter adaptive, which proves more effective than traditional initialization methods. *Thirdly*, we design a calibration strategy, Distributed Quantization Calibration (DQC), that stabilizes the training of quantized parameters and promotes rapid convergence, achieving QAT-level effectiveness with PTQ efficiency.

Comprehensive experiments (*i.e.*, Figs. [1](#page-0-0) and [2\)](#page-1-1) indicate that PassionSR incurs a performance drop of less than 1% at 8-bit precision and maintains relatively high performance even at 6-bit precision. Compared to recent leading diffusion quantization methods, PassionSR demonstrates significant performance advantages across various bit widths in one-step diffusion (OSD) model quantization. PassionSR delivers outstanding visual quality over other quantization methods. When compared with OSEDiff, our PassionSR achieves about 80∼85% parameters compression and operations reduction. Overall, our contributions are as follows:

- We propose a low-bit quantized OSDSR model, PassionSR. To the best of our knowledge, this is the first work to investigate low-bit quantization (*e.g.*, 6-bit and 8-bit) for OSDSR in a PTQ manner.
- We design a UNet-VAE model (*i.e.*, PassionSR-FP) structure that maintains high performance while simplifying the overall model to only UNet and VAE.
- We propose the Learnable Equivalent Transformation (LET) for the quantization of OSD models. Distributed Quantization Calibration (DQC) is designed to stabilize the training and accelerate the convergence.
- Our PassionSR achieves perceptual performance largely comparable to that of a full-precision model at 8-bit and 6-bit precision and obtains better performance and higher scores over other quantization methods.

# <span id="page-2-1"></span>2. Related Work

### 2.1. Single Image Super-Resolution

Single image super-resolution (SR) aims to recover highresolution (HR) images from low-resolution (LR) inputs with unknown and complex degradation patterns. Numerous models have been developed to address this challenge. In addition to early SR models [\[3,](#page-8-15) [15,](#page-8-16) [52\]](#page-9-13) and GAN-based approaches [\[21,](#page-8-3) [37,](#page-9-1) [49\]](#page-9-2), stable diffusion (SD) [\[27\]](#page-8-17) has emerged as a powerful technique due to its robust capability in capturing complex data distributions and providing strong generative priors. Related methods, including StableSR [\[36\]](#page-9-4), DiffBIR [\[23\]](#page-8-4), and SeeSR [\[43\]](#page-9-5), enhance the perceptual quality of generated images. However, their multistep processes introduce higher latency, which hinders realtime applications. To address this limitation, one-step diffusion (OSD) models, such as SinSR [\[38\]](#page-9-11) and OSEDiff [\[42\]](#page-9-3), have been developed to reduce inference latency by accelerating the process to a single step.

## 2.2. Model Quantization

Model quantization is a critical technique for accelerating models by reducing computational costs and inference time. Depending on whether the model's weights are retrained, quantization methods are divided into two categories: posttraining quantization (PTQ) and quantization-aware training (QAT). PTQ is highly time-efficient as it only calibrates the quantized parameters rather than finetunes the entire model. ZeroQuant [\[46\]](#page-9-14) calibrates quantized parameters without additional calibration datasets, and BRECQ [\[19\]](#page-8-18) introduces a block-wise reconstruction PTQ method. QAT can achieve higher accuracy but incurs high training costs. As a representative quantization method, LSQ [\[8\]](#page-8-13) improves low-bit quantization with a learnable step size.

### 2.3. Quantization of Diffusion Models

As diffusion models evolve rapidly, researchers have focused on improving their efficiency through quantization. PTQ4DM [\[28\]](#page-9-12) first investigates quantized diffusion models, identifying key challenges to overcome. Further works, including Q-Diffusion [\[18\]](#page-8-10), PTQD [\[10\]](#page-8-9), and QAT methods like Q-DM [\[20\]](#page-8-11), have made significant progress by developing specialized calibration strategies tailored to diffusion models. Notably, TDQ [\[30\]](#page-9-15) utilizes an MLP layer to predict quantized parameters, and APQ-DM [\[33\]](#page-9-16) designs a distribution-aware quantization approach to minimize quantization error. Additionally, QALoRA [\[44\]](#page-9-17), is a notable quantization method for large language models, reducing quantization error by finetuning LoRA layers along with quantized parameters. It is adopted to quantize diffusion models in EfficientDM [\[9\]](#page-8-14). QuEST [\[34\]](#page-9-18) finds that layers like the feedforward layer are sensitive to quantization. QuEST improves performance by selectively retraining these layers. These methods have advanced low-

<span id="page-2-0"></span>![](_page_2_Figure_7.jpeg)

Figure 3. Diffusion-based image SR acceleration.

bit quantization for multi-step diffusion models. However, there are few works specifically addressing the lowbit quantization of one-step diffusion (OSD) models, which are significantly different from multi-step diffusion models. We follow the strategy in Fig. [3](#page-2-0) to accelerate the diffusionbased SR models, especially the OSD models.

# 3. Methods

### 3.1. Preliminaries

Diffusion Models. Diffusion models [\[27\]](#page-8-17) are generative techniques that gradually introduce noise into data and then learn to invert this process to create new samples. The process starts with a real data distribution pdata(x), where data x<sup>0</sup> is gradually transformed into noise over several time steps, t = 1, 2, . . . , T. At each step, the data evolves towards randomness with noise controlled by a time-dependent parameter βt, which increases with each step. After enough steps, the data distribution becomes close to the standard normal distribution, i.e., x<sup>t</sup> ∼ N (0, I). Then process is reversed to recover the original data x<sup>0</sup> from the noisy data x<sup>t</sup> by training a neural model to predict xt−<sup>1</sup> from xt. The network learns the denoising function, parameterized by θ, which predicts the clean data mean µθ(xt, t) and the noise level σ 2 (t). By iterating this process, new samples are generated from random noise, gradually refined step by step. This denoising process allows the model to generate highly realistic samples.

Model Quantization. Model quantization uses both the scale factor and zero point bias to handle the shift in the data distribution. It reduces memory consumption and computation time by mapping model parameters and activations to low-bit integers. Given a floating-point vector x, the quantization operation is as follows:

$$\hat{x} = Q(\mathbf{x}, s, z) = s \cdot \text{Clip}\left(\frac{\mathbf{x} - z}{s}, l, u\right) + z,$$
 (1)

where s is the scaling factor that controls quantization precision, and the zero-point bias z shifts the data before scaling. Clip(·, l, u) bounds the quantized values within the range from the lower bounds l to upper bounds u.

Since quantization involves the non-differentiable rounding operations, the straight-through estimator (STE [\[24\]](#page-8-19)) is commonly used to approximate gradients:

$$\frac{\partial L(x)}{\partial x} \approx \begin{cases} 1 & \text{if } x \in [l, u], \\ 0 & \text{otherwise.} \end{cases}$$
 (2)

<span id="page-3-2"></span><span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Figure 4. Overview of our PassionSR. Step 1: we simplify OSEDiff [42] by removing DAPE and CLIP Encoder, obtaining PassionSR-FP. Step 2: the quantizer we use has two key trainable parts, consisting of the Learnable Boundary Quantizer and Learnable Equivalent Transformation. Step 3: we design a distributed calibration strategy and special loss function to accelerate convergence of calibration.

#### 3.2. UNet-VAE Model Structure

Based on OSEDiff [42], we obtain a full-precision (FP) model PassionSR-FP by simplifying the original design to only UNet and VAE while maintaining comparable performance. The details of the simplified model structure are shown in Step 1 of Fig. 4. Compared to OSEDiff, we replace the DAPE-CLIPEncoder branch with a constant embedding, preprocessed by the ClipEncoder using an empty string. Owing to the similar basic components, we can adopt similar calibration strategy on them.

#### 3.3. Learnable Quantized Parameter Strategy

To minimize performance drop in quantization, we introduce learnable quantized parameters and use a preconstructed small calibration dataset to guide the training, which is both time- and memory-efficient. It is important to note that only the quantized parameters are trained, while the original weight parameters remain unchanged. Step 2 in Fig. 4 visualizes the two key components with trainable parameters of our quantizer, LBQ and LET.

#### 3.3.1 Learnable Boundary Quantizer (LBQ)

To simulate the quantization loss, we apply fake quantization [12] to both activations and weights. The quantization and dequantization processes are defined as Eq. (3). Using

it we define the Learnable Boundary Quantizer

<span id="page-3-1"></span>
$$\begin{cases} X_{c} = \mathbf{Clip}(X, B_{l}, B_{u}), \alpha = \frac{B_{u} - B_{l}}{2^{N} - 1}, \beta = B_{l}, \\ X_{I} = \left\lfloor \frac{X_{c} - \beta}{\alpha} \right\rfloor, X_{q} = \alpha X_{I} + \beta, \end{cases}$$
(3)

where  $X_q$  is the fake-quantized value used to simulate quantization error. The function  $\mathbf{Clip}(X, B_1, B_u)$  is defined as  $\max(\min(X, B_u), B_1)$ , and  $\lfloor x \rceil$  rounds the input x to the nearest integer. The parameters  $B_1$  and  $B_u$ , representing the lower and upper boundaries, are the only trainable parameters, deciding the function of whole LBQ.

#### 3.3.2 Learnable Equivalent Transformation (LET)

Inspired by SmoothQuant [25] and OmniQuant [29], we introduce channel-wise trainable scale and shift factors to adjust the activation distribution, effectively addressing challenges posed by outliers during quantization.

The fundamental layers to be quantized in diffusion models include the linear layer, convolution layer, and matrix multiplication within the attention layer. We apply equivalent transformations to these layers, balancing the distribution of activations and weights and making the model more suitable for quantization. To maximize its potential to reduce performance drop, finetuning is required to obtain better parameters in LET.

<span id="page-4-4"></span>Linear Layer. In a typical linear layer, whose input dimension is  $C_{in}$  and output dimension is  $C_{out}$ , the input matrix  $X \in \mathbb{R}^{N \times C_{\text{in}}}$  is multiplied by the weight matrix  $W \in \mathbb{R}^{C_{\text{in}} \times C_{\text{out}}}$ , along with a bias matrix  $B \in \mathbb{R}^{1 \times C_{\text{out}}}$ , resulting in an output matrix  $Y \in \mathbb{R}^{N \times C_{\text{out}}}$ .

To introduce equivalent transformations in the linear layer, we apply a learnable scale factor  $s \in \mathbb{R}^{1 \times C_{\mathrm{in}}}$  and an offset factor  $\delta \in \mathbb{R}^{1 \times C_{\text{in}}}$  to the input X. To preserve the output Y, corresponding transformations are applied to Wand B, resulting in transformed input X, weight W, and bias B, as shown in Eq. (4):

<span id="page-4-0"></span>
$$\tilde{W} = s \odot W, \tilde{X} = (X - \delta) \oslash s, \tilde{B} = B + \delta W,$$
 (4)

where  $\odot$ ,  $\oslash$  are element-wise multiplication and division.

After applying the quantizer LBQ to the linear layer, the simulated quantized output  $Y_q$  is expressed as Eq. (5). And Eq. (6) refers to full precision output  $Y_{\rm fp}$ :

<span id="page-4-1"></span>
$$\begin{cases} Y_{\rm q} = Q_{\rm a}(\tilde{X})Q_{\rm w}(\tilde{W}) + Q_{\rm w}(\tilde{B}), & (5) \\ Y_{\rm fp} = \tilde{X}\tilde{W} + \tilde{B} = XW + B = Y, & (6) \\ \text{where } Q_{\rm a} \text{ and } Q_{\rm w} \text{ are the activation and weight quantizers.} \end{cases}$$

Convolution Layer. Equivalent transformations are applied along the channel dimension. The operations and quantization methods are similar to those in the linear layer by replacing matrix multiplication with convolution.

Attention Operation. In diffusion models, the attention operation is crucial for quantization due to the emergence of outliers following the softmax operation. It has been observed in vision transformers [48]. We apply similar transformations to the matrix multiplication of the Q, K, and Vmatrices in the transformer blocks. For example, the multiplication of the Q and K matrices can be transformed as shown in Eq. (7). After this transformation, the quantizer is applied to the transformed matrices Q and K, yielding the simulated quantized output  $P_q$ :

<span id="page-4-2"></span>
$$\begin{cases} \tilde{Q} = Q \oslash s, \quad s \odot K = \tilde{K}, \\ P_{q} = \text{Softmax}(Q_{\text{al}}(\tilde{Q})Q_{\text{a2}}(\tilde{K}^{\top})). \end{cases}$$
 (7)

It is worth noting that the scale factor s and offset  $\delta$  used in activation conversion can be merged into the preceding linear, convolution, or normalization layers. s and  $\delta$  used in weight and bias conversion can be incorporated directly into the weights and biases. This integration brings no additional memory or computation costs, making it hardwarefriendly. Hardware experiments by AWQ [22] have confirmed the compatibility of this equivalent transformation when employing quantized models to hardware devices.

### 3.4. Quantization Calibration Design

Obtaining optimal quantized parameters is critical in any quantization task. With trainable parameters in our quantizers, we design a quantization calibration strategy to minimize quantization error with low latency and memory costs. The calibration pipeline is shown in Step 3 of Fig. 4.

<span id="page-4-3"></span>![](_page_4_Figure_13.jpeg)

Figure 5. Loss comparison between w/ and w/o DQC

#### 3.4.1 Distributed Quantization Calibration (DQC)

Given the properties of the rounding function, the training process in model quantization tends to be unstable. This instability will get worse when calibrating the boundaries in LBQ and the scale factor in LET at the same time. Therefore, we propose a Distributed Quantization Calibration (DQC) strategy that divides the entire calibration process into two stages (i.e., Step 3 of Fig. 4). We reinitialize the LBQ to adapt to the new vector for quantization when the scale factors and offsets in LET are updated in Stage 1. Our DQC strategy significantly accelerates convergence and stabilizes the training process, as shown in Fig. 5.

#### 3.4.2 Loss Design for UNet and VAE

We further use a model-wise quantization calibration strategy, which ensures that each module's quantized output serves as input for the next module. It can avoid the accumulation of quantization errors across different modules.

The design of the loss function is crucial for calibrating quantized parameters. We use different loss functions for the UNet and VAE. We design the loss functions for the VAE encoder and decoder expressed as follows:

$$\begin{cases} \mathcal{L}_{\text{VAE}_{e}} &= \|V_{q_{e}}(X_{\text{fp}}) - V_{\text{fp}_{e}}(X_{\text{fp}})\|_{2}, \\ \mathcal{L}_{\text{VAE}_{d}} &= \|V_{q_{d}}(X_{q}) - V_{\text{fp}_{d}}(X_{\text{fp}})\|_{2}, \end{cases}$$
(8)

where  $V_{\rm q}$  denotes the quantized VAE,  $V_{\rm fp}$  represents the fullprecision VAE, and  $[]_e/[]_d$  indicates the encoder or decoder of the VAE.  $\|\cdot\|_2$  represents the mean square error (MSE) loss.  $X_q$  is the quantized input, the output of the previous quantized modules, while  $X_{\rm fp}$  is the full-precision input, the output of the previous full-precision modules.

For one-step diffusion (OSD) models, the time-step and noise level  $(1-\hat{\alpha})$  are constant. So the transformation function  $I(Z_1, \varepsilon)$  from the predicted noise  $\varepsilon(Z_1)$  and input latent feature  $Z_1$  to the output latent feature  $Z_h$  can be defined as:

$$I(Z_{\rm l},\varepsilon) = Z_{\rm h} = \sqrt{\frac{1}{\hat{\alpha}}} Z_{\rm l} - \sqrt{\frac{1-\hat{\alpha}}{\hat{\alpha}}} \varepsilon(Z_{\rm l}).$$
 (9)

We use MSE in latent feature space, leveraging the transformation function  $I(Z_1, \varepsilon)$  to facilitate smoother gradient descent and faster model convergence. We design the loss function for UNet as follows:

$$\mathcal{L}_{\text{Unet}} = \|I(Z_{\text{lq}}, \varepsilon_{\text{q}}) - I(Z_{\text{l}}, \varepsilon_{\text{fp}})\|_{2}. \tag{10}$$

<span id="page-5-2"></span><span id="page-5-0"></span>

| Datasets  | Bits   | Methods          | PSNR↑ | SSIM↑  | LPIPS↓ | DISTS↓ | NIQE↓ | MUSIQ↑ | MANIQA↑ | CLIP-IQA↑ |
|-----------|--------|------------------|-------|--------|--------|--------|-------|--------|---------|-----------|
| W32A32    |        | OSEDiff [42]     | 25.27 | 0.7379 | 0.3027 | 0.1808 | 4.355 | 67.43  | 0.4766  | 0.6835    |
|           |        | PassionSR-FP     | 25.39 | 0.7460 | 0.2984 | 0.1813 | 4.453 | 67.05  | 0.4680  | 0.6796    |
|           |        | MaxMin [12]      | 23.16 | 0.6875 | 0.5463 | 0.2879 | 7.932 | 32.92  | 0.1849  | 0.2363    |
|           |        | LSQ [8]          | 15.39 | 0.3375 | 0.9944 | 0.5427 | 10.08 | 50.11  | 0.3533  | 0.3173    |
| W8A8      |        | Q-Diffusion [18] | 24.88 | 0.6967 | 0.4993 | 0.2696 | 8.437 | 44.69  | 0.2352  | 0.5604    |
|           |        | EfficientDM [9]  | 14.77 | 0.4253 | 0.5478 | 0.3462 | 7.526 | 44.75  | 0.2568  | 0.4000    |
| RealSR    |        | PassionSR (ours) | 25.67 | 0.7499 | 0.3140 | 0.1932 | 5.654 | 65.88  | 0.4437  | 0.6912    |
|           |        | MaxMin [12]      | 15.55 | 0.2417 | 0.8018 | 0.4449 | 9.263 | 42.15  | 0.2791  | 0.4174    |
|           |        | LSQ [8]          | 13.73 | 0.1081 | 1.0900 | 0.5450 | 8.430 | 53.61  | 0.3036  | 0.4396    |
|           | W6A6   | Q-Diffusion [18] | 19.75 | 0.4727 | 0.6877 | 0.4024 | 7.381 | 56.46  | 0.4380  | 0.6439    |
|           |        | EfficientDM [9]  | 14.75 | 0.4386 | 0.5233 | 0.3451 | 7.497 | 42.97  | 0.2498  | 0.3740    |
|           |        | PassionSR (ours) | 25.15 | 0.7196 | 0.4199 | 0.2592 | 8.618 | 44.43  | 0.2131  | 0.4612    |
|           |        | OSEDiff [42]     | 25.57 | 0.7885 | 0.3447 | 0.1808 | 4.371 | 37.22  | 0.4794  | 0.7540    |
|           | W32A32 | PassionSR-FP     | 26.70 | 0.7978 | 0.3339 | 0.1765 | 4.336 | 37.03  | 0.4686  | 0.7520    |
|           |        | MaxMin [12]      | 24.97 | 0.7989 | 0.5091 | 0.2921 | 8.215 | 24.05  | 0.1846  | 0.3163    |
|           |        | LSQ [8]          | 14.56 | 0.1795 | 1.1661 | 0.592  | 10.19 | 29.07  | 0.4010  | 0.3970    |
|           | W8A8   | Q-Diffusion [18] | 27.14 | 0.7184 | 0.4765 | 0.2895 | 9.861 | 26.44  | 0.2284  | 0.5608    |
|           |        | EfficientDM [9]  | 15.55 | 0.4183 | 0.6291 | 0.3555 | 6.859 | 28.61  | 0.2468  | 0.4150    |
| DRealSR   |        | PassionSR (ours) | 27.41 | 0.8146 | 0.3422 | 0.1918 | 6.070 | 33.56  | 0.4286  | 0.7554    |
|           |        | MaxMin [12]      | 13.08 | 0.2291 | 0.8131 | 0.5077 | 10.51 | 35.83  | 0.2702  | 0.3864    |
|           |        | LSQ [8]          | 12.95 | 0.0934 | 1.1890 | 0.5833 | 8.591 | 26.39  | 0.2911  | 0.5600    |
|           | W6A6   | Q-Diffusion [18] | 21.75 | 0.6096 | 0.7008 | 0.4039 | 6.854 | 24.39  | 0.4109  | 0.6696    |
|           |        | EfficientDM [9]  | 15.07 | 0.4287 | 0.6127 | 0.357  | 6.690 | 28.37  | 0.2351  | 0.3973    |
|           |        | PassionSR (ours) | 26.62 | 0.7984 | 0.4429 | 0.2571 | 8.484 | 26.26  | 0.1824  | 0.4358    |
|           |        | OSEDiff [42]     | 24.95 | 0.7154 | 0.2325 | 0.1197 | 3.616 | 68.92  | 0.4340  | 0.6842    |
|           | W32A32 | PassionSR-FP     | 25.16 | 0.7221 | 0.2373 | 0.1185 | 3.573 | 69.27  | 0.4402  | 0.6958    |
| DIV2K val |        | MaxMin [12]      | 22.33 | 0.6618 | 0.5639 | 0.2731 | 7.563 | 33.68  | 0.1913  | 0.2818    |
|           |        | LSQ [8]          | 13.90 | 0.2537 | 0.9932 | 0.5515 | 9.578 | 48.11  | 0.3512  | 0.3246    |
|           | W8A8   | Q-Diffusion [18] | 24.20 | 0.6813 | 0.3997 | 0.2400 | 7.955 | 51.95  | 0.2709  | 0.6243    |
|           |        | EfficientDM [9]  | 15.24 | 0.4954 | 0.6041 | 0.3374 | 6.856 | 48.78  | 0.2685  | 0.4235    |
|           |        | PassionSR (ours) | 25.11 | 0.7199 | 0.2496 | 0.1277 | 4.424 | 67.92  | 0.3993  | 0.6939    |
|           |        | MaxMin [12]      | 11.66 | 0.1606 | 0.8509 | 0.4966 | 11.30 | 45.47  | 0.2764  | 0.3523    |
|           |        | LSQ [8]          | 12.21 | 0.0858 | 1.0695 | 0.5424 | 8.564 | 52.74  | 0.2872  | 0.4692    |
|           | W6A6   | Q-Diffusion [18] | 18.92 | 0.4939 | 0.6227 | 0.3718 | 6.162 | 51.50  | 0.3946  | 0.5814    |
|           |        | EfficientDM [9]  | 15.09 | 0.4991 | 0.5953 | 0.3292 | 6.900 | 46.01  | 0.2570  | 0.4007    |
|           |        | PassionSR (ours) | 24.34 | 0.7097 | 0.3440 | 0.2075 | 7.039 | 51.19  | 0.2267  | 0.4802    |

Table 2. Quantitative UNet-VAE quantization experiments results. PassionSR-FP is used as full-precision backbones rather than original OSEDiff. W8A8 denotes 8 bit weight and 8 bits activation quantization. The best results in the same setting are colored with red.

<span id="page-5-1"></span>

| Method       | Bit    | Params / M (↓ Ratio) | Ops / G (↓ Ratio) |
|--------------|--------|----------------------|-------------------|
| OSEDiff      | W32A32 | 1,303 (↓0%)          | 4,523 (↓0%)       |
| PassionSR-FT | W32A32 | 949 (↓27.13%)        | 4,240 (↓6.25%)    |
| PassionSR-U  | W8A8   | 300 (↓76.96%)        | 3,732 ↓17.50%)    |
|              | W6A6   | 246 (↓81.11%)        | 3,689 (↓18.44%)   |
| PassionSR-UV | W8A8   | 238 (↓81.77%)        | 1,060 (↓76.56%)   |
|              | W6A6   | 178 (↓86.32%)        | 795 (↓82.42%)     |

Table 3. Compression ratio of different quantization settings. PassionSR-U refers to UNet-only quantization while PassionSR-UV refers to UNet-VAE quantization.

# 4. Experiments

### 4.1. Experiment Setup

Data Construction. We randomly crop 500 LR and HR pairs, each of size 128×128, from DIV2K train [\[1\]](#page-8-22) to construct the calibration dataset. For the test datasets, we select RealSR [\[13\]](#page-8-23), DRealSR [\[41\]](#page-9-21), and DIV2K val [\[1\]](#page-8-22).

Evaluation Metrics. We employ reference-based evaluation metrics, including PSNR, SSIM [\[39\]](#page-9-22), LPIPS [\[51\]](#page-9-23), and DISTS [\[5\]](#page-8-24). We also utilize non-reference metrics, such as NIQE [\[50\]](#page-9-24), MUSIQ [\[14\]](#page-8-25), ManIQA [\[45\]](#page-9-25), and ClipIQA [\[35\]](#page-9-26). We evaluate all methods with full-size images.

Implementation Details. We quantize the weights and activations in the main components (*i.e.*, UNet and VAE) with low bit-widths (*e.g.*, 6 and 8 bits). We denote the quantization configuration w-bit weight quantization and a-bit activation quantization as WwAa. For calibration training, we set the learning rate of PassionSR as 1×10<sup>−</sup><sup>5</sup> and finetune for 4 epochs. It is worth demonstrating that we use the same initialization method as SmoothQuant [\[25\]](#page-8-20).

Compared Methods. We select representative quantization methods: MaxMin [\[12\]](#page-8-12), LSQ [\[8\]](#page-8-13), Q-Diffusion [\[18\]](#page-8-10), and EfficientDM [\[9\]](#page-8-14). We adopt these methods to quantize our full-precision PassionSR-FP based on their released code.

### 4.2. Main Results

Streamline Experiment. We replace the text embedding branch with a constant empty prompt embedding, preprocessed by the ClipEncoder using an empty string. Table [2](#page-5-0) shows a comparison between the original model OSEDiff and the streamline model PassionSR-FP. The results indicate minimal or even negligible performance drop. Additionally, as shown in Tab. [3,](#page-5-1) the total model parameters are reduced by 27.13%, and operations are reduced by 6.25%.

<span id="page-6-2"></span><span id="page-6-0"></span>![](_page_6_Figure_0.jpeg)

Figure 6. Visual comparison (×4) with high-resolution image, full-precision model's output and different quantization methods in some challenging cases at W8A8 and W6A6 UNet-VAE quantization. PassionSR gains significant visual advantages over other methods.

<span id="page-6-1"></span>

| Methods     | Efficiency |          | RealSR |        |        |        |       |        |         |           |
|-------------|------------|----------|--------|--------|--------|--------|-------|--------|---------|-----------|
|             | Time (h)   | GPU (GB) | PSNR↑  | SSIM↑  | LPIPS↓ | DISTS↓ | NIQE↓ | MUSIQ↑ | MANIQA↑ | CLIP-IQA↑ |
| MaxMin      | 0.00       | 0        | 15.55  | 0.2417 | 0.8018 | 0.4449 | 9.263 | 42.15  | 0.2791  | 0.4174    |
| LBQ         | 2.66       | 40       | 23.15  | 0.6621 | 0.5022 | 0.3115 | 7.234 | 47.75  | 0.3071  | 0.4787    |
| LBQ+LET     | 3.87       | 40       | 25.40  | 0.7529 | 0.3798 | 0.2584 | 6.604 | 44.26  | 0.2414  | 0.3224    |
| LBQ+LET+DQC | 1.07       | 28       | 24.41  | 0.7374 | 0.3427 | 0.2419 | 5.449 | 55.08  | 0.3083  | 0.4849    |

Table 4. Ablation study on our proposed components: LBQ, LET, and DQC. Our ablation experiments are in the setting of W6A6 UNet-VAE quantization. We test each ablation method on RealSR and record their calibration time and GPU costs.

Quantitative Results. In the UNet-VAE quantization experiment, Tab. [2](#page-5-0) shows that PassionSR significantly outperforms previous methods at the setting of W8A8 and W6A6. On each dataset, 8-bit PassionSR achieves comparable performance to the full-precision OSEDiff, or even better scores in some cases. Besides PassionSR, other quantization methods have an obvious decrease in quantitative results. For 6-bit quantization, while contrast methods like LSQ and Q-Diffusion exhibit low structural metrics (*e.g.*, PSNR and SSIM), they obtain high scores in non-reference IQA metrics. Our PassionSR achieves the highest reference IQA values and relatively lower non-reference IQA values than others. However, we find an interesting observation where images with substantial quantization noise also obtain relatively high non-reference IQA scores. It can explain why some quantized outputs by other methods have worse visual quality despite higher non-reference IQA scores. We provide more results and analyses in Fig. [6.](#page-6-0)

<span id="page-7-1"></span><span id="page-7-0"></span>![](_page_7_Figure_0.jpeg)

![](_page_7_Figure_1.jpeg)

![](_page_7_Figure_2.jpeg)

on of scale factor (b) Distribution of Original Activation (c) Distribution of Sn Figure 7. Distribution of scale factor and activation before and after smooth in the whole model.

**Visual Comparison.** We provide a visual comparison (×4) for UNet-VAE quantization in Fig. 6. Several challenging cases are selected for clearer visual contrast. Compared to previous methods, PassionSR generates clearer results and better textures, with a minimal gap from the full-precision model. Notably, PassionSR even surpasses the full-precision PassionSR-FP in certain cases.

Compression Ratio. To clearly present our compression and acceleration outcomes, we calculate the model's total size (Params / G) and the number of operations (Ops / G). The calculation follows the methods used in previous quantization studies [26]. The results are presented in Tab. 3, showing the compression and acceleration ratios for each setting. In the 8-bit setting, PassionSR-UV achieves an 81.77% compression ratio and a 76.56% acceleration ratio. Furthermore, with a 6-bit setting, we achieve a compression ratio of 86.32% and an acceleration ratio of 82.42%.

#### 4.3. Ablation Study

Learnable Equivalent Transformation (LET). In order to evaluate the effects of LET, we adopt the LBQ-only quantizer without LET to make comparison with LBQ and LET combined quantizer. The experiment results is presented in Tab. 4. Considering that the output images have lots of noise, the reference IQA metrics are more reasonable to measure the visual quality. The introduction of LET improves PSNR by over 2 dB and SSIM by about 0.1, with an additional hour of training, which indicates that LET brings huge performance increase.

**Distributed Quantization Calibration (DQC).** To research into the special calibration strategy DQC, we use collaborative calibration with LBQ and LET combined quantizer compared with DQC. The detailed experiments results are listed in Tab. 4. With slight performance enhancement, the integration of DQC mainly leads to faster converge and lower GPU memory cost, reducing the calibration time by about 2 hours and the GPU memory by over 10 G. It is because that the amount of parameters requiring gradients is smaller when applied DQC and the computational costs reduce during backward propagation. With distributed calibration, the training of scale factor in LET becomes stable, making it easier to find out the best parameters in LET.

#### 4.4. Distribution Visualization

By applying the equivalent transformation to UNet and VAE, we can adjust the values of scale factors to better control the distributions of weights and activations, making them easier to quantize. LET playes an important role in the quantization and we visualize the distribution of relative variables to observe LET's detailed effects in Fig. 7.

**Distribution of Scale Factor.** The dispersed distribution of scale factors in Fig. 7a implies that the scale factors in different channel or tensor take different values to address varying activation distributions. We obtain the best scale factor for different layer with different activation distributions through calibration process. Most of scale factor are larger than 1, which means that LET mainly alleviates the difficulties in activation quantization.

**Distribution of Activation.** Figure 7b shows that the distribution of original activations before LET is truly dispersed. A large amount of outliers have severe impacts on quantization. Under the help of LET, the distribution of activations is more centered and more friendly for quantization as shown in Fig. 7c. Moreover, the number of outliers largely decreases. This change in activation distribution indicates LET's strong ability to manipulate the distribution and great effects on minimizing the quantization errors.

#### 5. Conclusion

In this paper, we propose PassionSR, a novel post-training quantization method for one-step diffusion-based image super-resolution. By simplifying the model architecture to two core components, Variational Autoencoder (VAE) and UNet, we obtain a UNet-VAE structure with little performance loss. Our approach incorporates a Learnable Boundary Quantizer (LBQ) and Learnable Equivalent Transformation (LET) to manipulate activation distributions. And Distributed Quantization Calibration (DQC) strategy enhances training stability and accelerates convergence. Experiments show that PassionSR delivers perceptual performance comparable to full-precision models at 8-bit and 6bit. It gains significant advantages over recent leading diffusion quantization methods. This work paves the way for future one-step diffusion-based SR model quantization and practical deployment of advanced SR model applications.

# References

- <span id="page-8-22"></span>[1] Eirikur Agustsson and Radu Timofte. Ntire 2017 challenge on single image super-resolution: Dataset and study. In *CVPRW*, 2017. [6](#page-5-2)
- <span id="page-8-0"></span>[2] Zheng Chen, Yulun Zhang, Jinjin Gu, Linghe Kong, Xin Yuan, et al. Cross aggregation transformer for image restoration. *NeurIPS*, 2022. [1](#page-0-1)
- <span id="page-8-15"></span>[3] Zheng Chen, Yulun Zhang, Jinjin Gu, Linghe Kong, Xin Yuan, et al. Cross aggregation transformer for image restoration. *NeurIPS*, 2022. [3](#page-2-1)
- <span id="page-8-1"></span>[4] Zheng Chen, Yulun Zhang, Jinjin Gu, Linghe Kong, Xiaokang Yang, and Fisher Yu. Dual aggregation transformer for image super-resolution. *ICCV*, 2023. [1](#page-0-1)
- <span id="page-8-24"></span>[5] Keyan Ding, Kede Ma, Shiqi Wang, and Eero P Simoncelli. Image quality assessment: Unifying structure and texture similarity. *TPAMI*, 2020. [6](#page-5-2)
- <span id="page-8-6"></span>[6] Yifu Ding, Haotong Qin, Qinghua Yan, Zhenhua Chai, Junjie Liu, Xiaolin Wei, and Xianglong Liu. Towards accurate post-training quantization for vision transformer. *ACM MM*, 2022. [2](#page-1-2)
- <span id="page-8-2"></span>[7] Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang. Image super-resolution using deep convolutional networks. *TPAMI*, 2015. [1](#page-0-1)
- <span id="page-8-13"></span>[8] Steven K Esser, Jeffrey L McKinstry, Deepika Bablani, Rathinakumar Appuswamy, and Dharmendra S Modha. Learned step size quantization. In *ICLR*, 2020. [2,](#page-1-2) [3,](#page-2-1) [6,](#page-5-2) [7](#page-6-2)
- <span id="page-8-14"></span>[9] Yefei He, Jing Liu, Weijia Wu, Hong Zhou, and Bohan Zhuang. Efficientdm: Efficient quantization-aware fine-tuning of low-bit diffusion models. *arXiv preprint arXiv:2310.03270*, 2023. [2,](#page-1-2) [3,](#page-2-1) [6,](#page-5-2) [7](#page-6-2)
- <span id="page-8-9"></span>[10] Yefei He, Luping Liu, Jing Liu, Weijia Wu, Hong Zhou, and Bohan Zhuang. Ptqd: Accurate posttraining quantization for diffusion models. *NeurIPS*, 2023. [2,](#page-1-2) [3](#page-2-1)
- <span id="page-8-7"></span>[11] Itay Hubara, Yury Nahshan, Yair Hanani, Ron Banner, and Daniel Soudry. Accurate post training quantization with small calibration sets. *ICML*, 2021. [2](#page-1-2)
- <span id="page-8-12"></span>[12] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. Quantization and training of neural networks for efficient integer-arithmeticonly inference. *CVPR*, 2018. [2,](#page-1-2) [4,](#page-3-2) [6](#page-5-2)
- <span id="page-8-23"></span>[13] Xiaozhong Ji, Yun Cao, Ying Tai, Chengjie Wang, Jilin Li, and Feiyue Huang. Real-world superresolution via kernel estimation and noise injection. In *CVPRW*, 2020. [6](#page-5-2)
- <span id="page-8-25"></span>[14] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In *ICCV*, 2021. [6](#page-5-2)

- <span id="page-8-16"></span>[15] Jiwon Kim, Jung Kwon Lee, and Kyoung Mu Lee. Accurate image super-resolution using very deep convolutional networks. *CVPR*, 2016. [3](#page-2-1)
- <span id="page-8-8"></span>[16] Eli Kravchik, Fan Yang, Pavel Kisilev, and Yoni Choukroun. Low-bit quantization of neural networks for efficient inference. *ICCVW*, 2019. [2,](#page-1-2) [7](#page-6-2)
- <span id="page-8-5"></span>[17] Jianze Li, Jiezhang Cao, Zichen Zou, Xiongfei Su, Xin Yuan, Yulun Zhang, Yong Guo, and Xiaokang Yang. Distillation-free one-step diffusion for real-world image super-resolution. *arXiv preprint arXiv:2410.04224*, 2024. [1](#page-0-1)
- <span id="page-8-10"></span>[18] Xiuyu Li, Yijiang Liu, Long Lian, Huanrui Yang, Zhen Dong, Daniel Kang, Shanghang Zhang, and Kurt Keutzer. Q-diffusion: Quantizing diffusion models. *ICCV*, 2023. [2,](#page-1-2) [3,](#page-2-1) [6,](#page-5-2) [7](#page-6-2)
- <span id="page-8-18"></span>[19] Yuhang Li, Ruihao Gong, Xu Tan, Yang Yang, Peng Hu, Qi Zhang, Fengwei Yu, Wei Wang, and Shi Gu. Brecq: Pushing the limit of post-training quantization by block reconstruction. *arXiv preprint arXiv:2102.05426*, 2021. [3](#page-2-1)
- <span id="page-8-11"></span>[20] Yanjing Li, Sheng Xu, Xianbin Cao, Xiao Sun, and Baochang Zhang. Q-dm: An efficient low-bit quantized diffusion model. *NeurIPS*, 2024. [2,](#page-1-2) [3](#page-2-1)
- <span id="page-8-3"></span>[21] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. Swinir: Image restoration using swin transformer. *ICCV*, 2021. [1,](#page-0-1) [3](#page-2-1)
- <span id="page-8-21"></span>[22] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. Awq: Activation-aware weight quantization for on-device llm compression and acceleration. *Proceedings of Machine Learning and Systems*, 2024. [5](#page-4-4)
- <span id="page-8-4"></span>[23] Xinqi Lin, Jingwen He, Ziyan Chen, Zhaoyang Lyu, Bo Dai, Fanghua Yu, Wanli Ouyang, Yu Qiao, and Chao Dong. Diffbir: Towards blind image restoration with generative diffusion prior. *ECCV*, 2024. [1,](#page-0-1) [3,](#page-2-1) [7](#page-6-2)
- <span id="page-8-19"></span>[24] Zechun Liu, Kwang-Ting Cheng, Dong Huang, Eric Xing, and Zhiqiang Shen. Nonuniform-to-uniform quantization: Towards accurate quantization via generalized straight-through estimation. 2022. [3](#page-2-1)
- <span id="page-8-20"></span>[25] Jiayi Pan, Chengcan Wang, Kaifu Zheng, Yangguang Li, Zhenyu Wang, and Bin Feng. Smoothquant+: Accurate and efficient 4-bit post-training weightquantization for llm. *arXiv preprint arXiv:2312.03788*, 2023. [4,](#page-3-2) [6](#page-5-2)
- <span id="page-8-26"></span>[26] Haotong Qin, Yulun Zhang, Yifu Ding, Xianglong Liu, Martin Danelljan, Fisher Yu, et al. Quantsr: accurate low-bit quantization for efficient image superresolution. *NeurIPS*, 2024. [8](#page-7-1)
- <span id="page-8-17"></span>[27] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High- ¨

- resolution image synthesis with latent diffusion models. *CVPR*, 2022. [3](#page-2-1)
- <span id="page-9-12"></span>[28] Yuzhang Shang, Zhihang Yuan, Bin Xie, Bingzhe Wu, and Yan Yan. Post-training quantization on diffusion models. *CVPR*, 2023. [2,](#page-1-2) [3](#page-2-1)
- <span id="page-9-19"></span>[29] Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng Gao, Yu Qiao, and Ping Luo. Omniquant: Omnidirectionally calibrated quantization for large language models. *ICLR*, 2024. [4](#page-3-2)
- <span id="page-9-15"></span>[30] Junhyuk So, Jungwon Lee, Daehyun Ahn, Hyungjun Kim, and Eunhyeok Park. Temporal dynamic quantization for diffusion models. *NeurIPS*, 2024. [3](#page-2-1)
- <span id="page-9-6"></span>[31] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. *ICLR*, 2021. [1](#page-0-1)
- <span id="page-9-8"></span>[32] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. *ICML*, 2023. [1](#page-0-1)
- <span id="page-9-16"></span>[33] Changyuan Wang, Ziwei Wang, Xiuwei Xu, Yansong Tang, Jie Zhou, and Jiwen Lu. Towards accurate posttraining quantization for diffusion models. *CVPR*, 2024. [3](#page-2-1)
- <span id="page-9-18"></span>[34] Haoxuan Wang, Yuzhang Shang, Zhihang Yuan, Junyi Wu, Junchi Yan, and Yan Yan. Quest: Low-bit diffusion model quantization via efficient selective finetuning. *arXiv preprint arXiv:2402.03666*, 2024. [3](#page-2-1)
- <span id="page-9-26"></span>[35] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In *AAAI*, 2023. [6](#page-5-2)
- <span id="page-9-4"></span>[36] Jianyi Wang, Zongsheng Yue, Shangchen Zhou, Kelvin CK Chan, and Chen Change Loy. Exploiting diffusion prior for real-world image super-resolution. *IJCV*, 2024. [1,](#page-0-1) [3](#page-2-1)
- <span id="page-9-1"></span>[37] Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-esrgan: Training real-world blind superresolution with pure synthetic data. *ICCV*, 2021. [1,](#page-0-1) [3](#page-2-1)
- <span id="page-9-11"></span>[38] Yufei Wang, Wenhan Yang, Xinyuan Chen, Yaohui Wang, Lanqing Guo, Lap-Pui Chau, Ziwei Liu, Yu Qiao, Alex C Kot, and Bihan Wen. Sinsr: diffusionbased image super-resolution in a single step. *CVPR*, 2024. [1,](#page-0-1) [3](#page-2-1)
- <span id="page-9-22"></span>[39] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. *TIP*, 2004. [6](#page-5-2)
- <span id="page-9-9"></span>[40] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. *NeurIPS*, 2024. [1](#page-0-1)
- <span id="page-9-21"></span>[41] Pengxu Wei, Ziwei Xie, Hannan Lu, Zongyuan Zhan, Qixiang Ye, Wangmeng Zuo, and Liang Lin. Component divide-and-conquer for real-world image superresolution. In *ECCV*, 2020. [6](#page-5-2)

- <span id="page-9-3"></span>[42] Rongyuan Wu, Lingchen Sun, Zhiyuan Ma, and Lei Zhang. One-step effective diffusion network for realworld image super-resolution. *NeurIPS*, 2024. [1,](#page-0-1) [2,](#page-1-2) [3,](#page-2-1) [4,](#page-3-2) [6,](#page-5-2) [7](#page-6-2)
- <span id="page-9-5"></span>[43] Rongyuan Wu, Tao Yang, Lingchen Sun, Zhengqiang Zhang, Shuai Li, and Lei Zhang. Seesr: Towards semantics-aware real-world image super-resolution. *CVPR*, 2024. [1,](#page-0-1) [3](#page-2-1)
- <span id="page-9-17"></span>[44] Yuhui Xu, Lingxi Xie, Xiaotao Gu, Xin Chen, Heng Chang, Hengheng Zhang, Zhengsu Chen, Xiaopeng Zhang, and Qi Tian. Qa-lora: Quantization-aware low-rank adaptation of large language models. *ICLR*, 2024. [3](#page-2-1)
- <span id="page-9-25"></span>[45] Sidi Yang, Tianhe Wu, Shuwei Shi, Shanshan Lao, Yuan Gong, Mingdeng Cao, Jiahao Wang, and Yujiu Yang. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In *CVPR*, 2022. [6](#page-5-2)
- <span id="page-9-14"></span>[46] Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, Xiaoxia Wu, Conglong Li, and Yuxiong He. Zeroquant: Efficient and affordable post-training quantization for large-scale transformers. *NeurIPS*, 2022. [3](#page-2-1)
- <span id="page-9-10"></span>[47] Tianwei Yin, Michael Gharbi, Richard Zhang, Eli ¨ Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. *CVPR*, 2024. [1](#page-0-1)
- <span id="page-9-20"></span>[48] Zhihang Yuan, Chenhao Xue, Yiqi Chen, Qiang Wu, and Guangyu Sun. Ptq4vit: Post-training quantization framework for vision transformers with twin uniform quantization. *ECCV*, 2022. [5](#page-4-4)
- <span id="page-9-2"></span>[49] Kai Zhang, Jingyun Liang, Luc Van Gool, and Radu Timofte. Designing a practical degradation model for deep blind image super-resolution. *ICCV*, 2021. [1,](#page-0-1) [3](#page-2-1)
- <span id="page-9-24"></span>[50] Lin Zhang, Lei Zhang, and Alan C. Bovik. A featureenriched completely blind image quality evaluator. *TIP*, 2015. [6](#page-5-2)
- <span id="page-9-23"></span>[51] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In *CVPR*, 2018. [6](#page-5-2)
- <span id="page-9-13"></span>[52] Yulun Zhang, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong, and Yun Fu. Image super-resolution using very deep residual channel attention networks. *ECCV*, 2018. [3](#page-2-1)
- <span id="page-9-0"></span>[53] Yulun Zhang, Yapeng Tian, Yu Kong, Bineng Zhong, and Yun Fu. Residual dense network for image superresolution. *CVPR*, 2018. [1](#page-0-1)
- <span id="page-9-7"></span>[54] Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. Unipc: A unified predictor-corrector framework for fast sampling of diffusion models. *NeurIPS*, 2024. [1](#page-0-1)