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

