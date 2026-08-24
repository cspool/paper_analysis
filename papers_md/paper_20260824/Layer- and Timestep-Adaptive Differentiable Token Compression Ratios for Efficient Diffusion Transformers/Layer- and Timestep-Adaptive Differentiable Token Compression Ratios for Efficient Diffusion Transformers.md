# <span id="page-0-0"></span>Layer- and Timestep-Adaptive Differentiable Token Compression Ratios for Efficient Diffusion Transformers

Haoran You<sup>1</sup>,2,\*, Connelly Barnes<sup>2</sup> , Yuqian Zhou<sup>2</sup> , Yan Kang<sup>2</sup> , Zhenbang Du<sup>1</sup> , Wei Zhou<sup>1</sup> , Lingzhi Zhang<sup>2</sup> , Yotam Nitzan<sup>2</sup> , Xiaoyang Liu<sup>2</sup> , Zhe Lin<sup>2</sup> , Eli Shechtman<sup>2</sup> , Sohrab Amirghodsi<sup>2</sup> , Yingyan (Celine) Lin<sup>1</sup>

<sup>1</sup>Georgia Institute of Technology <sup>2</sup>Adobe Research

# Abstract

*Diffusion Transformers (DiTs) have achieved state-of-theart (SOTA) image generation quality but suffer from high latency and memory inefficiency, making them difficult to deploy on resource-constrained devices. One major efficiency bottleneck is that existing DiTs apply equal computation across all regions of an image. However, not all image tokens are equally important, and certain localized areas require more computation, such as objects. To address this, we propose DiffCR, a dynamic DiT inference framework with differentiable compression ratios, which automatically learns to dynamically route computation across layers and timesteps for each image token, resulting in efficient DiTs. Specifically, DiffCR integrates three features: (1) A tokenlevel routing scheme where each DiT layer includes a router that is fine-tuned jointly with model weights to predict token importance scores. In this way, unimportant tokens bypass the entire layer's computation; (2) A layer-wise differentiable ratio mechanism where different DiT layers automatically learn varying compression ratios from a zero initialization, resulting in large compression ratios in redundant layers while others remain less compressed or even uncompressed; (3) A timestep-wise differentiable ratio mechanism where each denoising timestep learns its own compression ratio. The resulting pattern shows higher ratios for noisier timesteps and lower ratios as the image becomes clearer. Extensive experiments on text-to-image and inpainting tasks show that DiffCR effectively captures dynamism across token, layer, and timestep axes, achieving superior tradeoffs between generation quality and efficiency compared to prior works. The project website is available [here.](https://www.haoranyou.com/diffcr)*

# 1. Introduction

Diffusion models have recently demonstrated outstanding performance in image generation, with architectures evolving from U-Nets [\[13,](#page-8-0) [30,](#page-9-0) [34,](#page-9-1) [41\]](#page-9-2) to Transformers [\[1,](#page-8-1) [3,](#page-8-2) [4,](#page-8-3) [29\]](#page-9-3). Among these, Diffusion Transformers (DiTs) [\[1,](#page-8-1) [29\]](#page-9-3) stand out for their superior scalability. However, diffusion models, particularly DiTs, are hampered by substantial computational and memory demands, which limits their efficiency in generation and deployment. For instance, generating a 1024px image with full context on a single A100 GPU can take 19.48 seconds and require >40GB GPU memory [\[28,](#page-9-4) [30\]](#page-9-0). One major efficiency bottleneck in most DiTs stems from the uniform application of computation across all image regions, despite varying levels of complexity in different areas [\[28,](#page-9-4) [32,](#page-9-5) [57\]](#page-10-0). Such an efficiency bottleneck suggests an ideal DiT inference framework could have adaptive and dynamic computation across three key axes in DiTs: token, layer, and timestep.

Various techniques have been proposed to address the efficiency bottleneck along the three aforementioned key axes: (1) token merging [\[2\]](#page-8-4), pruning [\[45\]](#page-9-6), and downsampling [\[39\]](#page-9-7); (2) layer [\[17\]](#page-8-5) or channel [\[9\]](#page-8-6) pruning; and (3) few-step distillation and sampling [\[23,](#page-8-7) [35,](#page-9-8) [46,](#page-9-9) [56\]](#page-10-1). While promising, the techniques (1-2) for the most part rely on heuristics, such as heuristic rules for token importance and channel and layer pruning rules. Moreover, compression ratios are often uniform across layers or adjusted empirically based on prior experience. In addition, most approaches focus on a single efficiency axis, overlooking the compounded effect of combining optimizations across all three.

To achieve a unified and learnable dynamic DiT inference framework with differentiable compression ratios across layers and timesteps, three key challenges must be tackled: (1) *Token Perspective:* Developing a learnable token importance metric that adapts to visual content, as not all tokens are equally important. (2) *Layer Perspective:* Designing mechanisms to autonomously learn adaptive compression ratios for each layer, optimizing processing efficiency, since not all layers contribute equally. (3) *Timestep Perspective:* Developing methods to learn and apply compression ratio patterns effectively across timesteps, as not all timesteps are equally important. We make the following

<sup>\*</sup>Work done while interning at Adobe.

<span id="page-1-0"></span>contributions to address these three challenges:

- We propose a dynamic DiT inference framework with differentiable compression ratios, dubbed DiffCR, which automatically learns an optimal dynamic computation across layers and timesteps for each image token, resulting in efficient DiT models for content generation tasks.
- *Enabler 1*: We adopt a token-level routing scheme inspired by the mixture-of-depth (MoD) [\[33\]](#page-9-10), which automatically learns token importance scores. Each DiT layer includes a lightweight router that is fine-tuned jointly with the model weights. Based on the compression ratio, less important tokens bypass computation in the entire layer. To the best of our knowledge, we are the first to apply MoD to the vision domain. Our routing analysis reveals that token importance varies across layers and timesteps.
- *Enabler 2*: Based on our analysis, we introduce a novel DiffCR module that enables the token routing scheme to be differentiable with respect to compression ratios, allowing the model to learn adaptive compression ratios for each layer starting with zero initialization. Redundant layers learn higher compression ratios, while important layers remain less compressed or entirely uncompressed.
- *Enabler 3*: We further present a timestep-wise differentiable ratio mechanism, enabling each layer and denoising timestep to learn its own compression ratio. This results in a pattern where noisier timesteps adopt higher compression ratios, while clearer stages maintain lower ratios.

Our extensive experiments on both image inpainting and text-to-image (T2I) tasks consistently demonstrate that DiffCR achieves a superior trade-off between generation quality and efficiency, with an average FID reduction of 8.51 while maintaining comparable latency and memory usage, compared to the most competitive baseline.

# 2. Related Work

#### 2.1. Diffusion Models

Diffusion models [\[13,](#page-8-0) [40\]](#page-9-11) have demonstrated superior performance over prior SOTA generative adversarial networks (GANs) in image synthesis tasks [\[7\]](#page-8-8). Early diffusion models primarily utilized U-Net architectures. Subsequent work introduced several improvements, such as advanced sampling [\[16,](#page-8-9) [23,](#page-8-7) [41\]](#page-9-2) and classifier-free guidance [\[12\]](#page-8-10). Although effective, these models suffered from high generation latency due to processing directly in pixel space, thus limiting their practical applications. The introduction of Latent Diffusion Models (LDMs) [\[34\]](#page-9-1) marked a significant advancement by encoding pixel space into a more compact latent space through training a Variational Auto-Encoder (VAE). This reduced the computational cost of the diffusion process, paving the way for widely used models like Stable Diffusion Models (SDMs) [\[30\]](#page-9-0). More recently, researchers have explored Transformer [\[43\]](#page-9-12) architectures for diffusion, leading to the development of DiTs [\[1,](#page-8-1) [29\]](#page-9-3), which employ a pure Transformer backbone and exhibit improved scalability. Our DiffCR proposes a novel dynamic DiT inference framework with differentiable compression ratios and is compatible with all recent DiT models.

### 2.2. Efficient Diffusion and DiT Models

DiTs [\[29\]](#page-9-3) are resource-intensive due to the transformer architecture, with the attention module exhibiting quadratic complexity relative to the number of tokens. Previous work has mainly focused on optimizing DiTs' deployment efficiency along three dimensions: token, layer, and timestep. For tokens, researchers have introduced techniques like token merging [\[2\]](#page-8-4) to merge similar tokens, token pruning [\[45\]](#page-9-6) or image resolution downsampling [\[39\]](#page-9-7) to remove redundant tokens, and LazyDiffusion [\[28\]](#page-9-4), which is specialized for the inpainting task and bypasses generating background tokens. For layers, methods such as layer [\[17\]](#page-8-5) and channel [\[9\]](#page-8-6) pruning, as well as intermediate feature caching [\[22,](#page-8-11) [25,](#page-9-13) [54\]](#page-10-2), have been proposed to skip redundant computations. For timesteps, strategies include distillation to reduce the required number of timesteps, which has been explored for UNets [\[15,](#page-8-12) [23,](#page-8-7) [35,](#page-9-8) [36,](#page-9-14) [55,](#page-10-3) [56\]](#page-10-1) although there is no reason to believe these techniques cannot apply to Transformers, and asymmetric sampling, which has been applied to Transformer architectures and allocates more samples to undersampled stages and fewer to stages that have already converged [\[31,](#page-9-15) [46\]](#page-9-9). Additionally, to accelerate diffusion T2I models, more specialized techniques have been introduced [\[3,](#page-8-2) [4\]](#page-8-3). In contrast, our proposed DiffCR is a learnable and unified dynamic DiT inference framework with differentiable compression ratios across layers and timesteps, exploring the compounded effects of compression across all three axes. However, we do not explore few step distillation (e.g. [\[55\]](#page-10-3)) in this paper, since it is an orthogonal acceleration method that is complementary to ours.

#### 2.3. Dynamic Inference

Model compression [\[6\]](#page-8-13) offers a static approach to improving inference efficiency, while dynamic inference [\[33,](#page-9-10) [48,](#page-9-16) [49,](#page-9-17) [52,](#page-10-4) [58\]](#page-10-5) enables adaptive compression based on the input, layer, or other conditions. For example, early exiting methods [\[14,](#page-8-14) [24,](#page-8-15) [42\]](#page-9-18) predict the optimal point for early termination within intermediate layers, allowing the model to exit before completing all computations. Dynamic layerskipping methods [\[48,](#page-9-16) [49,](#page-9-17) [52\]](#page-10-4) selectively execute subsets of layers for each input, often utilizing a gating network to make decisions on the fly. At a finer granularity, researchers have also explored channel skipping [\[9,](#page-8-6) [26\]](#page-9-19) and mixture-ofdepths (MoD) approaches [\[33\]](#page-9-10), which select specific subsets of layers for individual tokens rather than processing the entire input uniformly. In contrast, our DiffCR is the first to introduce a unified dynamic DiT inference framework that optimizes across three axes: token, layer, and

<span id="page-2-3"></span><span id="page-2-1"></span>> **[图片提取文字 (无描述)]:**
> DiT Layer i w/ Diff. Ratio DiTs Lower Bin (LB): 10% **Text Prompt:** Diff. Ratio Query Discrete Ratio Bins Attn. Upper Bin (UB): 20% **Border Collie** e.g., 12% {0%, 10%, ..., 100%} Nearest Bin (NB): 10% MLP 12% Proximity 10% 20% 80% Closer Coefficients: 20% Closer **Token-level Routing Scheme** DIT Layer i **During Training During Inference** Top-k Input → Router → Attn. → MLP -MoD with Tokens LB Ratio Input Input Concat. MoD with 80% Tokens Tokens **NB** Ratio 20% Out. Out. Prev. Token MoD with 100% Output Skip Entire Layer's **UB Ratio Updated Token** Computation Tokens (a) Token-level Routing Scheme (b) Differentiable Compression Ratios
![](_page_2_Figure_0.jpeg)

Figure 1. Overview of the proposed DiffCR framework: (a) token-level routing scheme and (b) differentiable compression ratios.

timestep. It also enables differentiable compression ratios that are fine-tuned jointly with the network, enhancing adaptability and efficiency.

#### <span id="page-2-2"></span>3. The Proposed DiffCR Framework

In this section, we present the proposed DiffCR framework. First, we provide an overview of the method. Then, we detail the three enablers: (1) the token-level routing scheme for DiTs in Sec. 3.2; (2) the layer-wise differentiable compression ratio scheme in Sec. 3.3; and (3) the timestep-wise differentiable compression ratio scheme in Sec. 3.4.

#### 3.1. Overview of DiffCR

Motivated by the need for unified and dynamic compression during DiT inference, DiffCR introduces a token-level routing scheme to dynamically learn the importance of each token on the fly. As illustrated in Fig. 1 (a), similar to previous mixture-of-depths (MoD) work [33] for NLP tasks, each DiT layer incorporates a lightweight router using a single linear layer to predict the importance of each token based on the input image/noise and text embedding. This allows us to bypass computations for less important tokens in each layer and to directly forward their activations to the layer's outputs. Consequently, each token is processed by only a selective subset of layers. Visualization of these routers' predictions reveals that different layers or timesteps favor varying compression ratios—for instance, some layers prioritize generating objects, while others focus on backgrounds—highlighting the need for adaptive compression across layers and timesteps. To achieve such dynamic compression, DiffCR incorporates a differentiable compression ratio scheme, as shown in Fig. 1 (b). This scheme includes a learnable scalar parameter that represents a continuous compression ratio, and predefined discrete ratio bins as proxy ratios. The scalar queries the bins to identify lower and upper bin ratios, creating two separate paths with distinct compression ratios. The final output

is a linear combination of these two paths, weighted by the proximity of the learned ratio to each bin. We apply a mean-squared error (MSE) loss to ensure that the average learned ratio across layers or timesteps converges to the target ratio. By doing so, DiffCR learns the adaptive compression ratios in a differentiable manner, resulting in efficient and dynamic mixture-of-depths DiTs.

#### <span id="page-2-0"></span>3.2. Enabler 1: Token-level Routing Scheme

**Motivation.** We are motivated by the varying computational demands across tokens, as many of them require fewer layers for efficient processing. We start from the same token-level routing scheme as MoD [33]. We remove from MoD two features that were specialized for the acausal NLP task: specifically, we remove the auxiliary loss and auxiliary MLP predictor from Section 3.5 of their paper. To the best of our knowledge, our paper is the first application of MoD to the vision domain, so we next review the routing mechanism, perform some visualizations, and report insights for vision tasks.

Token-level Routing. DiTs process noise and conditional text embeddings as inputs, aiming to denoise and generate images in an end-to-end manner. To predict token importance, we employ a simple yet effective token-level routing scheme from MoD [33]. As illustrated in Fig. 1 (a), each DiT layer incorporates a lightweight router composed of a single linear layer with a sigmoid activation function, predicting each token's importance on a scale from 0 to 1. After passing through the routers, we select the top-k most important tokens for this layer's processing, while the activations of other tokens are cached and concatenated with the layer outputs, bypassing the entire layer computation, including both attention and MLPs. To enable gradient flow to the router's weights during joint fine-tuning with pretrained DiT models, the same as MoD [33], we rescale the top-k token output activations by multiplying them with the router's predictions. This rescaling ensures that gradients are propa-

<span id="page-3-3"></span><span id="page-3-2"></span>> **[图片提取文字 (无描述)]:**
> "a pínk rose wíth water droplets" (a) Inpainting (b) Input: "an apple" Input: Text-to-Image Timestep: Timestep: **Router's Prediction Router's Prediction** 40 60 80 100 0 19 21 Layer: 13 17 26 Layer: 10 12 17 3 4
![](_page_3_Figure_0.jpeg)

Figure 2. Visualization of the router's predictions: (a) For inpainting tasks, where inputs are masked images with text prompts, we follow the previous SOTA method Lazy-Diffusion [28] to generate only the masked area rather than the entire image; (b) For text-to-image (T2I) tasks, where inputs are noise and text prompts, we follow PixArt- $\Sigma$  [4] for generation. Each visualization includes the router's prediction map with values ranging from 0 to 1. The generated image at each corresponding timestep is shown on the left, while the router's prediction maps across various layers and timesteps are displayed on the right. More visualizations are provided in the supplementary materials.

<span id="page-3-1"></span>> **[图片提取文字 (无描述)]:**
> ToMe -ToMe Memory (GB) 15 10 € 300 250 Latency (a) Compression Ratios (%) (b) Compression Ratios (%)
![](_page_3_Figure_2.jpeg)

Figure 3. Comparison of latency and memory savings between our DiffCR router and the previous token merging method (ToMe) [2] when applied to ViT-XL/2 [8, 28] on an A100 GPU.

gated effectively to the router during backpropagation. The value of k is determined based on compression ratios; we empirically find that 20% or 30% compression offers an optimal trade-off between latency/memory efficiency and minimal degradation in generation quality. Unlike previous token merging techniques [2], where computational savings are not directly proportional to the merge ratios due to the extra overhead, as shown in Fig. 3, the MoD in our approach yields more reductions in actual latency and memory usage due to negligible overhead.

Router Visualization and Insights. To test the router's effectiveness, we visualize its predictions in Fig. 2. Our observations reveal that: (1) The router effectively captures semantic information, clearly delineating object shapes and achieving an attention-like effect with significantly reduced computational costs; (2) The predicted token importance varies across layers and timesteps. For instance, some layers prioritize object generation, while others emphasize background areas. Additionally, as timesteps progress, the

router increasingly captures the semantic contours of objects, underscoring the need for dynamic token importance estimation; (3) *The optimal compression ratio differs across layers and timesteps*. For example, certain layers designate all tokens as high-importance, showing minimal redundancy, whereas other layers selectively prune object or background tokens with distinct shapes, requiring varying compression ratios. Similar variance is observed across timesteps. In the current MoD, a fixed global compression rate is applied equally to each layer and timestep, rather than adapting to its individual significance. Uniform pruning risks over-pruning critical layers or timesteps while leaving redundant ones less compressed. This motivates us to apply adaptive and dynamic compression ratios across both layers and timesteps.

#### <span id="page-3-0"></span>3.3. Enabler 2: Layer-wise Differentiable Ratio

**Motivation.** Recognizing that different layers prioritize different objects or background elements and thus benefit from distinct compression ratios, we propose a novel layer-wise differentiable compression ratio mechanism. This approach automatically learns each layer's compression ratio from a zero initialization in a differentiable manner, adapting to the varying redundancy levels across layers.

**Design Choice.** Before designing DiffCR, we address a key choice: a discrete proxy or a continuous ratio representation. Previous work [5] uses a discrete proxy with multiple compression ratio candidates and learnable probabilities, but this approach poses three challenges for MoD: (1) it lacks effective initialization, as the final ratio relies on the product of candidates and probabilities, making it

<span id="page-4-3"></span><span id="page-4-1"></span>> **[图片提取文字 (无描述)]:**
> (a) (b) Layer 1 Layer 15 **Compression Ratios** Layer 16 Layer 2 Compression Ratios Layer 17 0.25 Layer 18 Layer 19 Layer 20 0.20 Layer 21 Layer 22 Laver 8 Layer 23 Layer 9 0.15 Layer 24 Laver 25 Layer 11 Layer 12 Layer 26 Layer 13 Layer 27 0.10 - Layer 14 Layer 28 Avg. . Avg. Ratio 0.0 0.00 160 170 180 190 200 210 220 230 240 250 170 180 190 200 210 220 230 240 250 Training Iterations (k) Training Iterations (k) (c) **Compression Ratios** 0.8 0.6 0.4 0.2 0.0 2 3 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 Layers
![](_page_4_Figure_0.jpeg)

Figure 4. Visualization of the compression ratio trajectory during fine-tuning: (a) Trajectories for each of the 28 layers in DiT models; (b) Average ratio trajectory across all layers; and (c) The final learned ratio distribution across 28 layers.

difficult to initialize all ratios at zero; (2) MoD requires a differentiable and learnable router, incompatible with discrete proxies that need multiple sets of top-k tokens; and (3) it introduces numerous learnable parameters, complicating training and interpretability, and leading to hard-to-interpret ratio distributions. In contrast, we represent each layer's compression ratio with a single continuous scalar parameter, reducing the number of parameters to 28 for DiT's 28 layers [\[3\]](#page-8-2) and directly representing the compression ratio.

Layer-wise DiffCR. As illustrated in Fig. [1](#page-2-1) (b), we assign each layer a single learnable parameter and introduce discrete MoD compression ratio bins at 10% intervals, ranging from 0% to 100%. During training, the learnable MoD ratio queries the nearest two discrete bins to retrieve the lower and upper bin ratios. For example, a 22% learnable ratio would correspond to 20% as the lower bin and 30% as the upper bin. We then apply a forward pass through the DiT layer (with MoD routers) with each of these bin ratios, producing two output branches. The final output is a weighted linear combination of these branches, where the weights are determined by the proximity of the learnable ratio to each bin—e.g., for 12%, the output would be 80% weighted towards the 10% branch and 20% towards the 20% branch. Although this approach doubles the cost of a forward pass during training, we simply select the nearest bin as the final compression ratio during inference, eliminating any overhead. To ensure that the ratio converges to our target value, we incorporate an additional MSE loss between the current learned average ratios across all layers in the batch and the target ratio, which is a hyperparameter.

Ratio Trajectory Analysis. We visualize the training

<span id="page-4-2"></span>> **[图片提取文字 (无描述)]:**
> (a) Inpainting Task (b) T2I Task 0.5 0.40 25 25 0.35 0.4 0.30 21 21 0.25 Layers 13 0.3 17-0.20 13 0.2 0.15 9 9 0.10 0.1 5 0.05 0.0 0.00 10 9 **Timestep Regions** Timestep Regions
![](_page_4_Figure_5.jpeg)

Figure 5. Visualization of the learned ratio patterns across both timesteps and layers for the (a) inpainting task and (b) T2I task.

trajectory of compression ratios for all layers during finetuning of a LazyDiffusion model on an inpainting task in Fig. [4](#page-4-1) (a-c). The visualization reveals that: (1) Each layer learns its unique compression ratio, with redundant layers achieving higher compression and critical layers remaining less or entirely uncompressed; (2) The average ratio across layers gradually converges to the target ratio. In this example, with a target of 30%, the final achieved average ratio is approximately 29%, indicating a minor gap. Notably, a trade-off exists between convergence speed and generation quality: a higher MSE loss coefficient for the ratio accelerates convergence but may degrade quality due to overly rapid compression, while a smaller coefficient promotes gradual convergence and maintains quality, albeit with slower training. In practice, we set the coefficient to 0.3 to balance speed and quality effectively; (3) The middle layers exhibit greater redundancy, while the later layers generally have less redundancy and often cannot be compressed. The early layers show variable redundancy levels.

### <span id="page-4-0"></span>3.4. Enabler 3: Timestep-wise Differentiable Ratio

Motivation. In addition to layer-wise ratio variances, we also observe that the model exhibits varying levels of redundancy across timesteps. This motivates us to explore an approach for timestep-wise compression ratios as well.

Timestep-wise Differentiable Ratio. On top of the layer-wise DiffCR, we introduce learnable parameters specific to different timestep regions. For the image inpainting task, following the previous SOTA LazyDiffusion [\[28\]](#page-9-4), we use 1,000 training timesteps and 100 sampling timesteps, which we evenly divide into 10 regions. We assign 10 learnable ratios per layer accordingly, resulting in a total of 280 learnable parameters. Similarly, for T2I tasks, following PixArt-Σ [\[4\]](#page-8-3) with 20 timesteps, we divide them into 4 regions, assigning 4 learnable ratios per layer, yielding 112 learnable parameters. The same as before, we apply an MSE loss between the averaged learned ratios within the batch and the target ratio to ensure convergence.

Ratio Pattern Analysis. We visualize the learned compression ratio patterns across both timesteps and layers in Fig. [5.](#page-4-2) For both image inpainting and T2I tasks, we <span id="page-5-0"></span>consistently observe that noisy timesteps (corresponding to earlier sampling timesteps or later training timesteps) exhibit higher redundancy and allow for higher compression, whereas timesteps where images become clearer (corresponding to later sampling timesteps or earlier training timesteps) show less redundancy. This learned pattern aligns with previous empirical findings [\[46\]](#page-9-9), which suggest that high-noise timesteps are associated with convergence regions containing easier samples, allowing for fewer sampling timesteps, while low-noise timesteps involve harder samples and require more frequent sampling.

# 4. Experiments

#### 4.1. Experiment Settings

Tasks, Datasets, and Models. *Tasks & Datasets.* We evaluate our DiffCR on two representative image generation tasks using corresponding benchmark datasets: (1) an image inpainting task on an internal dataset of 220 million high-quality images, covering diverse objects and scenes. Masks and text prompts are generated following [\[28,](#page-9-4) [53\]](#page-10-6); and (2) a T2I task on the LAION-5B dataset [\[38\]](#page-9-20), restricted to image samples with high aesthetic scores, English text, and a minimum text similarity score of 0.24. *Models.* We integrate our proposed DiffCR approach with SOTA models. For the inpainting task, we use Lazy Diffusion (an adapted PixArt-α model with an additional ViT encoder) to generate images at 1024×1024 resolution. For the T2I task, we use PixArt-Σ to generate images at 512×512 resolution.

Training and Sampling Setting. For the inpainting task, we fine-tune the model parameters until convergence using the AdamW optimizer [\[20\]](#page-8-18) with a learning rate of 10<sup>−</sup><sup>4</sup> and weight decay of 3 × 10<sup>−</sup><sup>2</sup> . For sampling, images are generated using IDDPM [\[27\]](#page-9-21) with 100 timesteps and a CFG factor of 4.5. For the T2I task, we fine-tune the model using a LoRA adapter with a rank of 32 until both training and validation losses converge, and the MSE loss between the current and target compression ratios drops to approximately zero for DiffCR models. During training, we calculate the diffusion loss with IDDPM [\[27\]](#page-9-21) over 1K timesteps. For sampling, we generate images using DPMsolver [\[21\]](#page-8-19) with 20 timesteps and a CFG factor of 4.5. All training is conducted on a cluster of 8×A100-80GB GPUs.

Baselines and Evaluation Metrics. *Baselines.* For both the T2I and inpainting tasks, we compare the proposed DiffCR against SOTA baselines, including ToMe [\[2\]](#page-8-4), AT-EDM [\[45\]](#page-9-6), and our adapted MoD with uniform MoD compression ratio. For the inpainting task, we also compare against *RegenerateCrop*, which generates a tight square crop around the masked region, similar to popular software frameworks [\[44,](#page-9-22) [50\]](#page-10-7), and *RegenerateImage*, which generates the entire image, as commonly done in the literature [\[30,](#page-9-0) [34,](#page-9-1) [47,](#page-9-23) [53\]](#page-10-6). *Evaluation Metrics.* We assess the generated image quality using FID scores [\[11\]](#page-8-20), text-image alignment using CLIP scores [\[10\]](#page-8-21), and efficiency through inference FLOPs, latency, and memory usage, all measured on an A100 GPU. For inpainting and T2I models, we evaluate on 10K images from LAION-400M [\[37\]](#page-9-24) or LAION-5B [\[38\]](#page-9-20), excluding training samples, respectively.

### 4.2. DiffCR over SOTA Baselines

Text-to-Image. To assess the effectiveness of our proposed DiffCR, we apply our proposed DiffCR to the general textto-image task and compare it with previous token merging [\[2\]](#page-8-4) and pruning [\[45\]](#page-9-6) baselines. Specifically, we apply these compression methods on PixArt-Σ, a SOTA publicly accessible T2I model known for its high-resolution image generation quality and efficiency tradeoffs. As shown in Tab. [1,](#page-6-0) PixArt-Σ with DiffCR significantly improves generation quality, achieving 57.83 and 241.11 FID reductions over ToMe [\[2\]](#page-8-4) and AT-EDM [\[45\]](#page-9-6), respectively, with comparable or even lower latency (↓8.59%∼20.15%) and memory usage (↓-2.71%∼0.72%). Also, under similar latency compared to ToMe [\[2\]](#page-8-4) with 20% compression ratio, DiffCR achieves 335.23 FID reductions. Moreover, PixArt-Σ with DiffCR also achieves comparable image generation quality to uncompressed PixArt-Σ, while delivering 20.68% and 8.33% latency and memory savings. Note that we compare with fine-tuned PixArt-Σ on the LAION datasets for a fair comparison. This set of experiments demonstrates the effectiveness of DiffCR for general T2I tasks.

Image Inpainting. We further extend DiffCR to the inpainting task. Specifically, we apply it on top of the SOTA Lazy Diffusion (LD) [\[28\]](#page-9-4), which uses a DiT decoder to generate only the masked areas rather than the entire image, leveraging a separate ViT encoder to capture the global context of the input masked images. We compare our DiffCR approach against two types of baselines: (1) RegenerateImage and RegenerateCrop, and (2) LD with previous token merging [\[2\]](#page-8-4) or pruning [\[45\]](#page-9-6) techniques. As shown in Tab. [2,](#page-6-1) our DiffCR consistently outperforms all baselines in terms of accuracy-efficiency tradeoffs. For example, LD with DiffCR achieves FID reductions of 47.35 and 189.93 compared to LD with ToME [\[2\]](#page-8-4) or AT-EDM [\[45\]](#page-9-6), while achieving similar or up to 23.61% and 13.63% higher latency and memory savings. Also, under similar memory usage compared to ToMe [\[2\]](#page-8-4) with 30% compression ratio, LD with DiffCR achieve 265.94 FID reduction while delivering up to 21.54% latency savings. Moreover, compared to RegenerateImage, our method achieves 73.51%/60.26% FLOPs and latency savings when inpainting 256<sup>2</sup> mask sizes within 1024<sup>2</sup> images. Notably, like Lazy Diffusion, our method's complexity scales with mask size, while RegenerateImage generates based on full image resolution, making it less efficient for smaller mask sizes. In comparison to Regenerate-Crop, our method achieves significantly higher image generation quality (+41.01 FID) while also delivering 22.96%

<span id="page-6-3"></span><span id="page-6-0"></span>Table 1. Quantitative comparison of DiffCR with other baselines on the T2I task. All experiments are fine-tuned from the pre-trained PixArt- $\Sigma$  [4] on the LAION-5B [38] dataset. C.R. denotes compression ratios. We report FID ( $\downarrow$ ) and CLIP Score ( $\uparrow$ ) on 10K images (excluding training samples) as quality metrics and measure FLOPs ( $\downarrow$ ), latency ( $\downarrow$ ), and memory ( $\downarrow$ ) on an A100 GPU as efficiency metrics under batch sizes of 16 and 128. Memory is averaged across all layers. "-L" and "-LT" indicate layer-wise or layer/timestep-wise DiffCR. "TF" denotes training-free methods.

|                                      | DiT C.R. | Quality |                   | <b>DiT Efficiency (512<sup>2</sup>; BS = 16)</b> |          |           | <b>DiT Efficiency (512<sup>2</sup>; BS = 128)</b> |          |           |
|--------------------------------------|----------|---------|-------------------|--------------------------------------------------|----------|-----------|---------------------------------------------------|----------|-----------|
| Methods                              |          | FID     | <b>CLIP Score</b> | FLOPs (G)                                        | Lat. (s) | Mem. (GB) | FLOPs (G)                                         | Lat. (s) | Mem. (GB) |
| PixArt- $\Sigma$ [4]                 | 0%       | 151.0   | 0.173             | 17361.7                                          | 225.38   | 1.798     | 138893.9                                          | 1784.64  | 13.107    |
| PixArt- $\Sigma$ (Fine-tuned)        | 0%       | 11.93   | 0.242             | 17361.7                                          | 225.38   | 1.798     | 138893.9                                          | 1784.64  | 13.107    |
| PixArt- $\Sigma$ w/ ToMe (TF) [2]    | 10%      | 68.51   | 0.211             | 16391.4                                          | 224.04   | 1.674     | 131131.6                                          | 1772.84  | 12.087    |
| PixArt- $\Sigma$ w/ ToMe (TF) [2]    | 20%      | 345.91  | 0.122             | 15421.2                                          | 213.79   | 1.546     | 123369.2                                          | 1681.83  | 11.078    |
| PixArt- $\Sigma$ w/ AT-EDM (TF) [45] | 20%      | 251.79  | 0.129             | 15132.7                                          | 196.88   | 1.617     | 121061.2                                          | 1548.80  | 11.870    |
| PixArt- $\Sigma$ w/ <b>MoD</b>       | 20%      | 22.78   | 0.207             | 13949.3                                          | 178.89   | 1.659     | 111594.2                                          | 1402.96  | 11.987    |
| PixArt- $\Sigma$ w/ <b>DiffCR-L</b>  | 20%      | 12.28   | 0.232             | 13967.1                                          | 180.84   | 1.662     | 111737.0                                          | 1425.30  | 12.015    |
| PixArt- $\Sigma$ w/ <b>DiffCR-LT</b> | 20%      | 10.68   | 0.238             | 13957.2                                          | 179.71   | 1.664     | 111657.3                                          | 1415.63  | 12.021    |

<span id="page-6-1"></span>Table 2. Quantitative comparison of DiffCR with other baselines on the inpainting task. Scores for SDXL [30] are provided for reference only and are not directly comparable. C.R. denotes compression ratios. We report FID  $(\downarrow)$  [11] and CLIP Score  $(\uparrow)$  [10] on 10K images from LAION-400M [37] as quality metrics, and measure FLOPs  $(\downarrow)$ , latency  $(\downarrow)$ , and memory usage  $(\downarrow)$  on an A100 GPU as efficiency metrics for two inpainting mask sizes  $(512^2$  and  $256^2$  within  $1024^2$  images). "-L" and "-LT" indicate layer-wise or layer/timestep-wise DiffCR. "TF" denotes training-free methods.

| Methods                  | ViT En. | DiT De. | Quality |            | DiT Efficiency (512 $^2$ within 1024 $^2$ ) |          |           | DiT Efficiency (256 <sup>2</sup> within 1024 <sup>2</sup> ) |          |           |
|--------------------------|---------|---------|---------|------------|---------------------------------------------|----------|-----------|-------------------------------------------------------------|----------|-----------|
| THUMOUS                  | C.R.    | C.R.    | FID     | CLIP Score | FLOPs (G)                                   | Lat. (s) | Mem. (GB) | FLOPs (G)                                                   | Lat. (s) | Mem. (GB) |
| SDXL [30]                | N/A     | 0%      | 6.37    | 0.2112     | 5979.5                                      | 66.09    | OOM       | 5979.5                                                      | 66.09    | OOM       |
| RegenerateImage          | N/A     | 0%      | 9.53    | 0.1942     | 809.8                                       | 13.11    | OOM       | 809.8                                                       | 13.11    | OOM       |
| RegenerateCrop           | N/A     | 0%      | 54.43   | 0.1737     | 267.3                                       | 4.30     | 45.00     | 199.2                                                       | 4.19     | 13.66     |
| Lazy Diffusion (LD) [28] | 0%      | 0%      | 10.90   | 0.1882     | 1085.1                                      | 17.13    | 45.00     | 285.8                                                       | 5.68     | 13.66     |
| LD w/ Model Pruning      | 50%     | 30%     | 27.36   | 0.1796     | 775.5                                       | 15.33    | 45.00     | 204.5                                                       | 5.37     | 13.66     |
| LD w/ ToMe (TF) [2]      | 50%     | 10%     | 60.77   | 0.1756     | 979.0                                       | 16.87    | 40.14     | 259.7                                                       | 6.82     | 13.01     |
| LD w/ ToMe (TF) [2]      | 50%     | 30%     | 279.36  | 0.1496     | 765.7                                       | 16.82    | 32.61     | 206.7                                                       | 6.64     | 11.14     |
| LD w/ AT-EDM (TF) [45]   | 50%     | 30%     | 203.35  | 0.1459     | 751.4                                       | 15.49    | 32.94     | 202.8                                                       | 6.11     | 11.91     |
| LD w/ MoD                | 50%     | 30%     | 18.34   | 0.1850     | 764.6                                       | 13.02    | 32.58     | 205.1                                                       | 4.83     | 11.11     |
| LD w/ <b>DiffCR-L</b>    | 50%     | 30%     | 13.53   | 0.1839     | 822.9                                       | 13.71    | 34.67     | 220.4                                                       | 5.30     | 11.30     |
| LD w/ DiffCR-LT          | 50%     | 30%     | 13.42   | 0.1845     | 804.9                                       | 13.89    | 34.88     | 214.5                                                       | 5.21     | 11.52     |

memory savings. Note that all memory measurements are taken with a batch size of 128. This set of experiments validates the effectiveness of our DiffCR when applied to image inpainting tasks.

#### 4.3. Ablation Studies of DiffCR

We conduct ablation studies on DiffCR, analyzing the contributions of the three enablers described in Sec. 3. As shown in Tabs. 1 and 2, we report the performance of LD or PixArt-Σ with MoD routers (Sec. 3.2), DiffCR-L (Sec. 3.3), DiffCR-LT (Sec. 3.4) for T2I and inpainting tasks, respectively. The results consistently show that all components of our DiffCR contribute to the final performance. Specifically, MoD alone achieves average FID reductions of 323.13 and 229.01 compared to ToMe [2] and AT-EDM [45] for the T2I task, with comparable or even lower latency and memory usage. DiffCR-L and DiffCR-

LT further enhance the generation quality, achieving additional FID reductions of 4.81/4.92 for the inpainting task and 10.5/12.1 for the T2I task.

Also, a key benefit of our DiffCR is that during fine-tuning, the average compression ratios across all layers gradually converge to the target ratio, producing a series of "byproduct" models with a range of compression ra-

<span id="page-6-2"></span>> **[图片提取文字 (无描述)]:**
> Inpainting 딢 Compression Ratios (%)
![](_page_6_Figure_9.jpeg)

Figure 7. Model trajectories.

tios. As shown in Fig. 7, we visualize the model trajectory with corresponding FID scores and compression ratios for both inpainting and T2I tasks. The observations indicate that, for T2I, the FID gradually increases with compression ratio, achieving the desired results. In contrast, for inpaint-

<span id="page-7-1"></span><span id="page-7-0"></span>> **[图片提取文字 (无描述)]:**
> **Lazy Diffusion** Masked Regenerate Regenerate LD w/ LD w/ PixArt-Σ w/ PixArt-Σ w/ PixArt-Σ MoD DiffCR Image Crop (LD) MoD DiffCR Input Mushroom Healthy homemade candies with nuts and dry fruits BMV e36 wallpaper full hd A strawberry vedding styled loot in reds with arge feather neadpiece The Lake asylum is still standing and still creepy Croissant (a) Inpainting Task (b) T2I Task
![](_page_7_Figure_0.jpeg)

Figure 6. Visual comparisons of our DiffCR with previous uncompressed models and SOTA compression methods: (a) Inpainting tasks, where DiffCR is applied to the LD models [28], and (b) T2I tasks, where DiffCR is applied to PixArt- $\Sigma$  [4].

ing, the FID gradually decreases. This difference arises because, compared to T2I, inpainting tasks and LD models are more sensitive to pruning and require longer finetuning to boost the generation quality. This is also reflected in Tabs. 2 and 1, where FID increases post-pruning for inpainting, while it even decreases for T2I.

#### 4.4. Qualitative Visual Examples

**Visual Examples.** We select challenging input prompts to evaluate the qualitative results of our proposed DiffCR. As shown in Fig. 6, the examples demonstrate that DiffCR achieves comparable or even superior generation quality compared to the RegenerateCrop baseline and even uncompressed LD or PixArt- $\Sigma$  for inpainting and T2I tasks, respectively. Note that ToMe and AT-EDM are omitted here due to their poor generation quality when applied to DiTs, even at a mere 10% compression ratio.

Human Preference Scores. We use a computer vision model to estimate likely human preferences and assess the models' ability to generate high-quality, contextually relevant images. Specifically, we generated 2K samples for the T2I task and used HPSv2 [51] to evaluate human preferences for images generated by different methods. As shown in Tab. 5, for T2I, we apply all compression methods to PixArt- $\Sigma$  [4]. DiffCR achieves a higher human preference score of 4.685/0.847 compared to previous compression methods, ToMe [2] and vanilla MoD [33], respectively.

Table 3. Human preference score (HPS) (†) comparison of the proposed DiffCR with other baselines on the T2I task.

| Methods                       | DiT C.R. | HPS Score |
|-------------------------------|----------|-----------|
| PixArt- $\Sigma$ (Fine-tuned) | 0%       | 22.582    |
| PixArt- $\Sigma$ w/ ToMe      | 20%      | 16.742    |
| PixArt- $\Sigma$ w/ MoD       | 20%      | 20.580    |
| PixArt- $\Sigma$ w/ DiffCR    | 20%      | 21.427    |

#### 5. Conclusion

In this work, we present DiffCR, a dynamic DiT inference framework with differentiable compression ratios that adaptively routes computation across tokens, layers, and timesteps, resulting in efficient DiT models. Specifically, DiffCR incorporates a token-level routing scheme based on MoD that dynamically learns the importance scores for each token, alongside a novel module that makes MoD differentiable with respect to compression ratios, enabling the model to learn adaptive compression ratios for each layer and timestep. Redundant layers and timesteps learn higher compression ratios, while critical layers and timesteps remain minimally compressed or uncompressed. Extensive experiments on both image inpainting and text-toimage (T2I) tasks consistently demonstrate DiffCR's superior trade-off between image generation quality and efficiency compared to other compression works.

# Acknowledgment

The work is supported in part by an internship at Adobe and in part by CoCoSys, one of the seven centers in JUMP 2.0, a Semiconductor Research Corporation (SRC) program sponsored by DARPA.

# References

- <span id="page-8-1"></span>[1] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 22669–22679, 2023. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-8-4"></span>[2] Daniel Bolya and Judy Hoffman. Token merging for fast stable diffusion. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 4599–4603, 2023. [1,](#page-0-0) [2,](#page-1-0) [4,](#page-3-3) [6,](#page-5-0) [7,](#page-6-3) [8,](#page-7-1) [5](#page-4-3)
- <span id="page-8-2"></span>[3] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. *arXiv preprint arXiv:2310.00426*, 2023. [1,](#page-0-0) [2,](#page-1-0) [5](#page-4-3)
- <span id="page-8-3"></span>[4] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-sigma: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. *arXiv preprint arXiv:2403.04692*, 2024. [1,](#page-0-0) [2,](#page-1-0) [4,](#page-3-3) [5,](#page-4-3) [7,](#page-6-3) [8](#page-7-1)
- <span id="page-8-17"></span>[5] Mengzhao Chen, Wenqi Shao, Peng Xu, Mingbao Lin, Kaipeng Zhang, Fei Chao, Rongrong Ji, Yu Qiao, and Ping Luo. Diffrate: Differentiable compression rate for efficient vision transformers. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 17164– 17174, 2023. [4](#page-3-3)
- <span id="page-8-13"></span>[6] Lei Deng, Guoqi Li, Song Han, Luping Shi, and Yuan Xie. Model compression and hardware acceleration for neural networks: A comprehensive survey. *Proceedings of the IEEE*, 108(4):485–532, 2020. [2](#page-1-0)
- <span id="page-8-8"></span>[7] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. *Advances in neural information processing systems*, 34:8780–8794, 2021. [2](#page-1-0)
- <span id="page-8-16"></span>[8] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In *International Conference on Learning Representations*, 2020. [4](#page-3-3)
- <span id="page-8-6"></span>[9] Gongfan Fang, Xinyin Ma, and Xinchao Wang. Structural pruning for diffusion models. In *Advances in Neural Information Processing Systems*, 2023. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-8-21"></span>[10] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. *arXiv preprint arXiv:2104.08718*, 2021. [6,](#page-5-0) [7](#page-6-3)
- <span id="page-8-20"></span>[11] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a

- two time-scale update rule converge to a local nash equilibrium. *Advances in neural information processing systems*, 30, 2017. [6,](#page-5-0) [7](#page-6-3)
- <span id="page-8-10"></span>[12] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. *arXiv preprint arXiv:2207.12598*, 2022. [2](#page-1-0)
- <span id="page-8-0"></span>[13] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. *Advances in neural information processing systems*, 33:6840–6851, 2020. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-8-14"></span>[14] Gao Huang, Danlu Chen, Tianhong Li, Felix Wu, Laurens Van Der Maaten, and Kilian Q Weinberger. Multi-scale dense networks for resource efficient image classification. *arXiv preprint arXiv:1703.09844*, 2017. [2](#page-1-0)
- <span id="page-8-12"></span>[15] Minguk Kang, Richard Zhang, Connelly Barnes, Sylvain Paris, Suha Kwak, Jaesik Park, Eli Shechtman, Jun-Yan Zhu, and Taesung Park. Distilling diffusion models into conditional gans. *ECCV 2024*, 2024. [2](#page-1-0)
- <span id="page-8-9"></span>[16] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. *Advances in neural information processing systems*, 35:26565–26577, 2022. [2](#page-1-0)
- <span id="page-8-5"></span>[17] Bo-Kyeong Kim, Hyoung-Kyu Song, Thibault Castells, and Shinkook Choi. Bk-sdm: A lightweight, fast, and cheap version of stable diffusion. *arXiv preprint arXiv:2305.15798*, 2023. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-8-23"></span>[18] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C Lawrence ´ Zitnick. Microsoft coco: Common objects in context. In *Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13*, pages 740–755. Springer, 2014. [5](#page-4-3)
- <span id="page-8-22"></span>[19] Haozhe Liu, Wentian Zhang, Jinheng Xie, Francesco Faccio, Mengmeng Xu, Tao Xiang, Mike Zheng Shou, Juan-Manuel Perez-Rua, and Jurgen Schmidhuber. Faster diffu- ¨ sion via temporal attention decomposition. *arXiv preprint arXiv:2404.02747*, 2024. [5](#page-4-3)
- <span id="page-8-18"></span>[20] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In *ICLR (Poster)*, 2019. [6](#page-5-0)
- <span id="page-8-19"></span>[21] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: a fast ode solver for diffusion probabilistic model sampling in around 10 steps. In *Proceedings of the 36th International Conference on Neural Information Processing Systems*, Red Hook, NY, USA, 2024. Curran Associates Inc. [6](#page-5-0)
- <span id="page-8-11"></span>[22] Xinyin Ma, Gongfan Fang, Michael Bi Mi, and Xinchao Wang. Learning-to-cache: Accelerating diffusion transformer via layer caching. *arXiv preprint arXiv:2406.01733*, 2024. [2,](#page-1-0) [5](#page-4-3)
- <span id="page-8-7"></span>[23] Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 14297–14306, 2023. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-8-15"></span>[24] Taehong Moon, Moonseok Choi, EungGu Yun, Jongmin Yoon, Gayoung Lee, and Juho Lee. Early exiting for accelerated inference in diffusion models. In *ICML 2023 Workshop on Structured Probabilistic Inference* & *Generative Modeling*, 2023. [2](#page-1-0)

- <span id="page-9-13"></span>[25] Giovane CM Moura, John Heidemann, Ricardo de O Schmidt, and Wes Hardaker. Cache me if you can: Effects of dns time-to-live. In *Proceedings of the Internet Measurement Conference*, pages 101–115, 2019. [2,](#page-1-0) [5](#page-4-3)
- <span id="page-9-19"></span>[26] Ravi Teja Mullapudi, William R Mark, Noam Shazeer, and Kayvon Fatahalian. Hydranets: Specialized dynamic architectures for efficient inference. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 8080–8089, 2018. [2](#page-1-0)
- <span id="page-9-21"></span>[27] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In *Proceedings of the 38th International Conference on Machine Learning*, pages 8162–8171. PMLR, 2021. [6](#page-5-0)
- <span id="page-9-4"></span>[28] Yotam Nitzan, Zongze Wu, Richard Zhang, Eli Shechtman, Daniel Cohen-Or, Taesung Park, and Michael Gharbi. Lazy ¨ diffusion transformer for interactive image editing. *arXiv preprint arXiv:2404.12382*, 2024. [1,](#page-0-0) [2,](#page-1-0) [4,](#page-3-3) [5,](#page-4-3) [6,](#page-5-0) [7,](#page-6-3) [8](#page-7-1)
- <span id="page-9-3"></span>[29] William Peebles and Saining Xie. Scalable diffusion models with transformers. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 4195–4205, 2023. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-9-0"></span>[30] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Muller, Joe Penna, and ¨ Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. *arXiv preprint arXiv:2307.01952*, 2023. [1,](#page-0-0) [2,](#page-1-0) [6,](#page-5-0) [7](#page-6-3)
- <span id="page-9-15"></span>[31] Yifan Pu, Zhuofan Xia, Jiayi Guo, Dongchen Han, Qixiu Li, Duo Li, Yuhui Yuan, Ji Li, Yizeng Han, Shiji Song, et al. Efficient diffusion transformer with step-wise dynamic attention mediators. *arXiv preprint arXiv:2408.05710*, 2024. [2](#page-1-0)
- <span id="page-9-5"></span>[32] Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. Dynamicvit: Efficient vision transformers with dynamic token sparsification. *Advances in neural information processing systems*, 34:13937–13949, 2021. [1](#page-0-0)
- <span id="page-9-10"></span>[33] David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, and Adam Santoro. Mixture-of-depths: Dynamically allocating compute in transformer-based language models. *arXiv preprint arXiv:2404.02258*, 2024. [2,](#page-1-0) [3,](#page-2-3) [8,](#page-7-1) [1,](#page-0-0) [5](#page-4-3)
- <span id="page-9-1"></span>[34] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image ¨ synthesis with latent diffusion models. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 10684–10695, 2022. [1,](#page-0-0) [2,](#page-1-0) [6](#page-5-0)
- <span id="page-9-8"></span>[35] Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. *arXiv preprint arXiv:2202.00512*, 2022. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-9-14"></span>[36] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In *European Conference on Computer Vision*, pages 87–103. Springer, 2025. [2](#page-1-0)
- <span id="page-9-24"></span>[37] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. *arXiv preprint arXiv:2111.02114*, 2021. [6,](#page-5-0) [7](#page-6-3)

- <span id="page-9-20"></span>[38] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. *Advances in Neural Information Processing Systems*, 35:25278–25294, 2022. [6,](#page-5-0) [7](#page-6-3)
- <span id="page-9-7"></span>[39] Ethan Smith, Nayan Saxena, and Aninda Saha. Todo: Token downsampling for efficient generation of high-resolution images. *arXiv preprint arXiv:2402.13573*, 2024. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-9-11"></span>[40] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In *International conference on machine learning*, pages 2256–2265. PMLR, 2015. [2](#page-1-0)
- <span id="page-9-2"></span>[41] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. *arXiv preprint arXiv:2010.02502*, 2020. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-9-18"></span>[42] Surat Teerapittayanon, Bradley McDanel, and Hsiang-Tsung Kung. Branchynet: Fast inference via early exiting from deep neural networks. In *2016 23rd international conference on pattern recognition (ICPR)*, pages 2464–2469. IEEE, 2016. [2](#page-1-0)
- <span id="page-9-12"></span>[43] A Vaswani. Attention is all you need. *Advances in Neural Information Processing Systems*, 2017. [2](#page-1-0)
- <span id="page-9-22"></span>[44] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, Dhruv Nair, Sayak Paul, William Berman, Yiyi Xu, Steven Liu, and Thomas Wolf. Diffusers: State-of-the-art diffusion models. [https://github.com/huggingface/](https://github.com/huggingface/diffusers) [diffusers](https://github.com/huggingface/diffusers), 2022. [6](#page-5-0)
- <span id="page-9-6"></span>[45] Hongjie Wang, Difan Liu, Yan Kang, Yijun Li, Zhe Lin, Niraj K Jha, and Yuchen Liu. Attention-driven training-free efficiency enhancement of diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 16080–16089, 2024. [1,](#page-0-0) [2,](#page-1-0) [6,](#page-5-0) [7,](#page-6-3) [5](#page-4-3)
- <span id="page-9-9"></span>[46] Kai Wang, Yukun Zhou, Mingjia Shi, Zhihang Yuan, Yuzhang Shang, Xiaojiang Peng, Hanwang Zhang, and Yang You. A closer look at time steps is worthy of triple speed-up for diffusion model training. *arXiv preprint arXiv:2405.17403*, 2024. [1,](#page-0-0) [2,](#page-1-0) [6,](#page-5-0) [3](#page-2-3)
- <span id="page-9-23"></span>[47] Su Wang, Chitwan Saharia, Ceslee Montgomery, Jordi Pont-Tuset, Shai Noy, Stefano Pellegrini, Yasumasa Onoe, Sarah Laszlo, David J Fleet, Radu Soricut, et al. Imagen editor and editbench: Advancing and evaluating text-guided image inpainting. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 18359– 18369, 2023. [6](#page-5-0)
- <span id="page-9-16"></span>[48] Xin Wang, Fisher Yu, Zi-Yi Dou, Trevor Darrell, and Joseph E Gonzalez. Skipnet: Learning dynamic routing in convolutional networks. In *Proceedings of the European conference on computer vision (ECCV)*, pages 409– 424, 2018. [2](#page-1-0)
- <span id="page-9-17"></span>[49] Yue Wang, Jianghao Shen, Ting-Kuei Hu, Pengfei Xu, Tan Nguyen, Richard Baraniuk, Zhangyang Wang, and Yingyan Lin. Dual dynamic inference: Enabling more efficient, adaptive, and controllable deep inference. *IEEE Journal of Selected Topics in Signal Processing*, 14(4):623–633, 2020. [2](#page-1-0)

- <span id="page-10-7"></span>[50] Stable Diffusion WebUI. [https://github.com/](https://github.com/AUTOMATIC1111/stable-diffusion-webui) [AUTOMATIC1111 / stable - diffusion - webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui), 2023. Accessed: 2024-11-10. [6](#page-5-0)
- <span id="page-10-8"></span>[51] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. *arXiv preprint arXiv:2306.09341*, 2023. [8,](#page-7-1) [5](#page-4-3)
- <span id="page-10-4"></span>[52] Zuxuan Wu, Tushar Nagarajan, Abhishek Kumar, Steven Rennie, Larry S Davis, Kristen Grauman, and Rogerio Feris. Blockdrop: Dynamic inference paths in residual networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 8817–8826, 2018. [2](#page-1-0)
- <span id="page-10-6"></span>[53] Shaoan Xie, Zhifei Zhang, Zhe Lin, Tobias Hinz, and Kun Zhang. Smartbrush: Text and shape guided object inpainting with diffusion model. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 22428–22437, 2023. [6](#page-5-0)
- <span id="page-10-2"></span>[54] Mengwei Xu, Mengze Zhu, Yunxin Liu, Felix Xiaozhu Lin, and Xuanzhe Liu. Deepcache: Principled cache for mobile deep vision. In *Proceedings of the 24th annual international conference on mobile computing and networking*, pages 129–144, 2018. [2,](#page-1-0) [5](#page-4-3)
- <span id="page-10-3"></span>[55] Tianwei Yin, Michael Gharbi, Taesung Park, Richard Zhang, ¨ Eli Shechtman, Fredo Durand, and William T Freeman. Improved distribution matching distillation for fast image synthesis. *arXiv preprint arXiv:2405.14867*, 2024. [2](#page-1-0)
- <span id="page-10-1"></span>[56] Tianwei Yin, Michael Gharbi, Richard Zhang, Eli Shecht- ¨ man, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6613–6623, 2024. [1,](#page-0-0) [2](#page-1-0)
- <span id="page-10-0"></span>[57] Wang Zeng, Sheng Jin, Wentao Liu, Chen Qian, Ping Luo, Wanli Ouyang, and Xiaogang Wang. Not all tokens are equal: Human-centric visual analysis via token clustering transformer. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 11101– 11111, 2022. [1](#page-0-0)
- <span id="page-10-5"></span>[58] Wangbo Zhao, Yizeng Han, Jiasheng Tang, Kai Wang, Yibing Song, Gao Huang, Fan Wang, and Yang You. Dynamic diffusion transformer. *arXiv preprint arXiv:2410.03456*, 2024. [2](#page-1-0)

# Layer- and Timestep-Adaptive Differentiable Token Compression Ratios for Efficient Diffusion Transformers

# Supplementary Material

# A. More Visualization of Token Routers

In Sec. 3.2, we provided an example visualization of the router predictions to evaluate the effectiveness of our DiffCR router. Here, we present additional visualization examples in Fig. [9](#page-12-0) to further validate our findings. Our observations consistently demonstrate the following: (1) *The router effectively captures semantic information*, clearly delineating object shapes and achieving an attention-like effect while significantly reducing computational costs. (2) *The predicted token importance varies across layers and timesteps*. For example, some layers focus on object generation, while others emphasize background areas. Additionally, as timesteps progress, the router increasingly captures the semantic contours of objects, highlighting the importance of dynamic token importance estimation. (3) *The optimal compression ratio differs across layers and timesteps*. For instance, some layers assign high importance to all tokens, indicating minimal redundancy, while others selectively prune tokens from objects or backgrounds with distinct shapes, requiring different compression ratios. This variance is also observed across timesteps. In the previous MoD [\[33\]](#page-9-10) approach, a fixed global compression rate is uniformly applied across layers and timesteps, ignoring their individual significance. Such uniform pruning risks over-pruning critical layers or timesteps while undercompressing redundant ones. This observation underscores the need for adaptive and dynamic compression ratios tailored to both layers and timesteps.

# B. Ratio Trajectory Analysis for the T2I Task

In Sec. 3.3, we visualized the ratio trajectory for inpainting tasks trained with our proposed layer-wise DiffCR. Here, we also provide the training trajectory of compression ratios for all layers during fine-tuning of a PixArt-Σ model on a T2I task, as shown in Fig. [8](#page-11-0) (a-c). The visualization consistently reveals that: (1) Each layer learns its unique compression ratio, with redundant layers achieving higher compression and critical layers remaining less or entirely uncompressed; (2) The average ratio across layers gradually converges to the target ratio. In this example, with a target of 20%, the final achieved average ratio is approximately 19%, indicating a minor gap. Notably, a trade-off exists between convergence speed and generation quality: a higher MSE loss coefficient for the ratio accelerates convergence but may degrade quality due to overly rapid compression, while a smaller coefficient promotes gradual convergence

<span id="page-11-0"></span>> **[图片提取文字 (无描述)]:**
> (a) (b) 0.200 - Layer 15 Layer 1 0.200 0.175 0.150 Avg. Ratio Layer 2 Layer 16 Layer 17 Compression Ratios Layer 18 Layer 19 Layer 5 Layer 20 Compression Layer 21 Layer 22 0.125 Layer 8 Layer 23 Layer 9 Layer 24 Layer 10 0.100 Layer 11 Layer 25 Layer 12 Layer 26 0.075 Layer 13 Layer 27 Layer 14 Layer 28 0.050 0.025 0.000 0.0 0 10 20 30 40 50 60 70 80 90 100 10 20 30 40 50 60 70 80 90 100 Training Iterations (k) Training Iterations (k) Compression Ratios 0.0 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 8 9 Layers
![](_page_11_Figure_6.jpeg)

Figure 8. Visualization of the compression ratio trajectory during fine-tuning for a T2I task: (a) Trajectories for each of the 28 layers in the PixArt-Σ model; (b) Average ratio trajectory across all layers; and (c) The final learned ratio distribution across 28 layers.

and maintains quality, albeit with slower training. In practice, we set the initial coefficient to 0.3 and dynamically adjust it during training to balance speed and quality effectively; (3) The middle layers exhibit greater redundancy, while the later layers generally have lower redundancy and often cannot be compressed. The early layers show variable redundancy levels.

Note that to prevent the model from learning 0% compression ratios across all layers, we balance diffusion loss (favoring lower ratios for higher quality) and MSE loss (driving the target average ratio) using a coefficient, without additional regularization or penalties. A higher coefficient speeds up convergence but may compromise quality, while a smaller one ensures gradual convergence and preserves quality. Some layers naturally learn 0% ratios, underscoring their importance.

# C. Correlation Between Learned Compression Ratios and Router Predictions

We select three representative layers with high, medium, and low learned compression ratios to visualize the corresponding predictions of the DiffCR router and analyze potential correlations. As shown in Fig. [10,](#page-12-1) where "C.R." denotes the compression ratios, we observe a strong correla-

<span id="page-12-0"></span>> **[图片提取文字 (无描述)]:**
> "eel sushí (a) (b) Input: Input: Inpainting Text-to-Image **Router's Prediction Router's Prediction** Timestep: Timestep: 60 80 100 21 26 Layer: 13 19 20 Layer: 2 3 17 10 "sky" Input: Input: "a lake" **Router's Prediction** Timestep: Timestep: **Router's Prediction** 40 60 80 15 100 20 21 26 7 13 17 3 19 20 Layer: 2 10 14 Layer:
![](_page_12_Figure_0.jpeg)

<span id="page-12-1"></span>Figure 9. More visualizations of the router's predictions: (a) For inpainting tasks, where inputs are masked images with text prompts, we follow the previous SOTA method Lazy-Diffusion [28] to generate only the masked area rather than the entire image; (b) For text-to-image (T2I) tasks, where inputs are noise and text prompts, we follow  $PixArt-\Sigma$  [4] for generation. Each visualization includes the router's prediction map with values ranging from 0 to 1. The generated image at each corresponding timestep is shown on the left, while the router's prediction maps across various layers and timesteps are displayed on the right.

> **[图片提取文字 (无描述)]:**
> (a) Inpainting (b) Text-to-Image 22 Layer: Layer: C.R.: 2% C.R.: 0% 35% 87% 54% 90%
![](_page_12_Figure_2.jpeg)

Figure 10. Visualization and analysis of the correlation between the learned compression ratios and the DiffCR router's predictions.

<span id="page-13-0"></span>> **[图片提取文字 (无描述)]:**
> Lazy Diffusion (LD) (p=0%)LD w/ Model Pruning (p=30%) PixArt- $\Sigma$  (p=0%) LD w/ ToMe (p=30%) PixArt- $\Sigma$  w/ ToMe (p=20%) LD w/ ToMe (p=10%) PixArt- $\Sigma$  w/ ToMe (p=10%) LD w/ AT-EDM (p=30%) PixArt- $\Sigma$  w/ AT-EDM (p=20%) LD w/ MoD (p=30%) PixArt- $\Sigma$  w/ MoD (p=20%) LD w/ DiffCR (p=30%) PixArt- $\Sigma$  w/ **DiffCR** (p=20%) Diameter 0.77 0.98 13.9 16.4  $10^{2}$ **TFLOPS** Diameter  $10^{2}$ TFLOPS 24 18 22 4.92 FID Reduction 20 16 18 ₽ 14 12.10 FID Reduction 16 20% Latency Savings 20% Latency Savings 14 12 12 10 10 0.80 0.85 0.80 0.85 1.00 0.75 0.90 0.95 1.00 1.05 0.70 0.75 0.90 0.95 1.05 Relative Latency (BS=1) Relative Latency (BS=16) (b) Inpainting (a) T2I
![](_page_13_Figure_0.jpeg)

Figure 11. Overall comparison of DiffCR with baselines in terms of latency, FID, and TFLOPS for both T2I and inpainting tasks.

tion between the learned ratios and the router's predictions. For layers with high compression ratios, such as layer 1 in inpainting or layer 9 in T2I, the router consistently predicts lower importance scores for many semantic areas, adopting an extremely "lazy behavior" to save computations. Conversely, for layers with low compression ratios, the router assigns higher importance scores to most areas. This visualization validates the joint learning effect between our token-level routers and the differentiable ratios.

# D. Trade-offs for Choosing Timestep Regions

In Sec. 3.4, we introduced the timestep-wise DiffCR, where the timestep regions are evenly divided into 10 regions for inpainting tasks with a total of 100 sampling timesteps, and 4 regions for T2I tasks with 20 sampling timesteps. Here, we provide additional guidance on selecting the number of timestep regions and the associated trade-offs. A larger number of timestep regions allows for learning finer-grained and more precise compression ratios across all timesteps. However, too many regions can make training unstable and challenging. To reduce training complexity and enhance stability, we select a smaller number of regions, such as 4 for T2I tasks. Conversely, using too few regions risks oversimplifying the method, reducing it to heuristic approaches like SpeeD [\[46\]](#page-9-9), which manually defines three timestep regions. In practice, we choose between 4 and 10 timestep regions to balance granularity and stability. While our approach aligns with the general insights of SpeeD, it is more systematic and adaptive. Unlike manual exploration of a large design space, our method efficiently handles a significantly greater number of regions in a principled manner, balancing granularity and training stability.

# E. Overall Comparison Figure

In Sec. 4.2, we presented a comprehensive comparison of our DiffCR method against baseline approaches for both inpainting and T2I tasks. Here, we provide the overall comparison figures to better illustrate the achieved im-

<span id="page-14-1"></span>> **[图片提取文字 (无描述)]:**
> **Lazy Diffusion** LD w/ Masked Regenerate Regenerate LD w/ PixArt-Σ w/ PixArt-Σ w/ PixArt-Σ DiffCR MoD DiffCR MoD Input Image Crop (LD) Heber c. kimball, october 7, 1853 Strawberry Vermont lake Witch by artevoletia Petals Kakashi hatake white background anbu wallpaper Buns Jason momoa A cloudy sky sunset Hawaii fizzes with volcanic activity Cucumber A building with many windows and some clouds in the sky A little owl took advantage of a downpour, to spread its wings and bathe (b) T2I Task (a) Inpainting Task
![](_page_14_Figure_0.jpeg)

Figure 12. Additional visual comparisons of our DiffCR with previous uncompressed models and SOTA compression methods: (a) Inpainting tasks, where DiffCR is applied to LD models [28], and (b) T2I tasks, where DiffCR is applied to PixArt- $\Sigma$  [4].

provements in FID and latency reductions. As shown in Fig. 11, our DiffCR consistently delivers superior trade-offs between FID and latency, achieving FID reductions of 12.10 and 4.92 for T2I and inpainting tasks, respectively, at comparable GPU latency when compared to the most competitive baseline.

#### F. Model Trajectories of DiffCR

In Sec. 4.2, we visualized the model trajectories during the training of DiffCR-L for both T2I and inpainting tasks. This revealed a key benefit: during fine-tuning, the averaged compression ratios across all layers gradually converge to the target ratio, producing a series of "by-product" models with varying compression ratios. Here, we also supply the model trajectories of DiffCR-LT ("-LT" denotes layer-and timestep-wise DiffCR). As shown in Fig. 13, we visualize the FID scores and corresponding compression ratios during the fine-tuning of DiffCR-LT. The observations consistently validate the benefits of this approach, showing that it enables the generation of a series of models with diverse compression ratios. Also, we observe that inpainting tasks

<span id="page-14-0"></span>> **[图片提取文字 (无描述)]:**
> (a) DiffCR-L (b) DiffCR-LT Compression Ratios (%) Compression Ratios (%)
![](_page_14_Figure_5.jpeg)

Figure 13. Model trajectories of DiffCR.

and Latent Diffusion (LD) models [28] are more sensitive to pruning and require longer fine-tuning to improve generation quality effectively, compared to T2I tasks. Moreover, for T2I tasks, DiffCR-LT demonstrates slightly greater stability in model trajectory compared to DiffCR-L.

#### **G.** More Visualization of Visual Examples

In Sec. 4.4, we selected challenging input prompts to evaluate the qualitative performance of our proposed DiffCR.

Table 4. Characteristics of our method and caching-based baselines.

<span id="page-15-1"></span>

| Method         | Model | Skip / Cache | Granulariy | Learnable | Token<br>Pruning | Timestep-wise<br>Feature Cache |  |
|----------------|-------|--------------|------------|-----------|------------------|--------------------------------|--|
| DeepCache [54] | U-Net | Block        | Block      | Х         | Х                | ✓                              |  |
| CMYC [25]      | U-Net | Block        | Block      | X         | X                | ✓                              |  |
| L2C [22]       | DiT   | Attn. & MLP  | Layer      | ✓         | X                | ✓                              |  |
| TGATE [19]     | DiT   | Attn.        | Layer      | X         | X                | ✓                              |  |
| DiffCR (Ours)  | DiT   | Attn. & MLP  | Token      | ✓         | ✓                | X but compatible               |  |

Here, we provide additional visual examples, as shown in Fig. 12. The examples consistently demonstrate that DiffCR achieves comparable or even superior generation quality compared to the RegenerateCrop baseline and even uncompressed LD or PixArt-Σ for inpainting and T2I tasks, respectively. Note that ToMe [2] and AT-EDM [45] are omitted here due to their poor generation quality when applied to DiTs, even at a modest compression ratio of 10%.

### H. Comparison with Caching-based Baselines

We summarize the characteristics of our method and caching-based baselines in Tab. 4. DeepCache [54] and CMYC [25] are designed for U-Net-based models, making direct comparison challenging, while L2C [22] and TGATE [19] target DiTs by caching layer features to reduce recomputation in future timesteps. Unlike these approaches, our method focuses on token pruning with learnable layer- and timestep-dependent compression ratios, and while it does not employ temporal caching, it remains compatible with such techniques. To directly compare, we evaluate all methods using PixArt-Σ on the MS-COCO-30K dataset (T2I task) under approximately 25% latency savings, where L2C achieves an FID of 28.6 (with our trained routers reproducing a similar caching pattern as reported), TGATE yields 43.6 FID, and our DiffCR achieves 28.6 FID. These results show that our method performs comparably to or better than caching-based baselines, and it can be further combined with them to achieve an additional 15  $\sim$  30% latency reduction.

### I. Human Preference Score for Inpainting

In Sec. 4.4, we utilized a computer vision model to estimate likely human preferences and evaluate the ability of models to generate high-quality, contextually relevant images for the T2I task. Here, we also provide the evaluation for inpainting tasks. Specifically, we generated 2K samples for the inpainting task and used HPSv2 [51] to assess human preferences for images produced by different methods. As shown in Tab. 5, for inpainting tasks, we applied all compression methods to Lazy Diffusion (LD) [28]. DiffCR achieves a higher human preference score of 2.181/0.263 compared to previous compression methods, ToMe [2] and

<span id="page-15-0"></span>Table 5. Human Preference Score (HPS) (†) comparison of the proposed DiffCR with baselines for the inpainting task.

| Methods             | DiT C.R. | HPS Score |
|---------------------|----------|-----------|
| RegenerateImage     | 0%       | 21.056    |
| RegenerateCrop      | 0%       | 19.466    |
| Lazy Diffusion (LD) | 0%       | 20.464    |
| LD w/ ToMe          | 30%      | 18.187    |
| LD w/ MoD           | 30%      | 20.105    |
| LD w/ DiffCR        | 30%      | 20.368    |

Table 6. Ablation study on the impact of different compression ratios with a batch size of 16.

| <b>Metrics\Ratios</b>                              |       |       |       |       |       |       |
|----------------------------------------------------|-------|-------|-------|-------|-------|-------|
| FID Score (↓)                                      | 27.80 | 27.53 | 28.64 | 28.57 | 28.44 | 29.21 |
| CLIP Score (↑)                                     | 16.23 | 16.28 | 16.44 | 16.37 | 16.37 | 16.37 |
| FID Score (↓)<br>CLIP Score (↑)<br>T2I Latency (s) | 11.90 | 11.16 | 10.31 | 9.23  | 8.19  | 7.12  |

vanilla MoD [33], respectively.

# J. Ablation Analysis on Compression Ratios

In this work, we target lower latency as a step toward edge deployment. To analyze the effect of varying compression ratios, we conducted an ablation study using the PixArt- $\Sigma$  model on the MS-COCO-30K dataset [18]. Notably, 1/3 of the timesteps were allocated to full-model inference to preserve accuracy. The results in the table below show that our method scales effectively to larger compression ratios, with only a slight increase in FID (<1). A 30% compression ratio was previously selected for challenging generation tasks to maintain accuracy while building upon existing state-of-the-art efficient methods.

#### K. Is MSE Loss Alone Sufficient?

We found that simply using the MSE loss effectively guides ratios toward the target without additional regularization, so we fixed it to MSE loss, but other loss functions may also work well. In addition, although we did not enforce binary prediction, the routers tend to learn a polarized distribution in some layers, separating important tokens from unimportant ones, with the learned ratios aligning accordingly, as shown in Fig. 10.