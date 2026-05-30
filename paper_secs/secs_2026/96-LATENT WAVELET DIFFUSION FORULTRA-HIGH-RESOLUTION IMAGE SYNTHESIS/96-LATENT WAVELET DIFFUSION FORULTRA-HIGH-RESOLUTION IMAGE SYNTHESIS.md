## LATENT WAVELET DIFFUSION FOR ULTRA-HIGH-RESOLUTION IMAGE SYNTHESIS

#### Luigi Sigillo1,2,3<sup>∗</sup> , Shengfeng He<sup>2</sup> , Danilo Comminiello<sup>1</sup>

<sup>1</sup>Sapienza University of Rome, <sup>2</sup>Singapore Management University, <sup>3</sup>EMBL luigi.sigillo@uniroma1.it

![](_page_0_Picture_5.jpeg)

Figure 1: We propose Latent Wavelet Diffusion, achieving 4K image synthesis without architectural changes or additional inference cost to existing latent diffusion models.

## ABSTRACT

High-resolution image synthesis remains a core challenge in generative modeling, particularly in balancing computational efficiency with the preservation of finegrained visual detail. We present *Latent Wavelet Diffusion (LWD)*, a lightweight training framework that significantly improves detail and texture fidelity in ultrahigh-resolution (2K-4K) image synthesis. LWD introduces a novel, frequencyaware masking strategy derived from wavelet energy maps, which dynamically focuses the training process on detail-rich regions of the latent space. This is complemented by a scale-consistent VAE objective to ensure high spectral fidelity. The primary advantage of our approach is its efficiency: LWD requires no architectural modifications and adds zero additional cost during inference, making it a practical solution for scaling existing models. Across multiple strong baselines, LWD consistently improves perceptual quality and FID scores, demonstrating the power of signal-driven supervision as a principled and efficient path toward high-resolution generative modeling. The code is available at [https:](https://github.com/LuigiSigillo/LatentWaveletDiffusion) [//github.com/LuigiSigillo/LatentWaveletDiffusion](https://github.com/LuigiSigillo/LatentWaveletDiffusion).

## 1 INTRODUCTION

Diffusion models have become a dominant paradigm in generative modeling, achieving state-of-theart results in tasks such as image synthesis [\(Podell et al., 2024;](#page-13-0) [Sauer et al., 2024;](#page-14-0) [Lopez et al., 2025;](#page-13-1) [Wang et al., 2025\)](#page-15-0), super-resolution [\(Saharia et al., 2022;](#page-14-1) [Sigillo et al., 2024;](#page-14-2) [Wang et al., 2024\)](#page-15-1), and image editing [\(Brooks et al., 2023;](#page-11-0) [Meng et al., 2022\)](#page-13-2). Foundational methods like Denoising Diffusion Probabilistic Models (DDPM) [Ho et al.](#page-12-0) [\(2020\)](#page-12-0) and Denoising Diffusion Implicit Models

<sup>∗</sup>Work done during a visiting period at SMU. Now at European Molecular Biology Laboratory.

(DDIM) [Song et al.](#page-14-3) [\(2021a\)](#page-14-3) have enabled increasingly powerful variants. Latent Diffusion Models (LDMs) [Rombach et al.](#page-14-4) [\(2022\)](#page-14-4) improve efficiency by operating in a learned latent space, while recent architectures such as Diffusion Transformers (DiTs) [\(Peebles & Xie, 2023;](#page-13-3) [Esser et al., 2024\)](#page-12-1) further enhance scalability and modeling capacity.

Despite recent advances, generating ultra-high-resolution (UHR) images at resolutions of 2K to 4K and beyond remains a significant challenge [\(Tragakis et al., 2024\)](#page-15-2). Simply scaling models trained on lower-resolution data often fails to generalize, leading to repeated structures, blurred textures, and spatial inconsistencies [\(Bar-Tal et al., 2023;](#page-11-1) [He et al., 2023\)](#page-12-2). Naive alternatives, such as hierarchical generation pipelines and post-hoc super-resolution techniques [\(Du et al., 2024a;](#page-11-2) [He et al., 2023\)](#page-12-2), typically produce oversmoothed outputs. To overcome these limitations, several strategies have been explored. One direction involves direct UHR training or fine-tuning [\(Ren et al., 2024;](#page-14-5) [Chen et al.,](#page-11-3) [2024\)](#page-11-3), although this typically demands extensive computational resources, access to proprietary highresolution datasets [\(Liu et al., 2024b\)](#page-13-4), and substantial GPU memory for large model backbones [\(Labs,](#page-13-5) [2024;](#page-13-5) [Esser et al., 2024\)](#page-12-1). Other efforts focus on architectural modifications to improve long-range dependency modeling [\(Liu et al., 2024a\)](#page-13-6), or optimization techniques that enhance the quality of latent representations [\(Hahm et al., 2024\)](#page-12-3).

Across these varied approaches, a common limitation persists: most methods treat all spatial regions uniformly during generation, applying the same refinement process to areas with vastly different visual complexity. This uniformity disregards local frequency variation, failing to distinguish between smooth regions and areas rich in textures, edges, or semantic structure. The consequences are twofold. First, computation is wasted on low-detail areas that require minimal refinement. Second, high-detail regions are not sufficiently supervised, leading to artifacts or loss of fidelity. These issues stem from both architectural and algorithmic constraints: latent representations often lack the structural granularity required for UHR synthesis, and current diffusion training objectives do not incorporate spatial adaptivity into the denoising process. Together, these limitations present a core bottleneck for perceptually accurate ultra-high-resolution image generation.

In this work, we propose *Latent Wavelet Diffusion* (LWD), a general and modular framework that introduces frequency-sensitive spatial supervision into the latent denoising process of diffusion models. LWD is motivated by the observation that different regions of an image exhibit varying levels of structural complexity and perceptual importance. While some areas contain intricate textures or sharp edges, others are homogeneous or low in detail. Our goal is to exploit this spatial heterogeneity by allocating greater learning signal to regions with high visual complexity, and reducing supervision in low-detail areas. Importantly, LWD achieves this adaptivity without modifying the underlying architecture of the diffusion model, making it broadly applicable across model families.

The LWD framework consists of three key components:

- 1. A spectrally-aware VAE fine-tuning objective that improves the structure and diffusability of latent representations. By encouraging semantic consistency and frequency regularity, this objective enhances the suitability of latent spaces for high-resolution generation. It serves as the foundation for the subsequent components of LWD.
- 2. A wavelet-derived spatial saliency map, computed via a discrete wavelet transform (DWT) applied to the latent representation. This map aggregates the local energy of high-frequency subbands (LH, HL, HH) and is normalized to highlight spatial regions with strong structural detail. These saliency maps are interpretable, data-driven, and require no additional training, providing a principled measure of spatial importance directly from the signal.
- 3. A time-dependent masking strategy that leverages the frequency-based saliency maps to modulate the training loss. At each spatial location and timestep, a binary mask determines whether the denoising loss is applied. Regions with high wavelet energy receive supervision across more timesteps, while simpler areas are updated less frequently. This mechanism introduces spatial adaptivity into the learning process and improves the fidelity of fine-scale detail.

LWD is compatible with a broad range of latent diffusion models, including both classical diffusion and flow-matching formulations. Because it operates solely through the training objective, LWD can be seamlessly integrated into existing pipelines. While ultra-high-resolution generation is inherently computationally demanding due to the backbone's scaling properties, LWD incurs zero marginal cost relative to the baseline. It requires no architectural changes or cascaded upsamplers, making it a practical solution for improving the generation quality of existing models. We demonstrate its

flexibility and effectiveness by applying it to several state-of-the-art latent diffusion models and evaluating performance on ultra-high-resolution image synthesis (2K to 4K). Experimental results show that LWD consistently enhances perceptual quality, improves FID scores, and better preserves fine-grained detail, all without increasing inference complexity.

## 2 RELATED WORK

Diffusion Models and Latent Generation. Diffusion models have become foundational in generative modeling, particularly for image synthesis [\(Shen et al., 2025;](#page-14-6) [Zhan et al., 2025\)](#page-15-3), by progressively denoising Gaussian noise using a learned score function. Variants based on stochastic differential equations [\(Song et al., 2021b\)](#page-14-7), probability flow ordinary differential equations [\(Lipman et al., 2023\)](#page-13-7), and reinforcement-trained objectives [\(Black et al., 2024\)](#page-11-4) have expanded the design space with improved stability and sampling efficiency.

To reduce the cost of high-resolution generation, Latent Diffusion Models (LDMs) [\(Rombach et al.,](#page-14-4) [2022\)](#page-14-4) perform synthesis in a compressed space learned via variational autoencoders (VAEs). However, generation quality is closely tied to the spectral fidelity and structure of these latent representations. Prior work has sought to improve this through enhanced VAE architectures [\(Esser et al., 2021\)](#page-11-5), hierarchical compression [\(Takida et al., 2024\)](#page-14-8), and frequency-aware regularization [\(Skorokhodov](#page-14-9) [et al., 2025;](#page-14-9) [Kouzelis et al., 2025\)](#page-13-8). We build on this direction by integrating frequency-sensitive supervision both during encoding and throughout the denoising process.

Flow Matching and High-Resolution Diffusion. Flow matching [Lipman et al.](#page-13-7) [\(2023\)](#page-13-7); [Esser et al.](#page-12-1) [\(2024\)](#page-12-1) offers an alternative to classical diffusion by learning a continuous velocity field that maps noise to data in latent space, eliminating the need for fixed noise schedules. This formulation underlies models such as Flux [Labs](#page-13-5) [\(2024\)](#page-13-5), which, paired with DiT backbones, has demonstrated strong performance in ultra-high-resolution pipelines [\(Zhang et al., 2025;](#page-16-0) [Yu et al., 2025\)](#page-15-4). Our method extends this family by introducing frequency-based spatial masking into the flow-matching objective. Through wavelet decomposition of the latent space, LWD computes spatial saliency maps that guide targeted supervision toward detail-rich regions, enhancing fine structure without modifying model architecture.

Variational Autoencoder Optimization. The performance of latent diffusion models at high resolutions depends critically on the expressiveness and spectral consistency of the VAE. Improvements include multi-scale encoders [\(Vahdat & Kautz, 2020;](#page-15-5) [Takida et al., 2024\)](#page-14-8), spectral loss functions [\(Bjork](#page-11-6) ¨ [et al., 2022\)](#page-11-6), and scale-consistency constraints [\(Skorokhodov et al., 2025;](#page-14-9) [Kouzelis et al., 2025\)](#page-13-8). Wavelet-based methods [\(Esteves et al., 2025;](#page-12-4) [Lin et al., 2024;](#page-13-9) [Agarwal et al., 2025\)](#page-11-7) enrich latent expressiveness by isolating frequency components, while compression-oriented approaches [\(Xie](#page-15-6) [et al., 2025;](#page-15-6) [Tang et al., 2024;](#page-15-7) [HaCohen et al., 2024\)](#page-12-5) aim to reduce token count for improved sampling efficiency. Our LWD fine-tunes a pretrained VAE with a scale-consistent spectral loss that suppresses spurious high-frequency noise. This regularized latent space facilitates downstream wavelet decomposition and supports our spatially adaptive denoising objective.

Ultra High-Resolution Image Synthesis. Maintaining global structure and fine detail in ultraresolution generation is a persistent challenge. Standard diffusion models tend to produce repetitive patterns or distortions [\(He et al., 2023\)](#page-12-2). Existing solutions include cascaded generation [\(Ho et al.,](#page-12-6) [2022\)](#page-12-6), progressive upsampling [\(Gu et al., 2024\)](#page-12-7), domain-specific pipelines [\(Ren et al., 2024\)](#page-14-5), and latent-space super-resolution [\(Jeong et al., 2025\)](#page-12-8). However, training-based methods [\(Xie et al., 2023;](#page-15-8) [Zheng et al., 2024;](#page-16-1) [Guo et al., 2024;](#page-12-9) [Chen et al., 2024\)](#page-11-3) are resource-intensive, and training-free approaches [\(Bar-Tal et al., 2023;](#page-11-1) [Lee et al., 2023\)](#page-13-10) often yield local artifacts.

Methods such as ScaleCrafter [He et al.](#page-12-2) [\(2023\)](#page-12-2) mitigate repetition through dilated convolutions but may distort structure, while ResMaster [Shi et al.](#page-14-10) [\(2025\)](#page-14-10) uses low-resolution references for guided refinement. HiDiffusion [Zhang et al.](#page-16-2) [\(2023\)](#page-16-2) introduces architectural changes that risk performance trade-offs, and progressive strategies like DemoFusion [Du et al.](#page-11-2) [\(2024a\)](#page-11-2) suffer from slow inference and irregular patterns. Diffusion-4K [Zhang et al.](#page-16-0) [\(2025\)](#page-16-0) and URAE [Yu et al.](#page-15-4) [\(2025\)](#page-15-4) advance latent modeling and parameter-efficient adaptation at 4K resolution. Our LWD complements these approaches by introducing signal-driven, spatially adaptive supervision, which improves structural and perceptual fidelity at no additional cost.

Generative Modeling in the Frequency Domain. Frequency structure plays an increasingly important role in generative modeling [\(Yang et al., 2023\)](#page-15-9). Wavelet-based diffusion methods, such

![](_page_3_Figure_1.jpeg)

Figure 2: (a) Temporal evolution of latent zt, wavelet energy maps Awavelet, and attention map M<sup>t</sup> across diffusion timesteps. (b) Our wavelet-masked flow matching objective at a timestep t. The model computes a wavelet attention map M<sup>t</sup> from latent z<sup>t</sup> to modulate the prediction error between target velocity field (ϵ − z0) and predicted velocity vΘ(zt, t, y). This focuses optimization on high-frequency regions with greater perceptual importance. While operations occur in latent space, decoded visualizations are shown for interpretability.

as WaveDiff [Phung et al.](#page-13-11) [\(2023\)](#page-13-11), and spectral decomposition approaches have proven useful for efficient sampling [\(Qian et al., 2024\)](#page-13-12), super-resolution [\(Aloisi et al., 2026;](#page-11-8) [Sigillo et al., 2025\)](#page-14-11), and restoration [\(Huang et al., 2024b;](#page-12-10) [Zhao et al., 2024;](#page-16-3) [Jiang et al., 2023\)](#page-12-11). FouriScale [Huang et al.](#page-12-12) [\(2024a\)](#page-12-12) demonstrated that frequency filtering enhances coherence, and DiffuseHigh [Kim et al.](#page-12-13) [\(2025\)](#page-12-13) leveraged low-frequency DWT guidance to improve global structure in UHR synthesis.

Diffusion-4K [\(Zhang et al., 2025\)](#page-16-0) incorporated wavelet losses in the latent space to balance frequency bands, but applied them uniformly across all spatial locations. In contrast, LWD introduces waveletbased spatial conditioning through time-dependent masking. Rather than treating frequency as a passive loss signal, our LWD actively uses local frequency energy to modulate supervision across space and time, concentrating on learning where detail matters most. This enables sharper, more coherent synthesis without modifying the underlying model or increasing inference cost.

## 3 METHODOLOGY

*Latent Wavelet Diffusion (LWD)* introduces frequency-aware supervision into latent diffusion models by coupling signal-driven saliency analysis with adaptive training. Our key insight is that structural complexity in images is unevenly distributed in space, yet most denoising models refine all positions equally. LWD addresses this by modulating the supervision schedule based on local frequency content, improving detail fidelity without increasing computational cost.

LWD operates in two sequential stages. The first stage fine-tunes a variational autoencoder (VAE) using a scale-consistency objective. This independent step yields spectrally stable latent representations, preparing the space for the subsequent frequency-based modulation.

In the second stage, we fine-tune a latent diffusion model (e.g., Flux) using a modified flow-matching objective that integrates frequency-guided supervision. This stage incorporates three tightly coupled components: (1) extraction of wavelet-based spatial saliency maps from latent codes; (2) construction of a time-dependent mask that adapts the training signal based on local frequency energy; and (3) application of this mask to modulate the training loss dynamically across spatial positions and timesteps. Together, these mechanisms introduce spatial adaptivity into the denoising process, directing learning resources toward detail-rich regions. All components are model-agnostic and can be integrated into standard latent diffusion pipelines without architectural changes.

#### <span id="page-3-0"></span>3.1 VAE FINE-TUNING WITH SCALE-CONSISTENCY LOSS

High-resolution generation places unique demands on the latent space: it must retain both semantic structure and spectral coherence across scales. To ensure this, we fine-tune the variational autoencoder (VAE) using a multi-resolution reconstruction objective that regularizes frequency content while preserving perceptual fidelity.

Formally, let z = E(x) be the latent encoding of image x,  $x_{\text{down}}$  a downsampled version of x, and  $z_{\text{down}}$  the downsampled version of the latent z. Our loss combines four components:

<span id="page-4-0"></span>
$$\mathcal{L}_{\text{VAE}} = \underbrace{\|D(z) - x\|_2^2}_{\text{Reconstruction}} + \alpha \underbrace{\|D(E(z_{\text{down}})) - x_{\text{down}}\|_2^2}_{\text{Scale Consistency}} + \beta \underbrace{D_{\text{KL}}(q(z \mid x) \parallel p(z))}_{\text{Latent Regularization}} + \lambda \underbrace{\mathcal{L}_{\text{LPIPS}}(D(z), x)}_{\text{Perceptual Loss}},$$

Here, we incorporate a scale-consistency term (Skorokhodov et al., 2025; Kouzelis et al., 2025) that encourages the VAE to maintain structural coherence across resolution scales. While originally proposed for general reconstruction, we identify it as critical for wavelet-guided UHR synthesis. Without this regularization, standard VAEs exhibit spurious high-frequency noise that confounds downstream wavelet masking. This preprocessing naturally suppresses compression artifacts while preserving essential structural information in z, enabling our masking strategy to target meaningful details rather than noise.

Unlike recent approaches that inject frequency conditioning directly into the encoder (Aloisi et al., 2026; Zhang et al., 2025), our formulation decouples signal regularization and generation: we first sculpt the latent space to exhibit desirable

![](_page_4_Figure_6.jpeg)

Figure 3: Normalized Discrete Cosine Transform amplitudes over zigzag frequency indices. The 'RGB' curve represents the target spectrum of real images. Our tuning (+SE) suppresses high-frequency energy corresponding to artifacts, aligning the latent spectrum with the RGB reference to ensure a cleaner latent space.

frequency properties, and then use this structure to guide the training of the diffusion model. This preserves architectural modularity while enabling effective frequency-aware supervision.

#### 3.2 WAVELET-DERIVED FREQUENCY SALIENCY MAPS

To guide spatial supervision based on structural complexity, we extract saliency maps from latent representations using localized frequency analysis. Given a latent tensor  $z \in \mathbb{R}^{C \times H \times W}$ , we apply a single-level Discrete Wavelet Transform (DWT), producing four subbands:

$$DWT(z) \to \{z_{LL}, z_{LH}, z_{HL}, z_{HH}\},\tag{2}$$

where  $z_{LL}$  captures low-frequency approximations and  $\{z_{LH}, z_{HL}, z_{HH}\}$  encode directional high-frequency detail.

We compute the localized high-frequency energy as:

$$E(i,j) = \frac{1}{C} \sum_{c} \left[ (z_{LH}^{c,i,j})^2 + (z_{HL}^{c,i,j})^2 + (z_{HH}^{c,i,j})^2 \right], \tag{3}$$

where (i,j) denotes spatial position and  $c \in \{1,\ldots,C\}$  indexes feature channels. The resulting energy map  $E \in \mathbb{R}^{H/2 \times W/2}$  is bilinearly upsampled and min-max normalized per sample to obtain the final saliency map  $A_{\text{wavelet}} \in [0,1]^{H \times W}$ .

This map serves as a proxy for local structural richness, highlighting regions of the latent space associated with high-frequency content (e.g., textures, contours, transitions). Unlike learned attention mechanisms based on semantic similarity (e.g., DINO (Caron et al., 2021)), our approach is deterministic and directly derived from signal properties. While we refer to  $A_{\rm wavelet}$  as an "attention map" for interpretability, it is best understood as a frequency-aware saliency measure.

Notably, our VAE fine-tuning objective (Eq. 1) helps suppress high-frequency artifacts and stabilize spectral behavior. This preprocessing step ensures that high-frequency activations captured by the DWT correspond to meaningful structure, rather than encoding noise, thereby improving the utility of  $A_{\rm wavelet}$  for guiding spatial supervision.

#### 3.3 ADAPTIVE FLOW MATCHING WITH FREQUENCY-GUIDED MASKING

We adopt a continuous-time flow-matching formulation [\(Lipman et al., 2023;](#page-13-7) [Esser et al., 2024\)](#page-12-1) for training the latent diffusion model. Given a target latent z<sup>0</sup> and noise sample ϵ ∼ N (0, I), we define interpolated samples as:

$$z_t = (1 - t) z_0 + t \epsilon, \quad t \in [0, 1],$$
 (4)

and supervise the predicted velocity field vΘ(zt, t, y), conditioned on text y, via:

$$\mathcal{L}_{fm} = \|(\epsilon - z_0) - v_{\Theta}(z_t, t, y)\|_2^2.$$
 (5)

To incorporate frequency-awareness into the training objective, we apply spatially adaptive masking based on the wavelet saliency map Awavelet. Specifically, for each location (i, j), we define a timedependent binary mask:

$$M_t(i,j) = \begin{cases} 1 & \text{if } T \cdot (A_{\text{wavelet}}(i,j) + \ell) \ge t \\ 0 & \text{otherwise} \end{cases}$$
 (6)

where T is the total number of diffusion timesteps and ℓ ∈ (0, 1) sets a lower bound on refinement. This ensures that all regions receive at least ℓT steps of supervision, while high-frequency regions benefit from extended refinement.

While selective spatial supervision has been investigated using transformer attention [\(Moser et al.,](#page-13-13) [2025\)](#page-13-13), our wavelet-derived saliency offers a fundamentally different perspective based on signal processing principles rather than learned semantic features, offering computational advantages and a more generalizable solution.

The final masked loss becomes:

$$\mathcal{L}_{\text{masked}} = \|M_t \odot [(\epsilon - z_0) - v_{\Theta}(z_t, t, y)]\|_2^2, \tag{7}$$

where ⊙ denotes element-wise multiplication. This formulation focuses learning capacity on detailrich regions, improves fidelity in high-frequency content, and does so without increasing inference complexity. Crucially, this mechanism operates purely at the objective level and is compatible with any latent diffusion model using a flow-based or score-based trajectory.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Datasets. We evaluate LWD on two ultra-resolution datasets. Aesthetic-4K [Zhang et al.](#page-16-0) [\(2025\)](#page-16-0) is a curated 4K benchmark with GPT-4o-generated captions and high visual quality. LAION-High-Res is a filtered subset of LAION-5B [Schuhmann et al.](#page-14-12) [\(2022\)](#page-14-12), from which we sample 50K 2K-resolution and 20K 4K-resolution image-caption pairs. These two datasets differ in both visual and linguistic distributions, allowing us to assess both generation fidelity and generalization under caption variance.

Implementation Details. We implement LWD using PyTorch and pytorch-wavelets [\(Cotter,](#page-11-10) [2019\)](#page-11-10) for the Haar-based DWT. For the masking strategy, we set the lower bound ℓ = 0.3, selected via ablation to ensure each spatial location receives at least 30% of supervision.

Evaluation Protocol. To evaluate LWD as a holistic framework, all 'LWD + Model' variants utilize the Scale-Consistent VAE fine-tuning described in Section [3.1,](#page-3-0) while baseline models are evaluated using their original, off-the-shelf VAE checkpoints. Notably, we observed that LWD significantly accelerates convergence; models required only 10–50% of the training iterations suggested in their original papers to reach convergence. Detailed hyperparameters, training costs, and other configurations are provided in Appendix [D.](#page-27-0)

Evaluation Metrics. We evaluate ultra-resolution text-to-image generation across three key dimensions. For image quality, we use Frechet Inception Distance (FID) and LPIPS (lower is better), ´ alongside the Gray Level Co-occurrence Matrix (GLCM) Score for texture and JPEG Compression

![](_page_6_Picture_1.jpeg)

Figure 4: Visual comparison of 2K image generations. LWD demonstrates improved detail preservation in complex areas while avoiding over-sharpening or texture collapse.

<span id="page-6-1"></span><span id="page-6-0"></span>Table 1: Quantitative results on different metrics. The prompts are from the HPD (Wu et al., 2023) dataset. All images are at a resolution of  $2048 \times 2048$ .

| Model                              | FID↓  | LPIPS ↓ | MAN-IQA↑ | QualiCLIP ↑ | HPSv2.1↑ | PickScore ↑ |
|------------------------------------|-------|---------|----------|-------------|----------|-------------|
| SDEdit Meng et al. (2022)          | 35.59 | 0.6456  | 0.3736   | 0.4480      | 30.92    | 22.86       |
| I-Max Du et al. (2024b)            | 36.28 | 0.6750  | 0.3641   | 0.4139      | 30.62    | 23.02       |
| Diffusion-4K Zhang et al. (2025)   | 37.10 | 0.6920  | 0.3550   | 0.4815      | 30.55    | 22.80       |
| PixArt-Sigma-XL Chen et al. (2024) | 36.58 | 0.6801  | 0.2949   | 0.4438      | 30.66    | 22.92       |
| Sana-1.6B Xie et al. (2025)        | 35.75 | 0.7169  | 0.3666   | 0.5796      | 30.42    | 22.83       |
| Lumina-Image 2.0 Qin et al. (2025) | 54.96 | 0.6445  | 0.3663   | 0.4567      | 23.08    | 21.15       |
| FLUX-1.dev Labs (2024)             | 37.58 | 0.6371  | 0.4110   | 0.5468      | 28.73    | 22.68       |
| URAE Yu et al. (2025)              | 35.25 | 0.6717  | 0.4076   | 0.5423      | 31.15    | 22.41       |
| LWD + URAE                         | 32.88 | 0.6336  | 0.4099   | 0.5356      | 28.78    | 22.43       |

Ratio as a proxy for fine-grained detail. For semantic alignment, we report CLIPScore (Hessel et al., 2021) and QualiCLIP Agnolucci et al. (2024). Finally, for perceptual quality, we use MAN-IQA Yang et al. (2022), HPSv2.1 Wu et al. (2023), and PickScore Kirstain et al. (2023). Higher values indicate better performance for all metrics except FID and LPIPS.

### 4.2 QUANTITATIVE RESULTS

**2K Results.** Table 1 and the top block of Table 2 demonstrate consistent improvements from integrating LWD across multiple backbone models. On the HPD prompt dataset (Wu et al., 2023), LWD reduces FID by up to 7% and LPIPS up to 6% while also achieving comparable MAN-IQA and QualiCLIP, indicating improved semantic alignment and perceptual quality. Moreover, on the Aesthetic dataset (Zhang et al., 2025), these gains are observed across diverse architectures, reinforcing the generality and model-agnostic nature of our approach.

**4K Results.** On the Aesthetic-4K (the bottom block of Table 2) and HPD prompt dataset (Table 7), LWD outperforms baselines such as URAE and PixArt-Sigma, particularly in metrics like FID, CLIPScore and Aesthetics. The improvements are especially pronounced in regions with fine structural detail, such as hair, foliage, or architectural elements, highlighting LWD's ability to scale effectively to ultra-high resolutions. These results suggest that frequency-aware supervision provides

<span id="page-7-0"></span>Table 2: Quantitative benchmarks of latent diffusion models on Aesthetic-Eval at  $2048 \times 2048$  and  $4096 \times 4096$ . GLCM Score measures high-frequency texture richness using gray-level co-occurrence matrices, while Compression Ratio assesses visual complexity via JPEG file size heuristics, both introduced in Diffusion-4K Zhang et al. (2025).

|    | Model                                                                                  | FID↓               | CLIPScore ↑                    | Aesthetics ↑                | GLCM Score ↑                | Compression Ratio ↓            |
|----|----------------------------------------------------------------------------------------|--------------------|--------------------------------|-----------------------------|-----------------------------|--------------------------------|
|    | SD3-F16 Esser et al. (2024)                                                            | 43.82              | 31.50                          | 5.91                        | 0.75                        | <b>11.23</b>                   |
|    | SD3-Diff4k-F16 Zhang et al. (2025)                                                     | 40.18              | 34.04                          | 5.96                        | <b>0.79</b>                 | 11.73                          |
|    | LWD + SD3-F16                                                                          | <b>38.74</b>       | <b>34.94</b>                   | <b>6.17</b>                 | 0.74                        | 11.99                          |
| 2K | PixArt-Sigma-XL Chen et al. (2024)                                                     | 39.13              | 35.02                          | 6.43                        | 0.79                        | 13.66                          |
|    | LWD + PixArt-Sigma-XL                                                                  | <b>36.14</b>       | 35.21                          | 6.27                        | <b>0.87</b>                 | <b>6.05</b>                    |
|    | Sana-1.6B Xie et al. (2025)<br>LWD + Sana-1.6B                                         | <b>32.06</b> 34.30 | 35.28<br><b>35.58</b>          | 6.15<br><b>6.23</b>         | <b>0.93</b><br>0.78         | <b>24.01</b> 27.34             |
|    | SD3-F16 Esser et al. (2024)<br>  SD3-Diff4k-F16 Zhang et al. (2025)<br>  LWD + SD3-F16 | -                  | 33.12<br>33.41<br><b>34.08</b> | 5.97<br>5.97<br><b>6.03</b> | 0.73<br>0.70<br><b>0.77</b> | 11.97<br><b>11.90</b><br>12.27 |
| 4K | Sana-1.6B Xie et al. (2025)                                                            | -                  | 34.40                          | 6.14                        | 0.39                        | 48.36                          |
|    | LWD + Sana-1.6B                                                                        | -                  | 34.59                          | <b>6.21</b>                 | <b>0.60</b>                 | <b>32.62</b>                   |

meaningful guidance even in challenging high-frequency regimes where baseline methods often struggle. LWD also achieves the highest GLCM score when paired with SD3-F16, and substantially improves Sana's performance on both GLCM and compression ratio, indicating stronger fine-detail retention and texture fidelity, without compromising overall quality.

#### 4.3 QUALITATIVE RESULTS

Figures 4, 5, and 9 compare outputs from LWD and baseline models under identical prompts. LWD consistently renders sharper textures in high-frequency regions, such as fabric, skin, and hair, while preserving global structure.

Zoomed-in comparisons highlight improved reconstruction of fine details (e.g., hair strands, eyelashes, small objects) that are often blurred or omitted by baselines. These results suggest that frequency-aware masking not only enhances local precision but does so in a context-sensitive manner, avoiding over-sharpening or artifacts. This indicates that LWD effectively balances fine detail refinement with structural coherence. More full-resolution results can be found in Appendix B.

#### 4.4 ABLATION STUDIES

#### 4.4.1 EFFECT OF SCALE-CONSISTENCY LOSS ON RECONSTRUCTION QUALITY

Table 3 reports quantitative reconstruction metrics for various VAEs, with and without the proposed Scale-Consistency (SC) loss, evaluated on the Aesthetic-4K validation set. Across different architectures SD3-VAE, Flux-VAE, and Sana-AE, the addition of SC consistently improves performance, particularly in rFID and perceptual quality (LPIPS). For instance, Flux-VAE-SC outperforms its baseline with a significant reduction in rFID (0.50 vs. 0.73) and an increase in PSNR and SSIM, indicating sharper and more faithful reconstructions. Notably, SD3-VAE-F16-SC achieves a substantial LPIPS improvement (0.18 vs. 0.30), suggesting better perceptual fidelity despite using a more aggressive compression factor (F16). These results confirm that scale-consistent regularization enhances latent representations, making them more robust and structurally aligned, critical properties for downstream diffusion tasks.

#### 4.4.2 IMPACT OF VAE REGULARIZATION AND WAVELET MASKING

To rigorously evaluate the contribution of each component within the LWD framework, we conducted a detailed ablation study. Our method comprises two main stages: (1) fine-tuning a VAE with a scale-consistency (SC) loss and (2) fine-tuning the diffusion model with our proposed wavelet-masked loss ( $\mathcal{L}_{masked}$ ). The study presented here isolates the impact of each component on the final text-to-image generation quality.

The results, shown in Table 4, confirm that both the SC loss and the wavelet-masked loss contribute meaningfully to generation quality, with the full LWD framework yielding the strongest performance.

![](_page_8_Picture_1.jpeg)

Figure 5: 4K images generated by LWD with different architectures.

<span id="page-8-1"></span><span id="page-8-0"></span>Table 3: Quantitative reconstruction results of VAEs with and without the Scale-Consistency Loss [3.1](#page-3-0) on Aesthetic-4K [\(Zhang et al., 2025\)](#page-16-0) validation set.

| Model                            | rFID ↓ | NMSE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|----------------------------------|--------|--------|--------|--------|---------|
| SD3-VAE (Esser et al., 2024)     | 1.05   | 0.01   | 26.54  | 0.86   | 0.08    |
| SD3-VAE-F16 (Zhang et al., 2025) | 0.70   | 0.07   | 19.82  | 0.63   | 0.30    |
| SD3-VAE-F16-SC                   | 0.70   | 0.04   | 22.58  | 0.75   | 0.18    |
| Flux-VAE (Labs, 2024)            | 0.73   | 0.01   | 27.18  | 0.89   | 0.07    |
| Flux-VAE-SC                      | 0.50   | 0.01   | 28.14  | 0.90   | 0.06    |
| Sana-AE (Xie et al., 2025)       | 0.74   | 0.04   | 22.16  | 0.70   | 0.159   |
| Sana-AE-SC                       | 0.55   | 0.02   | 23.64  | 0.73   | 0.163   |

The results in Table [4](#page-9-0) show that the GLCM score slightly decreases with the full LWD framework. This reflects a known limitation of classical texture metrics like GLCM, which do not always correlate with perceptual coherence. Our method trades raw statistical complexity for more realistic details, a positive trade-off validated by significant improvements in perceptual metrics like FID and Aesthetics.

<span id="page-9-0"></span>Table 4: Ablation Study on the Contributions of LWD Components. We evaluate each component's impact on final generation quality using the Diffusion4k backbone on the Aesthetic dataset at 2048×2048 resolution.

| Configuration             | FID ↓ | CLIPScore ↑ | Aesthetics ↑ | GLCM Score ↑ |
|---------------------------|-------|-------------|--------------|--------------|
| Baseline (SD3-Diff4k-F16) | 40.18 | 34.04       | 5.96         | 0.79         |
| + VAE Scale-Consistency   | 39.50 | 34.10       | 6.05         | 0.78         |
| + Wavelet Masking         | 39.20 | 34.50       | 6.10         | 0.75         |
| Full LWD (Ours)           | 38.74 | 34.94       | 6.17         | 0.74         |

#### 4.5 DISCUSSION

Across quantitative and qualitative benchmarks, LWD enhances ultra-resolution image synthesis by integrating signal-derived saliency into the training loss. Compared to both conventional models and prior wavelet-based methods (e.g., Diffusion-4K), it improves perceptual fidelity, semantic alignment, and spectral regularity, without increasing inference cost or modifying the underlying architecture. Its plug-and-play nature makes it broadly compatible with modern latent diffusion pipelines.

LWD improves perceptual fidelity while maintaining comparable alignment scores. This reflects an intentional design choice: LWD prioritizes high-frequency detail recovery to prevent texture collapse at UHR scales, complementing the base model's semantic capabilities rather than replacing them.

Beyond quality gains, LWD represents a shift toward more interpretable and structure-aware supervision. Unlike attention mechanisms that rely on semantic priors, LWD leverages wavelet energy as a transparent, self-supervised signal to prioritize detail-rich regions.

This frequency-guided supervision introduces a form of spatial curriculum learning, where complex regions receive more focused updates. Such adaptive loss weighting opens avenues for dynamic training strategies, such as frequency-aware learning rates or hybrid spatial-frequency schedules. These mechanisms may be especially valuable in domains where structural detail is critical but semantic guidance is weak, such as scientific visualization, material design, or multimodal generation.

Limitations and Future Work. While LWD improves generation quality without architectural changes or inference overhead, it inherits limitations common to latent diffusion models. In particular, VAE compression can cause the loss of fine-grained semantic detail, potentially limiting performance in tasks requiring precise spatial alignment or photorealistic accuracy.

Future work could address this by incorporating higher-fidelity latent spaces or hybrid approaches that combine latent and pixel-space supervision. Extending LWD to domains such as video generation, depth-aware synthesis, or multimodal conditioning also represents a promising direction. The frequency-aware masking mechanism is general and could be adapted to guide temporal attention, cross-modal alignment, or resolution-specific sampling in broader generative contexts.

## 5 CONCLUSION

We introduced *Latent Wavelet Diffusion* (LWD), a general and modular framework that integrates frequency-based supervision into latent diffusion models for ultra-high-resolution image synthesis. By computing wavelet energy maps in the latent space and applying spatially and temporally adaptive masking, LWD selectively emphasizes high-detail regions during training. Without requiring architectural modifications or incurring additional inference cost, LWD consistently improves perceptual fidelity and semantic alignment across models such as Flux and SD3. It preserves high-frequency detail and structural coherence more effectively than prior methods, demonstrating the value of signal-aware supervision in guiding the generative process. By unifying principles from signal processing and diffusion modeling, LWD offers a scalable and interpretable approach applicable to a wide range of generative architectures.

Broader Impact. LWD promotes efficient and interpretable generation by aligning supervision with signal-level detail. This may benefit applications requiring controllable high-resolution synthesis, while raising familiar concerns around synthetic media misuse. Incorporating safeguards and provenance tools remains an important direction.

## REPRODUCIBILITY STATEMENT

To ensure the reproducibility of our work, we have included the core Python script detailing our wavelet-based masking algorithm in the supplementary material. Further implementation details, including key hyperparameters and computational requirements, are provided in Appendix section [D.](#page-27-0) We release our public GitHub repository[1](#page-10-0) , which contains the complete implementation, training scripts, evaluation code, and the final pre-trained model checkpoints.

## ETHICAL STATEMENT

Our work is built upon publicly available datasets commonly used in the field of generative modeling. We acknowledge that these large-scale datasets may contain inherent societal biases, which our model could potentially learn and reproduce. We have used these datasets in accordance with their original licenses. Below, we discuss the potential societal impacts of our work.

#### POSITIVE IMPACT

The primary goal of our research is to advance the state of high-resolution image generation, which has significant positive applications. These include empowering artists, designers, and content creators with more powerful creative tools; enhancing visual effects for entertainment and media; and potentially aiding in scientific visualization and data augmentation. By developing a method that improves quality with zero inference overhead, we aim to make high-fidelity generative AI more accessible and practical for a wider range of beneficial uses.

#### POTENTIAL RISKS AND MITIGATION

We recognize that generative models can be misused for malicious purposes, such as creating misinformation ("deepfakes"), generating harmful or explicit content, and perpetuating societal biases. To mitigate these risks, we are committed to the following measures:

- 1. Responsible Release: We release our code and models under a responsible AI license (e.g., a variant of the CreativeML Open RAIL-M license) that explicitly prohibits use for malicious, harmful, or unethical purposes.
- 2. Acknowledging Limitations: We are transparent about the limitations of our model and its potential to generate biased or factually incorrect content, as discussed in the main paper.
- 3. Encouraging Safe Deployment: We strongly encourage all downstream users of our models to implement their own safety filters, content moderation systems, and ethical guidelines before deploying any applications.

## LLM USAGE STATEMENT

A large language model (LLM) was used as a writing assistance tool during the preparation of this manuscript. The LLM's role was limited to improving grammar, clarity, and conciseness. All content was conceived and written by the authors, who take full responsibility for the paper's scientific integrity.

## ACKNOWLEDGEMENT

The work of L. Sigillo was partially supported by "Ricerca e innovazione nel Lazio - incentivi per i dottorati di innovazione per le imprese e per la PA - L.R. 13/2008" of Regione Lazio, Project "Deep Learning Generativo nel Dominio Ipercomplesso per Applicazioni di Intelligenza Artificiale ad Alta Efficienza Energetica", under grant number 21027NP000000136. The work of D. Comminiello was partly supported by Progetti di Ateneo of Sapienza University of Rome under grant numbers RM123188F75F8072 and RM1241910FC4BEEA and by the European Union under the Italian

<span id="page-10-0"></span><sup>1</sup><https://github.com/LuigiSigillo/LatentWaveletDiffusion>

National Recovery and Resilience Plan (NRRP) of NextGenerationEU, "Rome Technopole" (CUP B83C22002820006)—Flagship Project 5: "Digital Transition Through AESA Radar Technology, Quantum Cryptography and Quantum Communications".

## REFERENCES

- <span id="page-11-7"></span>Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. *arXiv preprint arXiv:2501.03575*, 2025.
- <span id="page-11-12"></span>Lorenzo Agnolucci, Leonardo Galteri, and Marco Bertini. Quality-aware image-text alignment for opinion-unaware image quality assessment. *arXiv preprint arXiv:2403.11176*, 2024.
- <span id="page-11-8"></span>Lorenzo Aloisi, Luigi Sigillo, Aurelio Uncini, and Danilo Comminiello. *A Wavelet Diffusion GAN for Image Super-Resolution*. Springer Nature Singapore, 2026. ISBN 978-981- 95-4072-3. doi: 10.1007/978-981-95-4072-3 36. URL [https://doi.org/10.1007/](https://doi.org/10.1007/978-981-95-4072-3_36) [978-981-95-4072-3\\_36](https://doi.org/10.1007/978-981-95-4072-3_36). <https://arxiv.org/abs/2410.17966>.
- <span id="page-11-1"></span>Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. MultiDiffusion: Fusing diffusion paths for controlled image generation. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pp. 1737–1752. PMLR, 23–29 Jul 2023. URL [https://proceedings.mlr.press/v202/](https://proceedings.mlr.press/v202/bar-tal23a.html) [bar-tal23a.html](https://proceedings.mlr.press/v202/bar-tal23a.html).
- <span id="page-11-6"></span>Sara Bjork, Jonas Nordhaug Myhre, and Thomas Haugland Johansen. Simpler is better: Spectral ¨ regularization and up-sampling techniques for variational autoencoders. In *ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pp. 3778– 3782, 2022. doi: 10.1109/ICASSP43922.2022.9746027.
- <span id="page-11-4"></span>Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In *The Twelfth International Conference on Learning Representations*, 2024. URL <https://openreview.net/forum?id=YCWjhGrJFD>.
- <span id="page-11-0"></span>Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 18392–18402, 2023.
- <span id="page-11-9"></span>Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J ´ egou, Julien Mairal, Piotr Bojanowski, and ´ Armand Joulin. Emerging properties in self-supervised vision transformers. In *Proceedings of the IEEE/CVF international conference on computer vision*, pp. 9650–9660, 2021.
- <span id="page-11-3"></span>Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-σ: Weak-to-strong training of diffusion transformer for 4k text-to-image generation. In *European Conference on Computer Vision*, pp. 74–91. Springer, 2024.
- <span id="page-11-10"></span>Fergal Cotter. *Uses of Complex Wavelets in Deep Convolutional Neural Networks*. PhD thesis, Apollo - University of Cambridge Repository, 2019. URL [https://www.repository.cam.ac.](https://www.repository.cam.ac.uk/handle/1810/306661) [uk/handle/1810/306661](https://www.repository.cam.ac.uk/handle/1810/306661).
- <span id="page-11-2"></span>Ruoyi Du, Dongliang Chang, Timothy Hospedales, Yi-Zhe Song, and Zhanyu Ma. Demofusion: Democratising high-resolution image generation with no \$\$\$. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 6159–6168, 2024a.
- <span id="page-11-11"></span>Ruoyi Du, Dongyang Liu, Le Zhuo, Qin Qi, Hongsheng Li, Zhanyu Ma, and Peng Gao. I-max: Maximize the resolution potential of pre-trained rectified flow transformers with projected flow, 2024b. URL <https://arxiv.org/abs/2410.07536>.
- <span id="page-11-5"></span>Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 12873–12883, 2021.

- <span id="page-12-1"></span>Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam ¨ Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In *Forty-first international conference on machine learning*, 2024.
- <span id="page-12-4"></span>Carlos Esteves, Mohammed Suhail, and Ameesh Makadia. Spectral image tokenizer. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 17181–17190, 2025.
- <span id="page-12-7"></span>Jiatao Gu, Shuangfei Zhai, Yizhe Zhang, Joshua M. Susskind, and Navdeep Jaitly. Matryoshka diffusion models. In *The Twelfth International Conference on Learning Representations*, 2024. URL <https://openreview.net/forum?id=tOzCcDdH9O>.
- <span id="page-12-9"></span>Lanqing Guo, Yingqing He, Haoxin Chen, Menghan Xia, Xiaodong Cun, Yufei Wang, Siyu Huang, Yong Zhang, Xintao Wang, Qifeng Chen, et al. Make a cheap scaling: A self-cascade diffusion model for higher-resolution adaptation. In *European Conference on Computer Vision*, pp. 39–55. Springer, 2024.
- <span id="page-12-5"></span>Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion. *arXiv preprint arXiv:2501.00103*, 2024.
- <span id="page-12-3"></span>Jaehoon Hahm, Junho Lee, Sunghyun Kim, and Joonseok Lee. Isometric representation learning for disentangled latent space of diffusion models. In *Forty-first International Conference on Machine Learning*, 2024. URL <https://openreview.net/forum?id=ufCptn28vG>.
- <span id="page-12-2"></span>Yingqing He, Shaoshu Yang, Haoxin Chen, Xiaodong Cun, Menghan Xia, Yong Zhang, Xintao Wang, Ran He, Qifeng Chen, and Ying Shan. Scalecrafter: Tuning-free higher-resolution visual generation with diffusion models. In *The Twelfth International Conference on Learning Representations*, 2023.
- <span id="page-12-14"></span>Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A referencefree evaluation metric for image captioning. In *EMNLP (1)*, 2021.
- <span id="page-12-0"></span>Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. *Advances in neural information processing systems*, 33:6840–6851, 2020.
- <span id="page-12-6"></span>Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. *Journal of Machine Learning Research*, 23(47):1–33, 2022.
- <span id="page-12-12"></span>Linjiang Huang, Rongyao Fang, Aiping Zhang, Guanglu Song, Si Liu, Yu Liu, and Hongsheng Li. Fouriscale: A frequency perspective on training-free high-resolution image synthesis. In *European Conference on Computer Vision*, pp. 196–212. Springer, 2024a.
- <span id="page-12-10"></span>Yi Huang, Jiancheng Huang, Jianzhuang Liu, Mingfu Yan, Yu Dong, Jiaxi Lv, Chaoqi Chen, and Shifeng Chen. Wavedm: Wavelet-based diffusion models for image restoration. *IEEE Transactions on Multimedia*, 26:7058–7073, 2024b. doi: 10.1109/TMM.2024.3359769.
- <span id="page-12-8"></span>Jinho Jeong, Sangmin Han, Jinwoo Kim, and Seon Joo Kim. Latent Space Super-Resolution for Higher-Resolution Image Generation with Diffusion Models . In *2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2355–2365, Los Alamitos, CA, USA, June 2025. IEEE Computer Society. doi: 10.1109/CVPR52734.2025.00225. URL [https:](https://doi.ieeecomputersociety.org/10.1109/CVPR52734.2025.00225) [//doi.ieeecomputersociety.org/10.1109/CVPR52734.2025.00225](https://doi.ieeecomputersociety.org/10.1109/CVPR52734.2025.00225).
- <span id="page-12-11"></span>Hai Jiang, Ao Luo, Haoqiang Fan, Songchen Han, and Shuaicheng Liu. Low-light image enhancement with wavelet-based diffusion models. *ACM Trans. Graph.*, 42(6), December 2023. ISSN 0730-0301. doi: 10.1145/3618373. URL <https://doi.org/10.1145/3618373>.
- <span id="page-12-13"></span>Younghyun Kim, Geunmin Hwang, Junyu Zhang, and Eunbyung Park. Diffusehigh: Training-free progressive high-resolution image synthesis through structure guidance. In *Proceedings of the AAAI conference on artificial intelligence*, volume 39, pp. 4338–4346, 2025.
- <span id="page-12-15"></span>Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Picka-pic: An open dataset of user preferences for text-to-image generation. *Advances in Neural Information Processing Systems*, 36:36652–36663, 2023.

- <span id="page-13-8"></span>Theodoros Kouzelis, Ioannis Kakogeorgiou, Spyros Gidaris, and Nikos Komodakis. EQ-VAE: Equivariance regularized latent space for improved generative image modeling. In *Forty-second International Conference on Machine Learning*, 2025. URL [https://openreview.net/](https://openreview.net/forum?id=UWhW5YYLo6) [forum?id=UWhW5YYLo6](https://openreview.net/forum?id=UWhW5YYLo6).
- <span id="page-13-5"></span>Black Forest Labs. Flux. <https://github.com/black-forest-labs/flux>, 2024.
- <span id="page-13-10"></span>Yuseung Lee, Kunho Kim, Hyunjin Kim, and Minhyuk Sung. Syncdiffusion: Coherent montage via synchronized joint diffusions. *Advances in Neural Information Processing Systems*, 36:50648– 50660, 2023.
- <span id="page-13-9"></span>Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, et al. Open-sora plan: Open-source large video generation model. *arXiv preprint arXiv:2412.00131*, 2024.
- <span id="page-13-7"></span>Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In *The Eleventh International Conference on Learning Representations*, 2023. URL <https://openreview.net/forum?id=PqvMRDCJT9t>.
- <span id="page-13-6"></span>Guangyi Liu, Yu Wang, Zeyu Feng, Qiyu Wu, Liping Tang, Yuan Gao, Zhen Li, Shuguang Cui, Julian McAuley, Zichao Yang, Eric P. Xing, and Zhiting Hu. Unified generation, reconstruction, and representation: Generalized diffusion with adaptive latent encoding-decoding. In *Forty-first International Conference on Machine Learning*, 2024a. URL [https://openreview.net/](https://openreview.net/forum?id=igRjCCAz2a) [forum?id=igRjCCAz2a](https://openreview.net/forum?id=igRjCCAz2a).
- <span id="page-13-4"></span>Yixin Liu, Kai Zhang, Yuan Li, Zhiling Yan, Chujie Gao, Ruoxi Chen, Zhengqing Yuan, Yue Huang, Hanchi Sun, Jianfeng Gao, Lifang He, and Lichao Sun. Sora: A review on background, technology, limitations, and opportunities of large vision models, 2024b.
- <span id="page-13-1"></span>Eleonora Lopez, Luigi Sigillo, Federica Colonnese, Massimo Panella, and Danilo Comminiello. Guess what i think: Streamlined eeg-to-image generation with latent diffusion models. In *ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pp. 1–5, 2025. doi: 10.1109/ICASSP49660.2025.10890059.
- <span id="page-13-14"></span>Stephane Mallat and Wen Liang Hwang. Singularity detection and processing with wavelets. *IEEE transactions on information theory*, 38(2):617–643, 1992.
- <span id="page-13-2"></span>Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In *International Conference on Learning Representations*, 2022. URL [https://openreview.net/forum?](https://openreview.net/forum?id=aBsCjcPu_tE) [id=aBsCjcPu\\_tE](https://openreview.net/forum?id=aBsCjcPu_tE).
- <span id="page-13-13"></span>Brian B. Moser, Stanislav Frolov, Federico Raue, Sebastian Palacio, and Andreas Dengel. Dynamic Attention-Guided Diffusion for Image Super-Resolution . In *2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, pp. 451–460, Los Alamitos, CA, USA, March 2025. IEEE Computer Society. doi: 10.1109/WACV61041.2025.00054. URL [https://doi.](https://doi.ieeecomputersociety.org/10.1109/WACV61041.2025.00054) [ieeecomputersociety.org/10.1109/WACV61041.2025.00054](https://doi.ieeecomputersociety.org/10.1109/WACV61041.2025.00054).
- <span id="page-13-3"></span>William Peebles and Saining Xie. Scalable diffusion models with transformers. In *Proceedings of the IEEE/CVF international conference on computer vision*, pp. 4195–4205, 2023.
- <span id="page-13-11"></span>Hao Phung, Quan Dao, and Anh Tran. Wavelet diffusion models are fast and scalable image generators. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 10199– 10208, 2023.
- <span id="page-13-0"></span>Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Muller, Joe ¨ Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In *The Twelfth International Conference on Learning Representations*, 2024. URL <https://openreview.net/forum?id=di52zR8xgf>.
- <span id="page-13-12"></span>Yurui Qian, Qi Cai, Yingwei Pan, Yehao Li, Ting Yao, Qibin Sun, and Tao Mei. Boosting diffusion models with moving average sampling in frequency domain. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 8911–8920, 2024.

- <span id="page-14-13"></span>Qi Qin, Le Zhuo, Yi Xin, Ruoyi Du, Zhen Li, Bin Fu, Yiting Lu, Xinyue Li, Dongyang Liu, Xiangyang Zhu, Will Beddow, Erwann Millon, Victor Perez, Wenhai Wang, Yu Qiao, Bo Zhang, Xiaohong Liu, Hongsheng Li, Chang Xu, and Peng Gao. Lumina-image 2.0: A unified and efficient image generative framework. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 20031–20042, October 2025.
- <span id="page-14-5"></span>Jingjing Ren, Wenbo Li, Haoyu Chen, Renjing Pei, Bin Shao, Yong Guo, Long Peng, Fenglong Song, and Lei Zhu. Ultrapixel: Advancing ultra high-resolution image synthesis to new peaks. In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024. URL <https://openreview.net/forum?id=voJCpdlw53>.
- <span id="page-14-4"></span>Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High- ¨ resolution image synthesis with latent diffusion models. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 10684–10695, 2022.
- <span id="page-14-1"></span>Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. *IEEE transactions on pattern analysis and machine intelligence*, 45(4):4713–4726, 2022.
- <span id="page-14-0"></span>Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In *European Conference on Computer Vision*, pp. 87–103. Springer, 2024.
- <span id="page-14-12"></span>Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. Laion-5b: an open large-scale dataset for training next generation image-text models. In *Proceedings of the 36th International Conference on Neural Information Processing Systems*, NIPS '22, Red Hook, NY, USA, 2022. Curran Associates Inc. ISBN 9781713871088.
- <span id="page-14-6"></span>Hui Shen, Jingxuan Zhang, Boning Xiong, Rui Hu, Shoufa Chen, Zhongwei Wan, Xin Wang, Yu Zhang, Zixuan Gong, Guangyin Bao, Chaofan Tao, Yongfeng Huang, Ye Yuan, and Mi Zhang. Efficient diffusion models: A survey. *Transactions on Machine Learning Research*, 2025. URL <https://openreview.net/forum?id=wHECkBOwyt>.
- <span id="page-14-10"></span>Shuwei Shi, Wenbo Li, Yuechen Zhang, Jingwen He, Biao Gong, and Yinqiang Zheng. Resmaster: Mastering high-resolution image generation via structural and fine-grained guidance. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 39, pp. 6887–6895, 2025.
- <span id="page-14-2"></span>Luigi Sigillo, Riccardo Fosco Gramaccioni, Alessandro Nicolosi, and Danilo Comminiello. Ship in sight: Diffusion models for ship-image super resolution. In *2024 International Joint Conference on Neural Networks (IJCNN)*, pp. 1–8, 2024. doi: 10.1109/IJCNN60899.2024.10650042.
- <span id="page-14-11"></span>Luigi Sigillo, Christian Bianchi, Aurelio Uncini, and Danilo Comminiello. Quaternion waveletconditioned diffusion models for image super-resolution. In *2025 International Joint Conference on Neural Networks (IJCNN)*, pp. 1–8, 2025. doi: 10.1109/IJCNN64981.2025.11228578.
- <span id="page-14-9"></span>Ivan Skorokhodov, Sharath Girish, Benran Hu, Willi Menapace, Yanyu Li, Rameen Abdal, Sergey Tulyakov, and Aliaksandr Siarohin. Improving the diffusability of autoencoders. In *Forty-second International Conference on Machine Learning*, 2025. URL [https://openreview.net/](https://openreview.net/forum?id=2hEDcA7xy4) [forum?id=2hEDcA7xy4](https://openreview.net/forum?id=2hEDcA7xy4).
- <span id="page-14-3"></span>Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In *International Conference on Learning Representations*, 2021a. URL [https://openreview.net/](https://openreview.net/forum?id=St1giarCHLP) [forum?id=St1giarCHLP](https://openreview.net/forum?id=St1giarCHLP).
- <span id="page-14-7"></span>Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In *International Conference on Learning Representations*, 2021b. URL [https://openreview.net/forum?](https://openreview.net/forum?id=PxTIG12RRHS) [id=PxTIG12RRHS](https://openreview.net/forum?id=PxTIG12RRHS).
- <span id="page-14-8"></span>Yuhta Takida, Yukara Ikemiya, Takashi Shibuya, Kazuki Shimada, Woosung Choi, Chieh-Hsin Lai, Naoki Murata, Toshimitsu Uesaka, Kengo Uchida, Wei-Hsiang Liao, and Yuki Mitsufuji. HQ-VAE: Hierarchical discrete representation learning with variational bayes. *Transactions on Machine*

- *Learning Research*, 2024. ISSN 2835-8856. URL [https://openreview.net/forum?](https://openreview.net/forum?id=xqAVkqrLjx) [id=xqAVkqrLjx](https://openreview.net/forum?id=xqAVkqrLjx).
- <span id="page-15-7"></span>Anni Tang, Tianyu He, Junliang Guo, Xinle Cheng, Li Song, and Jiang Bian. Vidtok: A versatile and open-source video tokenizer. *arXiv preprint arXiv:2412.13061*, 2024.
- <span id="page-15-2"></span>Athanasios Tragakis, Marco Aversa, Chaitanya Kaul, Roderick Murray-Smith, and Daniele Faccio. Is one gpu enough? pushing image generation at higher-resolutions with foundation models. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), *Advances in Neural Information Processing Systems*, volume 37, pp. 41242–41273. Curran Associates, Inc., 2024. doi: 10.52202/ 079017-1305. URL [https://proceedings.neurips.cc/paper\\_files/paper/](https://proceedings.neurips.cc/paper_files/paper/2024/file/48644509339cb3076f7b0407c7588af6-Paper-Conference.pdf) [2024/file/48644509339cb3076f7b0407c7588af6-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/48644509339cb3076f7b0407c7588af6-Paper-Conference.pdf).
- <span id="page-15-5"></span>Arash Vahdat and Jan Kautz. Nvae: A deep hierarchical variational autoencoder. *Advances in neural information processing systems*, 33:19667–19679, 2020.
- <span id="page-15-1"></span>Yufei Wang, Wenhan Yang, Xinyuan Chen, Yaohui Wang, Lanqing Guo, Lap-Pui Chau, Ziwei Liu, Yu Qiao, Alex C Kot, and Bihan Wen. Sinsr: diffusion-based image super-resolution in a single step. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 25796–25805, 2024.
- <span id="page-15-12"></span>Z. Wang, E.P. Simoncelli, and A.C. Bovik. Multiscale structural similarity for image quality assessment. In *The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers, 2003*, volume 2, pp. 1398–1402 Vol.2, 2003. doi: 10.1109/ACSSC.2003.1292216.
- <span id="page-15-0"></span>Zhendong Wang, Jianmin Bao, Shuyang Gu, Dong Chen, Wengang Zhou, and Houqiang Li. Designdiffusion: High-quality text-to-design image generation with diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 20906–20915, June 2025.
- <span id="page-15-10"></span>Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis, 2023. URL <https://arxiv.org/abs/2306.09341>.
- <span id="page-15-8"></span>Enze Xie, Lewei Yao, Han Shi, Zhili Liu, Daquan Zhou, Zhaoqiang Liu, Jiawei Li, and Zhenguo Li. Difffit: Unlocking transferability of large diffusion models via simple parameter-efficient fine-tuning. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 4230–4239, 2023.
- <span id="page-15-6"></span>Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, and Song Han. SANA: Efficient high-resolution text-to-image synthesis with linear diffusion transformers. In *The Thirteenth International Conference on Learning Representations*, 2025. URL <https://openreview.net/forum?id=N8Oj1XhtYZ>.
- <span id="page-15-11"></span>Sidi Yang, Tianhe Wu, Shuwei Shi, Shanshan Lao, Yuan Gong, Mingdeng Cao, Jiahao Wang, and Yujiu Yang. Maniqa: Multi-dimension attention network for no-reference image quality assessment. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 1191– 1200, 2022.
- <span id="page-15-9"></span>Xingyi Yang, Daquan Zhou, Jiashi Feng, and Xinchao Wang. Diffusion probabilistic model made slim. In *2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 22552–22562, 2023. doi: 10.1109/CVPR52729.2023.02160.
- <span id="page-15-4"></span>Ruonan Yu, Songhua Liu, Zhenxiong Tan, and Xinchao Wang. Ultra-resolution adaptation with ease. *International Conference on Machine Learning*, 2025.
- <span id="page-15-3"></span>Zheyuan Zhan, Defang Chen, Jian-Ping Mei, Zhenghe Zhao, Jiawei Chen, Chun Chen, Siwei Lyu, and Can Wang. Conditional image synthesis with diffusion models: A survey. *Transactions on Machine Learning Research*, 2025. ISSN 2835-8856. URL [https://openreview.net/](https://openreview.net/forum?id=ewwNKwh6SK) [forum?id=ewwNKwh6SK](https://openreview.net/forum?id=ewwNKwh6SK). Survey Certification.

- <span id="page-16-0"></span>Jinjin Zhang, Qiuyu Huang, Junjie Liu, Xiefan Guo, and Di Huang. Diffusion-4k: Ultra-highresolution image synthesis with latent diffusion models. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2025.
- <span id="page-16-4"></span>Lin Zhang, Lei Zhang, Xuanqin Mou, and David Zhang. Fsim: A feature similarity index for image quality assessment. *IEEE Transactions on Image Processing*, 20(8):2378–2386, 2011. doi: 10.1109/TIP.2011.2109730.
- <span id="page-16-2"></span>Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In *Proceedings of the IEEE/CVF international conference on computer vision*, pp. 3836–3847, 2023.
- <span id="page-16-3"></span>Chen Zhao, Weiling Cai, Chenyu Dong, and Chengwei Hu. Wavelet-based fourier information interaction with frequency diffusion adjustment for underwater image restoration. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 8281–8291, 2024.
- <span id="page-16-1"></span>Qingping Zheng, Yuanfan Guo, Jiankang Deng, Jianhua Han, Ying Li, Songcen Xu, and Hang Xu. Any-size-diffusion: Toward efficient text-driven synthesis for any-size hd images. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 38, pp. 7571–7578, 2024.

