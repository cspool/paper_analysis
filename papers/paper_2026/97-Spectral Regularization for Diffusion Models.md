**Spectral Regularization for Diffusion Models** 

**Satish Chandran**[* 1] **N´ıcolas Roque dos Santos**[* 2] **Yunshu Wu**[2] **Greg Ver Steeg**[2] **Evangelos Papalexakis**[2] 

## **Abstract** 

Diffusion models are typically trained using pointwise reconstruction objectives that are agnostic to the spectral and multi-scale structure of natural signals. We propose a loss-level spectral regularization framework that augments standard diffusion training with differentiable Fourier- and wavelet-domain losses, without modifying the diffusion process, model architecture, or sampling procedure. The proposed regularizers act as soft inductive biases that encourage appropriate frequency balance and coherent multi-scale structure in generated samples. Our approach is compatible with DDPM, DDIM, and EDM formulations and introduces negligible computational overhead. Experiments on image and audio generation demonstrate consistent improvements in sample quality, with the largest gains observed on higher-resolution, unconditional datasets where fine-scale structure is most challenging to model. 

## **1. Introduction** 

Diffusion models have emerged as a powerful and versatile framework for generative modeling of high-dimensional signals. By learning to reverse a gradual noising process, diffusion models provide stable training, strong mode coverage, and state-of-the-art performance across a wide range of modalities, including natural images, audio waveforms, and graphical representations (Sohl-Dickstein et al., 2015; Ho et al., 2020; Dhariwal & Nichol, 2021; Liu et al., 2023). Their success has led to rapid adoption in image synthesis (Rombach et al., 2022), audio generation (Kong et al., 2021), and conditional generation tasks such as super-resolution and inpainting (Saharia et al., 2022). 

Despite their strong empirical performance, diffusion models are typically trained using pointwise reconstruction 

1Department of Mathematics, University of California Riverside, Riverside, California, USA[2] Department of Computer Science, University of California Riverside, Riverside, California, USA. Correspondence to: Evangelos Papalexakis _<_ epapalexcs@cs.ucr.edu _>_ . 

losses defined in the signal domain, most commonly meansquared error on predicted noise or clean signals (Ho et al., 2020; Song et al., 2021). While these objectives are well motivated from both a empirical and theoretical perspective, they are agnostic to the spectral and multi-scale structure that characterizes many natural signals. Images and audio often exhibit highly structured frequency content, longrange correlations, and scale-dependent patterns that are only implicitly captured through pixel/sample-level supervision. As a result, diffusion models often generate samples that match low-level statistics while exhibiting artifacts such as over-smoothing, incorrect frequency balance, or degraded fine-scale structure (Benita et al., 2025; Chen et al., 2025; Ding et al., 2024; Jiralerspong et al., 2025). 

Recent works have explored augmenting diffusion models with additional forms of structure or inductive bias. In scientific and engineering domains, this has included incorporating explicit constraint or residual losses into diffusion training to enforce known properties of the data-generating process (Shu et al., 2023; Bastek et al., 2025; Jacobsen et al., 2025). While effective in settings where such constraints are well defined, these approaches are less directly applicable to general image and audio tasks, where the structure is implicit, statistical, and perceptual rather than being defined explicitly as a set of differential equations. 

In contrast, frequency-domain representations have long played a central role in image and audio processing. Fourier analysis provides a global description of signal energy distribution across frequencies and is fundamental to understanding smoothness, noise, and periodic structure (Oppenheim et al., 1996). Wavelet representations extend this perspective by offering localized, multi-resolution decompositions that capture both spatial or temporal locality and scale (Mallat, 2008). These representations underpin classical methods in compression, denoising, and texture analysis, and they have also been used as inductive biases or auxiliary losses in deep learning models for images and audio (Bruna & Mallat, 2013; Gatys et al., 2016; Kong et al., 2021). 

We propose a spectral regularization framework for diffusion model training that augments the standard denoising objective with differentiable losses defined in the Fourier and wavelet domains. Rather than modifying model architectures, samplers, or imposing hard constraints, our approach 

_Preprint. March 4, 2026._ 

1 

**Spectral Regularization for Diffusion Models** 

introduces a soft inductive bias that encourages generated samples to match the frequency-dependent structure of the data. Fourier-based losses capture global spectral characteristics, while wavelet-based losses provide localized, scaleaware control well suited to non-stationary signals such as audio and textured images. The resulting regularization is domain-agnostic, requires no auxiliary networks or additional supervision, and incurs negligible computational overhead. Empirically, we find that incorporating spectral information complements pixel-level objectives, leading to sharper reconstructions, improved perceptual quality, and reduced overfitting, while preserving the diversity and expressiveness of diffusion models. The code is available at https://anonymous.4open.science/r/fourierdm-8B8E. 

## **2. Related Works** 

**Spectral-domain diffusion models.** A growing body of work explores incorporating frequency structure directly into diffusion models by redefining the diffusion process in spectral coordinates or operating on transformed representations. Crabbe´ et al. (2024) formulate diffusion for time series in the Fourier domain, explicitly modeling conjugatesymmetric complex coefficients to ensure real-valued reconstructions. Jiralerspong et al. (2025) shape the diffusion dynamics in frequency space to emphasize or suppress specific spectral bands through frequency-based noise control. Phillips et al. (2022) propose spectral process diffusion, performing score-based modeling over coefficients of stochastic processes expressed in a spectral basis. 

These approaches embed spectral structure directly into the diffusion state space or dynamics, requiring modified forward processes or basis-specific parameterizations. In contrast, our method preserves standard diffusion formulations (DDPM, DDIM, and EDM) and introduces spectral structure solely through auxiliary loss terms applied to reconstructions. Spectral bias is therefore imposed at the loss objective level rather than through changes to the generative process itself. 

**Wavelet-based and multi-resolution diffusion.** Wavelet representations have motivated several diffusion models that operate directly on multi-scale decompositions. Guth et al. (2022) perform score-based diffusion on wavelet coefficients and interpret the resulting hierarchy through renormalization group theory. Phung et al. (2022) and Hu et al. (2024) apply wavelet-domain diffusion for efficient image generation and 3D shape modeling, respectively. Related approaches selectively apply diffusion to low-frequency components while refining high-frequency content using auxiliary modules, primarily for restoration and efficiency (Huang et al., 2023; Zhao et al., 2024; Liu et al., 2025; Zhou et al., 2026). In these works, diffusion is typically 

performed in a transformed representation or coupled with frequency-specific architectural modules. Our approach instead retains standard pixel- or waveform-space diffusion and applies wavelet regularization purely at the loss level. This maintains architectural simplicity while encouraging multi-scale consistency through the training objective. 

**Hybrid Fourier-wavelet diffusion.** Several recent works explicitly combine Fourier and wavelet representations within diffusion pipelines. Luo et al. (2025) introduce crossfrequency fusion for trajectory modeling in reinforcement learning. Kiruluta & Lemos (2025) propose a hybrid forward process combining partial Fourier corruption with wavelet decomposition and multi-branch denoising networks. These hybrid approaches integrate spectral structure into the forward process or network design. In contrast, we treat Fourier and wavelet transforms as analysis operators used only for defining differentiable penalties. The diffusion dynamics remain unchanged, making our framework modular and directly compatible with existing implementations. 

**Constraint-augmented diffusion models.** More broadly, diffusion models have been augmented with auxiliary losses to encode known structure, particularly in scientific domains. Physics-informed diffusion models incorporate residualbased constraints derived from governing equations to enforce physical consistency (Shu et al., 2023; Bastek et al., 2025; Jacobsen et al., 2025). While effective when explicit constraints are available, such methods are less applicable to natural images and audio, where structure is statistical rather than rule-based. Our work adopts the broader idea of auxiliary regularization but replaces equation-based constraints with soft frequency-domain penalties derived from signal statistics. 

**Positioning of Our Approach.** Overall, existing frequency-aware diffusion methods typically modify the diffusion process, operate in transformed domains, or introduce task-specific architectures. In contrast, we propose a loss-level spectral regularization framework that preserves standard diffusion formulations (DDPM, DDIM, and EDM) while encouraging frequency balance and multi-scale coherence through differentiable Fourierand wavelet-domain losses. This design is domain-agnostic, architecture-independent, and directly compatible with existing diffusion training and sampling pipelines. 

## **3. Diffusion Models** 

Diffusion models define a class of generative models that construct complex data distributions by reversing a gradual stochastic noising process. The central idea is to transform data samples into noise through a forward process that is analytically tractable, and to learn a parameterized reverse 

2 

**Spectral Regularization for Diffusion Models** 

process that progressively removes noise. This framework has evolved through several closely related formulations, including denoising diffusion probabilistic models (DDPMs) (Ho et al., 2020), deterministic diffusion implicit models (DDIMs) (Song et al., 2021), and more recent formulations such as Elucidated Diffusion Models (EDMs) (Karras et al., 2022), which unify training and sampling under continuous noise parameterizations. 

## **3.1. Denoising Diffusion Probabilistic Models (DDPM)** 

Denoising Diffusion Probabilistic Models (DDPMs) define a discrete-time Markov chain that gradually corrupts data with Gaussian noise over _T_ steps. Given a data sample _x_ 0 _∼ p_ data, the forward process is defined as 

**==> picture [203 x 19] intentionally omitted <==**

where _{βt}[T] t_ =1[is a predefined variance schedule.][By con-] struction, this process admits a closed-form marginal defined by 

**==> picture [211 x 11] intentionally omitted <==**

with _α_ ¯ _t_ =[�] _[t] s_ =1[(1] _[ −][β][s]_[)][.][The reverse process is parameter-] ized by a neural network trained to predict either the mean of the reverse transition or, more commonly, the added noise _ε_ . This leads to the standard DDPM objective: 

**==> picture [195 x 18] intentionally omitted <==**

which can be shown to correspond to a variational bound on the negative log-likelihood. DDPMs established diffusion models as an alternative to GANs, offering stable training and strong mode coverage, at the cost of relatively slow sampling due to the large number of required reverse steps. 

## **3.2. Denoising Diffusion Implicit Models (DDIM)** 

Denoising Diffusion Implicit Models (DDIMs) reinterpret the DDPM framework by constructing a non-Markovian, deterministic sampling process that preserves the same marginal distributions as DDPMs while enabling faster generation. Rather than sampling from a stochastic reverse transition, DDIMs define a deterministic mapping: 

**==> picture [204 x 13] intentionally omitted <==**

where 

**==> picture [195 x 13] intentionally omitted <==**

This formulation reveals that diffusion models define a family of generative trajectories indexed by a stochasticity parameter, interpolating between fully stochastic DDPM sampling and deterministic DDIM sampling. Importantly, DDIMs use the same training objective as DDPMs with the only difference being the sampling procedure. 

## **3.3. Elucidated Diffusion Models (EDM)** 

Elucidated Diffusion Models (EDMs) further generalize diffusion modeling by formulating training and sampling in continuous noise space rather than discrete time steps. Instead of indexing noise by _t_ , EDMs parameterize corruption using the noise standard deviation _σ_ , defining noisy samples as 

**==> picture [186 x 11] intentionally omitted <==**

EDMs introduce a reweighted denoising objective of the form 

**==> picture [222 x 19] intentionally omitted <==**

where the weighting function _λEDM_ ( _σ_ ) is chosen to balance contributions across noise scales. A key advantage of the EDM framework is that it exposes diffusion models as learning scale-dependent denoisers across a continuum of noise levels. This perspective is particularly relevant for image and audio generation, where meaningful structure exists across a wide range of spatial or temporal scales. By explicitly decoupling noise level, loss weighting, and sampling trajectory, EDMs provide a flexible foundation for incorporating additional regularization terms without altering the core generative mechanism. 

## **3.4. Implications for Regularized Diffusion Training** 

Across DDPM, DDIM, and EDM formulations, diffusion models are trained using pointwise denoising objectives defined in the signal domain. While sufficient for likelihoodbased learning, these objectives do not explicitly constrain how reconstruction error is distributed across frequencies or scales. Since the learned denoiser defines a family of noise-dependent reconstruction operators, augmenting the training objective with spectral or multi-scale regularization naturally complements the diffusion framework without altering its probabilistic foundations. 

From an operator perspective, diffusion models recover coarse, low-frequency structure at high noise levels, while fine-scale, high-frequency components are reconstructed only in low-noise regimes. As a result, high-frequency errors are learned under weaker effective regularization and fewer samples, making them more susceptible to overfitting and instability. Standard diffusion objectives treat all reconstruction errors equally, allowing error to concentrate in perceptually or structurally undesirable frequency bands (Benita et al., 2025; Chen et al., 2025; Ding et al., 2024; Jiralerspong et al., 2025). 

In this work, we address this limitation by introducing Fourier- and wavelet-based regularization terms that operate entirely at the loss level. The proposed approach applies uniformly to DDPM, DDIM, and EDM training and sampling, 

3 

**Spectral Regularization for Diffusion Models** 

providing explicit control over frequency and scale without modifying the diffusion process or model architecture. 

## **3.5. Fourier and Wavelet Transformations** 

Fourier and wavelet transforms provide foundational tools for analyzing the frequency and multi-scale structure of signals. For image and audio data, these representations offer complementary perspectives on smoothness, oscillatory behavior, and localized structure that are not explicitly captured in the spatial/temporal domains. As such, they play a central role in signal processing, compression, and perceptual modeling, and thus serve as natural candidates for imposing inductive biases in generative models. 

## 3.5.1. FOURIER TRANSFORM 

The Fourier transform represents a signal as a linear superposition of global sinusoidal basis functions and generalizes naturally to signals defined on R _[n]_ . Let _x_ ( _ζ_ ) _∈ L_[2] (R _[n]_ ), with spatial/temporal coordinate _ζ ∈_ R _[n]_ . The _n_ -dimensional Fourier transform is defined as 

**==> picture [181 x 24] intentionally omitted <==**

limiting their expressiveness for non-stationary or localized phenomena common in images and audio. This limitation is especially relevant for diffusion models, whose denoising objectives constrain only the total squared error and not its spectral distribution. As a result, small overall losses may still correspond to disproportionate high-frequency errors, leading to over-smoothing or perceptual artifacts. 

## 3.5.2. WAVELET TRANSFORMS 

Wavelet transforms address the local limitations of the Fourier transform by providing a localized, multi-resolution representation of signals. Instead of global sinusoids, wavelets use basis functions that are localized in both space (or time) and frequency. Let _x_ ( _ζ_ ) _∈ L_[2] (R _[n]_ ) be a signal with coordinate _ζ ∈_ R _[n]_ . Given a mother wavelet _ψ_ ( _ζ_ ) satisfying suitable admissibility conditions, a family of wavelets is generated through isotropic dilation and translation: 

**==> picture [179 x 25] intentionally omitted <==**

where _a ∈_ R[+] controls scale and _b ∈_ R _[n]_ controls translation. The normalization factor ensures energy preservation across scales. 

with inverse transform 

**==> picture [178 x 24] intentionally omitted <==**

where _ω ∈_ R _[n]_ denotes the frequency vector and _ω · ζ_ is the usual Euclidean inner/dot product. In practice, for discrete signals defined on grid (e.g. images and audio), the discrete Fourier transform and can be computed efficiently using the fast Fourier transform (FFT) (Press et al., 2007). 

A central property of the Fourier transform is energy preservation, formalized by the Parseval–Plancherel theorem: 

**==> picture [162 x 12] intentionally omitted <==**

i.e. the total _L_[2] energy of a signal is invariant under transformation to the frequency domain. Consequently, minimizing a squared reconstruction loss in signal space is equivalent to minimizing it in Fourier space. Crucially, this equivalence holds only for _L_[2] norms and does not extend to the _L_[1] losses used in our spectral regularization. Parseval’s identity is therefore agnostic to how reconstruction error is distributed across frequencies. This motivates the introduction of explicit spectral penalties: by applying _L_[1] discrepancies to Fourier amplitude (and phase), we intentionally break Parseval invariance to directly control the allocation of error, penalizing spectral imbalance—particularly in high-frequency components that are weakly constrained by standard diffusion objectives. 

While the Fourier spectrum captures global structure such as smoothness, anisotropy, and periodicity, its globally supported basis functions lack spatial or temporal localization, 

The continuous wavelet transform (CWT) of _x_ is defined as 

**==> picture [184 x 23] intentionally omitted <==**

yielding a joint representation over both scale and spatial location. 

Discrete wavelet transforms (DWTs) provide a hierarchical, multi-scale decomposition of a signal into approximation and detail coefficients at dyadic scales. For multidimensional signals, this yields multiple oriented sub-bands per scale, capturing localized and directional structure (e.g., horizontal, vertical, and diagonal components in images). Low-frequency coefficients encode coarse content, while high-frequency coefficients capture edges, textures, and transient features, closely mirroring the hierarchical representations learned by deep neural networks. 

From a modeling perspective, wavelet-domain representations enable explicit control over both scale and spatial localization. Regularization applied to wavelet coefficients can target specific resolutions or regions, making wavelets particularly effective for non-stationary signals such as natural images and audio, where meaningful structure varies across space, time, and scale. 

## **3.6. Spectral and Multi-Scale Structure** 

Both Fourier and wavelet representations offer complementary views of signal structure. Fourier transforms emphasize global frequency content and energy distributions, while 

4 

**Spectral Regularization for Diffusion Models** 

wavelets provide localized, scale-aware representations. For natural images and audio, meaningful structure is often expressed across a range of scales and frequencies, suggesting that generative models should respect these properties. 

From a modeling perspective, losses or regularizers defined in spectral or wavelet domains can be interpreted as constraints on the geometry of the generated distribution in transformed spaces. Unlike pointwise signal-domain objectives, such regularization explicitly emphasizes frequency balance, scale consistency, and localized structure. These properties motivate the use of Fourier/wavelet-based regularization within diffusion models, where denoising already operates implicitly across multiple noise scales. 

## **4. Spectral Regularization** 

Rather than introducing spectral losses as ad-hoc auxiliary penalties, we derive them from a geometric reinterpretation of diffusion training. Diffusion objectives constrain the reconstruction error in an _L_[2] sense, which controls only the total spectral energy of the error. Our spectral regularizers are formulated using _L_[1] discrepancies in the Fourier and wavelet domains rather than squared _L_[2] losses. This emphasizes the distribution of reconstruction error across frequencies rather than its total energy. The _L_[1] penalties treat discrepancies across bands uniformly and remain sensitive to structured high-frequency mismatches. 

## **4.1. Fourier-Regularized Diffusion Models** 

## 4.1.1. PRELIMINARIES 

Let _x_ 0 _∼ p_ data denote a data sample, and let _xt_ denote a noisy version of _x_ 0 obtained at diffusion time _t_ (or equivalently, noise level ( _α_ ). We denote by _F_ [ _xt_ ]( _ω_ ) the _n_ - dimensional Fourier transform of _xt_ , where _ω ∈_ R _[n]_ denotes the frequency variable. We express the Fourier transform in polar form, 

**==> picture [186 x 13] intentionally omitted <==**

where _At_ ( _**ω**_ ) = _|F_ [ _xt_ ]( _**ω**_ ) _|_ is the amplitude and _θt_ ( _**ω**_ ) is the phase. 

� � Given a predicted denoised sample _x_ 0 = _xθ_ ( _xt, t_ ), we denote its Fourier amplitude and phase by _A_[�] 0 and _ϕ_[�] 0, respectively. 

## 4.1.2. AMPLITUDE-BASED FOURIER LOSSES 

We consider Fourier-domain regularization terms that penalize discrepancies between the spectral representations of the predicted clean sample � _x_ 0 and the ground-truth sample _x_ 0. In contrast to pixel-domain losses, these objectives explicitly control frequency-dependent structure and scale. 

**Amplitude Loss.** The first regularizer enforces agreement between the amplitude spectra of the generated and target samples: 

**==> picture [177 x 20] intentionally omitted <==**

where _A_ 0 denotes the amplitude spectrum of the groundtruth sample and the expectation is taken over data samples and diffusion times. The Fourier amplitude spectrum captures how signal energy is distributed across frequencies, independent of spatial alignment. This enforces a global structural constraint that is invisible to pointwise losses. Importantly, amplitude discrepancies correspond to mismatches in frequency-wise energy allocation rather than local phase misalignment. As a result, amplitude-based regularization directly addresses the incorrect redistribution of reconstruction error across frequency bands. 

**Amplitude-and-Phase Loss.** While amplitude matching enforces global spectral alignment, it does not explicitly account for relative scaling across frequencies. To address this, we introduce a second regularizer that incorporates both amplitude magnitude and phase information: 

**==> picture [229 x 29] intentionally omitted <==**

The amplitude–phase (AP) coupling is motivated by the observation that phase information becomes meaningful primarily when associated with non-negligible spectral energy and that simply using the phase information leads to unstable training due to the branch-cuts. Large phase discrepancies in frequency bands with vanishing amplitude are perceptually insignificant, while similar discrepancies in dominant bands correspond to coherent structural distortions. This formulation avoids over-penalizing inconsequential phase noise while stabilizing fine-scale structure. 

## **4.2. Wavelet Regularized Diffusion Models** 

## 4.2.1. PRELIMINARIES 

Let _x_ 0 _∼ p_ data denote a data sample, and let _xt_ be its noisy counterpart at diffusion time _t_ . We denote by _W_ [ _xt_ ] = _Wt_[(] _[s,ℓ]_[)] ( _b_ ) � � _s,ℓ_[the discrete wavelet transform of] _[ x][t]_[, where] _s_ indexes scale, _ℓ_ indexes orientation or sub-band, and **b** _∈_ R _[n]_ denotes spatial or temporal location. This decomposition yields a hierarchical set of wavelet coefficients corresponding to different resolutions and directions. Simi� larly, given a predicted denoised sample � _x_ 0 = _xθ_ ( _xt, t_ ), we denote its wavelet coefficients by _W_[�] 0[(] _[s,ℓ]_[)] ( _b_ ). 

## 4.2.2. WAVELET LOSSES 

We define wavelet-domain regularization terms that penalize discrepancies between the wavelet coefficients of the 

5 

**Spectral Regularization for Diffusion Models** 

predicted clean sample and those of the ground-truth data. These losses encourage agreement across scales and locations, directly targeting multi-resolution structure. 

**Wavelet Coefficient Matching Loss.** Our first wavelet loss enforces alignment between wavelet coefficients at all scales and orientations: 

**==> picture [219 x 37] intentionally omitted <==**

where the norm is taken over spatial locations _b_ and _γs,l_ is the weight corresponding to each scale and sub-band. This loss encourages the diffusion model to match localized features such as edges, textures, and transient events at each resolution level. 

## **4.3. Training Objective** 

The final training objective augments the standard diffusion loss with the proposed Fourier regularization: 

**==> picture [156 x 10] intentionally omitted <==**

where _L_ is the standard diffusion denoising objective defined by Eq. (3) and _L_ S denotes either _L_[A] F[,] _[L]_[AS] F[,][or] _[L]_[W][.] The hyperparameter _λ_ controls the spectral regularization. 

This formulation mirrors the structure of constraintaugmented diffusion models while remaining fully datadriven and domain-agnostic. Fourier regularization shapes the learned denoising operator to respect global spectral properties without restricting the generative process to satisfy explicit rules or equations. The Wavelet loss provide complementary global and local spectral biases that improve the fidelity and robustness of generative diffusion models. As a result, the proposed method integrates seamlessly with existing diffusion architectures and sampling procedures. 

## **5. Experiments** 

## **5.1. Checkerboard Toy Experiment** 

To isolate the effect of our spectral regularizer, we construct a toy dataset of 64 _×_ 64 grayscale checkerboard images, which concentrate energy at a small set of high spatial frequencies. Such patterns provide a controlled stress test for assessing whether diffusion models preserve dominant periodic structure during generation. We compare a standard DDPM trained with the Mean Squared Error (MSE) noise prediction loss to the same objective augmented with our amplitude-and-phase spectral loss (Eq. 15), keeping architectures and optimization settings fixed. 

Figure 1 shows representative generations and Figure 2 shows the radially averaged power spectra. The baseline 

model exhibits visible smoothing and spectral leakage, producing attenuated and broadened responses at the checkerboard frequencies. In contrast, the spectral regularizer concentrates energy near the correct frequency bands and yields sharper periodic structure, resulting in samples that more closely match the ground-truth spectrum, despite remaining less strictly binary than the target. 

## **5.2. Image Datasets** 

We study spectral regularization as a lightweight fine-tuning strategy applied to pretrained EDM models (Karras et al., 2022). For each dataset and EDM formulation, models are fine-tuned for 5 optimization steps using the standard EDM denoising objective augmented with a spectral loss. This setup isolates the effect of loss-level spectral biasing without modifying the model architecture, diffusion formulation, or sampler. More training details are in Appendix A.1. 

We consider both variance-preserving (VP) and varianceexploding (VE) EDM variants. For each, we evaluate four spectral losses: (i) Fourier amplitude, (ii) Fourier amplitude+phase, (iii) Haar wavelet, and (iv) bi-orthogonal 1.3 (bior13) wavelet regularization. Fourier transforms are computed using PyTorch FFTs, while wavelet transforms are implemented based on PyWavelets (Lee et al., 2019). Experiments are conducted on CIFAR-10 (32 _×_ 32), FFHQ, and AFHQv2 (64 _×_ 64), following the standard EDM evaluation protocol. CIFAR-10 is evaluated under conditional sampling, whereas FFHQ and AFHQv2 are evaluated unconditionally. We consider two choices for the regularization weight. The first being the “weighted” setting where _λ_ = _λ_ EDM (see Eq. 7), and the “unweighted” setting where _λ_ = 1. Generative quality is measured using Frechet Incep-´ tion Distance (FID), computed between 50,000 generated samples and the full real dataset. Results are averaged over three random seeds. Some sampled images for the FFHQ and AFHQ datasets are shown in Appendix C. 

Table 1 reports FID scores for CIFAR-10, AFHQ, and FFHQ under both VE- and VP-EDM formulations. Since all experiments start from strong pretrained EDM baselines, the scope for improvement is necessarily limited. On CIFAR-10 (conditional), spectral losses have negligible effect, with all methods performing within the standard deviation of the EDM baseline, indicating limited benefit when conditional structure is already well captured. 

On the higher-resolution AFHQ and FFHQ datasets, we observe small but reliable FID reductions (typically 0 _._ 02 _−_ 0 _._ 07) across multiple spectral losses and EDM variants, with no cases of systematic degradation. These gains are comparable in magnitude across AFHQ and FFHQ, indicating that the proposed losses behave similarly on distinct but equally challenging natural image distributions. 

6 

**Spectral Regularization for Diffusion Models** 

**==> picture [75 x 76] intentionally omitted <==**

**==> picture [76 x 76] intentionally omitted <==**

**==> picture [76 x 76] intentionally omitted <==**

**==> picture [327 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Ground truth (b) Baseline (original MSE loss) (c) Ours (amp+phase loss)<br>**----- End of picture text -----**<br>


_Figure 1._ Checkerboard toy experiment. Figures (a) to (c) show the ground-truth pattern, a sample from a model trained without spectral regularization, and a sample from a model trained with the proposed amplitude-and-phase loss. 

_Table 1._ Frechet Inception Distance (FID) scores for different spectral regularizers and EDM variants.´ All models are fine-tuned from pretrained EDM baselines. Lower values indicate better generative performance. 

||EDM|EDM SAMPLING|||WEIGHTED|UNWEIGHTED|EDM|
|---|---|---|---|---|---|---|---|
|DATASET|VARIANT|STEPS|CONDITIONING|REGULARIZER|FID|FID|FID|
|||||AMP|1.82_±_0.01|1.82_±_0.01||
|CIFAR|VE|18|COND|AMPPHASE<br>HAAR|1.82_±_0.01<br>**1.81**_±_**0.01**|1.82_±_0.01<br>**1.81**_±_**0.02**|1.81_±_0.01|
|||||BIOR13|1.82_±_0.01|**1.81**_±_**0.02**||
|||||AMP|1.84_±_0.02|**1.84**_±_**0.02**||
|CIFAR|VP|18|COND|AMPPHASE<br>HAAR|1.84_±_0.02<br>1.84_±_0.02|**1.84**_±_**0.02**<br>**1.84**_±_**0.02**|1.84_±_0.02|
|||||BIOR13|**1.83**_±_**0.02**|**1.84**_±_**0.02**||
|||||AMP|**2.13**_±_**0.00**|**2.14**_±_**0.00**||
|AFHQ|VE|40|UNCOND|AMPPHASE<br>HAAR|**2.13**_±_**0.01**<br>2.14_±_0.01|2.16_±_0.01<br>**2.14**_±_**0.00**|2.17_±_0.00|
|||||BIOR13|2.14_±_0.01|2.14_±_0.01||
|||||AMP|**2.03**_±_**0.02**|2.05_±_0.02||
|AFHQ|VP|40|UNCOND|AMPPHASE<br>HAAR|2.04_±_0.02<br>2.05_±_0.02|**2.03**_±_**0.02**<br>**2.03**_±_**0.03**|2.04_±_0.00|
|||||BIOR13|2.07_±_0.02|2.05_±_0.02||
|||||AMP|2.5_±_0.02|2.51_±_0.02||
|FFHQ|VE|40|UNCOND|AMPPHASE<br>HAAR|**2.49**_±_**0.02**<br>2.5_±_0.02|2.50_±_0.02<br>**2.49**_±_**0.02**|2.56_±_0.03|
|||||BIOR13|2.51_±_0.02|2.5_±_0.02||
|||||AMP|2.35_±_0.03|**2.31**_±_**0.03**||
|FFHQ|VP|40|UNCOND|AMPPHASE<br>HAAR|**2.33**_±_**0.03**<br>**2.33**_±_**0.04**|2.34_±_0.03<br>2.33_±_0.04|2.38_±_0.01|
|||||BIOR13|2.34_±_0.04|2.32_±_0.04||



Importantly, the improvements are achieved with only a handful of fine-tuning steps and without modifying the architecture or sampler, highlighting that spectral regularization acts as a stable and data-efficient bias rather than an aggressive optimization mechanism. 

Overall, amplitude-phase regularization is the most consistently competitive method, achieving the best or tied-best performance on FFHQ and remaining close to optimal elsewhere. These results suggest that spectral regularization is most effective in higher-resolution, unconditional settings where diffusion models struggle to capture fine-scale structure, and offer limited benefits when baseline performance is already near saturation. 

## **5.3. Audio Dataset** 

We additionally evaluate spectral regularization for audio generation by fine-tuning a pretrained DiffWave model using the same loss-level protocol as in our image experiments (Kong et al., 2021). Specifically, we optimize the standard DiffWave DDPM denoising objective and augment it with one of our proposed spectral losses, namely Fourier amplitude, Fourier amplitude-phase, and wavelet-based regularizers. We fine-tune the official implementation for 150,000 steps on the LJSpeech-1.1 dataset (Ito & Johnson, 2017) starting from the publicly released pretrained checkpoint. Additional details are presented in Appendix A.2. 

A key implementation detail is that we compute spectral representations using the predicted clean waveform that is 

7 

**Spectral Regularization for Diffusion Models** 

_Table 2._ Audio generation quality metrics for DiffWave fine-tuning with spectral regularization. FAD measures distributional similarity in audio embedding space (lower is better), UTMOS estimates perceptual naturalness (higher is better), PESQ measures perceptual speech quality (higher is better), MR-STFT measures multi-resolution spectral error (lower is better), and NDB evaluates distributional coverage and mode balance (lower is better). We report the average and standard deviation of five runs with different seeds. 

|METHOD|_λ_|FAD_↓_|UTMOS_↑_|PESQ_↑_|MR-STFT_↓_|NDB_↓_|
|---|---|---|---|---|---|---|
|DIFFWAVE|–|1.994_±_0.008|3.941_±_0.005|3.440_±_0.002|1.217_±_0.001|0.63_±_0.02|
||10_−_4|**1.462**_±_**0.006**|3.953_±_0.009|3.477_±_0.003|1.1802_±_0.0005|0.65_±_0.03|
|AMP|10_−_5|1.609_±_0.007|3.953_±_0.009|3.476_±_0.001|1.1930_±_0.0004|0.63_±_0.02|
||10_−_6|1.775_±_0.008|3.952_±_0.006|3.491_±_0.003|1.1958_±_0.0005|0.64_±_0.03|
||10_−_4|1.694_±_0.009|3.969_±_0.008|**3.516**_±_**0.002**|1.1896_±_0.0003|0.66_±_0.02|
|AMP+PHASE|10_−_5|1.543_±_0.012|**3.988**_±_**0.003**|3.495_±_0.002|1.1773_±_0.0004|**0.59**_±_**0.01**|
||10_−_6|1.539_±_0.007|3.976_±_0.008|3.344_±_0.003|1.1921_±_0.0003|0.65_±_0.02|
||10_−_4|1.729_±_0.016|3.965_±_0.006|3.466_±_0.003|**1.1708**_±_**0.0006**|**0.59**_±_**0.02**|
|HAAR|10_−_5|1.992_±_0.014|3.988_±_0.008|3.485_±_0.003|1.2163_±_0.0002|0.66_±_0.02|
||10_−_6|2.123_±_0.010|3.923_±_0.002|3.359_±_0.002|1.2343_±_0.0002|0.69_±_0.03|
||10_−_4|1.492_±_0.011|3.977_±_0.009|3.500_±_0.002|1.1768_±_0.0002|0.62_±_0.01|
|BIOR|10_−_5|2.649_±_0.011|3.927_±_0.007|3.303_±_0.001|1.2949_±_0.0004|0.67_±_0.04|
||10_−_6|1.520_±_0.008|3.985_±_0.008|3.466_±_0.003|1.1787_±_0.0003|0.64_±_0.02|



**==> picture [235 x 137] intentionally omitted <==**

_Figure 2._ Radially averaged power spectra (log scale) for the ground truth, baseline DDPM with MSE loss, and DDPM with our amplitude+phase spectral loss. 

obtained from DDIM sampling, rather than directly transforming the noisy input. Specifically, at a randomly sampled diffusion timestep _t_ , we first obtain a denoised estimate _x[∗]_ 0 by running a deterministic DDIM update initialized at _xt_ using Eq. 5, and then compute Fourier/wavelet transforms of _x[∗]_ 0[for the spectral loss.][This follows the same pattern used] in Algorithm 1 of (Bastek et al., 2025). This choice ensures that spectral supervision is applied to a sample-consistent estimate of the clean signal, aligning the spectral objective with the model’s generation pathway. 

Table 2 demonstrates that loss-level spectral regularization consistently improves DiffWave audio generation across perceptual, spectral, and distributional metrics, despite being applied as a lightweight fine-tuning procedure. All spectral losses outperform the DiffWave baseline in FAD and PESQ for certain choice of _λ_ , indicating that explicit spectral-domain biasing effectively corrects spectral mismatches that are weakly constrained by time-domain de- 

noising alone. Fourier amplitude regularization yields the strongest FAD improvements, achieving the best overall score at moderate regularization strength, suggesting that matching global magnitude statistics is sufficient to recover dominant spectral structure that drives perceptual distance. In contrast, the amplitude-phase loss produces the most balanced gains across metrics, attaining the highest UTMOS and PESQ values and the lowest NDB. This shows the benefit of our novel approach of incorporating phase into the loss. Wavelet-based regularization exhibits complementary behavior: Haar wavelets achieve the lowest MR-STFT distance at higher _λ_ , highlighting improved multi-resolution temporal coherence, while biorthogonal wavelets show increased sensitivity to the regularization weight, likely due to their redundant, non-orthogonal structure. Overall, no single spectral loss dominates across all criteria. Thus, spectral regularization acts a controllable inductive bias whose effect depends on both representation choice and loss weighting. 

## **6. Conclusion** 

We introduced a loss-level spectral regularization framework for diffusion models that augments standard denoising objectives with differentiable Fourier- and wavelet-domain penalties, while leaving the diffusion process, architecture, and sampler unchanged. By explicitly shaping how reconstruction error is distributed across frequencies and scales, the proposed regularizers act as soft, domain-agnostic inductive biases that promote frequency balance and multiscale coherence. Empirically, we demonstrated that spectral regularization can be applied as a lightweight fine-tuning procedure to pretrained diffusion models, yielding consistent improvements in image and audio generation quality. The largest gains arise in higher-resolution, unconditional 

8 

**Spectral Regularization for Diffusion Models** 

settings, where diffusion models are most prone to spectral imbalance and degradation of fine-scale structure. Overall, our results suggest that loss-level spectral structure provides a principled and practical mechanism for improving diffusion models without sacrificing their generality or flexibility. 

## **7. Impact Statement** 

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. 

## **References** 

- Bastek, J.-H., Sun, W., and Kochmann, D. Physics-informed diffusion models. In _The Thirteenth International Conference on Learning Representations_ , 2025. URL https: //openreview.net/forum?id=tpYeermigp. 

- Benita, R., Elad, M., and Keshet, J. Spectral analysis of diffusion models with application to schedule design, 2025. URL https://arxiv.org/abs/2502.00180. 

- Bruna, J. and Mallat, S. Invariant scattering convolution networks. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ , 35(8):1872–1886, 2013. doi: 10. 1109/TPAMI.2012.230. 

- Chen, Y., Orlandi, M., Rapa, P. M., Benatti, S., Benini, L., and Li, Y. Physiowave: A multi-scale wavelettransformer for physiological signal representation, 2025. URL https://arxiv.org/abs/2506.10351. 

- Crabbe, J., Huynh, N., Stanczuk, J., and Van Der Schaar, M.´ Time series diffusion in the frequency domain. In _Proceedings of the 41st International Conference on Machine Learning_ , ICML’24. JMLR.org, 2024. 

- Dhariwal, P. and Nichol, A. Diffusion models beat gans on image synthesis. In Ranzato, M., Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W. (eds.), _Advances in Neural Information Processing Systems_ , volume 34, pp. 8780–8794. Curran Associates, Inc., 2021. URL https://proceedings.neurips. cc/paper_files/paper/2021/file/ 

- 49ad23d1ec9fa4bd8d77d02681df5cfa-Paper. pdf. 

- Ding, Z., Zhang, M., Wu, J., and Tu, Z. Patched denoising diffusion models for high-resolution image synthesis. In _The Twelfth International Conference on Learning Representations_ , 2024. 

- Gatys, L. A., Ecker, A. S., and Bethge, M. Image style transfer using convolutional neural networks. In 

_2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 2414–2423, 2016. doi: 10.1109/CVPR.2016.265. 

- Guth, F., Coste, S., De Bortoli, V., and Mallat, S. Wavelet score-based generative modeling. In _Proceedings of the 36th International Conference on Neural Information Processing Systems_ , NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc. ISBN 9781713871088. 

- Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), _Advances in Neural Information Processing Systems_ , volume 33, pp. 6840–6851. Curran Associates, Inc., 2020. URL https://proceedings.neurips. cc/paper_files/paper/2020/file/ 

- 4c5bcfec8584af0d967f1ab10179ca4b-Paper. pdf. 

- Hu, J., Hui, K.-H., Liu, Z., Li, R., and Fu, C.-W. Neural wavelet-domain diffusion for 3d shape generation, inversion, and manipulation. _ACM Trans. Graph._ , 43(2), January 2024. ISSN 0730-0301. doi: 10.1145/3635304. URL https://doi.org/10.1145/3635304. 

- Huang, Y., Huang, J., Liu, J., Yan, M., Dong, Y., Lv, J., and Chen, S. Wavedm: Wavelet-based diffusion models for image restoration. _IEEE Transactions on Multimedia_ , 26: 7058–7073, 2023. 

- Ito, K. and Johnson, L. The lj speech dataset. https:// keithito.com/LJ-Speech-Dataset/, 2017. 

- Jacobsen, C., Zhuang, Y., and Duraisamy, K. Cocogen: Physically consistent and conditioned score-based generative models for forward and inverse problems. _SIAM Journal on Scientific Computing_ , 47(2):C399– C425, 2025. doi: 10.1137/24M1636071. URL https: //doi.org/10.1137/24M1636071. 

- Jiralerspong, T., Earnshaw, B., Hartford, J., Bengio, Y., and Scimeca, L. Shaping inductive bias in diffusion models through frequency-based noise control, 2025. URL https://arxiv.org/abs/2502.10236. 

- Karras, T., Aittala, M., Laine, S., and Aila, T. Elucidating the design space of diffusion-based generative models. In _Proceedings of the 36th International Conference on Neural Information Processing Systems_ , NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc. ISBN 9781713871088. 

- Kiruluta, A. and Lemos, A. A hybrid wavelet-fourier method for next-generation conditional diffusion models, 2025. URL https://arxiv.org/abs/2504. 03821. 

9 

**Spectral Regularization for Diffusion Models** 

- Kong, Z., Ping, W., Huang, J., Zhao, K., and Catanzaro, B. Diffwave: A versatile diffusion model for audio synthesis. In _ICLR_ . OpenReview.net, 2021. URL http://dblp.uni-trier.de/db/conf/ iclr/iclr2021.html#KongPHZC21. 

- Lee, G. R., Gommers, R., Waselewski, F., Wohlfahrt, K., and O’Leary, A. Pywavelets: A python package for wavelet analysis. _Journal of Open Source Software_ , 4(36): 1237, 2019. doi: 10.21105/joss.01237. URL https: //doi.org/10.21105/joss.01237. 

- Liu, C., Fan, W., Liu, Y., Li, J., Li, H., Liu, H., Tang, J., and Li, Q. Generative diffusion models on graphs: methods and applications. In _Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence_ , IJCAI ’23, 2023. ISBN 978-1-956792-03-4. doi: 10. 24963/ijcai.2023/751. URL https://doi.org/10. 24963/ijcai.2023/751. 

- Liu, S., Zhu, C., Peng, L., Su, X., Li, L., and Wen, G. Wavelet-based diffusion with spatial-frequency attention for hyperspectral anomaly detection. _International Journal of Applied Earth Observation and Geoinformation_ , 142:104662, 2025. ISSN 15698432. doi: https://doi.org/10.1016/j.jag.2025.104662. URL https://www.sciencedirect.com/ science/article/pii/S1569843225003097. 

- Luo, Y., Chang, Y., and Wang, X. Wavelet fourier diffuser: Frequency-aware diffusion model for reinforcement learning, 2025. URL https://arxiv.org/abs/2509. 19305. 

- Mallat, S. _A Wavelet Tour of Signal Processing, Third Edition: The Sparse Way_ . Academic Press, Inc., USA, 3rd edition, 2008. ISBN 0123743702. 

- Oppenheim, A. V., Willsky, A. S., and Nawab, S. H. _Signals & systems (2nd ed.)_ . Prentice-Hall, Inc., USA, 1996. ISBN 0138147574. 

- Phillips, A., Seror, T., Hutchinson, M., Bortoli, V. D., Doucet, A., and Mathieu, E. Spectral diffusion processes, 2022. URL https://arxiv.org/abs/ 2209.14125. 

- Phung, H., Dao, Q., and Tran, A. Wavelet diffusion models are fast and scalable image generators. _2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 10199–10208, 2022. 

- Press, W. H., Teukolsky, S. A., Vetterling, W. T., and Flannery, B. P. _Numerical Recipes 3rd Edition: The Art of Scientific Computing_ . Cambridge University Press, 3 edition, 2007. ISBN 0521880688. 

- Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 10684–10695, June 2022. 

- Saharia, C., Chan, W., Saxena, S., Lit, L., Whang, J., Denton, E., Ghasemipour, S. K. S., Ayan, B. K., Mahdavi, S. S., Gontijo-Lopes, R., Salimans, T., Ho, J., Fleet, D. J., and Norouzi, M. Photorealistic text-to-image diffusion models with deep language understanding. In _Proceedings of the 36th International Conference on Neural Information Processing Systems_ , NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc. ISBN 9781713871088. 

- Shu, D., Li, Z., and Barati Farimani, A. A physics-informed diffusion model for high-fidelity flow field reconstruction. _Journal of Computational Physics_ , 478:111972, April 2023. ISSN 0021-9991. doi: 10.1016/j.jcp.2023. 111972. URL http://dx.doi.org/10.1016/j. jcp.2023.111972. 

- Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., and Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics. In Bach, F. and Blei, D. (eds.), _Proceedings of the 32nd International Conference on Machine Learning_ , volume 37 of _Proceedings of Machine Learning Research_ , pp. 2256–2265, Lille, France, 07– 09 Jul 2015. PMLR. URL https://proceedings. mlr.press/v37/sohl-dickstein15.html. 

- Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. In _9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021_ . OpenReview.net, 2021. URL https://openreview.net/ forum?id=PxTIG12RRHS. 

- Zhao, C., Cai, W., Dong, C., and Hu, C. Wavelet-based fourier information interaction with frequency diffusion adjustment for underwater image restoration. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pp. 8281–8291, 2024. 

- Zhou, L., Li, W., Li, J., Gao, G., and Lin, C.-W. Diffusion-based laplacian frequency-aware network for low-light image enhancement. _Pattern Recognition_ , 175:113060, 2026. ISSN 0031-3203. doi: https://doi.org/10.1016/j.patcog.2026.113060. URL https://www.sciencedirect.com/ science/article/pii/S0031320326000233. 

10 

**Spectral Regularization for Diffusion Models** 

_Table 3._ EDM fine-tuning hyperparameters used for all experiments. 

|**Dataset**|**Duration**|**Batch**|**LR**|**CRes**|**Dropout**|**Augment**|
|---|---|---|---|---|---|---|
|CIFAR-10 (cond/uncond)|0.5|16|2_×_10_−_4|–|–|–|
|AFHQ-64 (uncond)|0.5|16|5_×_10_−_5|1,2,2,2|0.25|0.15|
|FFHQ-64 (uncond)|–|16|5_×_10_−_5|1,2,2,2|0.05|0.15|



## **A. Training Details** 

## **A.1. EDM Configuration** 

We adopt the EDM framework of (Karras et al., 2022) as our pretrained diffusion backbone. We fine-tune eight publicly released checkpoints corresponding to the continuous-time DDPM++ (VP) and NCSN++ (VE) variants, pretrained on CIFAR-10 (conditional and unconditional), FFHQ-64 (unconditional), and AFHQv2 (unconditional). The pretrained weights are obtained from the official release and correspond to the following datasets and variants: 

## • **AFHQv2-64 (unconditional)** 

**– VE** : https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-ve.pkl **– VP** : https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-vp.pkl 

## • **CIFAR-10 (conditional)** 

**– VE** : https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-ve.pkl 

**– VP** : https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl 

## • **CIFAR-10 (unconditional)** 

**– VE** : https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-ve.pkl 

**– VP** : https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl 

## • **FFHQ-64 (unconditional)** 

**– VE** : https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-ve.pkl 

**– VP** : https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl 

All models are fine-tuned using the same optimization protocol as in the original EDM work. We use a per-GPU batch size of 16 and run experiments on NVIDIA A4000 and A6000 GPUs. Dataset-specific hyperparameters for fine-tuning are summarized in Table 3. 

## **A.2. Diffwave Configuration** 

We adopt DiffWave (Kong et al., 2021) as our pretrained audio diffusion backbone and fine-tune the official checkpoint released by LMNT.[1] Across all loss variants, we keep the network architecture and optimization settings fixed, following the original implementation. Specifically, we use: 

- **Training:** batch size = 16, learning rate = 2 _×_ 10 _[−]_[4] . 

- **Data / preprocessing:** sample rate = 22050 Hz, _n_ mels = 80, _n_ fft = 1024, hop = 256 samples, crop mel frames = 62. 

- **Model:** 30 residual layers, 64 residual channels, dilation cycle length = 10, conditional training. 

- **Diffusion schedules:** inference noise schedule = [10 _[−]_[4] _,_ 10 _[−]_[3] _,_ 10 _[−]_[2] _,_ 0 _._ 05 _,_ 0 _._ 2 _,_ 0 _._ 5]. 

We fine-tune DiffWave for 150,000 steps on the LJSpeech-1.1 dataset (Ito & Johnson, 2017), which contains 13,100 short utterances from a single speaker reading passages from seven non-fiction books. Following the original DiffWave training protocol, we excluded the LJ001* and LJ002* subsets from the training split, which we used for evaluation. 

> 1https://github.com/lmnt-com/diffwave 

11 

**Spectral Regularization for Diffusion Models** 

## **B. More Wavelet Transform Background** 

This appendix provides additional background on the discrete wavelet transforms used in our experiments, with particular focus on the Haar and biorthogonal 1.3 (bior1.3) wavelets. We include this discussion to clarify the mathematical structure of the corresponding spectral regularizers and to highlight the differences between orthogonal and biorthogonal constructions. 

## **B.1. Discrete Wavelet Transform** 

Given a discrete signal _x ∈_ R _[N]_ (or an image _x ∈_ R _[N][×][N]_ ), the discrete wavelet transform (DWT) represents _x_ in terms of localized basis functions obtained via dilations and translations of a mother wavelet _ψ_ and a scaling function _φ_ . In one dimension, the DWT decomposes _x_ into approximation and detail coefficients at multiple resolution levels, 

**==> picture [317 x 11] intentionally omitted <==**

where _aJ_ denotes coarse-scale approximation coefficients and _dj_ captures detail information at scale 2 _[−][j]_ . 

For images, a separable two-dimensional DWT is applied by performing the one-dimensional transform independently along each spatial axis. This yields one low-frequency subband (LL) and three directional high-frequency subbands (LH, HL, HH) at each scale, corresponding to horizontal, vertical, and diagonal features. 

From a spectral perspective, wavelet coefficients encode localized frequency content: unlike the Fourier transform, which provides global frequency information, wavelets retain joint spatial–frequency localization. This property makes waveletbased losses particularly sensitive to localized oscillations, edges, and multiscale structure. 

## **B.2. Haar Wavelet** 

The Haar wavelet is the simplest orthogonal wavelet and is defined by the scaling function 

**==> picture [99 x 7] intentionally omitted <==**

**==> picture [299 x 92] intentionally omitted <==**

The corresponding filter bank consists of length-two low-pass and high-pass filters, resulting in a transform that is exactly orthogonal and energy preserving. In the discrete setting, the Haar transform computes local averages and differences, making it particularly sensitive to sharp discontinuities and piecewise-constant structure. 

In our experiments, Haar regularization emphasizes consistency in coarse-to-fine difference patterns and strongly penalizes spurious high-frequency oscillations. However, due to its limited smoothness and short support, the Haar wavelet provides only a crude approximation of smooth spectral behavior. 

## **B.3. Biorthogonal 1.3 Wavelet** 

Biorthogonal wavelets generalize orthogonal constructions by allowing distinct analysis and synthesis bases. Rather than a single scaling function and wavelet, biorthogonal systems employ dual pairs ( _φ, ψ_ ) and ( ˜ _φ, ψ_[˜] ), which satisfy biorthogonality conditions but are not individually orthonormal. This added flexibility permits linear-phase filters and improved smoothness. 

The biorthogonal 1.3 (bior1.3) wavelet is defined implicitly through its associated analysis and synthesis filter banks. In one dimension, the analysis low-pass and high-pass filters are given by 

**==> picture [314 x 13] intentionally omitted <==**

while the synthesis low-pass and high-pass filters are 

**==> picture [345 x 14] intentionally omitted <==**

12 

**Spectral Regularization for Diffusion Models** 

These filters define the scaling and wavelet functions through the refinement equations 

**==> picture [358 x 22] intentionally omitted <==**

with analogous relations for the synthesis pair ( ˜ _φ, ψ_[˜] ) using ( _h,_[˜] ˜ _g_ ). The resulting wavelets are compactly supported but asymmetric. 

By construction, the analysis wavelet _ψ_ has one vanishing moment, while the synthesis wavelet _ψ_[˜] has three vanishing moments. This asymmetry yields smoother reconstructions than the Haar wavelet while retaining sensitivity to localized features. 

From a spectral perspective, the bior1.3 transform produces a more graded separation between low- and high-frequency components than Haar. High-frequency coefficients capture oscillatory behavior over slightly larger spatial neighborhoods, leading to smoother multiscale regularization when used as a loss. 

## **B.4. Wavelet-Based Regularization** 

Given a wavelet transform _W_ and its inverse _W[−]_[1] , we define wavelet-domain regularization by comparing wavelet coefficients of the predicted sample � _x_ and the reference sample _x_ , 

**==> picture [317 x 13] intentionally omitted <==**

This loss penalizes discrepancies across multiple spatial scales and orientations. Haar-based losses emphasize sharp transitions and edge-like features, while bior1.3-based losses impose smoother multiscale consistency due to their higherorder vanishing moments. 

Importantly, wavelet losses can be interpreted as localized spectral constraints: they enforce agreement between samples not only in frequency magnitude but also in spatially localized frequency bands. This contrasts with Fourier-based losses, which operate on globally supported basis functions and therefore impose uniform constraints across the domain. 

## **B.5. Relation to Fourier Regularization** 

Both Fourier and wavelet regularizers enforce spectral consistency, but they differ in how frequency information is localized. Fourier regularization constrains global frequency amplitudes (and phases), while wavelet regularization constrains frequency content within localized spatial neighborhoods and across scales. 

In practice, this distinction leads to different inductive biases. Fourier losses encourage globally correct power spectra, whereas wavelet losses emphasize local texture, edges, and multiscale coherence. Our empirical results reflect this difference, with Haar and bior1.3 wavelets exhibiting distinct trade-offs between sharpness and smoothness depending on the dataset and EDM parameterization. 

## **C. Generated Images for AFHQ and FFHQ** 

Here we present some selected image samples from our models. 

13 

**Spectral Regularization for Diffusion Models** 

**==> picture [289 x 289] intentionally omitted <==**

**==> picture [74 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a)  VE-EDM results<br>**----- End of picture text -----**<br>


**==> picture [289 x 289] intentionally omitted <==**

**==> picture [74 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
(b)  VP-EDM results<br>**----- End of picture text -----**<br>


_Figure 3._ Generated AFHQ samples obtained by fine-tuning with the unweighted Fourier amplitude loss under different EDM formulations. 

14 

**Spectral Regularization for Diffusion Models** 

**==> picture [289 x 289] intentionally omitted <==**

**==> picture [74 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a)  VE-EDM results<br>**----- End of picture text -----**<br>


**==> picture [289 x 289] intentionally omitted <==**

**==> picture [74 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
(b)  VP-EDM results<br>**----- End of picture text -----**<br>


_Figure 4._ Generated AFHQ samples obtained by fine-tuning with the unweighted Fourier amplitude+phase loss under different EDM formulations. 

15 

**Spectral Regularization for Diffusion Models** 

**==> picture [289 x 289] intentionally omitted <==**

_(a)_ VE-EDM results 

**==> picture [289 x 289] intentionally omitted <==**

**==> picture [74 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
(b)  VP-EDM results<br>**----- End of picture text -----**<br>


_Figure 5._ Generated AFHQ samples obtained by fine-tuning with the unweighted Haar wavelet loss under different EDM formulations. 

16 

**Spectral Regularization for Diffusion Models** 

**==> picture [289 x 289] intentionally omitted <==**

_(a)_ VE-EDM results 

**==> picture [289 x 289] intentionally omitted <==**

**==> picture [74 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
(b)  VP-EDM results<br>**----- End of picture text -----**<br>


_Figure 6._ Generated AFHQ samples obtained by fine-tuning with the unweighted bi-orthogonal 1.3 wavelet loss under different EDM formulations. 

17 

**Spectral Regularization for Diffusion Models** 

**==> picture [289 x 289] intentionally omitted <==**

**==> picture [74 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a)  VE-EDM results<br>**----- End of picture text -----**<br>


**==> picture [289 x 289] intentionally omitted <==**

_(b)_ VP-EDM results 

_Figure 7._ Generated FFHQ samples obtained by fine-tuning with the unweighted Fourier amplitude loss under different EDM formulations. 

18 

**Spectral Regularization for Diffusion Models** 

**==> picture [289 x 289] intentionally omitted <==**

**==> picture [74 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a)  VE-EDM results<br>**----- End of picture text -----**<br>


**==> picture [289 x 289] intentionally omitted <==**

**==> picture [74 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
(b)  VP-EDM results<br>**----- End of picture text -----**<br>


_Figure 8._ Generated FFHQ samples obtained by fine-tuning with the unweighted Fourier amplitude+phase loss under different EDM formulations. 

19 

**Spectral Regularization for Diffusion Models** 

**==> picture [289 x 289] intentionally omitted <==**

**==> picture [74 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a)  VE-EDM results<br>**----- End of picture text -----**<br>


**==> picture [289 x 289] intentionally omitted <==**

**==> picture [74 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
(b)  VP-EDM results<br>**----- End of picture text -----**<br>


_Figure 9._ Generated FFHQ samples obtained by fine-tuning with the unweighted Haar wavelet loss under different EDM formulations. 

20 

**Spectral Regularization for Diffusion Models** 

**==> picture [289 x 289] intentionally omitted <==**

**==> picture [74 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a)  VE-EDM results<br>**----- End of picture text -----**<br>


**==> picture [289 x 289] intentionally omitted <==**

**==> picture [74 x 9] intentionally omitted <==**

**----- Start of picture text -----**<br>
(b)  VP-EDM results<br>**----- End of picture text -----**<br>


_Figure 10._ Generated FFHQ samples obtained by fine-tuning with the unweighted bi-orthogonal 1.3 wavelet loss under different EDM formulations. 

21 

