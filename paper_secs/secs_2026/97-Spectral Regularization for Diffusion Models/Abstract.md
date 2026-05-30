# Abstract

Diffusion models are typically trained using pointwise reconstruction objectives that are agnostic to the spectral and multi-scale structure of natural signals. We propose a loss-level spectral regularization framework that augments standard diffusion training with differentiable Fourier- and wavelet-domain losses, without modifying the diffusion process, model architecture, or sampling procedure. The proposed regularizers act as soft inductive biases that encourage appropriate frequency balance and coherent multi-scale structure in generated samples. Our approach is compatible with DDPM, DDIM, and EDM formulations and introduces negligible computational overhead. Experiments on image and audio generation demonstrate consistent improvements in sample quality, with the largest gains observed on higher-resolution, unconditional datasets where fine-scale structure is most challenging to model.

## 1. Introduction

Diffusion models have emerged as a powerful and versatile framework for generative modeling of high-dimensional signals. By learning to reverse a gradual noising process, diffusion models provide stable training, strong mode coverage, and state-of-the-art performance across a wide range of modalities, including natural images, audio waveforms, and graphical representations [\(Sohl-Dickstein et al.,](#page-9-0) [2015;](#page-9-0) [Ho et al.,](#page-8-0) [2020;](#page-8-0) [Dhariwal & Nichol,](#page-8-1) [2021;](#page-8-1) [Liu et al.,](#page-9-1) [2023\)](#page-9-1). Their success has led to rapid adoption in image synthesis [\(Rombach et al.,](#page-9-2) [2022\)](#page-9-2), audio generation [\(Kong et al.,](#page-9-3) [2021\)](#page-9-3), and conditional generation tasks such as super-resolution and inpainting [\(Saharia et al.,](#page-9-4) [2022\)](#page-9-4).

Despite their strong empirical performance, diffusion models are typically trained using pointwise reconstruction

*Preprint. March 4, 2026.*

losses defined in the signal domain, most commonly meansquared error on predicted noise or clean signals [\(Ho et al.,](#page-8-0) [2020;](#page-8-0) [Song et al.,](#page-9-5) [2021\)](#page-9-5). While these objectives are well motivated from both a empirical and theoretical perspective, they are agnostic to the spectral and multi-scale structure that characterizes many natural signals. Images and audio often exhibit highly structured frequency content, longrange correlations, and scale-dependent patterns that are only implicitly captured through pixel/sample-level supervision. As a result, diffusion models often generate samples that match low-level statistics while exhibiting artifacts such as over-smoothing, incorrect frequency balance, or degraded fine-scale structure [\(Benita et al.,](#page-8-2) [2025;](#page-8-2) [Chen et al.,](#page-8-3) [2025;](#page-8-3) [Ding et al.,](#page-8-4) [2024;](#page-8-4) [Jiralerspong et al.,](#page-8-5) [2025\)](#page-8-5).

Recent works have explored augmenting diffusion models with additional forms of structure or inductive bias. In scientific and engineering domains, this has included incorporating explicit constraint or residual losses into diffusion training to enforce known properties of the data-generating process [\(Shu et al.,](#page-9-6) [2023;](#page-9-6) [Bastek et al.,](#page-8-6) [2025;](#page-8-6) [Jacobsen et al.,](#page-8-7) [2025\)](#page-8-7). While effective in settings where such constraints are well defined, these approaches are less directly applicable to general image and audio tasks, where the structure is implicit, statistical, and perceptual rather than being defined explicitly as a set of differential equations.

In contrast, frequency-domain representations have long played a central role in image and audio processing. Fourier analysis provides a global description of signal energy distribution across frequencies and is fundamental to understanding smoothness, noise, and periodic structure [\(Oppenheim](#page-9-7) [et al.,](#page-9-7) [1996\)](#page-9-7). Wavelet representations extend this perspective by offering localized, multi-resolution decompositions that capture both spatial or temporal locality and scale [\(Mallat,](#page-9-8) [2008\)](#page-9-8). These representations underpin classical methods in compression, denoising, and texture analysis, and they have also been used as inductive biases or auxiliary losses in deep learning models for images and audio [\(Bruna &](#page-8-8) [Mallat,](#page-8-8) [2013;](#page-8-8) [Gatys et al.,](#page-8-9) [2016;](#page-8-9) [Kong et al.,](#page-9-3) [2021\)](#page-9-3).

We propose a spectral regularization framework for diffusion model training that augments the standard denoising objective with differentiable losses defined in the Fourier and wavelet domains. Rather than modifying model architectures, samplers, or imposing hard constraints, our approach

<sup>1</sup>Department of Mathematics, University of California Riverside, Riverside, California, USA <sup>2</sup>Department of Computer Science, University of California Riverside, Riverside, California, USA. Correspondence to: Evangelos Papalexakis <epapalexcs@cs.ucr.edu>.

introduces a soft inductive bias that encourages generated samples to match the frequency-dependent structure of the data. Fourier-based losses capture global spectral characteristics, while wavelet-based losses provide localized, scaleaware control well suited to non-stationary signals such as audio and textured images. The resulting regularization is domain-agnostic, requires no auxiliary networks or additional supervision, and incurs negligible computational overhead. Empirically, we find that incorporating spectral information complements pixel-level objectives, leading to sharper reconstructions, improved perceptual quality, and reduced overfitting, while preserving the diversity and expressiveness of diffusion models. The code is available at [https://anonymous.4open.science/r/fourierdm-8B8E.](https://anonymous.4open.science/r/fourierdm-8B8E)

## 2. Related Works

Spectral-domain diffusion models. A growing body of work explores incorporating frequency structure directly into diffusion models by redefining the diffusion process in spectral coordinates or operating on transformed representations. [Crabbe et al.](#page-8-10) ´ [\(2024\)](#page-8-10) formulate diffusion for time series in the Fourier domain, explicitly modeling conjugatesymmetric complex coefficients to ensure real-valued reconstructions. [Jiralerspong et al.](#page-8-5) [\(2025\)](#page-8-5) shape the diffusion dynamics in frequency space to emphasize or suppress specific spectral bands through frequency-based noise control. [Phillips et al.](#page-9-9) [\(2022\)](#page-9-9) propose spectral process diffusion, performing score-based modeling over coefficients of stochastic processes expressed in a spectral basis.

These approaches embed spectral structure directly into the diffusion state space or dynamics, requiring modified forward processes or basis-specific parameterizations. In contrast, our method preserves standard diffusion formulations (DDPM, DDIM, and EDM) and introduces spectral structure solely through auxiliary loss terms applied to reconstructions. Spectral bias is therefore imposed at the loss objective level rather than through changes to the generative process itself.

Wavelet-based and multi-resolution diffusion. Wavelet representations have motivated several diffusion models that operate directly on multi-scale decompositions. [Guth](#page-8-11) [et al.](#page-8-11) [\(2022\)](#page-8-11) perform score-based diffusion on wavelet coefficients and interpret the resulting hierarchy through renormalization group theory. [Phung et al.](#page-9-10) [\(2022\)](#page-9-10) and [Hu et al.](#page-8-12) [\(2024\)](#page-8-12) apply wavelet-domain diffusion for efficient image generation and 3D shape modeling, respectively. Related approaches selectively apply diffusion to low-frequency components while refining high-frequency content using auxiliary modules, primarily for restoration and efficiency [\(Huang et al.,](#page-8-13) [2023;](#page-8-13) [Zhao et al.,](#page-9-11) [2024;](#page-9-11) [Liu et al.,](#page-9-12) [2025;](#page-9-12) [Zhou et al.,](#page-9-13) [2026\)](#page-9-13). In these works, diffusion is typically

performed in a transformed representation or coupled with frequency-specific architectural modules. Our approach instead retains standard pixel- or waveform-space diffusion and applies wavelet regularization purely at the loss level. This maintains architectural simplicity while encouraging multi-scale consistency through the training objective.

Hybrid Fourier-wavelet diffusion. Several recent works explicitly combine Fourier and wavelet representations within diffusion pipelines. [Luo et al.](#page-9-14) [\(2025\)](#page-9-14) introduce crossfrequency fusion for trajectory modeling in reinforcement learning. [Kiruluta & Lemos](#page-8-14) [\(2025\)](#page-8-14) propose a hybrid forward process combining partial Fourier corruption with wavelet decomposition and multi-branch denoising networks. These hybrid approaches integrate spectral structure into the forward process or network design. In contrast, we treat Fourier and wavelet transforms as analysis operators used only for defining differentiable penalties. The diffusion dynamics remain unchanged, making our framework modular and directly compatible with existing implementations.

Constraint-augmented diffusion models. More broadly, diffusion models have been augmented with auxiliary losses to encode known structure, particularly in scientific domains. Physics-informed diffusion models incorporate residualbased constraints derived from governing equations to enforce physical consistency [\(Shu et al.,](#page-9-6) [2023;](#page-9-6) [Bastek et al.,](#page-8-6) [2025;](#page-8-6) [Jacobsen et al.,](#page-8-7) [2025\)](#page-8-7). While effective when explicit constraints are available, such methods are less applicable to natural images and audio, where structure is statistical rather than rule-based. Our work adopts the broader idea of auxiliary regularization but replaces equation-based constraints with soft frequency-domain penalties derived from signal statistics.

Positioning of Our Approach. Overall, existing frequency-aware diffusion methods typically modify the diffusion process, operate in transformed domains, or introduce task-specific architectures. In contrast, we propose a loss-level spectral regularization framework that preserves standard diffusion formulations (DDPM, DDIM, and EDM) while encouraging frequency balance and multi-scale coherence through differentiable Fourierand wavelet-domain losses. This design is domain-agnostic, architecture-independent, and directly compatible with existing diffusion training and sampling pipelines.

## 3. Diffusion Models

Diffusion models define a class of generative models that construct complex data distributions by reversing a gradual stochastic noising process. The central idea is to transform data samples into noise through a forward process that is analytically tractable, and to learn a parameterized reverse process that progressively removes noise. This framework has evolved through several closely related formulations, including denoising diffusion probabilistic models (DDPMs) (Ho et al., 2020), deterministic diffusion implicit models (DDIMs) (Song et al., 2021), and more recent formulations such as Elucidated Diffusion Models (EDMs) (Karras et al., 2022), which unify training and sampling under continuous noise parameterizations.

#### 3.1. Denoising Diffusion Probabilistic Models (DDPM)

Denoising Diffusion Probabilistic Models (DDPMs) define a discrete-time Markov chain that gradually corrupts data with Gaussian noise over T steps. Given a data sample  $x_0 \sim p_{\rm data}$ , the forward process is defined as

$$q(x_t \mid x_{t-1}) = \mathcal{N}\left(\sqrt{1 - \beta_t} x_{t-1}, \ \beta_t I\right), \tag{1}$$

where  $\{\beta_t\}_{t=1}^T$  is a predefined variance schedule. By construction, this process admits a closed-form marginal defined by

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I), \quad (2)$$

with  $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$ . The reverse process is parameterized by a neural network trained to predict either the mean of the reverse transition or, more commonly, the added noise  $\varepsilon$ . This leads to the standard DDPM objective:

<span id="page-2-0"></span>
$$\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{x_0, \, \varepsilon, \, t} \left[ \left\| \varepsilon - \varepsilon_{\theta}(x_t, t) \right\|_2^2 \right], \tag{3}$$

which can be shown to correspond to a variational bound on the negative log-likelihood. DDPMs established diffusion models as an alternative to GANs, offering stable training and strong mode coverage, at the cost of relatively slow sampling due to the large number of required reverse steps

### 3.2. Denoising Diffusion Implicit Models (DDIM)

Denoising Diffusion Implicit Models (DDIMs) reinterpret the DDPM framework by constructing a non-Markovian, deterministic sampling process that preserves the same marginal distributions as DDPMs while enabling faster generation. Rather than sampling from a stochastic reverse transition, DDIMs define a deterministic mapping:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \, \widehat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \, \varepsilon_{\theta}(x_t, t), \qquad (4)$$

where

<span id="page-2-2"></span>
$$\widehat{x}_0 = \left(x_t - \sqrt{1 - \bar{\alpha}_t} \,\varepsilon_\theta(x_t, t)\right) / \sqrt{\bar{\alpha}_t}.\tag{5}$$

This formulation reveals that diffusion models define a family of generative trajectories indexed by a stochasticity parameter, interpolating between fully stochastic DDPM sampling and deterministic DDIM sampling. Importantly, DDIMs use the same training objective as DDPMs with the only difference being the sampling procedure.

#### 3.3. Elucidated Diffusion Models (EDM)

Elucidated Diffusion Models (EDMs) further generalize diffusion modeling by formulating training and sampling in continuous noise space rather than discrete time steps. Instead of indexing noise by t, EDMs parameterize corruption using the noise standard deviation  $\sigma$ , defining noisy samples as

$$x_{\sigma} = x_0 + \sigma \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I).$$
 (6)

EDMs introduce a reweighted denoising objective of the form

<span id="page-2-1"></span>
$$\mathcal{L}_{\mathrm{EDM}} = \mathbb{E}_{x_0, \, \varepsilon, \, \sigma} \left[ \lambda_{EDM}(\sigma) \left\| \varepsilon - \varepsilon_{\theta}(x_{\sigma}, \sigma) \right\|_{2}^{2} \right], \quad (7)$$

where the weighting function  $\lambda_{EDM}(\sigma)$  is chosen to balance contributions across noise scales. A key advantage of the EDM framework is that it exposes diffusion models as learning scale-dependent denoisers across a continuum of noise levels. This perspective is particularly relevant for image and audio generation, where meaningful structure exists across a wide range of spatial or temporal scales. By explicitly decoupling noise level, loss weighting, and sampling trajectory, EDMs provide a flexible foundation for incorporating additional regularization terms without altering the core generative mechanism.

#### 3.4. Implications for Regularized Diffusion Training

Across DDPM, DDIM, and EDM formulations, diffusion models are trained using pointwise denoising objectives defined in the signal domain. While sufficient for likelihood-based learning, these objectives do not explicitly constrain how reconstruction error is distributed across frequencies or scales. Since the learned denoiser defines a family of noise-dependent reconstruction operators, augmenting the training objective with spectral or multi-scale regularization naturally complements the diffusion framework without altering its probabilistic foundations.

From an operator perspective, diffusion models recover coarse, low-frequency structure at high noise levels, while fine-scale, high-frequency components are reconstructed only in low-noise regimes. As a result, high-frequency errors are learned under weaker effective regularization and fewer samples, making them more susceptible to overfitting and instability. Standard diffusion objectives treat all reconstruction errors equally, allowing error to concentrate in perceptually or structurally undesirable frequency bands (Benita et al., 2025; Chen et al., 2025; Ding et al., 2024; Jiralerspong et al., 2025).

In this work, we address this limitation by introducing Fourier- and wavelet-based regularization terms that operate entirely at the loss level. The proposed approach applies uniformly to DDPM, DDIM, and EDM training and sampling,

providing explicit control over frequency and scale without modifying the diffusion process or model architecture.

#### 3.5. Fourier and Wavelet Transformations

Fourier and wavelet transforms provide foundational tools for analyzing the frequency and multi-scale structure of signals. For image and audio data, these representations offer complementary perspectives on smoothness, oscillatory behavior, and localized structure that are not explicitly captured in the spatial/temporal domains. As such, they play a central role in signal processing, compression, and perceptual modeling, and thus serve as natural candidates for imposing inductive biases in generative models.

#### 3.5.1. FOURIER TRANSFORM

The Fourier transform represents a signal as a linear superposition of global sinusoidal basis functions and generalizes naturally to signals defined on  $\mathbb{R}^n$ . Let  $x(\zeta) \in L^2(\mathbb{R}^n)$ , with spatial/temporal coordinate  $\zeta \in \mathbb{R}^n$ . The n-dimensional Fourier transform is defined as

$$X(\omega) = \int_{\mathbb{R}^n} x(\zeta) e^{-i2\pi\omega\cdot\zeta} d\zeta, \tag{8}$$

with inverse transform

$$x(\zeta) = \int_{\mathbb{R}^n} X(\omega) \, e^{i2\pi\omega\cdot\zeta} \, d\zeta,\tag{9}$$

where  $\omega \in \mathbb{R}^n$  denotes the frequency vector and  $\omega \cdot \zeta$  is the usual Euclidean inner/dot product. In practice, for discrete signals defined on grid (e.g. images and audio), the discrete Fourier transform and can be computed efficiently using the fast Fourier transform (FFT) (Press et al., 2007).

A central property of the Fourier transform is energy preservation, formalized by the Parseval–Plancherel theorem:

$$||x(\zeta)||_2^2 = ||X(\omega)||_2^2,$$
 (10)

i.e. the total  $L^2$  energy of a signal is invariant under transformation to the frequency domain. Consequently, minimizing a squared reconstruction loss in signal space is equivalent to minimizing it in Fourier space. Crucially, this equivalence holds only for  $L^2$  norms and does not extend to the  $L^1$  losses used in our spectral regularization. Parseval's identity is therefore agnostic to how reconstruction error is distributed across frequencies. This motivates the introduction of explicit spectral penalties: by applying  $L^1$  discrepancies to Fourier amplitude (and phase), we intentionally break Parseval invariance to directly control the allocation of error, penalizing spectral imbalance—particularly in high-frequency components that are weakly constrained by standard diffusion objectives.

While the Fourier spectrum captures global structure such as smoothness, anisotropy, and periodicity, its globally supported basis functions lack spatial or temporal localization, limiting their expressiveness for non-stationary or localized phenomena common in images and audio. This limitation is especially relevant for diffusion models, whose denoising objectives constrain only the total squared error and not its spectral distribution. As a result, small overall losses may still correspond to disproportionate high-frequency errors, leading to over-smoothing or perceptual artifacts.

#### 3.5.2. WAVELET TRANSFORMS

Wavelet transforms address the local limitations of the Fourier transform by providing a localized, multi-resolution representation of signals. Instead of global sinusoids, wavelets use basis functions that are localized in both space (or time) and frequency. Let  $x(\zeta) \in L^2(\mathbb{R}^n)$  be a signal with coordinate  $\zeta \in \mathbb{R}^n$ . Given a mother wavelet  $\psi(\zeta)$  satisfying suitable admissibility conditions, a family of wavelets is generated through isotropic dilation and translation:

$$\psi_{a,b}(\zeta) = \frac{1}{|a|^{n/2}} \psi\left(\frac{\zeta - b}{a}\right),\tag{11}$$

where  $a \in \mathbb{R}^+$  controls scale and  $b \in \mathbb{R}^n$  controls translation. The normalization factor ensures energy preservation across scales.

The continuous wavelet transform (CWT) of x is defined as

$$W_x(a,b) = \int_{\mathbb{R}^n} x(\zeta) \, \psi_{a,b}(\zeta) \, d\zeta \tag{12}$$

yielding a joint representation over both scale and spatial location.

Discrete wavelet transforms (DWTs) provide a hierarchical, multi-scale decomposition of a signal into approximation and detail coefficients at dyadic scales. For multi-dimensional signals, this yields multiple oriented sub-bands per scale, capturing localized and directional structure (e.g., horizontal, vertical, and diagonal components in images). Low-frequency coefficients encode coarse content, while high-frequency coefficients capture edges, textures, and transient features, closely mirroring the hierarchical representations learned by deep neural networks.

From a modeling perspective, wavelet-domain representations enable explicit control over both scale and spatial localization. Regularization applied to wavelet coefficients can target specific resolutions or regions, making wavelets particularly effective for non-stationary signals such as natural images and audio, where meaningful structure varies across space, time, and scale.

## 3.6. Spectral and Multi-Scale Structure

Both Fourier and wavelet representations offer complementary views of signal structure. Fourier transforms emphasize global frequency content and energy distributions, while

wavelets provide localized, scale-aware representations. For natural images and audio, meaningful structure is often expressed across a range of scales and frequencies, suggesting that generative models should respect these properties.

From a modeling perspective, losses or regularizers defined in spectral or wavelet domains can be interpreted as constraints on the geometry of the generated distribution in transformed spaces. Unlike pointwise signal-domain objectives, such regularization explicitly emphasizes frequency balance, scale consistency, and localized structure. These properties motivate the use of Fourier/wavelet-based regularization within diffusion models, where denoising already operates implicitly across multiple noise scales.

## 4. Spectral Regularization

Rather than introducing spectral losses as ad-hoc auxiliary penalties, we derive them from a geometric reinterpretation of diffusion training. Diffusion objectives constrain the reconstruction error in an  $L^2$  sense, which controls only the total spectral energy of the error. Our spectral regularizers are formulated using  $L^1$  discrepancies in the Fourier and wavelet domains rather than squared  $L^2$  losses. This emphasizes the distribution of reconstruction error across frequencies rather than its total energy. The  $L^1$  penalties treat discrepancies across bands uniformly and remain sensitive to structured high-frequency mismatches.

### 4.1. Fourier-Regularized Diffusion Models

### 4.1.1. Preliminaries

Let  $x_0 \sim p_{\text{data}}$  denote a data sample, and let  $x_t$  denote a noisy version of  $x_0$  obtained at diffusion time t (or equivalently, noise level  $(\alpha)$ . We denote by  $\mathcal{F}[x_t](\omega)$  the n-dimensional Fourier transform of  $x_t$ , where  $\omega \in \mathbb{R}^n$  denotes the frequency variable. We express the Fourier transform in polar form,

$$\mathcal{F}[x_t](\omega) = A_t(\omega) \exp(i\phi_t(\omega)),$$
 (13)

where  $A_t(\omega) = |\mathcal{F}[x_t](\omega)|$  is the amplitude and  $\theta_t(\omega)$  is the phase.

Given a predicted denoised sample  $\widehat{x}_0 = \widehat{x}_{\theta}(x_t, t)$ , we denote its Fourier amplitude and phase by  $\widehat{A}_0$  and  $\widehat{\phi}_0$ , respectively.

### 4.1.2. AMPLITUDE-BASED FOURIER LOSSES

We consider Fourier-domain regularization terms that penalize discrepancies between the spectral representations of the predicted clean sample  $\hat{x}_0$  and the ground-truth sample  $x_0$ . In contrast to pixel-domain losses, these objectives explicitly control frequency-dependent structure and scale.

**Amplitude Loss.** The first regularizer enforces agreement between the amplitude spectra of the generated and target samples:

$$\mathcal{L}_{\mathrm{F}}^{\mathrm{A}} = \mathbb{E}_{x_0, t} \left[ \left\| A_0 - \widehat{A}_0 \right\|_{1} \right], \tag{14}$$

where  $A_0$  denotes the amplitude spectrum of the ground-truth sample and the expectation is taken over data samples and diffusion times. The Fourier amplitude spectrum captures how signal energy is distributed across frequencies, independent of spatial alignment. This enforces a global structural constraint that is invisible to pointwise losses. Importantly, amplitude discrepancies correspond to mismatches in frequency-wise energy allocation rather than local phase misalignment. As a result, amplitude-based regularization directly addresses the incorrect redistribution of reconstruction error across frequency bands.

**Amplitude-and-Phase Loss.** While amplitude matching enforces global spectral alignment, it does not explicitly account for relative scaling across frequencies. To address this, we introduce a second regularizer that incorporates both amplitude magnitude and phase information:

<span id="page-4-0"></span>
$$\mathcal{L}_{F}^{AP} = \mathbb{E}_{x_{0}, t} \left[ \left\| A_{0} - \widehat{A}_{0} \right\|_{1} \left( 1 + \left\| \left( \phi_{0} - \widehat{\phi}_{0} \right) \right\|_{1} \right) \right].$$
(15)

The amplitude–phase (AP) coupling is motivated by the observation that phase information becomes meaningful primarily when associated with non-negligible spectral energy and that simply using the phase information leads to unstable training due to the branch-cuts. Large phase discrepancies in frequency bands with vanishing amplitude are perceptually insignificant, while similar discrepancies in dominant bands correspond to coherent structural distortions. This formulation avoids over-penalizing inconsequential phase noise while stabilizing fine-scale structure.

## 4.2. Wavelet Regularized Diffusion Models

#### 4.2.1. Preliminaries

Let  $x_0 \sim p_{\text{data}}$  denote a data sample, and let  $x_t$  be its noisy counterpart at diffusion time t. We denote by  $\mathcal{W}[x_t] = \left\{W_t^{(s,\ell)}(b)\right\}_{s,\ell}$  the discrete wavelet transform of  $x_t$ , where s indexes scale,  $\ell$  indexes orientation or sub-band, and  $\mathbf{b} \in \mathbb{R}^n$  denotes spatial or temporal location. This decomposition yields a hierarchical set of wavelet coefficients corresponding to different resolutions and directions. Similarly, given a predicted denoised sample  $\widehat{x}_0 = \widehat{x}_{\theta}(x_t, t)$ , we denote its wavelet coefficients by  $\widehat{W}_0^{(s,\ell)}(b)$ .

#### 4.2.2. WAVELET LOSSES

We define wavelet-domain regularization terms that penalize discrepancies between the wavelet coefficients of the predicted clean sample and those of the ground-truth data. These losses encourage agreement across scales and locations, directly targeting multi-resolution structure.

**Wavelet Coefficient Matching Loss.** Our first wavelet loss enforces alignment between wavelet coefficients at all scales and orientations:

$$\mathcal{L}_{W} = \mathbb{E}_{x_{0}, t} \left[ \sum_{s, \ell} \gamma_{s, l} \left\| W_{0}^{(s, \ell)} - \widehat{W}_{0}^{(s, \ell)} \right\|_{1} \right], \quad (16)$$

where the norm is taken over spatial locations b and  $\gamma_{s,l}$  is the weight corresponding to each scale and sub-band. This loss encourages the diffusion model to match localized features such as edges, textures, and transient events at each resolution level.

#### 4.3. Training Objective

The final training objective augments the standard diffusion loss with the proposed Fourier regularization:

$$\mathcal{L}_{\text{total}} = \mathcal{L} + \lambda \, \mathcal{L}_{S}, \tag{17}$$

where  $\mathcal{L}$  is the standard diffusion denoising objective defined by Eq. (3) and  $\mathcal{L}_{\mathrm{S}}$  denotes either  $\mathcal{L}_{\mathrm{F}}^{\mathrm{A}}$ ,  $\mathcal{L}_{\mathrm{F}}^{\mathrm{AS}}$ , or  $\mathcal{L}_{\mathrm{W}}$ . The hyperparameter  $\lambda$  controls the spectral regularization.

This formulation mirrors the structure of constraintaugmented diffusion models while remaining fully datadriven and domain-agnostic. Fourier regularization shapes the learned denoising operator to respect global spectral properties without restricting the generative process to satisfy explicit rules or equations. The Wavelet loss provide complementary global and local spectral biases that improve the fidelity and robustness of generative diffusion models. As a result, the proposed method integrates seamlessly with existing diffusion architectures and sampling procedures.

#### 5. Experiments

## 5.1. Checkerboard Toy Experiment

To isolate the effect of our spectral regularizer, we construct a toy dataset of  $64 \times 64$  grayscale checkerboard images, which concentrate energy at a small set of high spatial frequencies. Such patterns provide a controlled stress test for assessing whether diffusion models preserve dominant periodic structure during generation. We compare a standard DDPM trained with the Mean Squared Error (MSE) noise prediction loss to the same objective augmented with our amplitude-and-phase spectral loss (Eq. 15), keeping architectures and optimization settings fixed.

Figure 1 shows representative generations and Figure 2 shows the radially averaged power spectra. The baseline

model exhibits visible smoothing and spectral leakage, producing attenuated and broadened responses at the checker-board frequencies. In contrast, the spectral regularizer concentrates energy near the correct frequency bands and yields sharper periodic structure, resulting in samples that more closely match the ground-truth spectrum, despite remaining less strictly binary than the target.

#### 5.2. Image Datasets

We study spectral regularization as a lightweight fine-tuning strategy applied to pretrained EDM models (Karras et al., 2022). For each dataset and EDM formulation, models are fine-tuned for 5 optimization steps using the standard EDM denoising objective augmented with a spectral loss. This setup isolates the effect of loss-level spectral biasing without modifying the model architecture, diffusion formulation, or sampler. More training details are in Appendix A.1.

We consider both variance-preserving (VP) and varianceexploding (VE) EDM variants. For each, we evaluate four spectral losses: (i) Fourier amplitude, (ii) Fourier amplitude+phase, (iii) Haar wavelet, and (iv) bi-orthogonal 1.3 (bior13) wavelet regularization. Fourier transforms are computed using PyTorch FFTs, while wavelet transforms are implemented based on PyWavelets (Lee et al., 2019). Experiments are conducted on CIFAR-10 (32  $\times$  32), FFHQ, and AFHQv2 ( $64 \times 64$ ), following the standard EDM evaluation protocol. CIFAR-10 is evaluated under conditional sampling, whereas FFHQ and AFHQv2 are evaluated unconditionally. We consider two choices for the regularization weight. The first being the "weighted" setting where  $\lambda = \lambda_{\rm EDM}$  (see Eq. 7), and the "unweighted" setting where  $\lambda = 1$ . Generative quality is measured using Fréchet Inception Distance (FID), computed between 50,000 generated samples and the full real dataset. Results are averaged over three random seeds. Some sampled images for the FFHQ and AFHQ datasets are shown in Appendix C.

Table 1 reports FID scores for CIFAR-10, AFHQ, and FFHQ under both VE- and VP-EDM formulations. Since all experiments start from strong pretrained EDM baselines, the scope for improvement is necessarily limited. On CIFAR-10 (conditional), spectral losses have negligible effect, with all methods performing within the standard deviation of the EDM baseline, indicating limited benefit when conditional structure is already well captured.

On the higher-resolution AFHQ and FFHQ datasets, we observe small but reliable FID reductions (typically 0.02-0.07) across multiple spectral losses and EDM variants, with no cases of systematic degradation. These gains are comparable in magnitude across AFHQ and FFHQ, indicating that the proposed losses behave similarly on distinct but equally challenging natural image distributions.

<span id="page-6-0"></span>![](_page_6_Picture_1.jpeg)

![](_page_6_Picture_2.jpeg)

![](_page_6_Picture_3.jpeg)

(a) Ground truth (b) Baseline (original MSE loss) (c) Ours (amp+phase loss)

*Figure 1.* Checkerboard toy experiment. Figures (a) to (c) show the ground-truth pattern, a sample from a model trained without spectral regularization, and a sample from a model trained with the proposed amplitude-and-phase loss.

<span id="page-6-1"></span>*Table 1.* Frechet Inception Distance (FID) scores for different spectral regularizers and EDM variants. All models are fine-tuned from ´ pretrained EDM baselines. Lower values indicate better generative performance.

| DATASET | EDM<br>VARIANT | EDM SAMPLING<br>STEPS | CONDITIONING | REGULARIZER                        | WEIGHTED<br>FID                                  | UNWEIGHTED<br>FID                                | EDM<br>FID |
|---------|----------------|-----------------------|--------------|------------------------------------|--------------------------------------------------|--------------------------------------------------|------------|
| CIFAR   | VE             | 18                    | COND         | AMP<br>AMP PHASE<br>HAAR<br>BIOR13 | 1.82±0.01<br>1.82±0.01<br>1.81±0.01<br>1.82±0.01 | 1.82±0.01<br>1.82±0.01<br>1.81±0.02<br>1.81±0.02 | 1.81±0.01  |
| CIFAR   | VP             | 18                    | COND         | AMP<br>AMP PHASE<br>HAAR<br>BIOR13 | 1.84±0.02<br>1.84±0.02<br>1.84±0.02<br>1.83±0.02 | 1.84±0.02<br>1.84±0.02<br>1.84±0.02<br>1.84±0.02 | 1.84±0.02  |
| AFHQ    | VE             | 40                    | UNCOND       | AMP<br>AMP PHASE<br>HAAR<br>BIOR13 | 2.13±0.00<br>2.13±0.01<br>2.14±0.01<br>2.14±0.01 | 2.14±0.00<br>2.16±0.01<br>2.14±0.00<br>2.14±0.01 | 2.17±0.00  |
| AFHQ    | VP             | 40                    | UNCOND       | AMP<br>AMP PHASE<br>HAAR<br>BIOR13 | 2.03±0.02<br>2.04±0.02<br>2.05±0.02<br>2.07±0.02 | 2.05±0.02<br>2.03±0.02<br>2.03±0.03<br>2.05±0.02 | 2.04±0.00  |
| FFHQ    | VE             | 40                    | UNCOND       | AMP<br>AMP PHASE<br>HAAR<br>BIOR13 | 2.5±0.02<br>2.49±0.02<br>2.5±0.02<br>2.51±0.02   | 2.51±0.02<br>2.50±0.02<br>2.49±0.02<br>2.5±0.02  | 2.56±0.03  |
| FFHQ    | VP             | 40                    | UNCOND       | AMP<br>AMP PHASE<br>HAAR<br>BIOR13 | 2.35±0.03<br>2.33±0.03<br>2.33±0.04<br>2.34±0.04 | 2.31±0.03<br>2.34±0.03<br>2.33±0.04<br>2.32±0.04 | 2.38±0.01  |

Importantly, the improvements are achieved with only a handful of fine-tuning steps and without modifying the architecture or sampler, highlighting that spectral regularization acts as a stable and data-efficient bias rather than an aggressive optimization mechanism.

Overall, amplitude-phase regularization is the most consistently competitive method, achieving the best or tied-best performance on FFHQ and remaining close to optimal elsewhere. These results suggest that spectral regularization is most effective in higher-resolution, unconditional settings where diffusion models struggle to capture fine-scale structure, and offer limited benefits when baseline performance is already near saturation.

#### 5.3. Audio Dataset

We additionally evaluate spectral regularization for audio generation by fine-tuning a pretrained DiffWave model using the same loss-level protocol as in our image experiments [\(Kong et al.,](#page-9-3) [2021\)](#page-9-3). Specifically, we optimize the standard DiffWave DDPM denoising objective and augment it with one of our proposed spectral losses, namely Fourier amplitude, Fourier amplitude-phase, and wavelet-based regularizers. We fine-tune the official implementation for 150,000 steps on the LJSpeech-1.1 dataset [\(Ito & Johnson,](#page-8-16) [2017\)](#page-8-16) starting from the publicly released pretrained checkpoint. Additional details are presented in Appendix [A.2.](#page-10-1)

A key implementation detail is that we compute spectral representations using the predicted clean waveform that is

<span id="page-7-1"></span>*Table 2.* Audio generation quality metrics for DiffWave fine-tuning with spectral regularization. FAD measures distributional similarity in audio embedding space (lower is better), UTMOS estimates perceptual naturalness (higher is better), PESQ measures perceptual speech quality (higher is better), MR-STFT measures multi-resolution spectral error (lower is better), and NDB evaluates distributional coverage and mode balance (lower is better). We report the average and standard deviation of five runs with different seeds.

| METHOD    | λ    | FAD ↓       | UTMOS ↑     | PESQ ↑      | MR-STFT ↓     | NDB ↓     |
|-----------|------|-------------|-------------|-------------|---------------|-----------|
| DIFFWAVE  | –    | 1.994±0.008 | 3.941±0.005 | 3.440±0.002 | 1.217±0.001   | 0.63±0.02 |
| AMP       | 10−4 | 1.462±0.006 | 3.953±0.009 | 3.477±0.003 | 1.1802±0.0005 | 0.65±0.03 |
|           | 10−5 | 1.609±0.007 | 3.953±0.009 | 3.476±0.001 | 1.1930±0.0004 | 0.63±0.02 |
|           | 10−6 | 1.775±0.008 | 3.952±0.006 | 3.491±0.003 | 1.1958±0.0005 | 0.64±0.03 |
| AMP+PHASE | 10−4 | 1.694±0.009 | 3.969±0.008 | 3.516±0.002 | 1.1896±0.0003 | 0.66±0.02 |
|           | 10−5 | 1.543±0.012 | 3.988±0.003 | 3.495±0.002 | 1.1773±0.0004 | 0.59±0.01 |
|           | 10−6 | 1.539±0.007 | 3.976±0.008 | 3.344±0.003 | 1.1921±0.0003 | 0.65±0.02 |
| HAAR      | 10−4 | 1.729±0.016 | 3.965±0.006 | 3.466±0.003 | 1.1708±0.0006 | 0.59±0.02 |
|           | 10−5 | 1.992±0.014 | 3.988±0.008 | 3.485±0.003 | 1.2163±0.0002 | 0.66±0.02 |
|           | 10−6 | 2.123±0.010 | 3.923±0.002 | 3.359±0.002 | 1.2343±0.0002 | 0.69±0.03 |
| BIOR      | 10−4 | 1.492±0.011 | 3.977±0.009 | 3.500±0.002 | 1.1768±0.0002 | 0.62±0.01 |
|           | 10−5 | 2.649±0.011 | 3.927±0.007 | 3.303±0.001 | 1.2949±0.0004 | 0.67±0.04 |
|           | 10−6 | 1.520±0.008 | 3.985±0.008 | 3.466±0.003 | 1.1787±0.0003 | 0.64±0.02 |

<span id="page-7-0"></span>![](_page_7_Figure_3.jpeg)

*Figure 2.* Radially averaged power spectra (log scale) for the ground truth, baseline DDPM with MSE loss, and DDPM with our amplitude+phase spectral loss.

obtained from DDIM sampling, rather than directly transforming the noisy input. Specifically, at a randomly sampled diffusion timestep t, we first obtain a denoised estimate x ∗ 0 by running a deterministic DDIM update initialized at x<sup>t</sup> using Eq. [5,](#page-2-2) and then compute Fourier/wavelet transforms of x ∗ 0 for the spectral loss. This follows the same pattern used in Algorithm 1 of [\(Bastek et al.,](#page-8-6) [2025\)](#page-8-6). This choice ensures that spectral supervision is applied to a sample-consistent estimate of the clean signal, aligning the spectral objective with the model's generation pathway.

Table [2](#page-7-1) demonstrates that loss-level spectral regularization consistently improves DiffWave audio generation across perceptual, spectral, and distributional metrics, despite being applied as a lightweight fine-tuning procedure. All spectral losses outperform the DiffWave baseline in FAD and PESQ for certain choice of λ, indicating that explicit spectral-domain biasing effectively corrects spectral mismatches that are weakly constrained by time-domain denoising alone. Fourier amplitude regularization yields the strongest FAD improvements, achieving the best overall score at moderate regularization strength, suggesting that matching global magnitude statistics is sufficient to recover dominant spectral structure that drives perceptual distance. In contrast, the amplitude-phase loss produces the most balanced gains across metrics, attaining the highest UTMOS and PESQ values and the lowest NDB. This shows the benefit of our novel approach of incorporating phase into the loss. Wavelet-based regularization exhibits complementary behavior: Haar wavelets achieve the lowest MR-STFT distance at higher λ, highlighting improved multi-resolution temporal coherence, while biorthogonal wavelets show increased sensitivity to the regularization weight, likely due to their redundant, non-orthogonal structure. Overall, no single spectral loss dominates across all criteria. Thus, spectral regularization acts a controllable inductive bias whose effect depends on both representation choice and loss weighting.

