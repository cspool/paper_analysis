## Spectral Regularization for Diffusion Models

- 属于算法pipeline的实现是什么？实验比较什么？
  提出loss-level spectral regularization框架，在标准diffusion training objective（DDPM/DDIM/EDM）基础上augment可微分的Fourier-domain和wavelet-domain L1 penalty terms，不改动diffusion process、模型架构或sampling procedure。核心设计：(1) Fourier Amplitude Loss (LA_F)：在predicted clean sample x̂₀与ground-truth x₀之间计算Fourier amplitude spectrum的L1差异——amplitude discrepancy对应frequency-wise energy allocation mismatch而非local phase misalignment，直接控制reconstruction error在frequency bands间的分布。(2) Fourier Amplitude-and-Phase Loss (LAP_F)：将phase信息通过amplitude coupling引入：LAP_F = E[||A₀−Â₀||₁·(1+||ϕ₀−ϕ̂₀||₁)]，利用amplitude加权phase penalty，避免对low-amplitude band的insignificant phase noise过度penalize，同时稳定fine-scale structure。(3) Wavelet Coefficient Matching Loss (LW)：对Haar和bior1.3两种wavelet，在predicted和ground-truth sample的所有scales和orientations上计算wavelet coefficient的L1差异：LW = E[Σ_{s,ℓ} γ_{s,l}||W₀^{(s,ℓ)}−Ŵ₀^{(s,ℓ)}||₁]，提供localized、scale-aware的multi-resolution control。(4) 最终objective：L_total = L + λ L_S，L为standard denoising loss，λ控制regularization强度。所有spectral loss使用L1而非L2（有意break Parseval invariance以直接控制error的spectral distribution）。Fourier transforms用PyTorch FFT实现，wavelet transforms用PyWavelets。训练方式为lightweight fine-tuning：图像仅需5 optimization steps from pretrained EDM checkpoint，音频需150K steps。实验比较：图像在CIFAR-10/FFHQ/AFHQ上对比EDM baseline (VE/VP variants)，度量FID；音频在LJSpeech上对比DiffWave baseline，度量FAD/UTMOS/PESQ/MR-STFT/NDB。还包含checkerboard toy experiment验证spectral regularizer对高频periodic structure的preservation能力。

- 硬件平台是什么，配置是什么。
  NVIDIA A4000和A6000 GPU。Per-GPU batch size=16。EDM fine-tuning duration=0.5（CIFAR/AFHQ）、learning rate=2×10⁻⁴ (CIFAR)/5×10⁻⁵ (AFHQ/FFHQ)。DiffWave fine-tuning：sample rate=22050 Hz, nmels=80, nfft=1024, hop=256 samples, learning rate=2×10⁻⁴, batch size=16, 150K steps。

- 模型是什么。数据集和bench分别是什么。
  图像：EDM (Karras et al., 2022)，DDPM++ (VP) 和 NCSN++ (VE) variants，pretrained on CIFAR-10 32×32 (conditional+unconditional)、FFHQ 64×64 (unconditional)、AFHQv2 64×64 (unconditional)。预训练权重从NVIDIA官方release获取。采样步数：CIFAR 18 steps, AFHQ/FFHQ 40 steps。FID on 50K generated samples vs full real dataset，3 random seeds average。音频：DiffWave (Kong et al., 2021)，30 residual layers, 64 residual channels, dilation cycle length=10, conditional training。LJSpeech-1.1 dataset (13,100 utterances, single speaker)，排除LJ001*/LJ002*用于evaluation。Inference noise schedule=[10⁻⁴,10⁻³,10⁻²,0.05,0.2,0.5]。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源：https://anonymous.4open.science/r/fourierdm-8B8E。算法pipeline（以FFHQ 64×64 unconditional + VP-EDM + Fourier amplitude loss fine-tuning为例）：
  1. 加载pretrained EDM VP checkpoint（NVIDIA官方release的edm-ffhq-64x64-uncond-vp.pkl）。
  2. Fine-tuning loop（仅5 optimization steps）：对每个batch采样diffusion timestep t→加噪得x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε→用DDIM一步reconstruction得x̂₀（Eq.5: x̂₀ = (x_t − √(1−ᾱ_t)·ε_θ(x_t,t)) / √ᾱ_t）→对x̂₀和x₀分别计算Fourier transform（PyTorch FFT）→计算amplitude spectrum L1 loss LAF = |||F[x₀]| − |F[x̂₀]|||₁→总loss L_total = L_EDM + λ·LAF→反向传播更新模型参数。
  3. 关键实现细节：spectral loss在predicted clean waveform x̂₀上计算（通过DDIM一步得到），而非直接在noisy input x_t上做transform。这确保spectral supervision作用在sample-consistent estimate of clean signal上，与model generation pathway对齐。
  4. 采样：与standard EDM完全一致，无额外overhead——spectral regularization仅在训练时作为auxiliary loss生效，不改动sampler或architecture。
  5. 对比baseline（无spectral loss的标准EDM fine-tuning），仅5步fine-tuning即可在FFHQ上获得0.02-0.07的FID improvement，证明spectral bias作为data-efficient inductive prior的有效性。
