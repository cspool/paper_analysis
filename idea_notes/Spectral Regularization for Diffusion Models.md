## Spectral Regularization for Diffusion Models

- baseline方法是什么？
  Baseline是standard diffusion model training，使用pointwise signal-domain reconstruction objective（DDPM的L_DDPM = E[||ε−ε_θ(x_t,t)||²₂]，EDM的L_EDM = E[λ(σ)||ε−ε_θ(x_σ,σ)||²₂]）。这些objective均在signal domain（pixel/waveform space）以MSE/L2形式定义reconstruction error，不显式约束error在frequency bands或spatial scales间的分布。核心问题：L2 objective只控制total spectral energy of error（Parseval identity: ||x||²₂ = ||X(ω)||²₂恒成立），但对error的spectral distribution完全agnostic——small overall loss仍可对应disproportionate high-frequency errors。

  全栈执行例子（以FFHQ 64×64 unconditional + VP-EDM + A6000 GPU为例）：
  - 算法层：Standard EDM training/sampling。模型NCSN++/DDPM++接收noisy image x_σ = x₀ + σε→预测denoised ε_θ(x_σ,σ)→L2 loss对所有frequencies赋予equal weight。High-frequency components在低noise regime（σ小）才被学习，但此时effective regularization更弱、sample更少→HF errors更易overfit。结果：生成的FFHQ样本可能出现over-smoothing、incorrect frequency balance、degraded fine-scale texture。
  - 系统框架/Serving层：论文未明确说明（research training setting，非production serving）。
  - 编译框架层：论文未明确说明（标准PyTorch + CUDA训练路径）。
  - kernel调度层：论文未明确说明（标准PyTorch FFT via cuFFT backend）。
  - 硬件架构层：NVIDIA A4000/A6000 GPU，无定制硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出loss-level spectral regularization框架，在standard denoising objective上augment differentiable Fourier-domain和wavelet-domain L1 penalty，不改动diffusion process、architecture或sampler。这是一种"soft inductive bias"——通过训练objective而非硬约束来引入spectral awareness。

  **缺陷1：Standard L2 denoising loss对所有frequencies赋予equal weight，无法区分低频和高频的reconstruction quality**
  → Fourier Amplitude Loss (LA_F)：在predicted clean sample x̂₀和ground-truth x₀间计算Fourier amplitude spectrum的L1差异。L1 intentionally breaks Parseval invariance——Parseval identity（||x||²₂=||X(ω)||²₂）仅对L2成立，对L1不成立，因此L1 amplitude loss可直接控制error在frequency bands间的distribution，而非仅控制total energy。Checkerboard toy experiment（64×64 grayscale, dominant high spatial frequencies）验证：baseline MSE model产生attenuated/broadened spectral responses和visible smoothing，而spectral regularizer correctly concentrates energy near correct frequency bands。

  **缺陷2：Fourier spectrum捕获global frequency structure，但缺乏spatial/temporal localization——对non-stationary signals（textured images、audio transients、edges）的local spectral structure不敏感**
  → Wavelet Coefficient Matching Loss (LW)：对Haar和bior1.3两种wavelet，在multi-scale decomposition的所有scales和orientations上计算wavelet coefficient L1 difference。Wavelet提供localized、scale-aware control：Haar强调sharp discontinuities和edge-like features，bior1.3由于higher-order vanishing moments提供smoother multi-scale consistency。Audio实验验证：Haar wavelets achieve lowest MR-STFT distance（improved multi-resolution temporal coherence），而不同wavelet type对应不同的sharpness-vs-smoothness trade-off。

  **缺陷3：Phase information对perceptual quality critical，但直接使用phase difference会导致training instability（branch-cuts、low-amplitude band noise）**
  → Amplitude-Phase Coupled Loss (LAP_F)：phase penalty通过amplitude weighting引入：LAP_F = E[||A₀−Â₀||₁·(1+||ϕ₀−ϕ̂₀||₁)]。关键设计：large phase discrepancies in bands with vanishing amplitude are perceptually insignificant（被amplitude因子抑制），而similar discrepancies in dominant bands correspond to coherent structural distortions（被amplitude因子放大）。Audio实验验证：Amplitude-phase loss produces most balanced gains——达highest UTMOS和PESQ、lowest NDB（mode coverage best），说明phase coupling稳定fine-scale structure且不引入instability。

  **缺陷4：修改diffusion process或architecture的spectral方法（如wavelet-domain diffusion、frequency-based noise control）需要specialized implementation，与现有DDPM/DDIM/EDM pipeline不兼容**
  → Loss-level-only design：所有spectral regularization仅在training objective层面添加auxiliary term，不需modified forward process、basis-specific parameterization或architecture change。与DDPM/DDIM/EDM fully compatible。Image实验仅需5 optimization steps fine-tuning（非full retraining），证明spectral bias是data-efficient的——在strong pretrained EDM baseline上仍获0.02-0.07 FID improvement，尤其在高分辨率unconditional setting（FFHQ/AFHQ）收益最大。

  **缺陷5：Time-domain waveform loss in audio diffusion难以capture perceptually important spectral structure**
  → Audio实验：DiffWave在LJSpeech上fine-tune 150K steps with spectral losses。Fourier amplitude regularization yields strongest FAD improvement（1.994→1.462 at λ=10⁻⁴），证明matching global magnitude statistics足以恢复驱动perceptual distance的dominant spectral structure。所有spectral losses在FAD和PESQ上均outperform DiffWave baseline，指示explicit spectral-domain biasing有效correct weakly-constrained spectral mismatches。

  论文方法全栈执行例子（以FFHQ 64×64 unconditional + VP-EDM + Fourier amplitude loss fine-tuning + A6000为例）：
  - 算法层：加载pretrained VP-EDM checkpoint→仅5 step fine-tuning。每step：采样timestep t→forward process得x_t→DDIM一步得x̂₀→对x̂₀和x₀做PyTorch FFT→计算amplitude L1 loss→总loss L_total = L_EDM + λ·LAF→backprop update。采样阶段与standard EDM完全相同，无extra compute。
  - 系统框架/Serving层：论文未明确说明（属于training-time modification, sampling time unchanged）。
  - 编译框架层：论文未明确说明（标准PyTorch FFT + PyWavelets库，training-time auxiliary loss仅增加可忽略的computational overhead）。
  - kernel调度层：论文未明确说明（PyTorch FFT底层使用cuFFT）。
  - 硬件架构层：NVIDIA A4000/A6000 GPU，无定制硬件。论文强调compute overhead "negligible"——FFT和DWT在GPU上高度优化。
