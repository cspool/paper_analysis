## Fourier Spectral Regularization for Diffusion Training（扩散训练的傅里叶谱正则化）

术语是什么？通过联网搜索让回答具体和精准。

Fourier Spectral Regularization是一种loss-level训练正则化方法，在标准扩散模型denoising objective上附加可微分的Fourier域L1 penalty。与signal-domain L2 loss（如DDPM的MSE noise prediction loss）仅控制error total energy不同，Fourier spectral loss显式约束reconstruction error在frequency bands间的distribution。定义两种Fourier损失：(1) Fourier Amplitude Loss (LA_F)：LA_F = E[|| |F[x₀]| − |F[x̂₀]| ||₁]，仅匹配幅度谱(amplitude spectrum)，对spatial alignment不敏感，直接控制frequency-wise energy allocation mismatch；(2) Fourier Amplitude-and-Phase Loss (LAP_F)：LAP_F = E[||A₀−Â₀||₁·(1+||ϕ₀−ϕ̂₀||₁)]，将phase penalty通过amplitude magnitude加权，避免对low-amplitude band的insignificant phase noise过度penalize，同时稳定high-amplitude band的fine-scale structure。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Fourier Spectral Regularization的training step（以FFHQ + VP-EDM + LA_F为例）：

```
// 输入：x_0 (ground-truth batch), λ (regularization weight)
1: σ ~ p_train(σ), ε ~ N(0,I)
2: x_σ = x_0 + σ·ε                          // EDM forward corruption
3: ε_hat = ε_θ(x_σ, σ)                      // 网络预测噪声
4: x̂_0 = x_σ − σ·ε_hat                      // DDIM一步reconstruction得干净估计
5: 
6: // Standard EDM denoising loss
7: L_EDM = λ_EDM(σ)·||ε − ε_hat||²₂
8:
9: // Fourier Amplitude Loss
10: F_x0 = torch.fft.fft2(x_0)               // 2D FFT of ground-truth
11: F_x̂0 = torch.fft.fft2(x̂_0)              // 2D FFT of prediction
12: A_0 = torch.abs(F_x0)                      // amplitude spectrum
13: Â_0 = torch.abs(F_x̂0)
14: LA_F = ||A_0 − Â_0||₁                     // L1 amplitude discrepancy
15:
16: L_total = L_EDM + λ·LA_F
17: θ ← θ − η·∇_θ L_total
```

关键设计决策：(1) Spectral loss在**predicted clean sample x̂₀**上计算（DDIM一步reconstruction），而非直接在noisy input x_t上计算——确保spectral supervision与model generation pathway对齐。(2) 使用L1而非L2：有意break Parseval invariance。Parseval-Plancherel identity (||x||²₂ = ||X(ω)||²₂)仅对L2成立，对L1不成立——因此L1 amplitude loss可直接控制error的spectral distribution而非仅total energy。(3) Fourier amplitude loss对spatial translation invariant，amplitude spectrum captures frequency-wise energy分配与spatial shift无关，是天然的"global structural constraint"。(4) PyTorch torch.fft.fft2底层使用cuFFT，GPU计算高效，overhead negligible。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现仅需~20行PyTorch代码。在Spectral Regularization论文中：对pretrained EDM checkpoint做5-step lightweight fine-tuning, λ为唯一新增hyperparameter。Checkerboard toy experiment直观验证：baseline MSE model产生attenuated/broadened spectral responses + visible smoothing，spectral regularizer correctly concentrates energy near correct frequency bands。FFHQ/AFHQ上FID改善0.02-0.07（仅5步fine-tuning）。Audio (DiffWave on LJSpeech)上Fourier amplitude regularization yields strongest FAD improvement (1.994→1.462 at λ=10⁻⁴)。

涉及论文标题：
- Spectral Regularization for Diffusion Models

---

