## Wavelet Coefficient Matching Loss for Diffusion Training（扩散训练的小波系数匹配损失）

术语是什么？通过联网搜索让回答具体和精准。

Wavelet Coefficient Matching Loss (LW) 是一种基于离散小波变换(DWT)的loss-level训练正则化方法，在predicted clean sample x̂₀和ground-truth x₀的所有尺度(scales)和方向(orientations)上计算小波系数的L1差异：LW = E[Σ_{s,ℓ} γ_{s,l}·||W₀^{(s,ℓ)} − Ŵ₀^{(s,ℓ)}||₁]，其中s索引尺度，ℓ索引方向/子带(LL/LH/HL/HH)，γ_{s,l}为各尺度/子带的权重。与Fourier spectral loss（基于全局正弦基函数，提供uniform frequency constraint）不同，wavelet coefficients提供localized、scale-aware的representation——每个coefficient同时编码spatial location和frequency content。这使wavelet loss能explicitly target localized oscillations、edges、textures和transient features。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Wavelet Coefficient Matching Loss计算流程（以FFHQ + VP-EDM + Haar wavelet loss为例）：

```
// 输入：x_0 (ground-truth batch), λ, wavelet='haar', level=J
1: σ ~ p_train(σ), ε ~ N(0,I)
2: x_σ = x_0 + σ·ε
3: x̂_0 = x_σ − σ·ε_θ(x_σ, σ)              // DDIM一步干净估计
4:
5: L_EDM = λ_EDM(σ)·||ε − ε_θ(x_σ,σ)||²₂   // 标准EDM loss
6:
7: // Wavelet Coefficient Matching Loss
8: coeffs_x0 = DWT_2D(x_0, wavelet='haar', level=J)
   // coeffs: [(LL_J, (LH_J, HL_J, HH_J)), ..., (LH_1, HL_1, HH_1)]
9: coeffs_x̂0 = DWT_2D(x̂_0, wavelet='haar', level=J)
10: L_W = 0
11: for each scale s, orientation ℓ:
12:     L_W += γ_{s,ℓ} · ||coeffs_x0[s][ℓ] − coeffs_x̂0[s][ℓ]||₁
13:
14: L_total = L_EDM + λ·L_W
15: θ ← θ − η·∇_θ L_total
```

两种小波基对比：(1) Haar wavelet（最简正交小波）：support长度=2（等价于local average vs difference），强调sharp discontinuities和edge-like features，对piecewise-constant structure敏感，但limited smoothness导致对smooth spectral behavior的approximation较粗略。(2) Biorthogonal 1.3 (bior1.3)：非对称双正交小波，analysis wavelet有1个vanishing moment、synthesis wavelet有3个vanishing moment，提供smoother multi-scale separation，high-frequency coefficients捕获larger spatial neighborhoods上的oscillatory behavior。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

PyWavelets (Lee et al., 2019, https://pywavelets.readthedocs.io) 提供Python原生DWT实现，支持Haar/Daubechies/Biorthogonal等wavelet families。在Spectral Regularization论文中，wavelet loss作为training-time auxiliary penalty：图像实验5步fine-tuning，音频(DiffWave on LJSpeech) 150K步fine-tuning。音频实验上Haar wavelet在较高λ下achieve lowest MR-STFT distance (improved multi-resolution temporal coherence)，bior1.3 shows increased sensitivity to λ due to redundant non-orthogonal structure。

与Wavelet Energy Saliency Map (LWD论文)的区别：Wavelet Energy Saliency Map用DWT从latent计算spatial energy map→生成binary mask→modulate diffusion loss，是"where to supervise"的spatial selection机制。Wavelet Coefficient Matching Loss直接比较wavelet coefficients作为loss term，是"what to supervise"的frequency-domain constraint，不涉及masking或spatial selection。

涉及论文标题：
- Spectral Regularization for Diffusion Models

---

