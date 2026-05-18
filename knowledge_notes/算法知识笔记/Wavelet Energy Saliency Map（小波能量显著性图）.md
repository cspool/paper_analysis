## Wavelet Energy Saliency Map（小波能量显著性图）

术语是什么？通过联网搜索让回答具体和精准。
Wavelet Energy Saliency Map是LWD论文提出的、基于Discrete Wavelet Transform (DWT)从latent representation中计算的空间显著性度量。给定一个latent tensor z∈R^{C×H×W}，应用单层DWT分解为四个子带z_LL（低频近似）、z_LH（水平细节）、z_HL（垂直细节）、z_HH（对角细节），计算每个空间位置(i,j)在三个高频子带上的channel-pooled能量：E(i,j) = (1/C) Σ_c[(z_LH^{c,i,j})² + (z_HL^{c,i,j})² + (z_HH^{c,i,j})²]，然后bilinear upsampling + per-sample min-max normalization得到Awavelet∈[0,1]^{H×W}。该图突出latent space中与high-frequency content（纹理、边缘、轮廓）关联的区域，作为频率感知的spatial saliency proxy。不同于基于learned attention的saliency（如DINO），wavelet saliency是deterministic、无训练、直接从信号属性导出的。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Wavelet Energy Saliency Map计算流程：
```
// 输入：latent tensor z_t ∈ R^{C×H×W}
1: z_LL, z_LH, z_HL, z_HH = DWT_2D(z_t, wavelet='haar')
   // 每个子带 ∈ R^{C×H/2×W/2}

2: // 计算高频能量图（Eq.3）
   E_hf = zeros(H/2, W/2)
   for i in 0..H/2-1, j in 0..W/2-1:
       for c in 0..C-1:
           E_hf[i,j] += z_LH[c,i,j]² + z_HL[c,i,j]² + z_HH[c,i,j]²
       E_hf[i,j] /= C                    // channel平均

3: // 上采样 + 归一化
   E_full = bilinear_upsample(E_hf, scale=2)   // 匹配原latent分辨率H×W
   A_wavelet = (E_full - min(E_full)) / (max(E_full) - min(E_full) + ε)
   // A_wavelet ∈ [0,1]^{H×W}
```

关键设计选择：
- **仅用HF子带（LH, HL, HH）**：LL子带编码coarse spatial content，包含局部复杂度的信息极少。类似Sobel/Laplacian边缘检测中gradient magnitude只反映high-frequency transitions。
- **Haar Wavelet选择**：最紧凑support（2 coefficients）→最精确的空间定位、最小cross-position interference。Daubechies wavelet (db2)的wider receptive field在mask边界产生"gray area"→dilute supervision；FFT High-Pass虽计算快但sacrifice spatial localization→Gibbs ringing artifacts污染mask边界（GLCM 0.71 vs Haar 0.74）。
- **无需训练**：wavelet saliency完全基于信号属性计算，不依赖learned attention，零额外参数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
pytorch-wavelets库（Cotter, 2019）提供PyTorch原生的2D DWT实现，支持Haar/Daubechies等多种wavelet basis。在LWD中使用：每个training step对当前latent z_t做DWT→计算HF energy→生成mask→modulate loss。关键前提：VAE latent space需预先通过scale-consistency loss净化——未经regularization的标准VAE latent中"high-frequency energy"多对应spurious artifacts而非真实结构，使wavelet saliency失效。LWD论文的VAE fine-tuning stage (stage 1)通过抑制cross-scale inconsistent高频伪影将latent spectral distribution对齐到clean RGB reference，使得wavelet energy map有意义。该技术也可泛化到其他需要信号驱动的spatial adaptive supervision的场景（如video generation中的temporal attention guidance、depth-aware synthesis）。

涉及论文标题：
- Latent Wavelet Diffusion for Ultra-High-Resolution Image Synthesis

---

