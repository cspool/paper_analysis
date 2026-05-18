## Scale-Consistency VAE Objective（尺度一致性VAE目标函数）

术语是什么？通过联网搜索让回答具体和精准。
Scale-Consistency VAE Loss是LWD论文stage 1中微调预训练VAE的目标函数，旨在提升latent space在高分辨率下的spectral fidelity和cross-scale coherence。传统VAE训练仅最小化reconstruction loss + KL divergence，对cross-scale frequency consistency无约束，导致UHR下latent representation包含尺度间不一致的高频伪影。Scale-consistency loss引入多尺度强约束：对原图x和其降采样版本x_down同时做encode-decode，要求两者reconstruction质量一致。完整loss：L_VAE = ||D(z)-x||²₂ + α||D(E(z_down))-x_down||²₂ + β D_KL(q(z|x)||p(z)) + λ L_LPIPS(D(z),x)，其中α=0.25, β=0.001, λ=0.05。该loss最早由Skorokhodov et al. (2025)和Kouzelis et al. (2025)为通用reconstruction提出，LWD将其识别为wavelet-guided UHR synthesis的关键前提。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Scale-Consistency VAE训练流程（以Flux-VAE fine-tuning为例）：
```
// 训练一个batch
1: x = batch_of_high_res_images        // 如1024×1024, 2048×2048
2: x_down = downsample(x, scale=2)     // 半分辨率版本（如512×512, 1024×1024）

3: z = E(x)                            // latent of full-res image
4: z_down = E(x_down)                  // latent of downsampled image

5: // 四项loss
6: L_recon = MSE(D(z), x)              // 标准reconstruction
7: L_scale  = MSE(D(downsample(z)), x_down)  // scale-consistency: 对降采样z做decode后与降采样x比较
   // 或等价：L_scale = MSE(D(E(x_down)), x_down)
8: L_kl = KL(q(z|x) || p(z))           // latent regularization
9: L_percep = LPIPS(D(z), x)           // perceptual loss (AlexNet features)

10: L_total = L_recon + 0.25*L_scale + 0.001*L_kl + 0.05*L_percep
11: optimizer.step(L_total)
```

Scale-consistency的核心机制：
- 对z_down（低分辨率latent）decode后与x_down（低分辨率原图）比较→强制encoder在不同分辨率下产生结构一致的latent
- 惩罚"跨尺度不一致的高频分量"→即那些在full-res中存在但在downsampled version中不应出现（或pattern不同）的高频信号
- 效果：将latent frequency spectrum对齐到clean natural image的DCT spectrum（Figure 3）→抑制spurious high-frequency noise

VAE reconstruction metrics（Table 3, 论文Appendix B）：
```
Flux-VAE:       rFID=0.73, LPIPS=0.07, PSNR=27.18, SSIM=0.89
Flux-VAE-SC:    rFID=0.50, LPIPS=0.06, PSNR=28.14, SSIM=0.90  (+SC improvement)

SD3-VAE-F16:    rFID=0.70, LPIPS=0.30, PSNR=19.82, SSIM=0.63
SD3-VAE-F16-SC: rFID=0.70, LPIPS=0.18, PSNR=22.58, SSIM=0.75  (LPIPS大降)
```

Scale-consistency对SD3-VAE-F16（aggressive f=16压缩）的LPIPS改进特别显著（0.30→0.18），说明高度压缩的VAE更容易产生cross-scale artifacts。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在LWD中，Scale-Consistency VAE fine-tuning是stage 1（独立于stage 2的扩散模型微调）。两个stage解耦是设计的核心：先净化latent space使其呈现良好频率特性→再用净化后的latent指导扩散模型训练（wavelet masking）。未经SC调优的VAE中high-frequency energy多对应spurious noise→wavelet mask会指向噪声而非真实结构→"frequency-guided supervision"失效。这就是LWD论文中"Synergy of Frequency Suppression and Utilization"分析的核心：VAE loss压制跨尺度不一致的高频伪影→剩余的高频能量与视觉salient features（edges/textures）相关性更强→增强wavelet attention机制的signal-to-noise ratio。

涉及论文标题：
- Latent Wavelet Diffusion for Ultra-High-Resolution Image Synthesis

---

