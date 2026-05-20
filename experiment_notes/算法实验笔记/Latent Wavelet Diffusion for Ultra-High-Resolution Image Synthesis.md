## Latent Wavelet Diffusion for Ultra-High-Resolution Image Synthesis

- 属于算法pipeline的实现是什么？实验比较什么？
  属于算法pipeline的实现是Latent Wavelet Diffusion (LWD)，一个纯训练阶段的频率感知框架，不做任何架构修改、推理期零额外开销。包含两个阶段：(1) 用Scale-Consistency (SC) loss微调VAE，抑制跨尺度不一致的高频伪影，使latent空间更适合下游wavelet分解；(2) 用wavelet-masked flow matching objective微调扩散模型——对latent z_t做单层Haar DWT得到LL/LH/HL/HH四个子带，计算高频能量图Awavelet∈[0,1]^{H×W}作为空间显著性度量，再用time-dependent binary mask Mt(i,j) = 1 if T·(Awavelet+l)≥t else 0来调制训练loss，使得高频区域（纹理/边缘）在更多timestep接受监督，平滑区域接受较少监督。l=0.3为消融选出的下界。实验比较baseline：Flux-1.dev、SD3-F16、SD3-Diff4k-F16、PixArt-Sigma-XL、Sana-1.6B、URAE，以及外部baseline SDEdit、I-Max、Diffusion-4K、Lumina-Image 2.0。指标包括FID、LPIPS、MAN-IQA、QualiCLIP、HPSv2.1、PickScore、CLIPScore、Aesthetics、GLCM Score、Compression Ratio、频率域指标HLFR/RDR/WQS/HFE/HFEI。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 GPU（4×A100, 64GB each）。VAE fine-tuning: 60K steps, batch size=4, lr=1×10^{-5}。各backbone训练：LWD+URAE(Flux) 2K约4h/4K约24h, batch size=1; LWD+Diff4K(SD3) 2K约48h, batch size=8; LWD+SANA 2K/4K约24h, batch size=2/1; LWD+PixArt-Σ 2K约24h, batch size=2。训练期peak memory增加~3-4%（如Sana从90.5%到93.9% A100 memory），每20step时间几乎不变（~47s），推理期零开销。

- 模型是什么。数据集和bench分别是什么。
  模型：Flux（flow-matching backbone）、SD3（MMDiT backbone, 16ch VAE）、PixArt-Sigma-XL、Sana-1.6B（linear DiT）、URAE（基于Flux-1.dev的参数高效适配）。数据集：Aesthetic-4K（策展4K benchmark, GPT-4o captions）、LAION-High-Res（50K 2K + 20K 4K image-caption pairs, LAION-5B子集）、HPD prompts（Wu et al., 2023）。评估bench：Aesthetic-Eval、HPD prompt dataset。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源地址：https://github.com/LuigiSigillo/LatentWaveletDiffusion。使用PyTorch + pytorch-wavelets（Cotter, 2019）实现Haar DWT。LWD算法pipeline：
  1. Stage 1 — VAE微调：用scale-consistency loss L_VAE = ||D(z)-x||²₂ + α||D(E(z_down))-x_down||²₂ + β D_KL(q(z|x)||p(z)) + λ L_LPIPS(D(z),x) 微调预训练VAE（Flux-VAE/SD3-VAE/Sana-AE），权重α=0.25, β=0.001, λ=0.05。该阶段被证明是wavelet masking有效的前提——抑制跨尺度不一致的高频噪声，使后续DWT提取的高频能量对应有意义结构而非伪影。
  2. Stage 2 — Wavelet-masked flow matching：对每个training step的latent z_t = (1-t)z_0 + tε，执行单层Haar DWT→计算HF energy map E(i,j) = (1/C) Σ_c [(z_LH)²+(z_HL)²+(z_HH)²]→bilinear上采样+min-max归一化得Awavelet→按Mt(i,j)=1 if T·(Awavelet+l)≥t else 0生成binary mask→计算masked loss L_masked = ||M_t ⊙ [(ε-z_0) - v_Θ(z_t,t,y)]||²₂。Haar wavelet选型的原因：最紧凑support（2 coefficients）提供最精确空间定位，计算效率最高，避免FFT的Gibbs ringing导致的mask边界模糊（Haar GLCM 0.74 vs FFT 0.71）。
  3. 推理期：与baseline完全相同——LWD仅在训练期修改objective，不改变模型参数结构，推理zero overhead。LWD模型与baseline参数量相同、inference time相同。
  4. 收敛加速：LWD仅需baseline论文建议原始训练iteration的10-50%即达收敛。LWD+URAE仅需2k steps（约4h 2K/24h 4K），而Diff4K需要10k steps（约48h）。
