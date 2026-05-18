## Latent Wavelet Diffusion for Ultra-High-Resolution Image Synthesis

- baseline方法是什么？
  Baseline是标准Latent Diffusion Models (LDMs)，如Flux-1.dev（flow-matching backbone）、SD3-F16（MMDiT backbone + 16ch VAE）、PixArt-Sigma-XL（DiT backbone）、Sana-1.6B（linear DiT）、URAE（基于Flux的参数高效适配）。这些模型在训练时对所有spatial regions使用uniform denoising supervision：L_fm = ||(ε-z_0) - v_Θ(z_t,t,y)||²₂，每个空间位置(i,j)在每一timestep接受等量优化信号。这导致两个缺陷：(1) 计算浪费——低细节平滑区域（如天空、纯色背景）被过度监督；(2) 高细节区域监督不足——纹理、边缘、毛发等高频结构得不到足够关注，在UHR（2K-4K）下产生模糊或纹理坍缩。此外，标准VAE在高分辨率下latent representation包含跨尺度不一致的高频伪影，进一步恶化UHR生成质量。

  Baseline全栈执行例子（以Flux-1.dev在A100上生成4K图像为例）：
  - 算法层：Flux flow-matching模型，给定text prompt y，VAE encoder将输入图像x映射到latent z_0∈R^{C×H×W}，前向扩散z_t = (1-t)z_0 + tε，模型预测velocity field v_Θ(z_t,t,y)，loss对所有(H,W)位置等权：每个空间位置在每个timestep接受相同的||(ε-z_0)-v_Θ||²损失信号。平滑区域和高细节区域在训练中获得完全相同频次的refinement。
  - 系统框架层：PyTorch DataParallel/FSDP训练pipeline，HuggingFace Diffusers加载预训练模型，标准训练loop：encode→sample noise→forward→uniform loss→backward→optimizer step。无空间或时间自适应调制。
  - 编译框架层：PyTorch eager mode + torch.compile（可选），使用标准CUDA kernel（FlashAttention for DiT backbone）。无额外编译优化。
  - kernel层：标准PyTorch CUDA kernel——DiT attention用FlashAttention-2，卷积/线性层用cuBLAS。所有操作以完整tensor/layer为粒度，无空间选择性计算。
  - 硬件层：NVIDIA A100 GPU (64GB HBM2e, 108 SM)。训练时GPU memory约57.9-60.1GB（取决于backbone），训练batch=1-8。推理时相同资源消耗。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Latent Wavelet Diffusion (LWD)，通过在训练阶段引入signal-derived frequency saliency来增强latent diffusion model的UHR生成质量。核心insight：图像中不同区域的结构复杂度分布不均匀，但现有denoising model对所有位置等同refine。LWD通过wavelet-derived spatial saliency maps + time-dependent masking实现频率感知的spatially adaptive supervision，两步走：(1) 用scale-consistency loss微调VAE以净化latent空间；(2) 用wavelet-masked flow matching objective微调扩散模型。

  LWD全栈执行例子（以LWD+Flux在A100上生成4K图像为例）：
  - 算法层：Stage 1——Flux-VAE经scale-consistency loss微调：L_VAE = ||D(z)-x||²₂ + α||D(E(z_down))-x_down||²₂ + β D_KL(q||p) + λ L_LPIPS。该loss惩罚跨尺度不一致的高频分量（即伪影），使latent空间的频谱分布对齐clean natural image的RGB频谱。Stage 2——对每个training step，z_t输入Haar DWT→计算HF energy E(c,i,j) = (1/C) Σ_c[(z_LH)²+(z_HL)²+(z_HH)²]→bilinear upsample+min-max normalize得Awavelet→time-dependent binary mask M_t(i,j)=1 if T·(Awavelet+l)≥t else 0→最终L_masked = ||M_t ⊙ [(ε-z_0)-v_Θ(z_t,t,y)]||²₂。高频区域（纹理/边缘/轮廓）在更多timestep收到监督信号，平滑区域在较少timestep收到监督。l=0.3确保所有区域至少30%监督覆盖。Haar wavelet因其最紧凑support（2 coefficients）提供最精确空间定位，避免Daubechies的mask边界模糊和FFT高通的Gibbs ringing伪影。
  - 系统框架层：与baseline相同——PyTorch训练pipeline，额外依赖pytorch-wavelets库（Cotter, 2019）做Haar DWT。Mask计算开销极小（Haar DWT per step），训练memory仅增~3%（Sana 90.5%→93.9%），每step时间几乎不变。完全不需要推理期修改——训练好的LWD模型权重可直接替换baseline checkpoint，推理pipeline不变。
  - 编译框架层：论文未明确说明（与baseline相同，PyTorch eager mode）。DWT通过pytorch-wavelets的CUDA kernel实现，mask生成与loss modulation为纯PyTorch tensor操作。
  - kernel层：论文未明确说明（与baseline相同，标准PyTorch CUDA kernel）。Wavelet DWT通过pytorch-wavelets库，基于标准卷积操作实现。
  - 硬件层：NVIDIA A100 GPU (4×A100, 64GB each)。训练period不变，推理period不变。LWD+URAE 4K训练约24h (batch=1)，LWD+Diff4K 2K约48h (batch=8)。推理期与baseline identical——相同参数量、相同inference time、相同GPU memory。

  **缺陷1：Uniform supervision——所有空间位置接受相同频次refinement，低细节区域浪费计算，高细节区域监督不足**
  → LWD方案：wavelet energy map Awavelet捕捉每个空间位置的局部high-frequency能量。time-dependent masking M_t(i,j)使高Awavelet区域在更多timestep参与loss计算（mask=1更长时间），低Awavelet区域仅在l·T个timestep强制参与（l=0.3）。这本质上是spatial curriculum learning——模型在训练早期关注所有region建立全局结构，后期将capacity集中在细节丰富区域。GLCM score从baseline 0.79提升到0.74（更接近真实分布），表明masking改善了而非简单增加高频纹理。

  **缺陷2：标准VAE latent空间包含跨尺度不一致的高频伪影，污染wavelet-based saliency的准确性**
  → LWD方案：scale-consistency loss在VAE微调中引入多尺度reconstruction约束——对原图x和其降采样版本x_down分别reconstruct，强制encoder在跨尺度下保持结构一致性。这抑制了"spurious high-frequency noise"（跨尺度不一致的伪影），使后续DWT提取的HF energy对应真实结构而非噪声。消融证明：仅VAE-SC (+2.5% Aesthetics） 和仅Wavelet Masking (+2.3% FID) 各自有效，但组合 (Full LWD) 效果最优（+4.1% CLIPScore, +3.5% Aesthetics），验证了两阶段协同的必要性。

  **缺陷3：UHR生成中纹理坍缩——现有方法（Diffusion-4K wave loss等）虽引入频率约束但uniform施加，不区分空间区域**
  → LWD方案：相较于Diffusion-4K将wavelet loss作为uniform空间上的passive frequency signal，LWD将频率能量转化为active spatial condition——直接在flow-matching loss上施加spatially adaptive binary masking，使模型"知道哪里需要更多学习"。在4K Aesthetic-Eval上LWD+SD3-F16在FID/CLIPScore/Aesthetics三项指标和GLCM上全面持平或超越SD3-Diff4k-F16，且不需Diff-4K的额外wave loss计算开销。LWD+URAE在4K HPD上MAN-IQA 0.4011（最高）、GLCM 0.74（最高）。

  **缺陷4：UHR训练资源消耗巨大——full UHR fine-tuning需大量GPU内存和长时间训练，训练数据稀缺**
  → LWD方案：通过加速收敛缓解——仅需baseline原始训练iteration的10-50%即达收敛（如LWD+URAE仅2k steps vs baseline建议值远超此数），大幅降低训练成本。LWD在50K 2K + 20K 4K LAION子集上即可有效训练，不依赖专有UHR数据集。训练期memory overhead仅~3%（额外存储DWT中间tensor，尺寸为latent map级别）。

  **关键trade-off**：LWD以训练时marginal overhead（每step DWT+mask计算，约3% memory）换取UHR生成质量significant improvement，且零推理开销。但继承LDM通用限制——VAE compression导致fine-grained semantic detail的丢失可能限制需要精确spatial alignment的任务（论文建议未来工作可探索更高保真度的latent space或latent+pixel联合监督）。Haar wavelet对sharp edge/discontinuity定位好，但可能对gradual texture transition不敏感。GLCM score在full LWD下轻微下降（0.79→0.74）反映perceptual realism对raw texture complexity的trade-off——论文认为这是有意义的trade-off，经FID/Aesthetics/HPS等感知指标验证。
