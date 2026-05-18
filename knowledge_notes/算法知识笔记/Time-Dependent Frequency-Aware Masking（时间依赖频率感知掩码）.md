## Time-Dependent Frequency-Aware Masking（时间依赖频率感知掩码）

术语是什么？通过联网搜索让回答具体和精准。
Time-Dependent Frequency-Aware Masking是LWD论文的核心训练机制，将wavelet-based spatial saliency转化为时变二元mask来调制扩散模型训练loss。具体地，给定wavelet saliency map Awavelet∈[0,1]^{H×W}和当前的扩散timestep t，对每个spatial位置(i,j)生成binary mask：M_t(i,j) = 1 if T·(Awavelet(i,j)+ℓ) ≥ t else 0，其中T为总timestep数，ℓ∈(0,1)为lower bound（论文设ℓ=0.3）。物理含义：高saliency区域（Awavelet大）的mask=1的timestep范围更广——这些区域在整个扩散过程中被监督更久；低saliency区域仅在t ≤ T·ℓ的早期timestep被监督（至少30%的基础监督）。最终loss：L_masked = ||M_t ⊙ [(ε-z_0) - v_Θ(z_t,t,y)]||²₂。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Time-Dependent Frequency-Aware Masking完整算法：
```
// 输入：latent z_0, noise ε, model v_Θ
// 参数：T（总timesteps），ℓ（lower bound=0.3）

// 扩散混合
1: t ~ Uniform(0, T]                          // 或从noise schedule采样
2: z_t = (1 - t/T) * z_0 + (t/T) * ε          // Flow matching插值

// Wavelet Saliency计算（见Wavelet Energy Saliency Map）
3: z_LL, z_LH, z_HL, z_HH = DWT_2D(z_t, 'haar')
4: A_wavelet = normalize(upsample(LH²+HL²+HH², channel_mean))

// Time-Dependent Mask生成（Eq.6）
5: M_t = zeros_like(z_t[:,0,:,:])              // binary mask
6: for each spatial position (i, j):
7:     if T * (A_wavelet[i,j] + ℓ) >= t:      // 当前timestep t - high saliency or early step
8:         M_t[i,j] = 1                       // 参与loss
9:     else:
10:        M_t[i,j] = 0                        // 跳过（不贡献梯度）

// Masked Loss（Eq.7）
11: target = ε - z_0
12: pred = v_Θ(z_t, t/T, text_conditioning)
13: diff = (target - pred)²                    // per-element squared error
14: loss = mean(M_t ⊙ diff)                    // 仅mask=1位置贡献loss
```

Lower bound ℓ的消融（Table 6, 论文Appendix A）：
```
ℓ=0.0: FID=34.15, GLCM=0.68  ← 平滑区域欠训练
ℓ=0.1: FID=33.21, GLCM=0.72
ℓ=0.3: FID=32.88, GLCM=0.74  ← 最优trade-off
ℓ=0.5: FID=33.46, GLCM=0.71
ℓ=0.7: FID=34.02, GLCM=0.69  ← 退化为接近uniform loss
```

策略的物理意义：
- **高Awavelet位置**（纹理/边缘）：T·(Awavelet+ℓ)大→覆盖更多timestep→被监督更久→detail refinement更充分
- **低Awavelet位置**（平滑区域）：仅在early timestep (t≤T·ℓ) 被监督→只保证基本结构建立
- **空间curriculum learning**：类似课程学习——先全部区域建立全局结构（early steps），后聚焦detail-rich regions精细雕刻（later steps）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
该masking机制是纯training-time操作，推理期完全不需要——训练好的LWD模型与baseline model参数量相同、inference pipeline不变。训练期overhead：每step需做一次Haar DWT（通过pytorch-wavelets的CUDA kernel）生成mask，额外memory约3%（存储DWT中间tensor + mask，均为latent map量级，比diffusion backbone参数小3个数量级）。论文设计确保masking仅作用在objective level，不与任何特定模型架构耦合——可应用于任意使用flow-based或score-based trajectory的latent diffusion model。masking策略的通用性暗示可扩展到video generation (temporal attention)，depth-aware synthesis (depth-aligned masking)，multimodal conditioning等场景。

涉及论文标题：
- Latent Wavelet Diffusion for Ultra-High-Resolution Image Synthesis

---

