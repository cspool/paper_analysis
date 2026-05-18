## Latent Diffusion Model (LDM)（潜空间扩散模型）

术语是什么？通过联网搜索让回答具体和精准。
Latent Diffusion Model (LDM) 由Rombach et al. (2022)提出，是当前主流的扩散模型框架。核心思想：将扩散过程从高维pixel space迁移到低维learned latent space。由一个预训练的VAE (encoder E + decoder D)提供压缩——encoder将图像x∈R^{H×W×3}压缩到latent z∈R^{C×h×w}（典型f=8下采样，h=H/8, w=W/8），扩散模型在latent space中做denoising/flow matching，decoder将生成结果从latent恢复到pixel。比pixel-space diffusion大大降低计算开销，使高分辨率生成在单GPU上可行。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

LDM完整生成pipeline（以Flux text-to-image为例）：
```
// 离线阶段：VAE预训练
1: VAE.encoder: x ∈ R^{H×W×3} → z ∈ R^{C×h×w}     (f=8压缩)
2: VAE.decoder: z ∈ R^{C×h×w} → x ∈ R^{H×W×3}

// 在线生成
3: z_T ~ N(0, I)                                     // 在latent space初始化噪声
4: for t = T, ..., 1:                                // T个denoising steps
5:     v = DiT(z_t, t, text_embedding)               // transformer预测velocity/noise
6:     z_{t-1} = scheduler_step(z_t, v, t)           // ODE/SDE step
7: generated_image = VAE.decoder(z_0)                // latent→pixel
```

关键设计选择：
- **Compression factor f**：决定latent相对于pixel的缩小比。f=8（常用，如Flux-VAE、SD3-VAE 16ch）→latent token数=f²倍减少。更大f（如SD3-F16, f=16）进一步减少token但可能损失细节。
- **VAE quality对UHR生成至关重要**：LWD论文发现标准VAE在UHR下latent space包含cross-scale inconsistent high-frequency artifacts，通过scale-consistency fine-tuning压制伪影可显著提升后续wavelet masking的有效性。
- **LDM vs pixel-space diffusion**: LDM训练/推理快f²倍，但VAE compression可能丢失fine-grained semantic detail——这是LWD承认的limitation（future work建议探索higher-fidelity latent space或latent+pixel联合监督）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
主流实现：CompVis/latent-diffusion（GitHub）、HuggingFace Diffusers、Stable Diffusion系列。LDM的关键模块：1) VAE (encoder+decoder)，通常独立预训练后冻结，扩散模型仅操作在latent；2) U-Net或DiT backbone执行denoising/flow matching；3) text encoder（CLIP/T5）提供条件。训练通常在较低分辨率（256-512）pretrain→UHR fine-tune（LWD的stage 2）。LWD论文中VAE微调独立于扩散模型微调（stage 1 vs stage 2），保持模块化解耦。

涉及论文标题：
- Latent Wavelet Diffusion for Ultra-High-Resolution Image Synthesis

---

