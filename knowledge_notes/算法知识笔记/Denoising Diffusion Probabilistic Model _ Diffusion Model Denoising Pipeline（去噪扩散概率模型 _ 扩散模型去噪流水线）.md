## Denoising Diffusion Probabilistic Model / Diffusion Model Denoising Pipeline（去噪扩散概率模型 / 扩散模型去噪流水线）

术语是什么？

扩散模型 (Diffusion Model) 通过迭代去噪从纯噪声生成文本条件图像。其pipeline包含两个过程：(1) Forward Process（前向过程）：逐步向真实图像添加高斯噪声，经过T步后逼近标准正态分布；(2) Reverse Process（反向/去噪过程）：从纯噪声开始，逐步去噪恢复图像，每步将当前噪声latent送入完整模型（UNet或DiT），共需T步（通常T=50）。扩散模型评价指标包括FID（Frechet Inception Distance，越低越好，衡量与真实图像的分布距离）、CLIPScore（越高越好，衡量图文对齐）、IS（Inception Score，越高越好，衡量质量与多样性）、PickScore（越高越好，基于人类偏好训练的评分）。

从算法pipeline角度拆解术语：

扩散模型去噪pipeline（以Stable Diffusion 50步为例）：

```
// Forward Process (training only): x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * epsilon

// Reverse Process (inference):
1: latent_T ~ N(0, I)                          // 初始化纯噪声
2: for t = T, T-1, ..., 1 do                    // 50 denoising steps
3:     epsilon_theta = model(latent_t, t, prompt_embedding)  // UNet/DiT预测噪声
4:     latent_{t-1} = 1/sqrt(alpha_t) * (latent_t - (1-alpha_t)/sqrt(1-bar_alpha_t) * epsilon_theta) + sigma_t * z
5: end for
6: image = VAE.decode(latent_0)                 // VAE decoder将latent恢复为图像
```

扩散动态的关键特性：early denoising steps决定图像结构（layout、object position），later steps关注细节（texture、color、fine details）。这使small model可以处理cache-hit请求的later refinement steps——因为cached图像已提供结构，仅需细节调整。

术语一般如何实现？如何使用？

主流开源实现包括HuggingFace diffusers库的StableDiffusionPipeline、StableDiffusionXLPipeline、FluxPipeline等。MoDM利用扩散动态特性：cache-hit时对检索图像加噪到timestep t_k（公式2），然后仅用small model执行剩余T-k步。MoDM使用的模型：SD3.5L (8B, BF16), FLUX.1-dev (12B, BF16), SDXL (3B, FP16), SANA (1.6B, BF16), SD3.5L-Turbo (10步蒸馏版)。所有模型生成1024x1024图像，T=50步（除Turbo用10步）。

涉及论文标题：
- MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models
- Difflow: A Data-Characteristic-Aware Serving System for Diffusion Models（基于图文相似度的缓存检索）
- MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models（patch级并行去噪、operator taxonomy for serving）

