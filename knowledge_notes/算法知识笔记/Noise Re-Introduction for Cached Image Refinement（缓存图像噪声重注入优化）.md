## Noise Re-Introduction for Cached Image Refinement（缓存图像噪声重注入优化）

术语是什么？

Noise Re-Introduction是MoDM提出的将cached final image重新引入扩散去噪流程的机制：对检索到的缓存图像I*按扩散模型的noise schedule在timestep t_k处加噪，生成中间状态I_tilde = sigma_{t_k} * epsilon + (1-sigma_{t_k}) * I*（公式2），然后使用small model执行剩余的T-k步去噪。这与SDEdit和标准image-to-image diffusion使用相同的公式，但MoDM将其应用于serving cache pipeline：cached image提供high-level structure，noise添加variation，后续denoising将图像refine到匹配新prompt。

从算法pipeline角度拆解术语：

Noise Re-Introduction算法：

```
// 输入: cached image I*, target timestep t_k, diffusion model noise schedule sigmas[]
// 输出: noised intermediate latent I_tilde

Function ReIntroduceNoise(I*, t_k, sigmas):
    sigma_tk = sigmas[t_k]                      // 从noise schedule查表
    epsilon ~ N(0, I)                           // 标准高斯噪声
    I_tilde = sigma_tk * epsilon + (1 - sigma_tk) * I*  // 线性插值(Eq.2)

// 之后执行denoising:
    latent_tk = VAE.encode(I_tilde)             // 或直接在pixel space（取决于模型设计）
    for step = t_k-1, t_k-2, ..., 0:
        latent_step = diffusion_denoise_step(latent_{step+1}, step, prompt, small_model)
    output_image = VAE.decode(latent_0)
```

核心特性：(1) sigma_{t_k}控制噪声量——越大则越接近纯噪声（更多variation），越小则越接近原图（更多preservation）；(2) noise schedule sigmas[]由diffusion model预定义（通常为线性或cosine schedule）；(3) 不同k值对应不同的sigma_{t_k}，higher k → larger sigma → more variation，但起始于更好的prior（cached image）。

术语一般如何实现？如何使用？

MoDM直接使用扩散模型内置的noise schedule（Stable Diffusion系列使用DDPM schedule），在Request Scheduler中调用模型的scheduler.sigmas[t_k]获得sigma值。加噪操作是简单的element-wise linear interpolation，计算成本极低（可忽略 vs denoising cost）。与从头生成（从纯噪声t=T开始）相比，从t_k加噪再refine的compute savings来自两方面：跳过k步denoising + 用小模型替代大模型执行剩余T-k步。

涉及论文标题：
- MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models

