## Diffusion Transformer (DiT)

术语是什么？

Diffusion Transformer (DiT) 是以Transformer架构替代传统UNet作为denoising backbone的扩散模型。由Peebles和Xie在ICCV 2023提出，将扩散过程的每一步操作在完整的图像latent token序列上通过Transformer的self-attention机制进行去噪。DiT已取代UNet成为主流高质量图像生成模型的backbone，代表性模型包括Stable Diffusion 3 (SD3)和FLUX.1-dev (12B参数)。与LLM不同，DiT模型参数通常较小（最大开源DiT仅12B），能够fit在单张80GB H100 GPU上，因此DiT推理中的并行化主要目的是降低延迟而非解决显存容量问题。

从算法pipeline角度拆解术语：

DiT推理pipeline（以FLUX.1-dev 50步去噪为例）：

```
1: latent ← VAE.encode(input_image)        // 将图像压缩到latent space
2: latent_noisy ← add_noise(latent, t=T)    // 从纯噪声开始
3: for step t = T, T-1, ..., 1 do           // 50个denoising steps
4:     latent_noisy ← DiT_block(latent_noisy, t, text_conditioning)
5:        // 每个DiT block: self-attention over all latent tokens
6:        // latent tokens数取决于分辨率: 256→256 tokens, 512→1024, 1024→4096, 2048→16384
7: end for
8: output_image ← VAE.decode(latent_noisy)  // VAE decoder从latent恢复图像
```

关键特征：(a) **Stateless**：无KV cache，每步独立计算全部latent tokens；(b) **Compute-bound**：多步去噪在全量latent tokens上执行，计算量由分辨率决定（2048×2048约25 TFLOPs per step on FLUX）；(c) **Step数固定**：通常50步，每步耗时高度可预测（CV < 0.7%）；(d) **异构输入**：DiT serving workload由少量离散分辨率组成（256/512/1024/2048），但不同分辨率计算量差异巨大（256→556 TFLOPs vs 2048→24965 TFLOPs）。

术语一般如何实现？如何使用？

DiT模型由一系列Transformer block组成，每block包含multi-head self-attention + MLP。主流开源实现包括HuggingFace diffusers库中的DiTPipeline。Sequence Parallelism (SP)是DiT推理加速的主要并行方式：通过Ulysses attention沿token序列维度切分数据到多GPU，attention前通过all-to-all collective转换layout，attention后再通过all-to-all转回。DiT推理中SP的scaling efficiency是sublinear的——小分辨率（256×256）在SP=8时通信占比超30%导致效率极低，大分辨率（2048×2048）则受益于更多GPU。xDiT是目前主要的开源DiT推理框架，支持固定degree的SP。

MixFusion论文补充了U-Net与DiT在serving角度的关键差异：(1) U-Net包含Convolution算子（kernel size 1-3），后者需要跨patch context→需要Patch Edge Stitcher处理边界依赖；DiT仅含Transformer blocks（无Convolution）→patched inference时accuracy自然达到100%（PSNR inf/SSIM 1.0, Table 4）；(2) U-Net模型（SDXL）单denoising step含~7 blocks，DiT模型（SD3）含~24 blocks——更多blocks意味着更频繁的cache操作和更高的cache management overhead（Figure 17，SD3 cache overhead显著高于SDXL）；(3) SDXL不同分辨率间latency差距仅1.3×，SD3超2.4×——更大的variance为scheduling optimization留下更多空间（SD3上MixFusion scheduling相对baseline的收益更大）；(4) U-Net中Convolution引入non-linear complexity→SD3的throughput更易于MLP prediction（Table 5, MLP accuracy SD3 0.96 vs SDXL 0.81）。

涉及论文标题：
- TetriServe: Efficiently Serving Mixed DiT Workloads
- MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models
- Latent Wavelet Diffusion for Ultra-High-Resolution Image Synthesis

---

