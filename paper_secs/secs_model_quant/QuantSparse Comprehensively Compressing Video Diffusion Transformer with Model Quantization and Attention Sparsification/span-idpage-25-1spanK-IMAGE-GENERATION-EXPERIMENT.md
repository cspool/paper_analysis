# <span id="page-25-1"></span>K IMAGE GENERATION EXPERIMENT

QuantSparse is designed as a general framework for Diffusion Transformers (DiTs) and is not limited to video generation. To validate its generalizability, we conducted an experiment on Hunyuan-DiT [\(Li et al.,](#page-11-15) [2024c\)](#page-11-15), a 1.5B parameters model targeting image generation. We evaluate on Draw-Bench [\(Saharia et al.,](#page-12-15) [2022\)](#page-12-15) under W4A8 quantization and present the results in Tab. [17.](#page-25-2) Even for image-generation DiTs, QuantSparse outperforms SOTA quantization baselines QuaRot [\(Ashkboos](#page-10-7) [et al.,](#page-10-7) [2024\)](#page-10-7) and Q-VDiT [\(Feng et al.,](#page-11-12) [2025c\)](#page-11-12) while using only 40% attention density. This confirms that our framework generalizes to DiT-based visual generation tasks and not limited to video generation.

Method Density<sup>↓</sup> PSNR<sup>↑</sup> SSIM<sup>↑</sup> LPIPS<sup>↓</sup> Hunyuan-DiT QuaRot 100% 17.30 0.627 0.460 Q-VDiT 100% 19.32 0.658 0.347 QuantSparse 40% 20.34 0.692 0.289

<span id="page-25-2"></span>Table 17: Image generation experiment results on Hunyuan-DiT.

