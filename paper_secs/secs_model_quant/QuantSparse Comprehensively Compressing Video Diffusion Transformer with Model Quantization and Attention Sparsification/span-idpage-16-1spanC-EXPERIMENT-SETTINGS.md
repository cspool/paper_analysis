# <span id="page-16-1"></span>C EXPERIMENT SETTINGS

Same with prior works (Zhao et al., 2024; Ashkboos et al., 2024; Feng et al., 2025c), we adopt channel-wise weight quantization and dynamic token-wise activation quantization. And we use uniform symmetry quantization for both weight and activation for better hardware acceleration and memory saving. For fair comparison, we apply the same quantization granularity for all quantization

<span id="page-17-2"></span><span id="page-17-1"></span>

|                                                              | #Bits |                      |                      |                  | Qualit                       | y                |              |        |
|--------------------------------------------------------------|-------|----------------------|----------------------|------------------|------------------------------|------------------|--------------|--------|
| Method                                                       | (W/A) | Density <sub>↓</sub> | Video                | Quality N        | 1etrics                      | FP Diff. Metrics |              |        |
|                                                              |       |                      | CLIPSIM <sub>↑</sub> | VQA↑             | $\Delta$ FSCore $\downarrow$ | PSNR↑            | SSIM↑        | LPIPS↓ |
| ${\it Hunyuan Video~13B~(CFG=6.0,720\times1280p,frames=60)}$ |       |                      |                      |                  |                              |                  |              |        |
| Full Prec.                                                   | 16/16 | 100%                 | 0.184                | 81.23            | 0.000                        | -                | -            | -      |
| SmoothQuant                                                  | 6/6   | 100%                 | 0.180                | 69.55            | 1.406                        | 15.91            | 0.553        | 0.411  |
| QuaRot                                                       | 6/6   | 100%                 | 0.182                | 72.28            | 0.546                        | 16.99            | 0.590        | 0.378  |
| ViDiT-Q                                                      | 6/6   | 100%                 | 0.182                | 72.36            | 0.937                        | 18.24            | 0.623        | 0.335  |
| Q-VDiT                                                       | 6/6   | 100%                 | 0.182                | 73.68            | 1.232                        | 21.02            | 0.675        | 0.264  |
| QuaRot+SVG                                                   | 6/6   | 25%                  | 0.181                | 72.57            | 0.718                        | 16.85            | 0.581        | 0.385  |
| Q-VDiT+SVG                                                   | 6/6   | 25%                  | 0.181                | 72.59            | 1.405                        | 20.38            | 0.658        | 0.284  |
| QuaRot+SVG                                                   | 6/6   | 15%                  | 0.181                | 72.60            | 0.997                        | 16.85            | 0.578        | 0.394  |
| Q-VDiT+SVG                                                   | 6/6   | 15%                  | 0.181                | 72.04            | 1.763                        | 19.94            | 0.644        | 0.307  |
| QuantSparse                                                  | 6/6   | 25%                  | 0.183                | 81.17            | 0.435                        | 22.71            | 0.720        | 0.221  |
| QuantSparse                                                  | 6/6   | 15%                  | 0.183                | 82.26            | 0.328                        | 22.68            | 0.720        | 0.224  |
|                                                              | M     | an2.1 14B            | (CFG = 5.0, 7)       | $20 \times 1280$ | )p, frames =                 | = 80)            |              |        |
| Full Prec.                                                   | 16/16 | 100%                 | 0.182                | 90.79            | 0.000                        | -                | -            | -      |
| SmoothQuant                                                  | 6/6   | 100%                 | 0.178                | 62.25            | 0.363                        | 13.06            | 0.404        | 0.656  |
| QuaRot                                                       | 6/6   | 100%                 | 0.180                | 66.56            | 0.313                        | 13.59            | 0.409        | 0.566  |
| ViDiT-Q                                                      | 6/6   | 100%                 | 0.180                | 71.26            | 0.251                        | 15.30            | 0.513        | 0.376  |
| Q-VDiT                                                       | 6/6   | 100%                 | 0.180                | 89.10            | 0.082                        | 18.13            | 0.610        | 0.264  |
| QuaRot+SVG                                                   | 6/6   | 25%                  | 0.179                | 67.64            | 0.336                        | 13.60            | 0.407        | 0.555  |
| Q-VDiT+SVG                                                   | 6/6   | 25%                  | 0.179                | 88.29            | 0.091                        | 16.69            | 0.563        | 0.323  |
| QuaRot+SVG                                                   | 6/6   | 15%                  | 0.180                | 60.14            | 0.396                        | 13.55            | 0.399        | 0.567  |
| Q-VDiT+SVG                                                   | 6/6   | 15%                  | 0.179                | 85.26            | 0.182                        | 15.94            | 0.532        | 0.367  |
| QuantSparse                                                  | 6/6   | 25%                  | 0.182                | 89.96            | 0.002                        | 18.67            | 0.622        | 0.240  |
| QuantSparse                                                  | 6/6   | 15%                  | 0.181                | 92.87            | 0.060                        | 18.67            | <u>0.616</u> | 0.277  |

Table 6: Text-to-Video generation experiments on more huge models.

methods. We adopt channel-wise scale used in (Xiao et al., 2023a; Wu et al., 2024; Zhao et al., 2024; Feng et al., 2025c) and rotation-based matrix used in (Ashkboos et al., 2024; Zhao et al., 2024; Sun et al., 2024b) for quantization. We follow block-wise post-training strategy used in (Wu et al., 2024; Chen et al., 2024; Sun et al., 2024b) for calibration. All the experiments are conducted on a single NVIDIA A800 GPU.

During calibration, we set channel-wise scale, rotation matrix, and quantization scale as learnable following (Feng et al., 2025c; Sun et al., 2024b). We use 20 random generated samples and train 15 epoch for each transformer block. We apply the same calibration samples and epochs for all methods for fair comparison. We use AdamW (Loshchilov & Hutter, 2017) optimizer and cosine learning rate scheduler. For the channel-wise scale and rotation matrix, we use a learning rate of  $5e^{-3}$ . For the learnable quantization scale, we use a learning rate of  $5e^{-2}$ . For distillation, we use r=128 for global distillation pooling, k=256 for salient query selection, and  $\lambda_{\rm global}=1e^{-4}$ ,  $\lambda_{\rm global}=1e^{-4}$  for Wan2.1-1.3B, Wan2.1-14B, and  $\lambda_{\rm global}=1.0$ ,  $\lambda_{\rm global}=1e^{2}$  for HunyuanVideo, respectively. The selection of distillation balancing factor is based on the order of magnitude of the loss. For sparse attention, we use a fixed cache refreshing interval of 5, and use k=16 for SVD.

For deployment, we quantize the weight and absorb all the quantization parameters following (Zhao et al., 2024; Sun et al., 2024b; Feng et al., 2025c; Ashkboos et al., 2024). For activation, we use dynamic online quantization same as (Feng et al., 2025c; Sun et al., 2024b; Zhao et al., 2024).

