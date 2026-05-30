# <span id="page-21-0"></span>H MORE ABLATION STUDY

Here, we provide more ablation study about the proposed Multi-Scale Salient Attention Distillation (MSAD) and Second-Order Sparse Attention Reparameterization (SSAR).

We first study the pooling stride s used in Eq. 6 and salient token k in Eq. 8 to verify the hyperparameter selection of both global and local distillation. We present the results in Tab. 10. It can be seen that different hyperparameters can improve the distillation performance. This shows that our distillation method is both effective and robust, which is insensitive to hyperparameters. This also demonstrates that the memory-efficient distillation are effective enough and we do not have to use the giant complete attention map to supervise the attention module. Higher s and lower k can reduce memory, but typically harm performance. Yet we identify that decreasing s and increasing k also brings little improvement. Since s = 128 and k

<span id="page-21-4"></span>Table 10: Ablation on s and k used in attention distillation.

| Value | VQA↑  | PSNR↑        | SSIM↑ | LPIPS↓ |
|-------|-------|--------------|-------|--------|
| None  | 81.92 | 14.35        | 0.486 | 0.425  |
|       | p     | ooling stric | de s  |        |
| s=64  | 85.19 | 16.05        | 0.543 | 0.348  |
| s=128 | 85.26 | 16.01        | 0.547 | 0.349  |
| s=256 | 85.12 | 15.93        | 0.545 | 0.355  |
|       | S     | alient toke  | n k   |        |
| k=128 | 86.21 | 16.72        | 0.551 | 0.349  |
| k=256 | 86.95 | 16.82        | 0.561 | 0.325  |
| k=512 | 86.93 | 16.95        | 0.561 | 0.324  |

increasing k also brings little improvement. Since s=128 and k=256 are both effective and efficient as shown in Fig. 3d, we choose this balanced selection.

We then study the top-r components in SVD used in Eq. 16, and present the results in Tab. 11. Compared with the original second-order residual, it can be seen that the different selection of r in SVD can improve the temporal stability of the second-order residual and bring better performance. In our experiment, we chose r=16 as it achieves good performance. We further explore higher-order residual effectiveness on attention approximation. Compared with the Second-Order residual, Third-

Table 11: Ablation on SVD used in SSAR.

<span id="page-22-2"></span><span id="page-22-1"></span>

| Method | VQA↑                    | PSNR↑           | SSIM↑               | LPIPS↓           |
|--------|-------------------------|-----------------|---------------------|------------------|
| None   | 68.00                   | 14.16           | 0.470               | 0.445            |
| First  | 70.82                   | 17.08           | 0.572               | 0.285            |
| Second | 89.73                   | 18.68           | 0.616               | 0.258            |
| Third  | 89.71                   | 18.70           | 0.620               | 0.263            |
| top-8  | 91.12                   | 18.69           | 0.621               | 0.253            |
| top-16 | 91.98 <sub>+23.98</sub> | $18.72_{+4.56}$ | $0.630_{\pm 0.160}$ | $0.240_{-0.205}$ |
| top-32 | 91.75                   | 18.72           | 0.628               | 0.242            |

Order residual only slightly improves PSNR from 18.70 to 18.68 and decreases the performance on VQA, SSIM, and LPIPS. This indicates that the stability brought by higher-order residuals will gradually saturate, and we attribute it to the additional noise brought by longer time series information on higher-order residuals. The second-order residual not only stabilizes the first-order residual, but SVD can further reduce spatiotemporal noise.

We then study the weight factor used in Eq. 9 to verify the distillation robustness of hyperparameters. We present the results in Tab. 12. The values are selected by controlling the distillation term to be of the same order of magnitude as  $\mathcal{L}_{\text{quant}}$ . It can be seen that different weight factors improve the model performance. This shows that our distillation method is not only effective but also insensitive to the choice of hyperparameters, indicating its generalization and effectiveness. Since  $\lambda_{\text{global}} = 1e - 4$  and  $\lambda_{\text{local}} = 1e - 4$  are good enough, and the hyperparameter selection is robust, we do not further fine-tune the hyperparameter selection.

<span id="page-22-3"></span>Table 12: Ablation on  $\lambda_*$  used in Eq. 9.

| $\lambda_*$ | VQA <sub>↑</sub> | PSNR↑    | SSIM↑ | LPIPS↓ |
|-------------|------------------|----------|-------|--------|
| None        | 81.92            | 14.35    | 0.486 | 0.425  |
|             |                  | * = glob | al    |        |
| 5e-3        | 84.76            | 15.79    | 0.518 | 0.362  |
| 1e-4        | 85.26            | 16.01    | 0.547 | 0.349  |
| 5e-4        | 85.33            | 15.72    | 0.540 | 0.351  |
|             |                  | * = loca | ıl    |        |
| 5e-3        | 86.73            | 16.86    | 0.547 | 0.336  |
| 1e-4        | 86.95            | 16.82    | 0.561 | 0.325  |
| 5e-4        | 86.54            | 16.72    | 0.562 | 0.328  |
|             |                  |          |       |        |

We further compare our Multi-Scale Salient Attention Distillation (MSAD) with full-attention distillation (using the complete FP attention map as the target) on Wan2.1-1.3B (Wan et al., 2025) under W4A8 quantization. The results are shown in Tab. 13. MSAD achieves nearly identical performance to full-attention distillation. The results highlight MSAD's efficiency advantages while maintaining comparable performance.

Table 13: Ablation study on full-attention distillation.

<span id="page-22-4"></span>

| Method                    | Resolution     | PSNR↑ | LPIPS↓ | Attention Memory Cost (GB)↓ | Calibration Time (Hours)↓ |  |  |  |
|---------------------------|----------------|-------|--------|-----------------------------|---------------------------|--|--|--|
| Wan2.1 1.3B               |                |       |        |                             |                           |  |  |  |
| Full Attention            | (17472, 17472) | 15.25 | 0.338  | 6.82                        | 1.86                      |  |  |  |
| MSAD (s = 64)             | (273, 273)     | 15.23 | 0.338  | 0.17                        | 0.66                      |  |  |  |
| <b>MSAD</b> ( $s = 128$ ) | (137, 137)     | 15.22 | 0.338  | 0.14                        | 0.64                      |  |  |  |
| <b>MSAD</b> ( $s = 256$ ) | (69, 69)       | 15.21 | 0.339  | 0.13                        | 0.63                      |  |  |  |

**Effect of attention density.** We conduct an ablation study on attention density, analyzing the trade-off between performance and inference speed. The results are presented in Tab. 5. As shown, a 25% density offers a good balance, achieving a significant 1.55× speedup with minimal performance degradation (PSNR of 18.72). A 15% density further boosts the speedup to 1.74× while maintaining acceptable performance (PSNR of 18.22). Based on these results, we selected 25% and 15% density for the experiments presented in the main paper. The 25% density provides a strong baseline for high performance with good acceleration, while the 15% density demonstrates the potential for even greater inference speedup at a slightly decreased performance trade-off.

