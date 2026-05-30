# <span id="page-22-0"></span>I CALIBRATION COMPUTATION RESOURCE

We study the calibration computation resource of each of our proposed methods and the overall pipeline. As Second-Order Sparse Attention Reparameterization (SSAR) is used for only inference, for calibration, we only add Multi-Scale Salient Attention Distillation (MSAD) compared to naive Post-Training Quantization (PTQ). We present the calibration resource in Tab. 14. Compared with

<span id="page-23-2"></span><span id="page-23-1"></span>Table 14: Calibration computation resource report. PTQ denotes naive Post-Training Quantization without attention distillation.

| Method            | Calibration                        | Performance           |                 |                  |  |  |  |  |
|-------------------|------------------------------------|-----------------------|-----------------|------------------|--|--|--|--|
| 1/10/11/04        | GPU Memory (GB)↓ GPU Time (Hours)↓ |                       | PSNR↑           | LPIPS↓           |  |  |  |  |
| Wan2.1 1.3B       |                                    |                       |                 |                  |  |  |  |  |
| PTQ               | 16.21                              | 0.62                  | 10.57           | 0.587            |  |  |  |  |
| +Global           | 16.27 <sub>+0.4%</sub>             | $0.63_{+0.2\%}$       | 13.27           | 0.452            |  |  |  |  |
| +Local            | 16.28 <sub>+0.4%</sub>             | $0.63_{+0.2\%}$       | 13.85           | 0.421            |  |  |  |  |
| QuantSparse       | 16.34 <sub>+0.8%</sub>             | $0.64_{+1.6\%}$       | $15.22_{+4.65}$ | $0.338_{-0.249}$ |  |  |  |  |
| Hunyuan Video 13B |                                    |                       |                 |                  |  |  |  |  |
| PTQ               | 39.22                              | 5.08                  | 16.27           | 0.472            |  |  |  |  |
| +Global           | 39.33 <sub>+0.3</sub> %            | $5.10_{+0.4\%}$       | 18.42           | 0.357            |  |  |  |  |
| +Local            | 39.32 <sub>+0.3%</sub>             | $5.11_{+0.6\%}$       | 18.96           | 0.342            |  |  |  |  |
| QuantSparse       | 39.41 <sub>+0.5</sub> %            | 5.13 <sub>+1.0%</sub> | $20.86_{+4.59}$ | $0.272_{-0.200}$ |  |  |  |  |
| Wan2.1 14B        |                                    |                       |                 |                  |  |  |  |  |
| PTQ               | 47.39                              | 2.57                  | 14.35           | 0.425            |  |  |  |  |
| +Global           | 47.54 <sub>+0.3%</sub>             | $2.59_{+0.8\%}$       | 16.01           | 0.349            |  |  |  |  |
| +Local            | 47.50 <sub>+0.2%</sub>             | $2.58_{+0.4\%}$       | 16.82           | 0.325            |  |  |  |  |
| QuantSparse       | 47.65 <sub>+0.5</sub> %            | 2.60 <sub>+1.1%</sub> | $18.72_{+4.37}$ | $0.240_{-0.185}$ |  |  |  |  |

naive PTQ, our *Global Distillation* only brings an average of 0.8% extra time burden and almost no additional memory consumption because of its efficient low-resolution attention operation. Also, our *Local Distillation* only needs to calculate the token saliency distribution once before each block calibration and reuse the salient token index in each optimization iteration, which is also very efficient. These two distillation methods are not only efficient but also can effectively alleviate the attention shift caused by quantization and improve the video generation effect. QuantSparse has significantly improved the model performance by combining two distillation methods, while ensuring high efficiency.

To further prove the effectiveness of proposed Second-Order Sparse Attention Reparameterization (SSAR), we present the inference burden brought by SSAR in Tab. 15. Compared with Non-Reparameterization, the cache-based method only requires one additional matrix addition operation for the sparse attention output, which is very efficient. Therefore, the cache-based method will only bring little additional latency and memory burden. Furthermore, the second-order residual can store and calculate the second-order term and the first-order term together. Therefore, compared with the first-order residual, the second-order residual only requires an additional second order term calculation, but significantly improves the sparse attention performance under quantization, and improves the PSNR from 17.08 to 18.68 under Wan2.1-14B (Wan et al., 2025). In addition, using SVD to extract the temporally stable component of second-order residuals brings almost no additional consumption, but can further improve the effect of second-order residuals, which further decreases LPIPS from 0.258 to 0.240 under Wan2.1-14B.

