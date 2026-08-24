# C. Correlation Between Learned Compression Ratios and Router Predictions

We select three representative layers with high, medium, and low learned compression ratios to visualize the corresponding predictions of the DiffCR router and analyze potential correlations. As shown in Fig. [10,](#page-12-1) where "C.R." denotes the compression ratios, we observe a strong correla-

<span id="page-12-0"></span>> **[图片提取文字 (无描述)]:**
> "eel sushí (a) (b) Input: Input: Inpainting Text-to-Image **Router's Prediction Router's Prediction** Timestep: Timestep: 60 80 100 21 26 Layer: 13 19 20 Layer: 2 3 17 10 "sky" Input: Input: "a lake" **Router's Prediction** Timestep: Timestep: **Router's Prediction** 40 60 80 15 100 20 21 26 7 13 17 3 19 20 Layer: 2 10 14 Layer:
![](_page_12_Figure_0.jpeg)

<span id="page-12-1"></span>Figure 9. More visualizations of the router's predictions: (a) For inpainting tasks, where inputs are masked images with text prompts, we follow the previous SOTA method Lazy-Diffusion [28] to generate only the masked area rather than the entire image; (b) For text-to-image (T2I) tasks, where inputs are noise and text prompts, we follow  $PixArt-\Sigma$  [4] for generation. Each visualization includes the router's prediction map with values ranging from 0 to 1. The generated image at each corresponding timestep is shown on the left, while the router's prediction maps across various layers and timesteps are displayed on the right.

> **[图片提取文字 (无描述)]:**
> (a) Inpainting (b) Text-to-Image 22 Layer: Layer: C.R.: 2% C.R.: 0% 35% 87% 54% 90%
![](_page_12_Figure_2.jpeg)

Figure 10. Visualization and analysis of the correlation between the learned compression ratios and the DiffCR router's predictions.

<span id="page-13-0"></span>> **[图片提取文字 (无描述)]:**
> Lazy Diffusion (LD) (p=0%)LD w/ Model Pruning (p=30%) PixArt- $\Sigma$  (p=0%) LD w/ ToMe (p=30%) PixArt- $\Sigma$  w/ ToMe (p=20%) LD w/ ToMe (p=10%) PixArt- $\Sigma$  w/ ToMe (p=10%) LD w/ AT-EDM (p=30%) PixArt- $\Sigma$  w/ AT-EDM (p=20%) LD w/ MoD (p=30%) PixArt- $\Sigma$  w/ MoD (p=20%) LD w/ DiffCR (p=30%) PixArt- $\Sigma$  w/ **DiffCR** (p=20%) Diameter 0.77 0.98 13.9 16.4  $10^{2}$ **TFLOPS** Diameter  $10^{2}$ TFLOPS 24 18 22 4.92 FID Reduction 20 16 18 ₽ 14 12.10 FID Reduction 16 20% Latency Savings 20% Latency Savings 14 12 12 10 10 0.80 0.85 0.80 0.85 1.00 0.75 0.90 0.95 1.00 1.05 0.70 0.75 0.90 0.95 1.05 Relative Latency (BS=1) Relative Latency (BS=16) (b) Inpainting (a) T2I
![](_page_13_Figure_0.jpeg)

Figure 11. Overall comparison of DiffCR with baselines in terms of latency, FID, and TFLOPS for both T2I and inpainting tasks.

tion between the learned ratios and the router's predictions. For layers with high compression ratios, such as layer 1 in inpainting or layer 9 in T2I, the router consistently predicts lower importance scores for many semantic areas, adopting an extremely "lazy behavior" to save computations. Conversely, for layers with low compression ratios, the router assigns higher importance scores to most areas. This visualization validates the joint learning effect between our token-level routers and the differentiable ratios.

