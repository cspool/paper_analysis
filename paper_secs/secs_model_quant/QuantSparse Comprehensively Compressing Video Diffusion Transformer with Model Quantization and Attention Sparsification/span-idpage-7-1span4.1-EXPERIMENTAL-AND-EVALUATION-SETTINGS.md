# <span id="page-7-1"></span>4.1 EXPERIMENTAL AND EVALUATION SETTINGS

**Evaluation Settings.** We apply QuantSparse to HunyuanVideo-13B (Kong et al., 2024), Wan2.1-1.3B and 14B (Wan et al., 2025) with 50 sampling steps. We employ two types of metrics: (1) Multi-aspects metrics evaluation: including CLIPSIM (Wu et al., 2021), VQA (Wu et al., 2023), FlowScore (Liu et al., 2024b), PSNR, SSIM, and LPIPS (Zhang et al., 2018). All metrics are evaluated on the prompt sets used in (Zhao et al., 2024; Feng et al., 2025c) (2) Benchmark evaluation: We select 8 major dimensions from Vbench (Huang et al., 2024b) following prior works (Zhao et al., 2024; Chen et al., 2025c). For bit setting, we use W6A6 and W4A8 following prior work (Zhao et al., 2024; Chen et al., 2024; Wu et al., 2024), since they can bring more compression effects and ensure the performance.

**Baseline Methods.** We select PTQ4DiT (Wu et al., 2024), Q-DiT (Chen et al., 2024), ViDiT-Q (Zhao et al., 2024), and Q-VDiT (Feng et al., 2025c) for diffusion baseline. We also compare with strong LLM baseline SmoothQuant (Xiao et al., 2023a) and QuaRot (Ashkboos et al., 2024). For sparsification, we compare with DiTFastAttn (DFT) (Yuan et al., 2024) (cache-based), Jenga (Zhang et al., 2025d) (dynamic pattern), and SparseVideoGen (SVG) (Xi et al., 2025) (static pattern).

**Implementation Detail.** Same with prior works (Zhao et al., 2024; Ashkboos et al., 2024; Feng et al., 2025c), we adopt channel-wise weight quantization and dynamic token-wise activation quantization. We follow block-wise post-training strategy used in (Wu et al., 2024; Chen et al., 2024; Sun et al., 2024b) for calibration. **More details can be found in Appendix C**.

### 4.2 MAIN RESULTS

<span id="page-8-2"></span>We present multi-aspects metrics evaluation results on HunyuanVideo (Kong et al., 2024) and Wan2.1-14B (Wan et al., 2025) in Tab. 2. It can be seen that the existing SOTA quantization methods have a significant performance degradation after applying sparse attention. But QuantSparse still maintains high generation performance even at high sparsity. It is worth mentioning that QuantSparse even surpasses the existing quantization-only methods under the low-bit settings of W6A6 and W4A8.

<span id="page-8-1"></span>Table 3: Ablation results of each component.

| Method | VQA↑             | PSNR↑           | SSIM↑            | LPIPS↓           |
|--------|------------------|-----------------|------------------|------------------|
|        | I                | Distillation An | alysis           |                  |
| None   | 81.92            | 14.35           | 0.486            | 0.425            |
| Global | 85.26            | 16.01           | 0.547            | 0.349            |
| Local  | 86.95            | 16.82           | 0.561            | 0.325            |
| MSAD   | $91.98_{+10.06}$ | $18.72_{+4.37}$ | $0.630_{+0.144}$ | $0.240_{-0.185}$ |
|        |                  | Cache Analy     | /sis             |                  |
| None   | 68.00            | 14.16           | 0.470            | 0.445            |
| First  | 70.82            | 17.08           | 0.572            | 0.285            |
| Second | 89.73            | 18.68           | 0.616            | 0.258            |
| SSAR   | $91.98_{+23.98}$ | $18.72_{+4.56}$ | $0.630_{+0.160}$ | $0.240_{-0.205}$ |

Compared with the Full-Precision (FP) model, QuantSparse even maintains almost lossless performance. For example, for HunyuanVideo under W6A6, QuantSparse achieved a VQA score of 82.26 with only 15% attention density, far exceeding current SOTA method Q-VDiT (Feng et al., 2025c) of 73.68, and even surpassing the FP model of 81.23. We present more baseline methods comparison in Appendix Sec. D, and comprehensive VBench evaluation results in Appendix Sec. E. We also observed that QuantSparse slightly outperforms Full Precision model on certain metrics. This slight outperformance of QuantSparse can be attributed to its focus on task-critical tokens and reduced attention to noisy or irrelevant tokens, as shown in our saliency analysis. Additionally, the SSAR module stabilizes sparse attention, reducing quantization noise and improving temporal consistency. These effects, combined with targeted compression, allow QuantSparse to maintain near-lossless quality while offering substantial compression and acceleration. We also visualized the generated videos in Fig. 5. Compared with FP model, QuantSparse achieves almost lossless generation performance while other methods have notable quality degradation. We provide more visual comparison results in Appendix Sec. M.

<span id="page-8-0"></span>![](_page_8_Figure_5.jpeg)

Figure 5: Visual comparison on Wan2.1-14B under W4A8 quantization setting. We uniformly sample two frames for visualization. '(xx%)' denotes the attention density.

### 4.3 ABLATION STUDY

We conduct ablation study on proposed Multi-Scale Salient Attention Distillation (MSAD) and Second-Order Sparse Attention Reparameterization (SSAR) on Wan2.1-14B under W4A8 in Tab. 3.

**Effect of attention distillation.** Compared with no distillation, both proposed attention guidance can enhance the model performance. The combined MSAD further improves PSNR from 14.35 to 18.72, demonstrating the effect of attention distillation.

**Effect of attention reparameterization.** Compared with naive sparse attention, first-order residual can reduce the attention error, demonstrating the effectiveness of attention reparameterization. Our proposed SSAR achieves the best approximation performance by reducing both the quantization-induced error and temporal variance.

Effect of cache-interval. We also supplement the ablation and the results are shown in Tab. 5. While shorter intervals yield higher PSNR and SSIM, indicating better performance, they also result in a reduced speedup  $(1.65 \times \text{ and } 1.69 \times \text{ respectively})$ . For instance, interval=3 achieves the highest PSNR (18.86) but sacrifices a noticeable amount of the potential speedup (9%). Longer intervals increasing the interval to 6 provides a slightly higher speedup  $(1.76 \times)$ . However, this comes at the cost of a degradation in performance (PSNR drops to 17.72). We choose interval=5 is based on its optimal balance between model performance and inference speedup. But we highlight that this

| TC 1 1 4 | D . 11 1 | cc ·       |             |
|----------|----------|------------|-------------|
| Table 4: | Defailed | efficiency | comparison. |
|          |          |            |             |

<span id="page-9-2"></span><span id="page-9-1"></span>

| Method      | Method   #Bits   Dens |             | Mod                                       | lel Overload                             | Latency   | & Speed       |
|-------------|-----------------------|-------------|-------------------------------------------|------------------------------------------|-----------|---------------|
| Memou       | (W/A)                 |             | Model Storage↓                            | Memory Consumption↓                      | DiT Time↓ | Speedup↑      |
|             | Hı                    | ınyuanVideo | 13B (CFG = 6.0,                           | $720 \times 1280 p, \texttt{frames} = 6$ | 60)       |               |
| Full Prec.  | 16/16                 | 100%        | 23.88GB                                   | 35.79GB                                  | 1264s     | 1.00×         |
| QuaRot      | 4/8                   | 100%        | 6.49GB                                    | 24.34GB                                  | 1149s     | 1.10×         |
| Q-VDiT      | 4/8                   | 100%        | 6.50GB                                    | 24.89GB                                  | 1155s     | $1.09 \times$ |
| DFT         | 16/16                 | 25%         | 23.88GB                                   | 40.11GB                                  | 792s      | 1.60×         |
| Jenga       | 16/16                 | 25%         | 23.88GB                                   | 36.92GB                                  | 846s      | $1.49 \times$ |
| SVG         | 16/16                 | 25%         | 23.88GB                                   | 40.10GB                                  | 786s      | $1.61 \times$ |
| SVG         | 16/16                 | 15%         | 23.88GB                                   | 40.10GB                                  | 707s      | $1.79 \times$ |
| QuantSparse | 4/8                   | 25%         | 6.49GB <sub>↓3.68×</sub>                  | 27.02GB <sub>↓1.32×</sub>                | 731s      | 1.73×         |
| QuantSparse | 4/8                   | 15%         | 6.49GB <sub>↓<b>3</b>.68×</sub>           | 27.02GB <sub>↓1.32</sub> ×               | 671s      | 1.88×         |
|             |                       | Wan2.1 14   | B (CFG = 5.0, 720)                        | $\times~1280p, \texttt{frames} = 80)$    |           |               |
| Full Prec.  | 16/16                 | 100%        | 26.61GB                                   | 42.48GB                                  | 4031s     | 1.00×         |
| QuaRot      | 4/8                   | 100%        | 7.00GB                                    | 26.04GB                                  | 3425s     | 1.18×         |
| Q-VDiT      | 4/8                   | 100%        | 7.02GB                                    | 26.73GB                                  | 3457s     | $1.17 \times$ |
| DFT         | 16/16                 | 25%         | 26.61GB                                   | 44.86GB                                  | 3015s     | $1.34 \times$ |
| Jenga       | 16/16                 | 25%         | 26.61GB                                   | 42.62GB                                  | 3087s     | 1.31×         |
| SVĞ         | 16/16                 | 25%         | 26.61GB                                   | 44.07GB                                  | 2987s     | 1.35×         |
| SVG         | 16/16                 | 15%         | 26.61GB                                   | 44.07GB                                  | 2661s     | 1.51×         |
| QuantSparse | 4/8                   | 25%         | 7.00GB <sub>\$\square\$3.80\times\$</sub> | 28.14GB <sub>↓1.51×</sub>                | 2594s     | 1.55×         |
| QuantSparse | 4/8                   | 15%         | 7.00GB <sub>↓3.80</sub> ×                 | $28.14GB_{\downarrow 1.51 \times}$       | 2315s     | 1.74×         |

is a trade-off based on computational resource and all interval settings offer reasonable results and notable acceleration.

More ablation study about pooling stride s, salient token k, weight factor  $\lambda$ , and SVD rank r in Eq. 9 and Eq. 16 in Appendix Sec. H.

### 4.4 EFFICIENCY ANALYSIS

We present the deployment efficiency in Tab. 4. All the experiments are conducted on a single NVIDIA A800 80G GPU with CUDA 12.4. We use CUTLASS (Thakkar et al., 2023) on top of PyTorch for performing INT matrix multiplication. Existing quantization methods can bring higher model compression, but the effect of inference acceleration is limited. Sparse attention brings significant acceleration, but has almost no model compression, and even brings more memory consumption. QuantSparse combines the advantages of both quantization and sparse attention, bringing significant model compression and acceleration. For Wan2.1-14B (Wan

<span id="page-9-0"></span>Table 5: Ablation study of cache-fresh interval and attention density on W4A8 Wan2.1-14B.

| -           | PSNR <sub>↑</sub> | SSIM↑      | LPIPS↓ | Speedup↑      |
|-------------|-------------------|------------|--------|---------------|
|             | Interv            | al Analys  | is     |               |
| Interval=3  | 18.86             | 0.631      | 0.243  | 1.65×         |
| Interval=4  | 18.48             | 0.617      | 0.260  | $1.69 \times$ |
| Interval=5  | 18.22             | 0.605      | 0.272  | $1.74 \times$ |
| Interval=6  | 17.72             | 0.566      | 0.321  | 1.76×         |
|             | Dens              | ity Analys | is     |               |
| Density=25% | 18.72             | 0.630      | 0.240  | 1.55×         |
| Density=20% | 18.45             | 0.622      | 0.252  | 1.63×         |
| Density=15% | 18.22             | 0.605      | 0.272  | $1.74 \times$ |
| Density=10% | 17.73             | 0.589      | 0.288  | $1.80 \times$ |

et al., 2025), QuantSparse (15% density) brings  $3.80\times$  storage compression,  $1.51\times$  memory saving, and  $1.74\times$  end-to-end acceleration. We further report the calibration resource consumption in Appendix Sec. I and report the performance combined with other acceleration methods in Appendix Sec. J.

### 5 Conclusion

In this paper, we propose QuantSparse, a unified compression framework that effectively combines model quantization and sparse attention. To address the amplified attention shift, we propose Multi-Scale Salient Attention Distillation to efficiently align the attention shift. To address the intrinsic sparsity loss, we propose Second-Order Sparse Attention Reparameterization to utilize decomposed second-order residual for attention approximation. Extensive experiments shown that QuantSparse achieves lossless performance while bringing significant model compression and acceleration.

