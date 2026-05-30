# <span id="page-15-1"></span>D.1. Results on Other Weight Precision

In the main paper, we present results for 8-bit weight quantization on LSUN-Churches and LSUN-Bedroom for the page limitation. In this section, we extend our analysis to 4-bit weight quantization and observe consistent conclusions. As shown in Table 12 and Table 13, our method successfully maintains generation quality at 4/3 bits for Churches and 4/4 bits for Bedrooms. In contrast, LCQ experiences a significant performance drop.

<span id="page-16-2"></span>Table 11. The Precision and Recall for Bedroom with LDM-4 under different Bits. The best performance is bolded.

| Methods                  | Bits (W/A) | Precision    | Recall       | Bits (W/A) | Precision    | Recall       |
|--------------------------|------------|--------------|--------------|------------|--------------|--------------|
| Full Prec. (Act)         | 8/32       | 0.65         | 0.45         | 4/32       | 0.66         | 0.41         |
| LCQ<br>LCQ+MoDiff (Ours) | 8/8        | 0.65<br>0.60 | 0.45<br>0.51 | 4/8        | 0.68<br>0.62 | 0.41<br>0.47 |
| LCQ<br>LCQ+MoDiff (Ours) | 8/6        | 0.17<br>0.59 | 0.13<br>0.51 | 4/6        | 0.63<br>0.62 | 0.43<br>0.47 |
| LCQ<br>LCQ+MoDiff (Ours) | 8/4        | 0.00<br>0.40 | 0.00<br>0.17 | 4/4        | 0.00<br>0.46 | 0.00<br>0.22 |

<span id="page-16-3"></span>Table 12. The IS, FID, sFID, and GBOPs for LSUN-Church with LDM under 4-bit weight quantization. The best performance is bolded.

| Methods                  | Bits (W/A) | GBops | FID ↓           | sFID ↓          |
|--------------------------|------------|-------|-----------------|-----------------|
| Full Prec. (Act)         | 8/32       | 5015  | 4.03            | 10.89           |
| LCQ<br>LCQ+MoDiff (Ours) | 8/8        | 1254  | 4.02<br>3.99    | 11.53<br>10.06  |
| LCQ<br>LCQ+MoDiff (Ours) | 8/6        | 940   | 4.50<br>3.89    | 12.90<br>10.12  |
| LCQ<br>LCQ+MoDiff (Ours) | 8/4        | 627   | 198.37<br>34.02 | 161.03<br>10.59 |
| LCQ<br>LCQ+MoDiff (Ours) | 8/3        | 470   | 341.62<br>12.05 | 407.68<br>35.29 |

<span id="page-16-4"></span>Table 13. The IS, FID, sFID, and GBOPs for LSUN-Bedrooms with LDM under 4-bit weight quantization. The best performance is bolded.

| Methods                  | Bits (W/A) | GBops | FID ↓           | sFID ↓          |
|--------------------------|------------|-------|-----------------|-----------------|
| Full Prec.               | 8/32       | 25560 | 3.45            | 8.45            |
| LCQ<br>LCQ+MoDiff (Ours) | 8/8        | 6390  | 3.61<br>3.57    | 8.65<br>8.44    |
| LCQ<br>LCQ+MoDiff (Ours) | 8/6        | 4609  | 64.17<br>3.57   | 63.18<br>6.53   |
| LCQ<br>LCQ+MoDiff (Ours) | 8/4        | 3195  | 372.30<br>27.88 | 262.11<br>77.85 |

## <span id="page-16-0"></span>D.2. Results on Tensor-Wise Quantization

In our main experiments, we present results using dynamic channel-wise quantization (LCQ). In this section, we extend our analysis to dynamic tensor-wise quantization (LTQ), which is more hardware-friendly. We conduct experiments on CIFAR-10 using DDIM, while continuing to use Q-Diffusion checkpoints for weight quantization. As shown in Table [14,](#page-17-1) our MoDiff framework is also effective for LTQ. However, the minimum activation bit-width achievable with LTQ is higher than that of LCQ. This is because tensor-wise quantization operates on higher-dimensional data, making accurate quantization more challenging.

## <span id="page-16-1"></span>D.3. Results on More Samplers

In the main paper, we demonstrate that MoDiff generalizes to the DDPM sampler. Here, we further show its applicability to additional solvers. Specifically, we perform tensor-wise dynamic quantization using DPM-Solver-2 [\(Lu et al.,](#page-9-18) [2022\)](#page-9-18) on CIFAR-10 with 20 sampling steps. Additionally, we evaluate MoDiff with the PLMS solver using 50 steps on Stable Diffusion with the MS-COCO 2014 dataset [\(Liu et al.,](#page-9-1) [2022\)](#page-9-1). As shown in Table [15](#page-17-2) and Table [7,](#page-14-2) MoDiff consistently

#### Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization

<span id="page-17-1"></span>Table 14. The IS, FID, sFID, and GBOPs for CIFAR-10 with DDIM using tensor-wise quantization under different precisions. The best performance is bolded.

| Methods                  | Bits (W/A) | IS ↑         | FID ↓            | sFID ↓          | Bits (W/A) | IS ↑         | FID ↓            | sFID ↓          |
|--------------------------|------------|--------------|------------------|-----------------|------------|--------------|------------------|-----------------|
| Full Prec. (Act)         | 8/32       | 9.00         | 4.24             | 4.41            | 4/32       | 8.78         | 5.09             | 5.19            |
| LTQ<br>LTQ+MoDiff (Ours) | 8/8        | 9.08<br>9.04 | 4.19<br>4.21     | 4.40<br>4.37    | 4/8        | 8.80<br>8.76 | 5.02<br>5.05     | 5.21<br>5.16    |
| LTQ<br>LTQ+MoDiff (Ours) | 8/6        | 8.98<br>9.09 | 9.93<br>4.00     | 8.69<br>4.27    | 4/6        | 8.89<br>8.80 | 9.96<br>5.04     | 8.07<br>4.42    |
| LTQ<br>LTQ+MoDiff(Ours)  | 8/4        | 2.27<br>8.37 | 306.06<br>28.19  | 94.28<br>19.90  | 4/4        | 2.37<br>8.35 | 294.88<br>26.17  | 90.91<br>18.94  |
| LTQ<br>LTQ+MoDiff (Ours) | 8/2        | 1.19<br>4.26 | 457.25<br>186.04 | 165.85<br>86.73 | 4/2        | 1.19<br>3.29 | 457.11<br>146.52 | 165.61<br>87.78 |

improves FID scores across different solvers.

<span id="page-17-2"></span>Table 15. The FID on CIFAR-10 with DDIM using DPM solver under different precisions. The best performance is bolded.

| Methods                  | Bits (W/A) | FID ↓           |
|--------------------------|------------|-----------------|
| DPM<br>DPM+MoDiff (Ours) | 8/8        | 3.92<br>3.91    |
| DPM<br>DPM+MoDiff (Ours) | 8/6        | 10.82<br>3.91   |
| DPM<br>DPM+MoDiff (Ours) | 8/4        | 299.72<br>26.54 |

## <span id="page-17-0"></span>D.4. Results on Fewer Generation Steps

<span id="page-17-3"></span>To demonstrate that MoDiff remains effective with fewer generation steps, we conduct experiments on CIFAR-10 using the DDIM sampler with only 20 steps. Tensor-wise dynamic quantization is applied throughout. As shown in Table [16,](#page-17-3) MoDiff maintains strong performance even under this reduced-step setting.

Table 16. FID on CIFAR-10 using the DDIM sampler in the ablation study of fewer steps.

| Methods                  | Bits (W/A) | FID ↓           |
|--------------------------|------------|-----------------|
| LTQ<br>LTQ+MoDiff (Ours) | 8/8        | 6.93<br>6.90    |
| LTQ<br>LTQ+MoDiff (Ours) | 8/6        | 20.28<br>6.75   |
| LTQ<br>LTQ+MoDiff (Ours) | 8/4        | 297.21<br>22.12 |

A line of research has focused on distilling diffusion models into few-step variants, which can achieve comparable generation quality within significantly fewer sampling steps. To evaluate the generalizability of MoDiff in this setting, we conduct experiments with MixDQ [\(Zhao et al.,](#page-10-13) [2024\)](#page-10-13), a method specifically designed for few-step diffusion. We use SDXL-Turbo as the backbone and apply 2, 4, and 8 sampling steps for image generation on the MS-COCO 2014 dataset [\(Lin et al.,](#page-9-14) [2014\)](#page-9-14), generating 10,000 images for FID computation. As shown in Table [17,](#page-18-2) our method is compatible with MixDQ and further improves performance in the few-step diffusion regime. The performance indicates that it is more challenging to lower the activation bit for SDXL-Turbo.

<span id="page-18-2"></span>Table 17. FID on MS-COCO using SDXL-Turbo and MixDQ across different generation steps. The best performance is bolded.

| Step | Bits(W/A) | MixDQ  | MixDQ+MoDiff |
|------|-----------|--------|--------------|
|      | 8/8       | 46.48  | 46.30        |
| 2    | 8/6       | 318.68 | 193.17       |
|      | 8/4       | 304.77 | 192.65       |
|      | 8/8       | 44.29  | 44.74        |
| 4    | 8/6       | 318.57 | 191.59       |
|      | 8/4       | 325.68 | 192.74       |
|      | 8/8       | 44.61  | 43.30        |
| 8    | 8/6       | 347.75 | 210.38       |
|      | 8/4       | 348.75 | 212.68       |

