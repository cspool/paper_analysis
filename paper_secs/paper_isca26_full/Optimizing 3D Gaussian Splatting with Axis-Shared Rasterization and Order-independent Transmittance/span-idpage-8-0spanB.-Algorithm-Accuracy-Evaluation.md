# <span id="page-8-0"></span>B. Algorithm Accuracy Evaluation

Render quality comparison. Our axis-shared rasterization is numerically identical to the original implementation; therefore, rendering quality differences arise solely from the MLPbased order-independent transmittance (OIT). Table IV reports the rendering quality of our MLP-based OIT and compares it with (i) the original 3DGS baseline and (ii) the state-of-theart sort-free weight-sum rendering method [18]. The weightsum method in [18] proposes several depth-based weighting functions; we report the best-performing variant, LC-WSR (without view-dependent opacity), for fair comparison. To further analyze the role of view information, we evaluate an ablation version of our model using depth-only input (denoted as OIT+d), while the full model incorporating both depth and view direction is denoted as OIT+d+view. As shown in Table IV, incorporating view information yields consistent improvements. PSNR increases from 26.17 (OIT+d) to 26.90 (OIT+d+view), accompanied by higher SSIM and lower LPIPS. Overall, our full model achieves a PSNR of 26.90, with only a minor 0.3 degradation compared to the original sorted baseline. SSIM remains nearly unchanged, and LPIPS is slightly improved. Compared with prior weightsum rendering [18], our approach consistently achieves better results across all three evaluation metrics. Fig. 12 visually compares our method with weight-sum rendering [18] and original 3DGS, as highlighted by the blue boxes, while the weight-sum method exhibits slight blending artifacts in these regions, our method better preserves local occlusion relationships and depth layering, producing results that more closely match the original 3DGS.

Methodological comparison with related works. Weightsum rendering [18] adopts a monotonic depth-based weighting

<span id="page-8-5"></span>

| Device                    | Tech. | Area | Power  | On-chip SRAM     | DRAM Bandwidth | Cores         |
|---------------------------|-------|------|--------|------------------|----------------|---------------|
| Jetson Orin Nano<br>(8GB) | 8 nm  | 200  | ~15 W  | ~3 MB<br>(L1+L2) | 68.2 GB/s      | 1024<br>CUDA  |
| RTX 3090                  | 8 nm  | 628  | 350 W  | 6 + 10.25 MB     | 936 GB/s       | 10496<br>CUDA |
| Ours                      | 28 nm | 3.85 | 1.64 W | 96 KB            | 38.4 GB/s      | 256 PEs       |

![](_page_8_Figure_10.jpeg)

<span id="page-8-4"></span>Fig. 12. Visual comparison among sort-free methods.

function. While computationally lightweight, it remains a handcrafted approximation that cannot fully capture complex Gaussian interactions in anisotropic 3DGS scenes. In contrast, our MLP-based OIT provides a data-driven representation. The MLP offers greater expressive capacity than prior methods and flexibly learns an optimized mapping for transmittance estimation. A key innovation of our work is the incorporation of view information into order-independent transmittance computation, thereby effectively leveraging the inherently view-dependent nature of 3DGS rendering. Moreover, the interaction between view direction and depth is effectively captured by the MLP, whereas such interactions are difficult to model using handcrafted depth-based functions. Table IV further demonstrates that incorporating view information (OIT+d+view) consistently outperforms the variant using only depth information (OIT+d).

**Advantages over related works.** (1) Higher accuracy. As shown in Table IV, our method consistently exhibits smaller degradation in PSNR and SSIM relative to weight-

<span id="page-8-3"></span> $\label{thm:equality} \textbf{TABLE IV}$  Image quality comparison with baseline and the sota solution.

| Scene           | Bicycle | Bonsai | Counter | Garden | Kitchen | Room   | Stump  | Avg.   |
|-----------------|---------|--------|---------|--------|---------|--------|--------|--------|
| Base. PSNR↑     | 23.71   | 29.66  | 27.14   | 26.30  | 28.86   | 29.21  | 25.62  | 27.21  |
| Weight-sum [18] | 23.05   | 26.65  | 24.96   | 24.67  | 26.24   | 28.11  | 24.37  | 25.43  |
| OIT+d           | 23.28   | 27.44  | 26.09   | 26.01  | 26.72   | 28.93  | 24.78  | 26.17  |
| OIT+d+view      | 23.87   | 28.80  | 26.83   | 26.25  | 28.34   | 28.97  | 25.23  | 26.90  |
| Base. SSIM ↑    | 0.6684  | 0.9223 | 0.8782  | 0.8333 | 0.9022  | 0.8930 | 0.7200 | 0.8309 |
| Weight-sum [18] | 0.6604  | 0.8772 | 0.8240  | 0.6251 | 0.8692  | 0.8811 | 0.6809 | 0.7700 |
| OIT+d           | 0.6643  | 0.8948 | 0.8668  | 0.8215 | 0.8745  | 0.8916 | 0.6826 | 0.8137 |
| OIT+d+view      | 0.6810  | 0.9081 | 0.8747  | 0.8276 | 0.8986  | 0.8966 | 0.6975 | 0.8263 |
| Base. LPIPS↓    | 0.3240  | 0.1623 | 0.2062  | 0.1232 | 0.1272  | 0.2171 | 0.2530 | 0.2017 |
| Weight-sum [18] | 0.2667  | 0.2037 | 0.2321  | 0.1930 | 0.1490  | 0.1964 | 0.2504 | 0.2122 |
| OIT+d           | 0.2620  | 0.1785 | 0.1860  | 0.1037 | 0.1451  | 0.1859 | 0.2490 | 0.1869 |
| OIT+d+view      | 0.2454  | 0.1581 | 0.1759  | 0.0969 | 0.1226  | 0.1844 | 0.2339 | 0.1739 |

sum approaches, while maintaining competitive LPIPS scores. (2) Higher training efficiency and practicality. Despite being MLP-based, our formulation remains lightweight, and it requires only approximately 30 minutes of additional training per scene within our training framework. In contrast, weightsum approaches typically require training from scratch with reduced learning rates to ensure stability, resulting in longer convergence times. (3) Lower hardware cost. Importantly, our MLP-based OIT is not merely an algorithmic alternative; it is developed from a hardware-algorithm co-design perspective. Although it involves more MAC operations than weight-sum rendering, it reuses the MAC datapath and exponential units already required for rasterization; therefore, MLP inference incurs nearly negligible additional hardware overhead. For example, to match the throughput of our MLP-based OIT within the same accelerator, deploying weight-sum rendering would require additional division units, incurring an extra 0.363 mm<sup>2</sup> area and 341 mW power, whereas our reconfiguration introduces only 0.147 mm<sup>2</sup> area and 88 mW power overhead.

