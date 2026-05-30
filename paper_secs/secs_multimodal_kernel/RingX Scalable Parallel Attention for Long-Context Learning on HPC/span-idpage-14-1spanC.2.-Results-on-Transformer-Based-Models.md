# <span id="page-14-1"></span>C.2. Results on Transformer-Based Models

To evaluate the generalizability of MoDiff across different architectures, we conduct experiments on the Diffusion Transformer [\(Peebles & Xie,](#page-10-16) [2023\)](#page-10-16). Following PTQ4DiT [\(Wu et al.,](#page-10-21) [2024\)](#page-10-21), we use DiT-XL/2 as the baseline model. The experiments are performed on the ImageNet 256×256 dataset [\(Russakovsky et al.,](#page-10-17) [2015\)](#page-10-17) using tensor-wise dynamic quantization. We generate 10,000 images using 50 sampling steps for evaluation. As shown in Table [8,](#page-14-3) MoDiff consistently enhances generation quality under low activation bit widths.

<span id="page-14-3"></span>Table 8. The IS, FID, and sFID for ImageNet 256x256 with DiT-XL/2 under different precisions. The best performance is bolded.

| Methods               | Bits (W/A) | IS ↑  | FID ↓  | sFID ↓ |
|-----------------------|------------|-------|--------|--------|
| PTQ4DiT               | 8/8        | 36.91 | 54.80  | 89.60  |
| PTQ4DiT+MoDiff (Ours) |            | 37.37 | 53.76  | 89.53  |
| PTQ4DiT               | 8/6        | 3.41  | 200.26 | 373.71 |
| PTQ4DiT+MoDiff (Ours) |            | 36.74 | 54.74  | 88.49  |
| PTQ4DiT               | 8/4        | 1.45  | 271.87 | 207.59 |
| PTQ4DiT+MoDiff (Ours) |            | 17.23 | 90.91  | 102.07 |

#### <span id="page-15-0"></span>C.3. More Measurements on Generation Quality

In the main paper, we evaluate the quality of generated outputs using Inception Score (IS), Fréchet Inception Distance (FID), and sFID. Here, we further assess the performance of our method using precision and recall.

The results are presented in Table 9, Table 10, and Table 11. These results demonstrate that MoDiff effectively preserves precision and recall even at low activation bit levels. For instance, on CIFAR-10, LCQ+MoDiff achieves a precision of 0.58 and a recall of 0.50, whereas LCQ alone results in 0 for both metrics.

<span id="page-15-2"></span>Table 9. The Precision and Recall for CIFAR-10 with DDIM under different Bits. The best performance is **bolded**.

| Methods                                                    | Bits (W/A) | Precision                    | Recall                       | Bits (W/A) | Precision                    | Recall                       |
|------------------------------------------------------------|------------|------------------------------|------------------------------|------------|------------------------------|------------------------------|
| Full Prec. (Act)                                           | 8/32       | 0.65                         | 0.55                         | 4/32       | 0.64                         | 0.56                         |
| Q-Diff<br>Q-Diff+MoDiff (Ours)<br>LCQ<br>LCQ+MoDiff (Ours) | 8/8        | 0.65<br>0.65<br>0.67<br>0.66 | 0.55<br>0.56<br>0.59<br>0.59 | 4/8        | 0.66<br>0.65<br>0.67<br>0.67 | 0.58<br>0.58<br>0.57<br>0.55 |
| Q-Diff<br>Q-Diff+MoDiff (Ours)<br>LCQ<br>LCQ+MoDiff (Ours) | 8/6        | 0.46<br>0.66<br>0.67<br>0.66 | 0.47<br>0.57<br>0.58<br>0.58 | 4/6        | 0.47<br>0.65<br>0.67<br>0.67 | 0.44<br>0.59<br>0.57<br>0.56 |
| Q-Diff<br>Q-Diff+MoDifff(Ours)<br>LCQ<br>LCQ+MoDiff (Ours) | 8/4        | 0.08<br>0.54<br>0.47<br>0.67 | 0.00<br>0.53<br>0.44<br>0.59 | 4/4        | 0.05<br>0.53<br>0.48<br>0.67 | 0.00<br>0.55<br>0.43<br>0.57 |
| Q-Diff<br>Q-Diff+MoDiff (Ours)<br>LCQ<br>LCQ+MoDiff (Ours) | 8/3        | 0.00<br>0.45<br>0.33<br>0.66 | 0.00<br>0.39<br>0.08<br>0.59 | 4/3        | 0.00<br>0.33<br>0.35<br>0.67 | 0.00<br>0.32<br>0.08<br>0.57 |
| Q-Diff<br>Q-Diff+MoDiff (Ours)<br>LCQ<br>LCQ+MoDiff (Ours) | 8/2        | 0.00<br>0.00<br>0.00<br>0.58 | 0.00<br>0.00<br>0.00<br>0.50 | 4/2        | 0.00<br>0.14<br>0.00<br>0.58 | 0.00<br>0.00<br>0.00<br>0.47 |

<span id="page-15-3"></span>Table 10. The Precision and Recall for Church with LDM-8 under different Bits. The best performance is bolded.

| Methods                  | Bits (W/A) | Precision    | Recall       | Bits (W/A) | Precision    | Recall       |
|--------------------------|------------|--------------|--------------|------------|--------------|--------------|
| Full Prec. (Act)         | 8/32       | 0.63         | 0.51         | 4/32       | 0.63         | 0.52         |
| LCQ<br>LCQ+MoDiff (Ours) | 8/8        | 0.62<br>0.63 | 0.47<br>0.53 | 4/8        | 0.62<br>0.63 | 0.46<br>0.53 |
| LCQ<br>LCQ+MoDiff (Ours) | 8/6        | 0.59<br>0.63 | 0.46<br>0.53 | 4/6        | 0.59<br>0.63 | 0.45<br>0.53 |
| LCQ<br>LCQ+MoDiff (Ours) | 8/4        | 0.03<br>0.63 | 0.14<br>0.53 | 4/4        | 0.02<br>0.63 | 0.07<br>0.5  |
| LCQ<br>LCQ+MoDiff (Ours) | 8/3        | 0.00<br>0.61 | 0.00<br>0.34 | 4/3        | 0.00<br>0.60 | 0.00<br>0.34 |

