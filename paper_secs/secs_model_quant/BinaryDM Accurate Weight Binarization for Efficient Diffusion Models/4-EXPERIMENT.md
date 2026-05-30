# 4 EXPERIMENT

We conduct experiments on various datasets, including CIFAR-10 32 × 32 [\(Krizhevsky et al., 2009\)](#page-11-14), LSUN-Bedrooms 256 × 256 [\(Yu et al., 2015\)](#page-13-4), LSUN-Churches 256 × 256 [\(Yu et al., 2015\)](#page-13-4), FFHQ 256 × 256 [\(Karras et al., 2019\)](#page-11-15) and ImageNet 256 × 256 [\(Deng et al., 2009\)](#page-10-12), for both unconditional and conditional image generation tasks over DDIM and LDM. The evaluation metrics used in our study encompass Inception Score (IS), Fréchet Inception Distance (FID) [\(Heusel et al., 2017\)](#page-10-13), Sliding Fréchet Inception Distance (sFID) [\(Salimans et al., 2016\)](#page-12-14), and Precision-and-Recall. We implement and evaluate the DMs binarized by our BinaryDM and the baseline presented in Section [3.1,](#page-2-1) where LSQ [\(Esser et al., 2019\)](#page-10-10) is employed uniformly as activations quantizers. Several SOTA quantization methods for DMs with 2∼8 bits weights are also considered [\(He et al., 2023;](#page-10-6) [Li et al., 2023a](#page-11-7)[;b;](#page-11-4) [So](#page-12-15) [et al., 2024\)](#page-12-15). Detailed settings are presented in Appendix [B.1.](#page-14-1)

#### 4.1 MAIN RESULTS

Unconditional Generation. We first conduct experiments on the CIFAR-10 dataset. As shown in Table [1,](#page-6-0) the binarized DM baseline suffers a severe breakdown in this low-resolution scenario, while our method significantly recovers the performance. Under the W1A4 bit-width, BinaryDM surpasses

<span id="page-7-0"></span>Table 2: Results for LDM on multiple datasets in unconditional generation by DDIM with 100 steps.

| Model | Dataset       | Method      | #Bits | Size(MB) | FID↓   | sFID↓  | Precision↑ | Recall↑ |
|-------|---------------|-------------|-------|----------|--------|--------|------------|---------|
|       |               | FP          | 32/32 | 1045.4   | 3.09   | 7.08   | 65.82      | 45.36   |
|       |               | LSQ         | 2/32  | 69.8     | 7.49   | 12.79  | 64.02      | 37.60   |
|       |               | Baseline    | 1/32  | 35.8     | 8.43   | 13.11  | 65.45      | 29.88   |
|       |               | BinaryDM    | 1/32  | 35.8     | 6.99   | 12.15  | 67.51      | 36.80   |
|       |               | Q-Diffusion | 2/8   | 69.8     | 62.01  | 33.56  | 16.48      | 14.12   |
|       |               | LSQ         | 2/8   | 69.8     | 6.48   | 11.66  | 62.55      | 38.92   |
| LDM-4 | LSUN-Bedrooms | Baseline    | 1/8   | 35.8     | 9.37   | 12.10  | 64.36      | 30.76   |
|       | 256 × 256     | BinaryDM    | 1/8   | 35.8     | 6.51   | 11.67  | 65.80      | 35.28   |
|       |               | Q-Diffusion | 4/4   | 134.9    | 427.46 | 277.22 | 0.00       | 0.00    |
|       |               | EfficientDM | 4/4   | 134.9    | 10.60  | -      | -          | -       |
|       |               | LSQ         | 2/4   | 69.8     | 12.95  | 12.79  | 55.97      | 34.30   |
|       |               | Baseline    | 1/4   | 35.8     | 10.87  | 15.46  | 64.05      | 26.50   |
|       |               | TDQ         | 1/4   | 35.8     | 11.28  | 12.80  | 55.14      | 27.32   |
|       |               | ReActNet    | 1/4   | 35.8     | 10.23  | 13.02  | 61.43      | 29.68   |
|       |               | Q-DM        | 1/4   | 35.8     | 9.99   | 11.96  | 57.62      | 29.30   |
|       |               | INSTA-BNN   | 1/4   | 35.8     | 9.42   | 12.39  | 60.05      | 31.08   |
|       |               | BI-DiffSR   | 1/4   | 35.8     | 8.58   | 11.81  | 62.61      | 30.86   |
|       |               | BinaryDM    | 1/4   | 35.8     | 7.74   | 10.80  | 64.71      | 32.98   |
|       |               | FP          | 32/32 | 1125.2   | 4.82   | 17.66  | 75.18      | 46.80   |
|       |               | LSQ         | 2/32  | 74.1     | 8.16   | 19.87  | 74.98      | 35.76   |
|       |               | Baseline    | 1/32  | 38.1     | 9.91   | 17.94  | 74.89      | 26.88   |
|       |               | BinaryDM    | 1/32  | 38.1     | 8.14   | 17.44  | 75.51      | 34.56   |
|       |               | Q-Diffusion | 2/8   | 74.1     | 201.23 | 238.70 | 2.39       | 8.60    |
|       | LSUN-Churches | LSQ         | 2/8   | 74.1     | 8.11   | 19.25  | 77.04      | 34.98   |
| LDM-8 | 256 × 256     | Baseline    | 1/8   | 38.1     | 10.94  | 16.95  | 74.30      | 25.66   |
|       |               | BinaryDM    | 1/8   | 38.1     | 8.63   | 15.13  | 77.74      | 33.48   |
|       |               | EfficientDM | 4/4   | 144.2    | 14.34  | -      | -          | -       |
|       |               | Q-Diffusion | 4/4   | 144.2    | 198.35 | 184.43 | 5.48       | 0.12    |
|       |               | LSQ         | 2/4   | 74.1     | 10.00  | 19.08  | 74.93      | 25.80   |
|       |               | Baseline    | 1/4   | 38.1     | 12.98  | 21.55  | 70.78      | 25.30   |
|       |               | BinaryDM    | 1/4   | 38.1     | 9.91   | 18.04  | 73.72      | 29.96   |
|       |               | FP          | 32/32 | 1045.4   | 6.64   | 14.16  | 76.88      | 50.82   |
|       |               | Q-Diffusion | 4/32  | 134.9    | 11.60  | 10.30  | -          | -       |
|       |               | Baseline    | 1/32  | 35.8     | 10.49  | 11.56  | 72.64      | 39.62   |
|       |               | BinaryDM    | 1/32  | 35.8     | 8.70   | 9.68   | 73.92      | 42.22   |
| LDM-4 | FFHQ          | Q-Diffusion | 8/8   | 265.0    | 10.87  | 10.01  | -          | -       |
|       | 256 × 256     | Q-Diffusion | 4/8   | 134.9    | 11.45  | 9.06   | -          | -       |
|       |               | Baseline    | 1/8   | 35.8     | 10.79  | 10.77  | 73.20      | 41.70   |
|       |               | BinaryDM    | 1/8   | 35.8     | 9.58   | 10.74  | 74.48      | 41.75   |
|       |               | Baseline    | 1/4   | 35.8     | 15.07  | 12.48  | 74.34      | 35.12   |
|       |               | BinaryDM    | 1/4   | 35.8     | 12.34  | 11.18  | 74.83      | 38.09   |

the binarized baseline by 9.46% in IS metrics on the CIFAR-10 and outperforms the LSQ under W2A4, where the latter involves several times of computation and storage.

Our LDM experiments encompass the evaluation of LDM-4 on LSUN-Bedrooms and FFHQ datasets, along with the assessment of LDM-8 on the LSUN-Churches dataset. The experiments utilized the DDIM sampler with 100 steps, and the detailed outcomes are presented in Table [2.](#page-7-0) We showcase results across various activation bit widths in the context of weight binarization, comparing them with the outcomes of some quantization methods at higher bit settings. The conventional binary baseline method exhibits subpar performance in the LDM context and experiences a further decline in the W1A4 experimental setup, particularly noticeable in the LSUN-Bedrooms dataset. In contrast, BinaryDM significantly enhances the generation quality, especially for LDM-4, exhibiting consistent performance across different activation bit settings. Notably, when compressing from W1A32 to W1A4 on the LSUN-Bedrooms dataset, the FID increased by a mere 0.75 for BinaryDM, showcasing its robustness. From the evaluation results of LDM-4 on FFHQ datasets, it can be observed that BinaryDM outperforms all other methods under various settings in terms of sFID, even surpassing W8A8 Q-Diffusion with a bit-width of W1A8. Moreover, BinaryDM demonstrates more significant

| Sampler    | Method   | #Bits | IS↑    | FID↓  | sFID↓ | Prec.↑ |
|------------|----------|-------|--------|-------|-------|--------|
|            | FP       | 32/32 | 235.84 | 12.96 | 25.99 | 92.63  |
|            | Baseline | 1/32  | 197.85 | 11.50 | 23.44 | 84.83  |
|            | BinaryDM | 1/32  | 215.55 | 10.86 | 21.10 | 88.43  |
| DDIM       | Baseline | 1/8   | 203.90 | 11.35 | 25.49 | 85.78  |
|            | BinaryDM | 1/8   | 211.43 | 11.23 | 24.12 | 88.09  |
|            | Baseline | 1/4   | 187.70 | 11.51 | 20.77 | 84.13  |
|            | BinaryDM | 1/4   | 208.42 | 10.78 | 20.40 | 87.61  |
|            | FP       | 32/32 | 247.38 | 13.54 | 18.85 | 94.22  |
|            | Baseline | 1/32  | 211.69 | 11.23 | 21.32 | 86.16  |
|            | BinaryDM | 1/32  | 226.86 | 11.00 | 19.01 | 91.17  |
| PLMS       | Baseline | 1/8   | 205.58 | 12.78 | 21.57 | 84.07  |
|            | BinaryDM | 1/8   | 225.18 | 11.33 | 19.18 | 90.35  |
|            | Baseline | 1/4   | 193.11 | 11.08 | 23.21 | 81.40  |
|            | BinaryDM | 1/4   | 218.06 | 10.36 | 18.85 | 88.74  |
|            | FP       | 32/32 | 242.27 | 13.10 | 19.82 | 93.53  |
|            | Baseline | 1/32  | 203.98 | 11.22 | 23.49 | 83.52  |
|            | BinaryDM | 1/32  | 214.91 | 11.07 | 20.61 | 87.71  |
| DPM-Solver | Baseline | 1/8   | 188.21 | 12.83 | 25.01 | 80.14  |
|            | BinaryDM | 1/8   | 216.27 | 11.68 | 20.52 | 88.36  |
|            | Baseline | 1/4   | 178.47 | 11.67 | 26.72 | 77.27  |
|            | BinaryDM | 1/4   | 206.80 | 10.83 | 20.68 | 85.34  |

<span id="page-8-0"></span>Table 3: Results on ImageNet 256 × 256 in conditional generation by DDIM with 20 steps.

advantages at lower activation bit-widths, achieving accurate generation with an FID of 12.34 under 4-bit activation. BinaryDM even approaches the generation quality of the full-precision model, with specifically generated image examples provided in Appendix [B.3.](#page-19-0)

Conditional Generation. For conditional generation, the performance of our BinaryDM is evaluated on the ImageNet dataset with a resolution of 256×256, focusing on LDM-4. We employ three distinct samplers to generate images: DDIM, PLMS, and DPM-Solver. The results in Table [3](#page-8-0) underscore the remarkable effectiveness of our BinaryDM on DDIM, surpassing the baseline consistently across almost all evaluation metrics and even outperforming the full-precision diffusion model in several cases. The binarized DM baseline performs relatively stable in configurations W1A32 and W1A8 but significantly declines under W1A4, with the IS decreasing to 187.70 when using the DDIM sampler. In contrast, our BinaryDM maintains an IS of 208.42 in W1A4. Specifically, when utilizing the DPM-Solver sampler, the IS plummets to 178.47, and the sFID experiences a sharp increase to 26.72. In stark contrast, our binarized DM maintains consistently high performance, achieving a 206.80 IS and a 20.68 FID and outperforming the baseline in most scenarios.

#### 4.2 ABLATION STUDY

We perform comprehensive ablation studies for LDM-4 on the LSUN-Bedrooms dataset to evaluate the effectiveness of our proposed EBB and LRM, and the results are presented in Table [4.](#page-9-0)

The performance has shown significant recovery when first applying our EBB to binarized diffusion models, with the FID decreasing from 8.43 to 7.39. This confirms that the degradation in parameter representational capacity due to binarization is a primary performance bottleneck in the binarized DM baseline. Solving this representation degradation is a prerequisite for improving model performance. From a structural perspective, EBB provides binarized diffusion models with an initial state with a higher information capacity, alleviating the degradation of representational ability in the early stages and guiding QAT toward a more easily optimizable direction.

With the application of LRM on this basis, the generative capability of the resulting binarized diffusion models is further enhanced, with the FID decreasing to 6.99. This indicates that the low-rank mimicking scheme, designed from a feature-matching perspective, effectively utilizes the representational information of the full-precision model, achieving supervision and alignment of intermediate features and better guiding the optimization of the binarized diffusion models.

Further substantiating this view, the detailed ablation experiments in Appendix [B.2](#page-15-0) delve into an in-depth discussion of the specifics concerning EBB and LRM. Combining these two techniques in

| Table 4: Ablation results on LSUN-Bedrooms 256 × 256. |  |  |  |
|-------------------------------------------------------|--|--|--|
|-------------------------------------------------------|--|--|--|

<span id="page-9-0"></span>

| Method                  | #Bits                | FID↓                 | sFID↓                   | Prec.↑                  | Recall↑                 |
|-------------------------|----------------------|----------------------|-------------------------|-------------------------|-------------------------|
| FP                      | 32/32                | 3.09                 | 7.08                    | 65.82                   | 45.36                   |
| Vanilla<br>+EBB<br>+LRM | 1/32<br>1/32<br>1/32 | 8.43<br>7.39<br>6.99 | 13.11<br>12.34<br>12.15 | 65.45<br>65.98<br>67.51 | 29.88<br>35.84<br>36.80 |

BinaryDM can significantly enhance performance, emphasizing that a better optimization process can improve the quality of generation when ensuring accurate representation.

### 4.3 EFFICIENCY ANALYSIS

For inference, we demonstrate the size and OPs of BinaryDM under different activation bit-widths. The results in Table [5](#page-9-1) indicate that our DM can achieve up to 29.2× space savings while obtaining up to 15.2× acceleration during inference, fully harnessing the advantages of binary computation. BinaryDM achieves optimal inference efficiency while surpassing the performance of advanced methods with higher bit widths. The W1A1 BinaryDM achieves a lower FID compared to the W4A4 EfficientDM, while its model size and OPs are only 26.5% and 25.9% of the latter, respectively.

For training, while the training process for our binarized DM typically incurs higher overhead compared to post-training quantization methods, practical observations reveal that our approach offers productivity advantages across various models and datasets. As shown in Table [6,](#page-9-2) despite having a training time shorter than the calibration time required by Q-Diffusion, our method attains significantly superior generation quality, particularly at lower bits.

<span id="page-9-1"></span>Table 5: Inference efficiency of our proposed BinaryDM of LDM-4 on LSUN-Bedrooms 256 × 256.

| Model | Method         | #Bits | Size(MB) | OPs(×109) | FID↓   |
|-------|----------------|-------|----------|-----------|--------|
|       | Full-Precision | 4/4   | 1045.4   | 96.0      | 3.09   |
|       | Q-Diffusion    | 4/4   | 134.9    | 24.3      | 427.46 |
| LDM-4 | EfficientDM    | 4/4   | 134.9    | 24.3      | 10.60  |
|       | LSQ            | 2/4   | 69.8     | 12.3      | 12.95  |
|       | BinaryDM       | 1/4   | 35.8     | 6.3       | 7.74   |

Table 6: Training time-cost of BinaryDM compared to the advanced PTQ method.

<span id="page-9-2"></span>

| Dataset       | Method      | #Bits | Size(MB) | Time(h) | FID↓   |
|---------------|-------------|-------|----------|---------|--------|
| LSUN-Bedrooms | Q-Diffusion | 4/4   | 134.9    | 13.7    | 427.46 |
|               | BinaryDM    | 1/4   | 35.8     | 11.3    | 13.93  |
| LSUN-Churches | Q-Diffusion | 4/4   | 144.2    | 10.9    | 198.35 |
|               | BinaryDM    | 1/4   | 38.1     | 9.0     | 15.11  |

