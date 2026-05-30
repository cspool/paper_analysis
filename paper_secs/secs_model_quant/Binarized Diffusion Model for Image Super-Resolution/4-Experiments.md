# 4 Experiments

### <span id="page-6-2"></span>4.1 Experimental Settings

**Data and Evaluation.** We take DIV2K [59] and Flickr2K [33] as the training dataset. Meanwhile, we evaluate the models with four benchmark datasets: Set5 [2], B100 [42], Urban100 [18], and Manga109 [43]. Experiments are conducted under two upscale factors:  $\times 2$  and  $\times 4$ . The LR images are generated from HR images through bicubic downsampling degradation. We apply two distortion-based metrics, PSNR and SSIM [64], which are calculated on the Y channel (*i.e.*, luminance) of the YCbCr space. We also use the perceptual metrics: LPIPS [12]. Following previous work [66, 49], the total parameters (**Params**) of the model are calculated as Params=Params $^b$ +Params $^f$ , and the overall operations (**OPs**) as OPs=OPs $^b$ +OPs $^f$ , where Params $^b$ =Params $^f$ /32 and OPs $^b$ =OPs $^f$ /64; the superscripts f and b denote full-precision and binarized modules, respectively.

**Implementation Details.** For the noise estimation network, we set the encoder and decoder level to 4. In each level of the encoder, we use 2 Residual Blocks (ResBlocks), while in the decoder, we apply 3 ResBlocks. The number of channels C is set to 64. We set the number of bias and RPReLU in TaR and TaA as K=5. For the diffusion model, we set the total number of timesteps to T=2,000. During the inference phase, we employ the DDIM sampler with 50 timesteps.

**Training Settings.** We train models with the  $\mathcal{L}_1$  loss. We employ the Adam optimizer [22] with  $\beta_1$ =0.9 and  $\beta_2$ =0.99, and a learning rate of  $1\times10^{-4}$ . The batch size is set to 16, with a total of 1,000K iterations. Input LR images are randomly cropped to size 64×64. Random rotations of 90°, 180°, and 270° and horizontal flips are used for data augmentation. Our model is implemented based on PyTorch [47] with two Nvidia A100-80G GPUs.

### <span id="page-6-0"></span>4.2 Ablation Study

In this section, we conduct all experiments on the  $\times 2$  scale factor. We apply DIV2K [59] and Flickr2K [33] as the training dataset, and Manga109 [43] as the testing dataset. The training iterations are set to 500K. Other settings are the same as defined in Sec. 4.1. We test the computational complexity (*i.e.*, OPs) of one single sampling step on the output size  $3\times256\times256$ .

<span id="page-7-5"></span><span id="page-7-0"></span>

| Method     | Baseline | +Identity | +CP-Down&Up | +CS-Fusion | +TaR&TaA |
|------------|----------|-----------|-------------|------------|----------|
| Params (M) | 4.29     | 4.29      | 4.29        | 4.30       | 4.58     |
| OPs (G)    | 36.67    | 36.67     | 36.67       | 36.67      | 36.67    |
| PSNR (dB)  | 27.66    | 29.29     | 31.08       | 31.99      | 32.66    |
| LPIPS      | 0.0780   | 0.0658    | 0.0327      | 0.0261     | 0.0200   |

<span id="page-7-1"></span>

| Method    | Params (M) | OPs (G) | PSNR (dB) | LPIPS  |
|-----------|------------|---------|-----------|--------|
| Add       | 4.10       | 33.40   | 18.89     | 0.1695 |
| Concat    | 4.29       | 36.67   | 31.08     | 0.0327 |
| Split     | 4.30       | 36.67   | 29.67     | 0.0384 |
| CS-Fusion | 4.30       | 36.67   | 31.99     | 0.0261 |

#### (a) Break-down ablation.

<span id="page-7-3"></span>

| Method | TaR | TaA | Params (M) | Ops (G) | PSNR (dB) | LPIPS  |
|--------|-----|-----|------------|---------|-----------|--------|
| w/o    |     |     | 4.30       | 36.67   | 31.99     | 0.0261 |
| In     | ✓   |     | 4.37       | 36.67   | 29.27     | 0.0337 |
| Out    |     | ✓   | 4.51       | 36.67   | 29.13     | 0.0308 |
| All    | ✓   | ✓   | 4.58       | 36.67   | 32.66     | 0.0200 |

#### (b) Ablation on feature fusion.

<span id="page-7-4"></span>

| #Pair      | 1      | 2      | 5      |
|------------|--------|--------|--------|
| Params (M) | 4.30   | 4.37   | 4.58   |
| OPs (G)    | 36.67  | 36.67  | 36.67  |
| PSNR (dB)  | 31.99  | 32.42  | 32.66  |
| LPIPS      | 0.0261 | 0.0229 | 0.0200 |

(c) Ablation on time aware module (TaR and TaA).

(d) Numbers (#) of bias and RPReLU pair.

Table 1: Ablation study. We train models on DIV2K and Flickr2K, and evaluate on Manga109 (×2).

<span id="page-7-2"></span>![](_page_7_Figure_9.jpeg)

![](_page_7_Figure_10.jpeg)

![](_page_7_Figure_11.jpeg)

Figure 6: Activation distribution in the skip connection. Input 1(2):  $\mathbf{x}_1, \mathbf{x}_2$ . Sum:  $\mathbf{x}_1 + \mathbf{x}_2$ . Fusion 1(2):  $\mathbf{x}_1^{sh}, \mathbf{x}_2^{sh}$ .

Figure 7: Weights of biases  $\mathbf{b}^i$   $(i \in \{1, ..., 5\})$  in TaR.

**Break Down.** We first execute a break-down ablation on different components of our method. The results are listed in Tab. 1a. The baseline is established by using binarized convolution (BI-Conv) and Pixel-(Un)Shuffle for dimension scaling in the downsample, upsample, and fusion (skip connection) modules of the UNet. Meanwhile, the basic BI-Conv block (Fig. 5) is employed without the identity shortcut. The baseline performance is poor, with the PSNR of 27.66 dB. Then, we add identity shortcut, consistent-pixel-downsample (CP-Down) and upsample (CP-Up), channel-shuffle-fusion module (CS-Fusion), and timestep-aware redistribution (TaR) and activation function (TaA) in sequence. We can find that the performance gradually increases. Ultimately, the final model achieves gains of 5 dB in PSNR and 0.0580 in LPIPS, compared to the baseline.

Channel-Shuffle Fusion. We experiment on the fusion module for the skip connection. We attempt four methods: directly add two features (Add); concatenation and adjust dimension by binarized convolution (Concat); process each feature via binarized convolution and add them; and our proposed CS-Fusion. The results are shown in Tab. 1b. Due to the differences between features, direct addition (Add) can hardly work, even with convolution (Split). Moreover, since the concatenation changes the dimensions, the Method (Concat) also degrades the performance. In contrast, our proposed CS-Fusion, eliminates the distribution imbalances by channel fusion, thereby achieving effective fusion. The visualization in Fig. 6, further indicates that addition cannot fuse data with narrow value distributions, whereas channel shuffle can effectively integrate.

**Timestep-aware Module.** We conduct experiments on the time-aware redistribution (TaR) and activation function (TaA). Firstly, we experiment with the combinations of TaR and TaA in Tab. 1c. We find that effective improvements are only achieved when both TaR and TaA are employed. This may be because both input and output activation impact the learning of the binarized module. Then, in Tab. 1d, we experiment with the pair number (#Pair) of bias and RPReLU. The experiments show that 5 pairs already lead to effective improvements. Considering the additional parameters, we adopt 5 as the pair number in BI-DiffSR. Moreover, we present the weights of five learnable biases in the TaR (module position shown at the image top) in Fig. 7. The difference in weights indicates that TaR can effectively adapt to the varying activation distributions at different timesteps.

### 4.3 Comparison with State-of-the-Art Methods

We compare our proposed BI-DiffSR with recent binarization methods, including BNN [19], DoReFa [71], XNOR [50], IRNet [48], ReActNet [38], and BBCU [66]. To ensure a fair comparison, we set the parameters (Params) and complexity (OPs) of all binarization methods to be similar. We also compare our BI-DiffSR with the full-precision (FP) model, SR3 [54].

<span id="page-8-2"></span><span id="page-8-0"></span>

| Method                                                                                               | Scale                                  | Params<br>(M)                                        | Ops<br>(G)                                                  |       | Set5<br>PSNR SSIM |        | LPIPS PSNR SSIM | B100 |                                                                              | LPIPS PSNR SSIM | Urban100 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | LPIPS PSNR SSIM | Manga109 | LPIPS                          |
|------------------------------------------------------------------------------------------------------|----------------------------------------|------------------------------------------------------|-------------------------------------------------------------|-------|-------------------|--------|-----------------|------|------------------------------------------------------------------------------|-----------------|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|----------|--------------------------------|
| Bicubic<br>SR3 [54]                                                                                  | ×2<br>×2                               | N/A<br>55.41                                         | N/A                                                         |       |                   |        |                 |      |                                                                              |                 |          | 33.67 0.9303 0.1274 29.55 0.8431 0.2508 26.87 0.8403 0.2064 30.82 0.9349 0.1025<br>176.41 36.69 0.9513 0.0310 30.41 0.8683 0.0700 30.29 0.9060 0.0430 35.11 0.9682 0.0161                                                                                                                                                                                                                                                                                                                                      |                 |          |                                |
| BNN [19]<br>DoReFa [71]<br>XNOR [50]<br>IRNet [48]<br>ReActNet [38]<br>BBCU [66]<br>BI-DiffSR (ours) | ×2<br>×2<br>×2<br>×2<br>×2<br>×2<br>×2 | 4.78<br>4.78<br>4.78<br>4.78<br>4.85<br>4.82<br>4.58 | 37.93<br>37.93<br>37.93<br>37.93<br>37.93<br>37.75<br>36.67 |       |                   |        |                 |      | 13.97 0.5210 0.4529 13.73 0.4553 0.5784 12.75 0.4236 0.5575                  |                 |          | 16.43 0.6553 0.2662 16.11 0.5912 0.3972 15.09 0.5495 0.4055 12.35 0.4609 0.5047<br>32.34 0.8661 0.0782 27.94 0.7548 0.1665 27.47 0.8225 0.1153 31.99 0.9428 0.0326<br>32.55 0.9340 0.0446 27.76 0.8199 0.1115 26.34 0.8452 0.0913 23.89 0.7621 0.1820<br>34.30 0.9271 0.0351 28.36 0.8158 0.0943 27.43 0.8563 0.0731 32.16 0.9441 0.0379<br>34.31 0.9281 0.0393 28.39 0.8202 0.0905 28.05 0.8669 0.0620 32.88 0.9508 0.0272<br>35.68 0.9414 0.0277 29.73 0.8478 0.0682 28.97 0.8815 0.0522 33.99 0.9601 0.0172 | 9.29            |          | 0.3035 0.7489                  |
| Bicubic<br>SR3 [54]                                                                                  | ×4<br>×4                               | N/A<br>55.41                                         | N/A                                                         |       |                   |        |                 |      |                                                                              |                 |          | 28.43 0.8111 0.3398 25.95 0.6678 0.5244 23.14 0.6579 0.4729 24.90 0.7876 0.3210<br>176.41 31.03 0.8798 0.1127 26.11 0.6933 0.2247 25.52 0.7702 0.1438 28.77 0.8854 0.0646                                                                                                                                                                                                                                                                                                                                      |                 |          |                                |
| BNN [19]<br>DoReFa [71]<br>XNOR [50]<br>IRNet [48]<br>ReActNet [38]<br>BBCU [66]<br>BI-DiffSR (ours) | ×4<br>×4<br>×4<br>×4<br>×4<br>×4<br>×4 | 4.78<br>4.78<br>4.78<br>4.78<br>4.85<br>4.82<br>4.58 | 37.93<br>37.93<br>37.93<br>37.93<br>37.93<br>37.75<br>36.67 | 10.40 | 0.246             | 0.9855 | 9.78            |      | 12.21 0.3103 0.8310 12.30 0.2128 0.9519 11.30 0.2191 0.9592<br>0.1709 1.0793 | 8.79            |          | 0.1614 1.1186<br>28.06 0.8274 0.1381 25.25 0.6552 0.3101 23.13 0.6647 0.2564 23.84 0.7839 0.1559<br>15.52 0.3514 0.7548 16.38 0.3121 0.7072 15.23 0.3043 0.7068 11.82 0.2442 0.8354<br>29.23 0.8362 0.1472 23.56 0.5670 0.3339 22.32 0.6440 0.2276 25.32 0.7854 0.1721<br>25.44 0.7795 0.1650 21.46 0.5472 0.3206 20.52 0.6293 0.2290 23.02 0.7966 0.1496<br>29.63 0.8374 0.1109 25.84 0.6779 0.2754 24.11 0.7177 0.1823 26.95 0.8548 0.0889                                                                   | 8.96<br>7.52    |          | 0.1833 1.0117<br>0.1464 1.1169 |

Table 2: Quantitative comparison with state-of-the-art binarization methods. The best and second best results are coloured with red and blue. Our method surpasses current approaches.

<span id="page-8-1"></span>![](_page_8_Figure_2.jpeg)

Figure 8: Visual comparison (×4) in some challenge cases.

Quantitative Results. We provide the quantitative comparisons in Tab. [2.](#page-8-0) We test OPs of single-step sampling on the output size 3×256×256. Compared to other binarization methods, our BI-DiffSR achieves the best performance. Specifically, on Urban100 and Manga109 (×2), BI-DiffSR surpasses the second-best method, BBCU, with a PSNR gain of 0.92 and 1.11 dB, respectively. Moreover, compared to the full-precision model, SR3, our method achieves comparable or even better perceptual performance with only 8.3% Params and 20.8% OPs. For instance, BI-DiffSR achieves 93.6% LPIPS results of SR3 on Manga109. These results demonstrate the superiority of our method.

Visual Results. We present visual comparisons (×4) in Fig. [8.](#page-8-1) Previous binarization methods struggle to recover image details in challenging cases. In contrast, our method can restore clearer results with more texture details. Meanwhile, the difference between our BI-DiffSR and the full-precision model results is small. More visual results are provided in the supplementary material.

