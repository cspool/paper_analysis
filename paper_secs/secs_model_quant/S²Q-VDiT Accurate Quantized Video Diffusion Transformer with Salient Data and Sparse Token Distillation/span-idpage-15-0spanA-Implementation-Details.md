# <span id="page-15-0"></span>A Implementation Details

In the main experiment, we use 10 random prompts for generating the candidate calibration samples. We finally selected 40 samples for post-training quantization for all methods. For our method, we use a channel-wise scale used in [\[55,](#page-13-8) [62,](#page-13-5) [54\]](#page-13-4) and a rotation matrix used in [\[47\]](#page-12-14) for linear quantization. We further use a learnable threshold for clipping the weight and activation min-max value as prior work [\[30,](#page-11-5) [18,](#page-11-10) [47\]](#page-12-14). We also use GPTQ weight quantizer [\[13\]](#page-10-11) for our experiment, following prior work [\[2\]](#page-10-4). We conduct all the experiments on a single NVIDIA A800 GPU.

For optimization, we train the diag-balancing scale, rotation-based matrix, and learnable clipping threshold following the layer-wise post-training quantization framework as prior works [\[30,](#page-11-5) [54\]](#page-13-4). We use 30 samples and train 15 epochs for each layer. We use AdamW optimizer and cosine learning rate scheduler. For the diag-balancing scale and rotation-based matrix, we use a learning rate of 5e-3. For the learnable clipping threshold, we use a learning rate of 5e-2.

For deployment, we absorb all weight quantization parameters as prior works [\[54,](#page-13-4) [55,](#page-13-8) [62\]](#page-13-5), which brings no extra burden. For activation quantization, we apply online dynamic quantization following [\[62,](#page-13-5) [1\]](#page-10-10).

### <span id="page-15-1"></span>B More Ablation on Hessian-aware Salient Data Selection

<span id="page-15-2"></span>Table 6: Performance of both 4-bit weight and activation quantization on CogVideoX-2B under three random seeds.

| Method    | Imaging                       | Aesthetic    | Motion       | Dynamic      | BG           | Subject      | Scene        | Overall      |
|-----------|-------------------------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
|           | Quality<br>Quality<br>Smooth. |              | Degree       | Consist.     | Consist.     | Consist.     | Consist.     |              |
| -         | 58.69                         | 55.25        | 97.95        | 50.00        | 96.40        | 94.30        | 33.79        | 25.91        |
| ATOS      | 51.65±(1.76)                  | 49.79±(0.59) | 98.09±(0.16) | 29.17±(3.40) | 95.82±(0.35) | 93.24±(0.19) | 29.94±(1.35) | 24.31±(0.37) |
| ATDS      | 50.63±(0.81)                  | 50.13±(0.25) | 98.05±(0.11) | 29.63±(2.62) | 95.94±(0.16) | 93.16±(0.41) | 30.98±(2.14) | 24.11±(0.27) |
| DTDS      | 50.66±(1.04)                  | 50.33±(0.19) | 98.03±(0.14) | 31.48±(4.58) | 96.01±(0.16) | 93.07±(0.18) | 30.47±(1.77) | 24.75±(0.25) |
| DS        | 52.73±(0.98)                  | 50.62±(0.81) | 98.15±(0.19) | 31.75±(2.73) | 96.06±(0.18) | 93.29±(0.15) | 31.38±(0.98) | 24.78±(0.22) |
| QS        | 52.34±(0.85)                  | 51.17±(0.23) | 98.11±(0.12) | 32.01±(2.97) | 96.10±(0.17) | 93.57±(0.19) | 31.86±(0.90) | 24.79±(0.23) |
| SDS(Ours) | 52.95±(0.69)                  | 51.58±(0.11) | 98.16±(0.09) | 32.87±(2.36) | 96.13±(0.15) | 93.89±(0.17) | 32.75±(0.77) | 24.84±(0.26) |

In this section, we investigate the random seed influence on the quantization performance of different calibration datasets mentioned in Sec. [3.2](#page-3-2) and Sec. [4.4.](#page-8-4) We compare our proposed Hessian-aware Salient Data Selection (SDS) with All Timesteps from One Prompt (ATOP), All Timesteps from Five Prompts (ATFP), and Random Timesteps from Five Prompts (RTFP) using three different random seeds. We further decoupled SDS into Diffusion Salience (DS) in Eq. [\(3\)](#page-3-3) and Quantization Salience (QS) in Eq. [\(6\)](#page-4-4) and reported the performance. We present the average results and variance in Tab. [6.](#page-15-2)

Other straightforward sampling methods have lower average performance and larger variances, proving the influence of random seeds in these random sampling methods. Using our proposed diffusion salience (DS) or quantization salience (QS) can all improve the performance and reduce the impact of random seeds. Only using DS and QS can improve Scene Consistency to over 31 with variances less than 1, while other random sampling methods cannot achieve. By jointly considering two saliences, Hessian-aware Salient Data Selection (SDS) can achieve the best quantization performance with minimal impact from randomness. SDS achieved an average Imaging Quality of 52.95 with only 0.69 variance, while the random sampling only achieved the best average of 51.65 with 1.67 variance.

### C Detailed Description of Selected Evaluation Metrics

