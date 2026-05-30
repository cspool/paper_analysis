# 3 Methodology

#### 3.1 Problem Formulation

Model quantization aims to learn appropriate clipping ranges [\[36,](#page-12-5) [57,](#page-13-8) [67\]](#page-14-7) for each tensor (e.g., weights or activations) in order to minimize the discrepancy between the outputs of the full-precision model and the quantized model. Following previous work [\[36,](#page-12-5) [37,](#page-12-10) [57\]](#page-13-8), we use fake quantization [\[22\]](#page-11-9) to simulate the quantization process. Given a pre-learned clipping range [lb, ub] for a tensor x, the quantization and dequantization process is defined as follows:

$$x_{\text{clip}} = \text{clamp}(x, lb, ub) = \min(\max(x, lb), ub), \tag{1}$$

$$x_{\text{int}} = \text{round}\left(\frac{x_{\text{clip}} - lb}{\Delta}\right), \quad \Delta = \frac{ub - lb}{2^N - 1}, \quad \hat{x} = x_{\text{int}} \cdot \Delta + lb,$$
 (2)

Where xclip is the clipped tensor, xint ∈ {0, 1, . . . , 2 <sup>N</sup> − 1} is the quantized integer, ∆ is the quantization step size, and xˆ is the dequantized approximation of the original value. Linear and MatMul layers are the most computationally intensive components in Transformer-based architectures. We follow existing work [\[36,](#page-12-5) [38\]](#page-12-6) to focus quantization on these modules. Since the quantization function is non-differentiable due to rounding, we also adopt the Straight-Through Estimator (STE) [\[10\]](#page-10-13) during training to approximate gradients and enable end-to-end optimization under quantized settings. More details are in the Appendix.

#### 3.2 Observations and Motivation

Observation 1: Inability to allocate varying representational capacity across frames. Previous studies have thoroughly examined the statistical properties of activations in Transformer-based architectures, uncovering long-tailed distributions and a mix of symmetric and asymmetric behaviors across layers [\[30,](#page-11-7) [36,](#page-12-5) [45,](#page-12-11) [67,](#page-14-7) [73\]](#page-14-10). However, these analyses are largely confined to single-frame scenarios. In the context of quantizing multi-frame video enhancement networks, it is essential

<span id="page-4-0"></span>![](_page_4_Figure_0.jpeg)

Figure 3: The overall framework of our proposed method.

to preserve the capability of the full-precision network to effectively integrate inter-frame texture information and motion cues—a challenge that vanilla methods fail to address.

Our analysis reveals that multi-frame networks perceive and process each frame in the input tensor differently. As shown in Figure 2(a), we collect per-frame activation statistics and observe significant disparities in activation distributions across frames. In particular, the value ranges (i.e., the minimum and maximum activation values) vary considerably, indicating that the network allocates representational capacity unevenly across frames. These differences are influenced by the model's frame-dependent attention dynamics. In particular, the network assigns different attention weights to each frame, resulting in varied activation distributions. Consequently, applying existing Transformer quantization methods [30, 36, 45, 67, 73], which typically assume a unified activation distribution, can be suboptimal for multi-frame models. Such methods overlook inter-frame variation, leading to inefficient quantization and increased error, ultimately degrading the quality of the output.

Observation 2: Over-reliance on full-precision teachers limits low-bit model learning. Low-bit quantization inevitably reduces the representational capacity of models, which brings significant challenges to multi-frame video enhancement tasks. Under low-bit quantization, both activation values and network precision degrade noticeably, making it difficult to maintain the clear motion trajectories and rich texture details required for these tasks. As shown in Figure 2(c), directly quantizing the network to 4-bit/2-bit and applying it without adaptation leads to severe artifacts and motion blur. The main reason for this degradation lies in the large quantization errors introduced when networks are directly quantized to low-bit precision. Furthermore, traditional methods often overlook the capacity gap between the teacher model (FP32) and the student model (e.g., 4-bit or 2-bit), relying solely on full-precision teachers for knowledge distillation [14, 36, 57]. This makes it difficult for the student model to learn high-quality mappings within its limited capacity, further increasing the challenge of achieving satisfactory performance under low-bit constraints.

#### 3.3 Proposed Method

Based on the above observations, we propose a two-stage quantization framework. The coarse stage uses Backtracking-based Multi-Frame Quantization (BMFQ) to initialize asymmetric bounds efficiently. The fine stage applies Progressive Multi-Teacher Distillation (PMTD) to refine bounds.

#### 3.3.1 Backtracking-based Multi-Frame Quantization

Motivated by Observation 1, we adopt a per-frame quantization strategy to handle frame-wise variations in activation distributions and representational capacity. Given a multi-frame activation tensor  $X \in \mathbb{R}^{N \times C \times H \times W}$ , where N denotes the number of frames, we independently search the clipping bounds for each frame  $X_i = X[i,:,:]$ , yielding frame-specific clipping parameters  $(lb_i, ub_i)$  for  $i = 1, \ldots, N$ . This strategy enables the quantizer to adapt to frame-specific activation

statistics, thereby improving quantization accuracy.

To robustly estimate the clipping bounds during the coarse stage, we formulate the selection of  $(lb_i, ub_i)$  as a constrained optimization problem that minimizes quantization-induced distortion over a percentile-based search space  $S_i$  derived from the empirical distribution of  $X_i$ :

<span id="page-5-0"></span>
$$(lb_i^*, ub_i^*) = \underset{(lb, ub) \in S_i}{\min} \ \mathbb{E}_{x \sim X_i} \left[ (x - Q_{lb, ub}(x))^2 \right],$$
 (3)

Here,  $Q_{lb,ub}(\cdot)$  denotes a uniform quantizer [39] with clipping range [lb, ub]. To mitigate the influence of outliers, we constrain the search space  $S_i$ using percentiles:  $lb \in [p_{0.1}(X_i), p_{10}(X_i)], ub \in$  $[p_{90}(X_i), p_{99.9}(X_i)]$ , where  $p_k(X)$  denotes the k-th percentile of the tensor X. To efficiently solve the optimization problem in Eq. (3), we introduce a Backtracking-based Bound Initialization (BTBI) algorithm. Starting from an initial estimate derived from the percentiles of  $X_i$ , the algorithm recursively explores candidate bounds by adjusting  $lb_i$  and  $ub_i$ within the search space  $S_i$ . At each step, it evaluates the quantization error and updates the optimal bounds if a better configuration is found. To avoid redundant searches, previously visited bounds are skipped. The algorithm backtracks to explore alternative adjustments when no further improvement is achieved, terminating when all candidates are evaluated or a convergence threshold is met. In contrast to traditional methods that uniformly shrink bounds [36] or adjust them sequentially [57], BTBI is less sensitive to outliers and explores a richer set of candidate configurations. By combining frame-wise adaptation with recursive backtracking search, BTBI robustly converges to optimal clipping parameters for each frame. To enhance understanding, we provide a detailed algorithm of our BTBI in Algorithm 1.

```
Algorithm 1: Backtracking-based Bound
Initialization (BTBI) pipeline
Input: X, step sizes \Delta L, \Delta U, threshold
Output: Optimal bounds lb^*, ub^*
visited \leftarrow \emptyset, error_{\min} \leftarrow \infty
Function Backtrack (lb, ub):
     if (lb, ub) \in visited or out-of-range
       then
       return
      visited \leftarrow visited \cup \{(lb, ub)\}
     X_q \leftarrow \text{Quantize } X \text{ using } (lb, ub)
     err \leftarrow ||X - X_q||_2
     if err > error_{\min} + \varepsilon then
       return
     \begin{array}{l} \textbf{if } err < error_{\min} \textbf{ then} \\ | \ error_{\min} \leftarrow err, lb^* \leftarrow lb, \end{array}
           ub^* \leftarrow ub
     foreach (\delta_l, \delta_u) \in \{\pm \Delta L, \pm \Delta U\}
       Backtrack(lb + \delta_l, ub + \delta_u)
Backtrack(lb_0, ub_0)
```

## 3.3.2 Progressive Multi-Teacher Distillation

As revealed by Observation 2, training accurate quantized models under extremely low-bit settings (e.g., 4-bit or 2-bit) remains challenging due to limited capacity and optimization instability. To address this, we propose Progressive Multi-Teacher Distillation (PMTD), a hierarchical distillation framework that leverages both high-bit and full-precision teachers to facilitate low-bit training. Instead of directly distilling knowledge from a full-precision (FP) teacher to a low-bit student, which often suffers from large representational gaps, PMTD introduces intermediate-bit teacher models (e.g., 8-bit) between the full-precision (FP) teacher and the low-bit student. These intermediate teachers serve as quantization-aware approximations of the FP model, providing smoother supervision and easing the optimization process. This hierarchical approach effectively bridges the representational gap between student and teacher models, ensuring more stable training dynamics and improved performance under extreme quantization constraints. Specifically, to train a 4-bit quantized model, PMTD first uses the full-precision model as a teacher to train an 8-bit model. When training the 4-bit model, both the full-precision network and the 8-bit network are used as teacher models, as illustrated in Figure 3(c). The distillation process is formally defined by the following loss function:

$$\mathcal{L}_{PMTD} = (\mathcal{L}_{INT} + \alpha(t) \cdot \mathcal{L}_{FP}) / (1 + \alpha(t)), \tag{4}$$

return  $lb^*, ub^*$ 

where  $\mathcal{L}_{\text{INT}}$  denotes the total loss from intermediate-bit teachers (e.g., 8-bit), and  $\mathcal{L}_{\text{FP}}$  represents the loss from the full-precision teacher. The balancing coefficient  $\alpha(t)$  linearly increases over time and is defined as  $\alpha(t) = \min\left(1, \frac{t}{T_{\text{warmup}}}\right)$ , where  $T_{\text{warmup}}$  is a hyperparameter controlling the warm-up duration. Each teacher-specific loss consists of two components: an output-level reconstruction loss

<span id="page-6-0"></span>Table 1: Quantitative comparison of different methods on four STVSR benchmarks. The best and the second best results are in **bold** and bold.

| Method          | Bit   | Vid4         |                     | Vimeo-Fast   |                     | Vimeo-Medium |                     | Vimeo-Slow   |               |
|-----------------|-------|--------------|---------------------|--------------|---------------------|--------------|---------------------|--------------|---------------|
| Method          |       | PSNR↑        | $SSIM \!\!\uparrow$ | PSNR↑        | $SSIM \!\!\uparrow$ | PSNR↑        | $SSIM \!\!\uparrow$ | PSNR↑        | SSIM↑         |
| RSTT-S [12]     | 32/32 | 26.29        | 0.7941              | 36.58        | 0.9381              | 35.43        | 0.9358              | 33.30        | 0.9123        |
| Trilinear       | 32/32 | 22.90        | 0.5883              | 29.45        | 0.8019              | 29.16        | 0.8351              | 28.21        | 0.8091        |
| OpenVINO [13]   | 2/2   | 18.24        | 0.3151              | 23.39        | 0.6103              | 23.60        | 0.6227              | 23.87        | 0.6242        |
| TensorRT [58]   | 2/2   | 20.31        | 0.5118              | 23.41        | 0.6106              | 23.61        | 0.6236              | 23.88        | 0.6283        |
| SNPE [19]       | 2/2   | 15.22        | 0.2378              | 23.40        | 0.6106              | 23.61        | 0.6241              | 23.88        | 0.6281        |
| Percentile [27] | 2/2   | 12.67        | 0.1349              | 15.27        | 0.2274              | 14.80        | 0.2165              | 14.67        | 0.2156        |
| MinMax [21]     | 2/2   | 10.34        | 0.0138              | 10.52        | 0.0266              | 10.48        | 0.0289              | 10.45        | 0.0303        |
| NoisyQuant [38] | 2/2   | 12.06        | 0.1028              | 12.50        | 0.1669              | 11.81        | 0.1465              | 11.49        | 0.1508        |
| DBDC+Pac [57]   | 2/2   | 22.64        | 0.5695              | 28.94        | 0.8254              | 28.87        | 0.8214              | 27.86        | 0.7905        |
| 2DQuant [36]    | 2/2   | <u>22.91</u> | <u>0.5883</u>       | <u>29.38</u> | <u>0.8315</u>       | <u>29.14</u> | <u>0.8330</u>       | <u>28.18</u> | 0.8086        |
| Ours            | 2/2   | 23.48        | 0.6252              | 30.33        | 0.8424              | 30.19        | 0.8523              | 29.14        | 0.8316        |
| OpenVINO [13]   | 4/4   | 18.84        | 0.4591              | 21.82        | 0.6372              | 21.63        | 0.6315              | 18.84        | 0.4591        |
| TensorRT [58]   | 4/4   | 18.63        | 0.4578              | 21.74        | 0.6019              | 21.70        | 0.6324              | 18.63        | 0.4578        |
| SNPE [19]       | 4/4   | 17.84        | 0.3977              | 21.64        | 0.6018              | 21.56        | 0.6301              | 18.76        | 0.4584        |
| Percentile [27] | 4/4   | 23.26        | 0.6314              | 27.12        | 0.7664              | 27.16        | 0.7709              | 26.58        | 0.7531        |
| MinMax [21]     | 4/4   | 21.60        | 0.5242              | 26.41        | 0.6990              | 25.94        | 0.7059              | 25.44        | 0.6957        |
| NoisyQuant [38] | 4/4   | 24.26        | 0.6905              | 31.22        | 0.8719              | 30.64        | 0.8705              | 29.61        | 0.8462        |
| DBDC+Pac [57]   | 4/4   | 24.50        | 0.6923              | 32.64        | 0.8857              | 32.06        | 0.8866              | 30.64        | 0.8643        |
| 2DQuant [36]    | 4/4   | <u>25.04</u> | <u>0.7256</u>       | <u>33.59</u> | <u>0.9035</u>       | <u>32.83</u> | 0.9009              | <u>31.21</u> | <u>0.8766</u> |
| Ours            | 4/4   | 25.42        | 0.7501              | 34.69        | 0.9181              | 33.74        | 0.9150              | 31.94        | 0.8903        |

and an intermediate feature-matching loss:

$$\mathcal{L}_{\text{INT}} = \sum_{k=1}^{K} \left( \mathcal{L}_{\text{rec}}^{(k)} + \lambda \cdot \mathcal{L}_{\text{feat}}^{(k)} \right), \tag{5}$$

$$\mathcal{L}_{FP} = \mathcal{L}_{rec}^{FP} + \lambda \cdot \mathcal{L}_{feat}^{FP}, \tag{6}$$

where K is the number of intermediate-bit teachers (e.g., K=2 when using 4-bit and 8-bit teachers),  $\mathcal{L}_{rec}$  is the  $\ell_2$  loss [32] between the student and teacher outputs, and  $\mathcal{L}_{feat}$  is the mean squared error (MSE) [2] between selected intermediate feature representations. The balancing coefficient  $\lambda$  is set to 5 to emphasize the importance of internal consistency. By gradually transitioning supervision from intermediate-bit to full-precision teachers, PMTD effectively reduces the training difficulty of low-bit models, mitigates quantization errors, and offers a more stable optimization path. This hierarchical approach ensures high-quality quantized outputs, even under extreme quantization constraints.

## 4 Experiment and Analysis

