# **Granular-DQ for SISR**

The proposed Granular-DQ aims to cultivate a layer-invariant SR quantization approach that enables dynamic quantization of existing SR models for varying image contents with the awareness of multi-granularity clues. The overall pipeline is shown in Figure 2, which contains two steps: 1) granularity-aware bit allocation by the granularity-bit controller (GBC) and 2) entropy-based fine-grained bit-width adaption on the patches allocated with high bits in GBC based on an entropy-to-bit (E2B) mechanism.

**Granularity-Bit Controller.** Given an image X, as shown in Figure 3, the GBC first encodes it into hierarchical feature  $\mathbf{Z} = \mathcal{E}(X)$  by the encoder  $\mathcal{E}$ , where  $\mathbf{Z} = Z_1, Z_2, ..., Z_D$  via D-1 downsampling operations. Note that the resolution from  $Z_1$  to  $Z_D$  decreases progressively, where the largest  $Z_1$ corresponds to the finest-granularity feature and the smallest  $Z_D$  denotes the coarsest-granularity one (i.e. D granularities), forming multi-granularity representations for X. We implement GBC with the Gumbel-Softmax, a differentiable sampling scheme (Jang, Gu, and Poole 2017), to adaptively measure the proportional contribution of all patches to the entire image, and align this with potential quantization bit-widths. To be specific, all the granularity features are group normalized and then average pooled to the coarsest granularity, i.e., with the same resolution of  $Z_D$ , denoted by  $\hat{\mathbf{Z}} = \hat{Z}_1, \hat{Z}_2, ..., \hat{Z}_D$ . We concatenate  $\hat{\mathbf{Z}}$  along the channel dimension and squeeze the multi-granularity information by global average pooling  $GAP(\cdot)$  to generate a channel-wise statistics S of X, formulated by

$$\mathbf{S} = GAP(\|\hat{Z}_1, \hat{Z}_2, ..., \hat{Z}_D\|). \tag{3}$$

Assuming there are N total bit codes  $(b_1,...,b_n,...,b_N)$  with different bit-widths, a linear layer is employed to acquire a learnable weight  $\mathbf{W_g} \in \mathbb{R}^{(N \times D) \times N}$  that operates on

![](_page_3_Figure_10.jpeg)

Figure 4: The generalized distribution statistic of the entropy for all LR patches on DIV2K.

**S** to generate the gating logits  $\mathbf{G} \in \mathbb{R}^{1 \times 1 \times N}$  as

$$G = W_g S, (4)$$

For each patch  $X_i$ , its gating logit  $g_i \in \mathbb{R}^N$  is utilized to ascertain the granularity level through the gating index  $\theta_i$ :

$$\theta_i = \arg\max_n(g_{i,n}) \in \{1, 2, ..., n\}.$$
 (5)

Inspired by the end-to-end discrete methodology in (Xie et al. 2020), the fixed decision typically dictated by Eq.(5) is substituted with a probabilistic sampling approach. It hinges on the utilization of a categorical distribution characterized by unnormalized log probabilities, from which discrete gating indices are derived by integrating a noise sample  $\sigma_n$ , originating from the standard Gumbel distribution Gumbel(0, 1):

$$\theta_i = \arg\max_n (g_{i,n} + \sigma_n). \tag{6}$$

After that, we calculate the gating score  $p_i$  for each patch:

$$p_i = \frac{\exp((g_{i,\theta_i} + \sigma_{\theta_i}))/\tau}{\sum_n^N \exp((g_{i,n} + \sigma_n)/\tau)},$$
 (7)

where  $p_i \in [0,1]$  measures the probability of  $X_i$  contributing to the entire image X, thus determining the granularity level and pointing to a corresponding code  $b_n$ . In our experiments, we set the temperature coefficient  $\tau=1$ . Similar to the forward propagation approach in quantization, the gradients for such a gate are calculated using a straight-through estimator, derived from  $p_i$  during the backward pass. By incorporating GBC at the onset of SR networks, Granular-DQ only introduces negligible computational overhead.

Entropy-based Fine-grained Bit-width Adaption. In this work, since Granular-DQ is optimized by pixel-wise supervision, relying solely on the GBC for quantization will force the network to be optimized toward reconstruction accuracy with pixel-wise supervision, which can lead to excessively high bits on some patches. To tackle this problem, we propose an entropy-based scheme to fine-tune bit adaption on the patches less quantized by GBC.

Specifically, we capture a generalized distribution statistic of the entropy for all LR patches on the training set. We first

| Methods                 | Scale      |       | Urban100 |       |       | Test2K |       |       | Test4K |                                                                                                                                                                                                                                                                                  |  |
|-------------------------|------------|-------|----------|-------|-------|--------|-------|-------|--------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--|
| Methods                 | Scure      | FAB↓  | PSNR↑    | SSIM↑ | FAB↓  | PSNR↑  | SSIM↑ | FAB↓  | PSNR↑  | NR↑ SSIM↑  .04 0.823 .77 0.813 .72 0.812 .91 0.818 .62 0.809 .91 0.818 .93 0.820  .80 0.814 .77 0.813 .91 0.818 .96 0.819 .98 0.820  .54 0.806 .59 0.807 .54 0.806 .59 0.807 .59 0.807 .11 0.825 .08 0.823 .92 0.819 .02 0.821 .11 0.824 .56 0.836 .48 0.834 .32 0.830 .31 0.829 |  |
| SRResNet                | $\times 4$ | 32.00 | 26.11    | 0.787 | 32.00 | 27.65  | 0.776 | 32.00 | 29.04  | 0.823                                                                                                                                                                                                                                                                            |  |
| PAMS                    | $\times 4$ | 8.00  | 26.01    | 0.784 | 8.00  | 27.67  | 0.781 | 8.00  | 28.77  | 0.813                                                                                                                                                                                                                                                                            |  |
| CADyQ                   | $\times 4$ | 5.73  | 25.92    | 0.781 | 5.14  | 27.64  | 0.781 | 5.02  | 28.72  | 0.812                                                                                                                                                                                                                                                                            |  |
| CABM                    | $\times 4$ | 5.34  | 25.86    | 0.778 | 5.17  | 27.52  | 0.771 | 5.07  | 28.91  | 0.818                                                                                                                                                                                                                                                                            |  |
| AdaBM                   | $\times 4$ | 5.60  | 25.72    | 0.773 | 5.20  | 27.55  | 0.777 | 5.10  | 28.62  | 0.809                                                                                                                                                                                                                                                                            |  |
| RefQSR( $\delta$ -4bit) | $\times 4$ | 4.00  | 25.90    | 0.778 | 5.17  | 27.52  | 0.771 | 5.07  | 28.91  | 0.818                                                                                                                                                                                                                                                                            |  |
| Granular-DQ (Ours)      | $\times 4$ | 4.00  | 25.98    | 0.783 | 4.01  | 27.55  | 0.773 | 4.01  | 28.93  | 0.820                                                                                                                                                                                                                                                                            |  |
| EDSR                    | $\times 4$ | 32.00 | 26.03    | 0.784 | 32.00 | 27.59  | 0.773 | 32.00 | 28.80  | 0.814                                                                                                                                                                                                                                                                            |  |
| PAMS                    | $\times 4$ | 8.00  | 26.01    | 0.784 | 8.00  | 27.67  | 0.781 | 8.00  | 28.77  | 0.813                                                                                                                                                                                                                                                                            |  |
| CADyQ                   | $\times 4$ | 6.09  | 25.94    | 0.782 | 5.52  | 27.67  | 0.781 | 5.37  | 28.91  | 0.818                                                                                                                                                                                                                                                                            |  |
| CABM                    | $\times 4$ | 5.80  | 25.95    | 0.782 | 5.65  | 27.57  | 0.772 | 5.56  | 28.96  | 0.819                                                                                                                                                                                                                                                                            |  |
| Granular-DQ (Ours)      | $\times 4$ | 4.97  | 26.01    | 0.784 | 4.57  | 27.58  | 0.773 | 4.41  | 28.98  | 0.820                                                                                                                                                                                                                                                                            |  |
| IDN                     | $\times 4$ | 32.00 | 25.42    | 0.763 | 32.00 | 27.48  | 0.774 | 32.00 | 28.54  | 0.806                                                                                                                                                                                                                                                                            |  |
| PAMS                    | $\times 4$ | 8.00  | 25.56    | 0.768 | 8.00  | 27.53  | 0.775 | 8.00  | 28.59  | 0.807                                                                                                                                                                                                                                                                            |  |
| CADyQ                   | $\times 4$ | 5.78  | 25.65    | 0.771 | 5.16  | 27.54  | 0.776 | 5.03  | 28.61  | 0.808                                                                                                                                                                                                                                                                            |  |
| CABM                    | $\times 4$ | 4.28  | 25.57    | 0.768 | 4.25  | 27.42  | 0.766 | 4.23  | 28.74  | 0.813                                                                                                                                                                                                                                                                            |  |
| Granular-DQ (Ours)      | $\times 4$ | 4.18  | 25.68    | 0.772 | 4.29  | 27.47  | 0.767 | 4.23  | 28.83  | 0.816                                                                                                                                                                                                                                                                            |  |
| SwinIR-light            | $\times 4$ | 32.00 | 26.46    | 0.798 | 32.00 | 27.72  | 0.779 | 32.00 | 29.14  | 0.825                                                                                                                                                                                                                                                                            |  |
| PAMS                    | $\times 4$ | 8.00  | 26.31    | 0.793 | 8.00  | 27.67  | 0.776 | 8.00  | 29.08  | 0.823                                                                                                                                                                                                                                                                            |  |
| CADyQ                   | $\times 4$ | 5.15  | 25.87    | 0.779 | 5.01  | 27.54  | 0.772 | 5.01  | 28.92  | 0.819                                                                                                                                                                                                                                                                            |  |
| CABM                    | $\times 4$ | 5.34  | 25.88    | 0.780 | 4.92  | 27.62  | 0.774 | 4.91  | 29.02  | 0.821                                                                                                                                                                                                                                                                            |  |
| Granular-DQ (Ours)      | $\times 4$ | 4.79  | 26.42    | 0.796 | 4.74  | 27.67  | 0.778 | 4.76  | 29.11  | 0.824                                                                                                                                                                                                                                                                            |  |
| HAT-S                   | $\times 4$ | 32.00 | 27.81    | 0.833 | 32.00 | 28.07  | 0.791 | 32.00 | 29.56  | 0.836                                                                                                                                                                                                                                                                            |  |
| PAMS                    | $\times 4$ | 8.00  | 27.56    | 0.827 | 8.00  | 28.00  | 0.789 | 8.00  | 29.48  | 0.834                                                                                                                                                                                                                                                                            |  |
| CADyQ                   | $\times 4$ | 5.53  | 26.98    | 0.814 | 5.41  | 27.88  | 0.784 | 5.33  | 29.32  | 0.830                                                                                                                                                                                                                                                                            |  |
| CABM                    | $\times 4$ | 5.49  | 26.95    | 0.813 | 5.38  | 27.87  | 0.784 | 5.30  | 29.31  | 0.829                                                                                                                                                                                                                                                                            |  |
| Granular-DQ (Ours)      | $\times 4$ | 4.77  | 27.66    | 0.829 | 4.80  | 28.01  | 0.789 | 4.78  | 29.49  | 0.834                                                                                                                                                                                                                                                                            |  |

Table 1: Quantitative comparison (FAB, PSNR (dB)/SSIM) with full precision models, PAMS, CADyQ, CABM, RefQSR and our method on Urban100, Test2K, Test4K for  $\times 4$  SR.  $\times 2$  SR results are provided in the **supplementary material**.

discretize the total N pixels within a patch into multiple bin intervals B based on the pixel values, which can estimate the probability distribution of pixels smoothly. Then entropy is computed as

$$\mathcal{H} = -\sum_{i=1}^{N} \mathcal{P}(x_i) log(\mathcal{P}(x_i)). \tag{8}$$

We use Gaussian-weighted kernel to assign different importance to the pixels in a patch with the formulation of  $\sum_{i=1}^{N} \sum_{j=1}^{B} exp(-\frac{(r_i)^2}{2\sigma^2}) + \epsilon$ , where  $r_i$  denotes the residual between the pixel value of the *i*-th pixel  $x_i$  and the segment values for bin intervals. Thus, one can obtain its kernel den-

sity 
$$\mathcal{P}(x_i)$$
 by  $\frac{\sum_{j=1}^B \exp(-\frac{(r_i)^2}{2\sigma^2})}{\sum_{i=1}^N \sum_{j=1}^B \exp(-\frac{(r_i)^2}{2\sigma^2}) + \epsilon}$ . In this way, we can get the entropy statistic across the overall training set, rep-

get the entropy statistic across the overall training set, represented by  $\mathbf{H} = \mathcal{H}_1, \mathcal{H}_2, ..., \mathcal{H}_M$  sorted in ascending order with M patches, as shown in Figure 4.

We establish an entropy-to-bit (E2B) mechanism based on the entropy statistic  $\mathbf{H}$  and conduct fine-grained bit-width adjustment. Firstly, serial quantiles are inserted on  $\mathbf{H}$  to divide it into multiple subintervals V by  $\mathcal{I}_t = \lceil \frac{M \cdot t}{V} \rceil$ , where  $\mathcal{I}_t$  denotes the patch indice at the t-th quantile, which points to a certain entropy  $\mathcal{H}_t$  in  $\mathbf{H}$ . The quantiles can be seen as thresholds, thus we provide candidate bit configurations according to the thresholds for all the patches. Given a patch

with its entropy E, one can find the index of the subinterval in  $\mathbf{H}$ , and finally determine the adapted bit-width. Taking two quantiles  $t_1$  and  $t_2$  as an example, we can get two patch indices  $\mathcal{I}_{t_1}$  and  $\mathcal{I}_{t_2}$  which corresponds to the entropy values  $\mathcal{H}_{t_1}$  and  $\mathcal{H}_{t_2}$  respectively, *i.e.*  $\mathbf{H}$  will be divided into three discrete subintervals as

$$c_{n} = \begin{cases} c_{1} & \text{if } E \leq \mathcal{H}_{t_{1}}, \\ c_{2} & \text{if } \mathcal{H}_{t_{1}} < E \leq \mathcal{H}_{t_{2}}, \\ c_{3} & \text{if } \mathcal{H}_{t_{2}} < E \leq \mathcal{H}_{M} \end{cases}$$
 (9)

where  $c_n$  denotes the adapted bit codes.

To further improve the flexibility and robustness of E2B for various contents, we present an adaptive threshold calibration (ATC) scheme on E2B. During the training iterations J, we leverage the exponential moving average (EMA) to dynamically calibrate the threshold t, formulated by

$$t^{(j)} = t^{(j-1)} \cdot \gamma + Norm(E) \cdot (1 - \gamma), \qquad (10)$$

where  $Norm(\cdot) = \frac{\mathcal{H}_t - \mathcal{H}_{min}}{\mathcal{H}_{max} - \mathcal{H}_{min}}$ , and  $\mathcal{H}_{max}$  and  $\mathcal{H}_{min}$  denotes the maximum and minimum entropy of all the patches in the current mini-batch at the j-th iteration.  $\gamma$  represents the smoothing parameter of EMA, which is set to 0.9997. It should be noted that the LR samples remain consistent across epochs during training. Hence, our method only necessitates the E2B with ATC at the initial epoch, circumventing significant computational expenditure with iterations.

Once the model is trained, as shown in Figure 2, our method enables to fine-grained adapt the bit-widths of the patches based on calibrated thresholds from the large training set, yielding preferable bit codes [c1, c2, ..., c<sup>N</sup> ].

In summary, by combining GBC and E2B, our method ensures optimal bit allocation for each patch individually while dispensing with the consideration for layer sensitivity as previous methods (Hong et al. 2022a; Tian et al. 2023).

## Loss Function

In previous SR quantization methods (Hong et al. 2022a; Tian et al. 2023; Lee, Yoo, and Jung 2024), the objective function is composed of L1 loss, knowledge distillation loss, and even bit regularization term to facilitate the bit adaption. In Granular-DQ, we only use L<sup>1</sup> loss to train all the models

$$L_1 = \|I_{HR} - I_{SR}\|_1 \tag{11}$$

where IHR is the HR ground truth of the LR input and ISR is the SR reconstruction by our Granular-DQ.

