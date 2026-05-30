# <span id="page-7-2"></span>5.3 ACCURACY IN VIT QUANTIZATION (Q2)

We investigate the effectiveness of SYNQ in enhancing ZSQ performance for Vision Transformers (ViTs). Table [2](#page-8-3) shows the ZSQ precision of four ViT models, DeiT-Tiny, DeiT-Small [\(Touvron](#page-13-12) [et al., 2021\)](#page-13-12), Swin-Tiny, and Swin-Small [\(Liu et al., 2021b\)](#page-12-15) pre-trained on ImageNet dataset. SYNQ enhances the quantization precision across various models, achieving up to 0.58%p increase in

<span id="page-8-3"></span>Table 2: Zero-shot Quantization accuracy [%] of ViT models on ImageNet dataset. WBAB indicates that both weights and activations are quantized to Bbit. Note that SYNQ shows consistent improvements in quantization performance across various models.

| Bits | Method                                        | DeiT-Tiny                            | DeiT-Small                           | Swin-Tiny                                       | Swin-Small                           | Average               |
|------|-----------------------------------------------|--------------------------------------|--------------------------------------|-------------------------------------------------|--------------------------------------|-----------------------|
|      | Full Precision                                | 72.21                                | 79.85                                | 81.35                                           | 83.20                                | 79.15                 |
| W4A8 | PSAQ-ViT (Li et al., 2022)<br>SYNQ (Proposed) | $65.57 \pm 0.10$<br>$65.90 \pm 0.07$ | $72.04 \pm 0.19$<br>$72.28 \pm 0.34$ | $69.78 \pm 1.67$ <b>70.76</b> $\pm$ <b>1.61</b> | $75.03 \pm 0.63$<br>$75.82 \pm 0.54$ | 70.61<br><b>71.19</b> |
| W8A8 | PSAQ-ViT (Li et al., 2022)<br>SYNQ (Proposed) | $71.56 \pm 0.03$<br>$71.74 \pm 0.03$ | $75.97 \pm 0.20$<br>$76.16 \pm 0.29$ | $73.54 \pm 1.61$ $74.11 \pm 1.82$               | $76.68 \pm 0.53$<br>$77.32 \pm 0.59$ | 74.44<br><b>74.83</b> |

average precision when applied to the recent method PSAQ-ViT (Li et al., 2022). The results show that SYNQ is an accurate ZSQ method not only tailored for CNN but also is effective in ViTs.

### <span id="page-8-0"></span>5.4 ANALYSIS ON CLASS ACTIVATION MAP TECHNIQUES (Q3)

We compare the quantization accuracy of SYNQ when utilizing different techniques to output the class activation map. We show the 3bit quantization accuracy of ResNet-18 model in Figure 6. Grad-CAM (Selvaraju et al., 2017) demonstrates higher performance over CAM (Zhou et al., 2016) and Grad-CAM++ (Chattopadhay et al., 2018). This is attributed to Grad-CAM++ being specialized in localizing multiple objects, whereas Grad-CAM focuses on a single object. Additionally, note that Grad-CAM also takes advantage over CAM in that it is a direct generalization of CAM which is applicable only to models with a global pooling layer. Thus, we utilize Grad-CAM to generate the saliency map for the CAM alignment loss  $\mathcal{L}_{CAM}$ , as described in Section 4.4.

<span id="page-8-4"></span>![](_page_8_Figure_6.jpeg)

Figure 6: ZSQ accuracy comparison on different CAM techniques. See Section 5.3 for details.

#### <span id="page-8-1"></span>5.5 ABLATION STUDY (Q4)

We perform an ablation study to show that each main idea of SYNQ, such as low-pass filter (I1) in Section 4.2, alignment of class activation map (I2) in Section 4.3, and soft labels for difficult samples (I3) in Section 4.4, improves the classification accuracy of the compressed model. We summarize the 3bit quantization results of ResNet-18 model on ImageNet dataset in Table 3. Note that the baseline denotes HAST (Li et al., 2023a) with layer-wise batch normalization loss from TexQ (Chen et al., 2023) as detailed in Appendix D. Our analysis shows that all proposed ideas contribute to improved performance, with low-pass filter (I1) having the strongest impact of 5.80%p.

## Table 3: Ablation study on the main ideas of SYNQ. All ideas contribute to the improved performance.

<span id="page-8-5"></span>

| I1 | 12     | 13 | Accuracy [%] |
|----|--------|----|--------------|
| В  | aselir | ne | 43.63        |
| 1  |        |    | 49.43        |
|    | 1      |    | 48.26        |
|    |        | 1  | 46.42        |
| 1  | 1      |    | 51.24        |
| 1  |        | 1  | 50.81        |
|    | 1      | 1  | 50.06        |
| /  | 1      | 1  | 52.02        |

#### <span id="page-8-2"></span>5.6 Hyperparameter Analysis (Q5)

We analyze the robustness of SYNQ concerning the newly introduced hyperparameters  $\lambda_{CE}$ ,  $\lambda_{CAM}$ ,  $D_0$ , and  $\tau$  in Figure 7. We report the 3bit quantization accuracy for the ResNet-18 model trained on the ImageNet dataset. We have three observations from the result. First, as shown in Figure 7(a), the classification accuracy remains robust across a range of  $\lambda_{CE}$  and  $\lambda_{CAM}$  values. This robustness indicates that SYNQ remains effective even when these hyperparameters are not precisely tuned. Second, Figure 7(b) illustrates the effect of varying the difficulty threshold  $\tau$ . Note that the classification accuracy increases as  $\tau$  increases from 0 to 0.5, since too low  $\tau$  excludes many useful samples for cross-entropy training. However, the classification accuracy starts to decrease as  $\tau$  becomes greater than 0.5, since it allows to use difficult and ambiguous samples for cross-entropy training. We observe that the  $\tau$  value of 0.5 gives the best trade-off between using more samples and not using ambiguous samples. We further conduct a deeper analysis on  $\tau$  in Appendix C.11, verifying its impact on different settings. Third, Figure 7(c) shows that an appropriate balance in  $D_0$  is necessary to maintain performance. Extremely low  $D_0$  values result in significant performance degradation due to excessive filtering, which oversmooths the images and results in the loss of crucial information. Overall, SYNQ consistently outperforms baselines across a diverse range of hyperparameter values.

<span id="page-9-0"></span>![](_page_9_Figure_1.jpeg)

Figure 7: Hyperparameter analysis on (a) balancing hyperparameters λCE and λCAM, (b) difficulty threshold τ , and (c) filtering hyperparameter D0. See Section [5.5](#page-8-1) for details.

