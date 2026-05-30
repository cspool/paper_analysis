# <span id="page-3-3"></span><span id="page-3-2"></span>4 PROPOSED METHOD

### 4.1 OVERVIEW

We propose SYNQ (Synthesis-aware Fine-tuning for Zero-shot Quantization), an accurate Zero-shot Quantization (ZSQ) method addressing the following three major challenges of existing methods that fine-tune with the synthetic dataset. These are the three main challenges that must be tackled:

- C1. Noise in the synthetic dataset. Previous methods fine-tune the quantized model with a noisy synthetic dataset, which exhibits a distribution discrepancy of frequency domain compared to real images. How can we minimize the effect of the noise within the generated samples?
- C2. Prediction based on off-target patterns. The quantized model predicts based on incorrect image regions that are unlike those observed in the pre-trained model. How can we optimize the quantized model to more accurately utilize on-target patterns?
- C3. Misguidance from erroneous hard labels. Despite the high error rate of difficult samples, existing works trust erroneous hard labels. How can we address the misguidance by hard labels?

<span id="page-4-1"></span>![](_page_4_Figure_1.jpeg)

Figure 4: Overall architecture of SYNQ. Our main ideas are 1) low-pass filter, 2) alignment of class activation map, and 3) soft labels for difficult samples. See Section 4 for details.

We address these challenges with the following main ideas:

- **I1. Low-pass filter (Section 4.2).** We directly reduce the noise from the dataset by exploiting a Gaussian low-pass filter in the frequency domain.
- **12. Alignment of class activation map** (**Section 4.3**). We align the class activation map between the pre-trained and quantized models, directly distilling knowledge to identify the correct image region from the pre-trained model to the quantized model.
- **I3. Soft labels for difficult samples (Section 4.4).** For difficult samples, we fine-tune only with soft labels or predictions from the pre-trained model to reduce ambiguity.

Figure 4 illustrates the overall process of SYNQ. SYNQ first generates a synthetic dataset from arbitrary labels. Then, SYNQ exploits a Gaussian low-pass filter to refine the samples by removing noise. With this filtered dataset, SYNQ fine-tunes the quantized model with KL divergence and cross-entropy losses, following the standard ZSQ framework. SYNQ also optimizes CAM alignment loss  $\mathcal{L}_{CAM}$  to enhance activation map alignment for better critical region detection. SYNQ exploits the threshold  $\tau$  to decide on the application of cross-entropy loss based on the difficulty of a sample.

#### <span id="page-4-0"></span>4.2 LOW-PASS FILTER

The first step of Zero-shot Quantization (ZSQ) is to produce the synthetic dataset which effectively mimics the real dataset. Existing methods leverage the prediction and batch normalization statistics of the pre-trained model to generate samples. However, their limitation is the noise in the synthetic dataset, as discussed in Figure 1. We investigate the intensity of this noise, by performing the Fourier transform on the datasets. Figure 5 illustrates the amplitude distribution of (a) ImageNet dataset, (b) the synthetic dataset by TexQ (Chen et al., 2023), and (c) Gaussian-filtered samples based on the distance from the center. The dark solid line indicates the mean distribution, and the surrounding colored region shows the standard deviation within a batch of 256 randomly chosen images. While the ImageNet dataset primarily exhibits lower frequency components, the synthetic dataset contains more high-frequency components, clearly indicating a higher level of sharpness and noise. This noise is observed in various ZSO methods, regardless of the setting (see Appendix C.3).

To mitigate this noise, we exploit a Gaussian low-pass filter on the generated samples. Given a sample  $\mathbf{x}_i$  with width W, height H, and filtering hyperparameter  $D_0$  which is related to cut-off frequency, we compute the filtered sample  $\mathbf{x}_i^F$  as shown in Equation (4).

<span id="page-4-2"></span>
$$\mathbf{x}_{i}^{F} = \mathcal{F}^{-1}\left(\mathbf{G} \odot \mathcal{F}(\mathbf{x}_{i})\right), \mathbf{G}_{uv} = \exp\left(-\frac{(D(u,v))^{2}}{2D_{0}^{2}}\right), D(u,v) = \sqrt{(u - \frac{W}{2})^{2} + (v - \frac{H}{2})^{2}}, \quad (4)$$

where D(u, v) denotes the distance from the coordinate (u, v) to the center in the frequency domain and  $\odot$  is an element-wise multiplication. This Gaussian low-pass filter G works in the frequency

<span id="page-5-0"></span>![](_page_5_Figure_1.jpeg)

Figure 5: Comparison of amplitude distribution among (a) ImageNet dataset, (b) synthetic dataset by TexQ, and (c) filtered samples. After filtering, the distribution closely aligns with that of real images.

domain from conducting Fourier transform  $\mathcal{F}$ , we then apply inverse Fourier transform  $\mathcal{F}^{-1}$  to obtain the filtered sample  $\mathbf{x}_i^F$ . Figure 5(c) clearly shows the positive effect of the filter: removal of noise in the high-frequency region resulting in an amplitude distribution aligning with that of real images. We further investigate the robustness of low-pass filter towards different types of noise in Appendix C.8.

### <span id="page-5-1"></span>4.3 ALIGNMENT OF CLASS ACTIVATION MAP

The next step is to fine-tune the quantized model to achieve high accuracy. The first challenge of this step is to ensure that the quantized model makes predictions using on-target image patterns. Existing methods fine-tune the model with classification and knowledge distillation from the pre-trained model, using the synthetic dataset. However, the quantized model from these methods fail to properly localize the object as depicted in Figure 2. To ensure the quantized model to make prediction based on correct image regions, we directly align the class activation map between pre-trained and quantized models. We optimize the class activation map (CAM) alignment loss  $\mathcal{L}_{CAM}$  by minimizing the mean square error between the saliency maps  $\mathbf{S}^{\theta}(\mathbf{x}_i)$  and  $\mathbf{S}^{\theta^q}(\mathbf{x}_i)$  of the pre-trained and quantized models, respectively. Among various techniques (Zhou et al., 2016; Selvaraju et al., 2017; Zagoruyko & Komodakis, 2017; Chattopadhay et al., 2018) to highlight the important region of the image, we select Grad-CAM (Selvaraju et al., 2017) due to its simplicity and superiority, which we discuss further in Section 5.4. Grad-CAM generates the saliency map  $\mathbf{S}^{\theta}(\mathbf{x}_i)$  by weighting the activations with their gradients, emphasizing the regions in the input image that have the greatest impact on the model's prediction. We formulate CAM alignment loss as Equation (5).

<span id="page-5-3"></span>
$$\mathcal{L}_{CAM}(\mathbf{x}_{i}; \theta, \theta^{q}) = \|\mathbf{S}^{\theta}(\mathbf{x}_{i}) - \mathbf{S}^{\theta^{q}}(\mathbf{x}_{i})\|_{F}^{2},$$

$$\mathbf{S}^{\theta}(\mathbf{x}_{i}) = \text{ReLU}\left(\sum_{k} \left(\frac{1}{W_{k}H_{k}} \sum_{w=1}^{W_{k}} \sum_{h=1}^{H_{k}} \frac{\partial y^{\mathbf{y}_{i}}}{\partial \mathbf{A}_{wh}^{k;\theta}(\mathbf{x}_{i})}\right) \mathbf{A}^{k;\theta}(\mathbf{x}_{i})\right),$$
(5)

where  $\mathbf{A}^{k;\theta}(\mathbf{x}_i)$  denotes the activations of the last layer at channel k with  $W_k$  and  $H_k$  representing its width and height, respectively. The gradient of the predicted score  $y^{\mathbf{y}_i}$  for the true class  $\mathbf{y}_i$  with respect to the activation  $\mathbf{A}^{k;\theta}_{wh}(\mathbf{x}_i)$  at spatial location (w,h) indicates the contribution of each activation to the model's prediction. Figure 2 clearly shows that  $\mathcal{L}_{CAM}$  enables SYNQ to accurately target the correct image regions as the pre-trained model does (see Appendix C.4 for further analysis).

### <span id="page-5-2"></span>4.4 SOFT LABELS FOR DIFFICULT SAMPLES

The second challenge of the fine-tuning step is the misguidance from possibly mislabeled samples. Existing works assign random classes as labels for generated samples, then minimize the Inception Loss (IL)  $\mathcal{L}_{IL}$  in Equation (1) to optimize the image so that the pre-trained model predicts the assigned labels. However, the pre-trained model frequently mislabels difficult samples in this approach, as the higher difficulty indicates that the pre-trained model assigns lower probabilities to the true label, following the definition in Equation (3).

To avoid this misguidance, we exclude the cross-entropy loss with the hard labels for difficult samples. We classify samples as easy or difficult based on a difficulty threshold  $\tau$ . For easy samples, we optimize both the cross-entropy loss with hard labels and the KL divergence with soft labels. In contrast, for difficult samples, we exclusively optimize the KL divergence with soft labels, completely omitting

the cross-entropy loss. This approach focuses on replicating the pre-trained model's responses to ambiguous images, minimizing performance degradation caused by overconfidence in hard labels. Note that previous methods apply both soft and hard labels irrespective of sample difficulty.

#### 4.5 Objective Function

Combining all three ideas of SYNQ, we modify the loss function for the fine-tuning phase from  $\mathcal{L}_{ZSO}$  in Equation (2) to  $\mathcal{L}_{SYNO}$  in Equation (6).

<span id="page-6-1"></span>
$$\mathcal{L}_{\text{SYNQ}} = \frac{1}{N} \sum_{i=1}^{N} \left( KL\left(q(\mathbf{x}_{i}^{F}; \boldsymbol{\theta}) || q(\mathbf{x}_{i}^{F}; \boldsymbol{\theta}^{q}) \right) + \mathbf{1}_{\left\{\delta(\mathbf{x}_{i}^{F}, \boldsymbol{\theta}) \leq \tau\right\}} \lambda_{CE} CE\left(q(\mathbf{x}_{i}^{F}; \boldsymbol{\theta}^{q}), \mathbf{y}_{i}\right) + \lambda_{CAM} \mathcal{L}_{CAM}\left(\mathbf{x}_{i}^{F}; \boldsymbol{\theta}, \boldsymbol{\theta}^{q}\right) \right), \tag{6}$$

where  $\lambda_{CE}$  and  $\lambda_{CAM}$  are balancing hyperparameters for cross-entropy loss and CAM alignment loss, respectively.  $\mathbf{1}_{\{\cdot\}}$  is the indicator function which returns 1 if the inner statement is true and 0 otherwise. We train with the filtered samples  $\mathbf{x}_i^F$  (Section 4.2) to remove the noise in the synthetic dataset. Then, we align the class activation map between the pre-trained and quantized models by optimizing  $\mathcal{L}_{CAM}$  (Section 4.3) to transfer the knowledge of finding an object on the image. We also exclude cross-entropy loss for difficult samples with a threshold  $\tau$  (Section 4.4) to mitigate the impact of misguidance from hard labels.

SYNQ is compatible with any ZSQ method utilizing synthetic datasets (Zhong et al., 2022b; Qian et al., 2023b; Jeon et al., 2023b). We adopt calibration center synthesis (Chen et al., 2023), difficult sample generation, and sample difficulty promotion (Li et al., 2023a) because we observe they generally perform better in ZSQ (refer to Appendix D for details). We visualize the generated images within synthetic dataset in Figure 12. The adaptability of SYNQ is clearly demonstrated through further experiments on other Zero-shot QAT and PTQ methods in Appendices C.6 and C.7, respectively. We formulate the overall algorithm of SYNQ in Algorithm 1.

Complexity Analysis. We analyze the time complexity of SYNQ, where N and L represent the numbers of training samples and layers, respectively.

<span id="page-6-2"></span>**Theorem 1** (Time Complexity of SYNQ). Given a model with an inference complexity of  $O(T_{\theta})$ , the time complexity for the quantization procedure (Algorithm 1) of SYNQ is  $O(NLT_{\theta})$ .

Theorem 1 demonstrates that SYNQ is an efficient approach, with a time complexity scaling linearly with the number of training samples N and model layers L. Furthermore, SYNQ generates only 5,120 samples, making it significantly faster than generator-based methods such as AdaSG (Qian et al., 2023b) and AdaDFQ (Qian et al., 2023a), which produce over 1 million samples (see Appendix C.9 for experiments with different sizes of dataset). We perform a runtime analysis of SYNQ in Appendix C.2 to analyze the computational overhead of SYNQ.

#### <span id="page-6-0"></span>5 EXPERIMENTS

We perform experiments to answer the following questions about SYNQ. Further discussions and experiments on SYNQ are discussed in Appendix C.

- **Q1.** Accuracy in Convolutional Neural Network (CNN) Quantization (Section 5.2). How accurate is the quantized CNN model from SYNQ compared to those from existing ZSQ methods?
- **Q2.** Accuracy in Vision Transformer (ViT) Quantization (Section 5.3). How effective is SYNQ in enhancing ViT Quantization performance?
- **Q3. Analysis on Class Activation Map Techniques (Section 5.4).** Which CAM technique demonstrates the highest performance?
- **Q4. Ablation Study** (Section 5.5). Are all components of SYNQ effective for enhancing the classification accuracy of the quantized model?
- **Q5.** Hyperparameter Analysis (Section 5.6). How robust are the performance gains by SYNQ in hyperparameters  $\lambda_{CE}$ ,  $\lambda_{CAM}$ ,  $D_0$ , and  $\tau$ ?

#### 5.1 EXPERIMENTAL SETUP

We briefly introduce the experimental setup. Further setups are detailed in Appendix D.

<span id="page-7-0"></span>Table 1: Zero-shot Quantization accuracy [%] of ResNet-20 (R-20) on CIFAR-10 and CIFAR-100, and ResNet-18 (R-20), ResNet-50 (R-50), and MobileNetV2 (MV2) on ImageNet. WBAB indicates that both weights and activations are quantized to Bbit. Note that SYNQ achieves the highest accuracy.

|                               |        | R-20 (CIFAR-10) | R-20 (CIFAR-100) |        | R-18 (ImageNet) |        | R-50 (ImageNet) |        | MV2 (ImageNet) |        |
|-------------------------------|--------|-----------------|------------------|--------|-----------------|--------|-----------------|--------|----------------|--------|
| Method                        | W4A4   | W3A3            | W4A4             | W3A3   | W4A4            | W3A3   | W4A4            | W3A3   | W4A4           | W3A3   |
| Full Precision (W32A32)       |        | 93.89           |                  | 70.33  |                 | 71.47  |                 | 77.73  |                | 73.03  |
| GDFQ (Xu et al., 2020)        | 90.11  | 75.11           | 63.75            | 47.61  | 60.60           | 20.23  | 54.16           | 0.31   | 59.43          | 1.46   |
| ARC (Zhu et al., 2021)        | 88.55  | -               | 62.76            | 40.15  | 61.32           | 23.37  | 64.37           | 1.63   | 60.13          | 14.30  |
| Qimera (Choi et al., 2021)    | 91.26  | 74.43           | 65.10            | 46.13  | 63.84           | 1.17   | 66.25           | -      | 61.62          | -      |
| ARC + AIT (Choi et al., 2022) | 90.49  | -               | 61.05            | 41.34  | 65.73           | -      | 68.27           | -      | 66.47          | -      |
| IntraQ (Zhong et al., 2022b)  | 91.49  | 77.07           | 64.98            | 48.25  | 66.47           | 45.51  | -               | -      | 65.10          | -      |
| AdaSG (Qian et al., 2023b)    | 92.10  | 84.14           | 66.42            | 52.76  | 66.50           | 37.04  | 68.58           | 16.98  | 65.15          | 26.90  |
| AdaDFQ (Qian et al., 2023a)   | 92.31  | 84.89           | 66.81            | 52.74  | 66.53           | 38.10  | 68.38           | 17.63  | 65.41          | 28.99  |
| HAST (Li et al., 2023a)       | 92.36  | 86.34           | 66.68            | 55.67  | 66.91           | 42.58  | -               | -      | 65.60          | -      |
| TexQ (Chen et al., 2023)      | 92.68  | 86.47           | 67.18            | 55.87  | 67.73           | 50.28  | 70.72           | 25.27  | 67.07          | 32.80  |
| PLF (Fan et al., 2024)        | 92.47  | 88.04           | 66.94            | 57.03  | 67.02           | -      | 68.97           | -      | -              | -      |
| SYNQ (Proposed)               | 92.76  | 88.11           | 67.34            | 57.28  | 67.90           | 52.02  | 71.05           | 26.89  | 67.27          | 34.21  |
| Standard Deviation            | ± 0.10 | ± 0.15          | ± 0.15           | ± 0.29 | ± 0.19          | ± 0.34 | ± 0.17          | ± 0.24 | ± 0.21         | ± 0.27 |

Setup. We evaluate our method across three datasets by reporting the top-1 accuracy for the validation sets of CIFAR-10, CIFAR-100 [\(Krizhevsky et al., 2009\)](#page-11-12) and ImageNet (ILSVRC 2012) [\(Deng et al.,](#page-10-9) [2009\)](#page-10-9) datasets. We select ResNet-20 [\(He et al., 2016\)](#page-11-10) model for CIFAR-10 and CIFAR-100, and ResNet-18, ResNet-50 [\(He et al., 2016\)](#page-11-10), and MobileNetV2 [\(Sandler et al., 2018\)](#page-12-14) model for ImageNet. We follow this prevalent experimental setup from existing works [\(Chen et al., 2023;](#page-10-5) [Qian et al.,](#page-12-13) [2023a](#page-12-13)[;b\)](#page-12-12) to correctly compare the performance of SYNQ.

Competitors. We compare SYNQ with existing ZSQ methods utilizing synthetic dataset, including GDFQ [\(Xu et al., 2020\)](#page-13-10), ARC [\(Zhu et al., 2021\)](#page-13-11), Qimera [\(Choi et al., 2021\)](#page-10-4), ARC + AIT [\(Choi](#page-10-8) [et al., 2022\)](#page-10-8), IntraQ [\(Zhong et al., 2022b\)](#page-13-6), AdaSG [\(Qian et al., 2023b\)](#page-12-12), AdaDFQ [\(Qian et al., 2023a\)](#page-12-13), HAST [\(Li et al., 2023a\)](#page-12-6), TexQ [\(Chen et al., 2023\)](#page-10-5), and PLF [\(Fan et al., 2024\)](#page-11-6). Both model weights and activation are quantized identically for all layers.

Implementation Details. We follow the settings from IntraQ [\(Zhong et al., 2022b\)](#page-13-6) and HAST [\(Li](#page-12-6) [et al., 2023a\)](#page-12-6) for equal comparison. We generate 5,120 images with a batch size of 256. The batch size for fine-tuning is 256 for CIFAR-10/100 and 16 for ImageNet with epochs uniformly set to 100. We search τ , D0, λCE, and λCAM within the ranges {0.5, 0.55, 0.6, 0.65, 0.7}, {20, 40, 60, 80, 100}, {0.005, 0.05, 0.5, 5}, and {20, 50, 100, 200, 300, 500, 2000}, respectively. All of our experiments were done at a workstation with Intel Xeon Silver 4214 and RTX 3090.

### <span id="page-7-1"></span>5.2 ACCURACY IN CNN QUANTIZATION (Q1)

We evaluate the quantization accuracy of SYNQ against existing ZSQ methods using CIFAR-10, CIFAR-100, and ImageNet datasets. Our method significantly enhances quantized model accuracy on all settings with 3bit and 4bit quantization as summarized in Table [1.](#page-7-0) We report the mean and standard deviation of 5 iterations, each using different random seed values. We have two observations from the result. First, SYNQ benefits the fine-tuning of quantized models consistently across diverse quantization bits, models, and datasets. Compared to state-of-the-art methods TexQ [\(Chen et al.,](#page-10-5) [2023\)](#page-10-5) and PLF [\(Fan et al., 2024\)](#page-11-6), SYNQ achieves higher accuracies of up to 1.74%p (ResNet-18 on ImageNet dataset). Second, SYNQ demonstrates increasing effectiveness as bit-width decreases. Considering that lower-bit quantization is inherently more challenging, our results clearly showcase the robustness of SYNQ due to its effective fine-tuning that overcomes the aforementioned limitations.

