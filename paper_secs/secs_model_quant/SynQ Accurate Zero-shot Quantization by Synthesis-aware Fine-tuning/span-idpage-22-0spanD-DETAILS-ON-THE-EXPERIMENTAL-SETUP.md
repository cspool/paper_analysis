# <span id="page-22-0"></span>D DETAILS ON THE EXPERIMENTAL SETUP

We describe the details on the experimental setup, including datasets, competitors, hyperparameters, implementation, and training.

Datasets. We utilize three benchmark datasets, CIFAR-10, CIFAR-100 [\(Krizhevsky et al., 2009\)](#page-11-12), and ImageNet (ILSVRC 2012) [\(Deng et al., 2009\)](#page-10-9) to evaluate the classification accuracy of the quantized model obtained by SYNQ. We directly use both CIFAR-10 and CIFAR-100 datasets in TorchVision package. Note that we utilize real datasets only for evaluation purposes.

Competitors. We briefly summarize the details of the competitors of SYNQ as follows:

- GDFQ [\(Xu et al., 2020\)](#page-13-10) is the first method to utilize a knowledge-matching generator to produce synthetic data which is guided by both batch normalization statistics loss and cross-entropy loss.
- ARC [\(Zhu et al., 2021\)](#page-13-11) or AutoReCon is a neural architecture search-based image reconstruction method.
- Qimera [\(Choi et al., 2021\)](#page-10-4) uses superposed latent embeddings to generate synthetic boundary supporting samples.
- AIT [\(Choi et al., 2022\)](#page-10-8) improves the loss function and gradients for ARC to generate better samples, which we denote it as AIT + ARC.
- IntraQ [\(Zhong et al., 2022b\)](#page-13-6) highlights the intra-class heterogeneity and retains this property in the synthetic dataset for better performance.

- AdaSG (Qian et al., 2023b) plots the ZSQ problem as a zero-sum game between two players, the generator and the quantized network to generate adaptive samples for the synthetic dataset.
- AdaDFQ (Qian et al., 2023a) further generalizes AdaSG to adaptively regulate the adaptability
  of the synthetic samples.
- **HAST** (Li et al., 2023a) pays more attention to difficult samples by generating difficult samples and further promoting the sample difficulty when training the quantized model.
- **TexQ** (Chen et al., 2023) retains the texture feature distributions within the synthetic dataset by using synthetic calibration centers to calibrate samples.
- PLF (Fan et al., 2024) evaluates synthetic data to assign pseudo-labels with different reliability to avoid misleading training.

Additionally, we compare with PSAQ-VIT (Li et al., 2022) and Genie (Jeon et al., 2023b) for the ViT (see Section 5.3) and PTQ (see Appendix C.7) experiments, respectively.

**Baseline.** We introduce the baseline method to produce the synthetic dataset for the main results and observations (e.g. Tables 3 and 6). We adopt calibration center synthesis (Chen et al., 2023), difficult sample generation, and sample difficulty promotion (Li et al., 2023a). Producing the synthetic dataset consists of three stages. First, we produce calibration centers following Chen et al. (2023), one center each for all possible classes. Second, we produce the synthetic dataset with two additional losses, hard-sample-enhanced inception loss  $\mathcal{L}_{HIL}$  from HAST and layered batch normalization statistics alignment loss  $\mathcal{L}_{L-BNS}^G$ , added on top of Equation 1. Lastly, we attach a perturbation to each image following sample difficulty promotion from HAST, to make generated samples more difficult for the quantized model. We select this baseline that combines only the synthetic dataset production part of the two papers HAST and TexQ, in order to intentionally set baselines only for the first step and replace the existing fine-tuning process with the proposed synthesis-aware fine-tuning. Refer to the original papers for further details.

<span id="page-23-0"></span>**Hyperparameters.** We conduct a grid search to validate hyperparameters, and select the set with the best performance. Table 10 reports the searched hyperparameter ranges of SYNQ for the ImageNet dataset experiment. For competitors, we search within the range described in each paper. We conduct 5 iteration for each experiments and report the mean and standard deviation of the results.

| Hyperparameter  | Range                              |
|-----------------|------------------------------------|
| $\alpha_1$      | [0.2, 0.4, 0.6, 0.8, 1]            |
| $\alpha_2$      | [0.01, 0.04, 0.1]                  |
| $\alpha^C$      | [0.4, 1, 2.5]                      |
| $\lambda_P$     | [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4] |
| $\lambda_{CE}$  | [5, 5e-1, 5e-2, 5e-3]              |
| $D_0$           | [20, 40, 60, 80, 100]              |
| $\tau$          | [0.5, 0.55, 0.6, 0.65, 0.7]        |
| $\lambda_{CAM}$ | [20, 50, 100, 200, 300, 500, 2000] |
| CAM Technique   | [CAM, Grad-CAM, Grad-CAM++]        |

Table 10: Hyperparameter ranges for SYNQ.

**Implementation and Machine.** We implement SYNQ with PyTorch and TorchVision libraries in Python. For the other methods, we reproduce the result using their open-source code if possible and implement them otherwise. All of our experiments were done at a workstation with Intel Xeon Silver 4214 and RTX 3090.

**Training Details.** We first generate the calibration centers with a constant learning rate of 0.05, following TexQ (Chen et al., 2023). Then, we optimize samples using the loss function described in Equation 1 with the Adam optimizer to generate the synthetic dataset. This optimizer has a momentum of 0.9 and an initial learning rate of 0.5. The synthetic images are updated over 1,000 iterations, with the learning rate decaying by a factor of 0.1 whenever the loss does not decrease for

50 consecutive iterations. For all datasets, a batch size of 256 is used, resulting in the generation of a total of 5,120 images. For the fine-tuning of the quantized model, the procedure follows Equation [6,](#page-6-1) employing SGD with a momentum of 0.9 and a weight decay of 1e-4. The batch size is set to 256 for CIFAR-10/100 and 16 for ImageNet. Initial learning rate is searched within the range of {1e-4, 1e-5, 1e-6} and is decayed by a factor of 0.1 over training epochs nep = 100.

ViT Quantization Experiment. We compare the ZSQ precision of PSAQ-ViT [\(Li et al., 2022\)](#page-12-16) with that of SYNQ applied on it. PSAQ-ViT generates the synthetic dataset based on the patch similarity, substituting the batch normalization statistics in CNN models. The pre-trained DeiT-Tiny, DeiT-Small [\(Touvron et al., 2021\)](#page-13-12), Swin-Tiny, and Swin-Small [\(Liu et al., 2021b\)](#page-12-15) models on ImageNet dataset is obtained from timm [\(Wightman, 2019\)](#page-13-20) library. We follow [Li et al.](#page-12-16) [\(2022\)](#page-12-16) for the experimental setup, where only 32 images are used from the synthetic dataset.