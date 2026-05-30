# Thinking in Granularity: Dynamic Quantization for Image Super-Resolution by Intriguing Multi-Granularity Clues

Mingshen Wang<sup>1</sup>, Zhao Zhang<sup>1,2\*</sup>, Feng Li<sup>1\*</sup>, Ke Xu<sup>3</sup>, Kang Miao<sup>1</sup>, Meng Wang<sup>1</sup>

<sup>1</sup>School of Computer Science and Information Engineering, Hefei University of Technology, Hefei, China
<sup>2</sup>Yunnan Key Laboratory of Software Engineering, Yunan, China
<sup>3</sup>School of Artificial Intelligence, Anhui University, Hefei, China

#### **Abstract**

Dynamic quantization has attracted rising attention in image super-resolution (SR) as it expands the potential of heavy SR models onto mobile devices while preserving competitive performance. Existing methods explore layer-to-bit configuration upon varying local regions, adaptively allocating the bit to each layer and patch. Despite the benefits, they still fall short in the trade-off of SR accuracy and quantization efficiency. Apart from this, adapting the quantization level for each layer individually can disturb the original inter-layer relationships, thus diminishing the representation capability of quantized models. In this work, we propose Granular-DQ, which capitalizes on the intrinsic characteristics of images while dispensing with the previous consideration for layer sensitivity in quantization. Granular-DQ conducts a multi-granularity analysis of local patches with further exploration of their information densities, achieving a distinctive patch-wise and layer-invariant dynamic quantization paradigm. Specifically, Granular-DQ initiates by developing a granularity-bit controller (GBC) to apprehend the coarse-to-fine granular representations of different patches, matching their proportional contribution to the entire image to determine the proper bit-width allocation. On this premise, we investigate the relation between bit-width and information density, devising an entropy-to-bit (E2B) mechanism that enables further fine-grained dynamic bit adaption of high-bit patches. Extensive experiments validate the superiority and generalization ability of Granular-DQ over recent state-ofthe-art methods on various SR models. Code and supplementary statement can be found at https://github.com/MmmingS/ Granular-DQ.git.

### Introduction

Single image super-resolution (SISR) has been a fundamental task in the computer vision community, aiming to recover high-resolution (HR) images from corrupted low-resolution (LR) input. Recently, from the pioneering deep learning-based method (Dong et al. 2014), convolutional neural networks (CNN) (Dong et al. 2014; Kim, Lee, and Lee 2016; Shi et al. 2016; Zhang et al. 2018; Ahn, Kang, and Sohn 2018; Li et al. 2022) and transformers (Liang et al. 2021;

![](_page_0_Figure_10.jpeg)

(b) Our dynamic granularity-aware quantization pipeline

Average bit:5.00

PSNR: 25.74dB

Figure 1: Visual comparison of (a) previous dynamic quantization pipeline (Hong et al. 2022a) that adapt the bit allocation for layers and patches simultaneously and (b) our Granular-DQ pipeline conducts patch-wise and layer-invariant dynamic quantization, which contains two steps: 1) granularity-aware bit allocation and 2) fine-grained bit-width adaption based on the entropy statistics. Our method recovers a better SR image with a lower average bit.

Lu et al. 2022; Zhang et al. 2022; Chen et al. 2023) have dominated SISR. While the SR performance continues to achieve breakthroughs, the model complexity of later methods also increases constantly, which limits their practical applications, especially tackling large-size images (*e.g.* 2K and 4K). This raises interest in compressing deep SR models to unlock their potential on resource-constrained devices.

Model quantization (Zhou et al. 2016) has emerged as a promising technology that reduces both computational overhead and memory cost with minimal performance sacrifice, where the effectiveness has been demonstrated in a wide range of high-level tasks (Zhou et al. 2016; Choi et al. 2018; Bhalgat et al. 2020; Chen et al. 2021; Gao et al. 2022; Luo et al. 2023). Some prior works design SR quantizers by adjusting the quantization range (Li et al. 2020; Zhong et al. 2022) or modeling the feature distribution (Hong et al. 2022b; Qin et al. 2024) for activations, assigning a fixed bit for diverse image regions. However, these methods overlook that the accuracy degradation from quantization can vary for different contents, where some are more sensitive to quantization, thus showing a worse tolerance for low bits.

To address this limitation, Hong *et al.* (Hong et al. 2022a) propose content-aware dynamic quantization (CADyQ)

<sup>\*</sup>Corresponding authors: Zhao Zhang (cszzhang@gmail.com) and Feng Li (fengli@hfut.edu.cn).

Copyright © 2025, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

which employs trainable bit selectors to measure the image and layer sensitivities for quantization simultaneously, as illustrated in Figure 1(a). Nevertheless, incorporating such selectors into each layer will cause additional computational costs, particularly pronounced in deep networks. Several methods (Tian et al. 2023; Lee, Yoo, and Jung 2024) improve the trained selectors in CADyQ by exploring different image characteristics of patches, which conduct once more patch-wise quantization to tackle the image sensitivity. Though some advancements have been made, such a layer-wise bit-width adaption in response to varying patches can introduce disturbances to the inter-layer relations within original models to some extent, which leads to disparities in the representations, consequently compromising the reconstruction after quantization.

These observations prompt us to consider a key question: *Can we straightly adapt quantization with the awareness of image contents while avoiding layer sensitivity?* In this context, deviating from existing methods, we rethink the quantization principle from two perspectives: 1) Granular characteristic, where fine-granularity representations reveal the texture complexity of local regions and coarse ones express structural semantics of the overall scene; 2) Entropy statistic, which reflects the average information density and the complexity of pixel distributions given patches (Shannon 1948), correlated with the image quality. Therefore, we propose a distinctive approach, dubbed Granular-DQ, which conducts low-bit dynamic quantization by harnessing the multi-granularity clues of diverse image contents to achieve efficient yet effective quantized SR models.

Granular-DQ consists of two sequential policies: one to conduct granularity-aware bit allocation for all the patches and the other is fine-grained bit-width adaption based on the entropy (see Figure 1(b)). For the former, we design a granularity-bit controller (GBC) that constructs a hierarchy of coarse-to-fine granularity representations for each patch. GBC then assigns an appropriate level of granularity to each patch, contingent upon its desired contribution percentage to the entire image, and aligns this with potential quantization bit-widths, enabling a tailored bit allocation. However, since Granular-DQ contains no bit constraint as CADyQ, relying solely on the GBC for quantization will force the network to be optimized toward reconstruction accuracy with pixel-wise supervision, leading to excessively high bits on some patches. To alleviate this, we present an entropybased fine-tuning approach on the premise of GBC, making a fine-grained bit adjustment for the patches less quantized. We capture generalized distribution statistics of the entropy across large-scale data, providing approximate entropy thresholds to establish an entropy-to-bit (E2B) mechanism. The resultant entropy thresholds are then dynamically calibrated and fine-tuned by exploiting the entropy of calibration patches as the adaption factor, achieving a more precise bit assignment. Experiments on representative CNNand transformer-based SR models demonstrate the superiority of Granular-DQ in the trade-off between accuracy and quantization efficiency over recent state-of-the-art methods. The main contributions are summarized as follows:

- For the first time, we propose Granular-DQ, a markedly different method with full explorations of the granularity and entropy statistic of images to quantization adaption, allowing complete patch-wise and layer-invariant dynamic quantization for SR models.
- We propose GBC which learns hierarchical granular representations of image patches and adaptively determines the granularity levels based on their contribution to the entire image, aligning these with suitable bit-widths.
- We propose an entropy-based fine-tuning approach upon GBC and build an E2B mechanism, which enables finegrained and precise bit adaption for the patches with excessively high bits. Granular-DQ shows preferable performance with existing methods.

