# <span id="page-5-2"></span>4 Experiments

### 4.1 Experimental Setups

Models. We conduct experiments using Vision Transformer (ViT) [\[7\]](#page-9-6) and ConvNeXt [\[8\]](#page-9-7) to validate our approach. Specifically, we transform the ImageNet 21K pre-trained dense checkpoints of ViT-S and ConvNeXt-T into the initialization weights of V-JetMoE-T and C-JetMoE-F through checkpoint recycling. As detailed in Sec. [3.1,](#page-2-0) V-JetMoE-T comprises dense layers in the first half and is equipped with SpheroMoE layers in the latter half. Each SpheroMoE layer consists of N/2 core experts and N universal experts, where N is the number of input tokens. Further details are in Appendix [A.](#page-13-0)

Datasets. We evaluate MoE Jetpack on 8 image classification datasets, including ImageNet-1K [\[25\]](#page-10-7), CIFAR-10, CIFAR-100 [\[26\]](#page-10-8), Flowers [\[27\]](#page-10-9), Pets [\[28\]](#page-10-10), STL-10 [\[29\]](#page-10-11), Food-101 [\[30\]](#page-10-12), and DTD [\[31\]](#page-10-13), encompassing a diverse range of tasks, including object classification, fine-grained species recognition, and texture classification.

Baseline Implementation. We follow the implementation details outlined by Xu et al. [\[22\]](#page-10-4) for comparisons of the dense models. For the MoE models, we employ Soft MoE [\[6\]](#page-9-5) as the baseline and have replicated it across all datasets. Our MoE Jetpack and Soft MoE utilize the same training strategies as the dense models to ensure comparison fairness. All implementations were executed using the MMpretrain framework [\[32\]](#page-10-14) on RTX4090. More information can be found in Appendix [B.](#page-13-1)

## 4.2 Main Results

Tab. [1](#page-5-0) compares the performance of the MoE Jetpack with Dense ViT models (trained from scratch and with pre-trained weights on ImageNet-21k) and Soft MoE models (trained from scratch) on various image classification datasets using ViT-T (a) and ConvNeXt-F (b) architectures. All models maintain approximately the same number of FLOPs. The MoE Jetpack, which inherits the knowledge from dense checkpoints pre-trained on ImageNet-21k, consistently outperforms MoE models trained from scratch, especially on smaller datasets. These results highlight the effectiveness of MoE Jetpack.

Table 1: Performance comparison on visual recognition tasks with ViT-T and ConvNeXt-F.

<span id="page-5-0"></span>

| Dataset (↓) |      |      | Dense Dense (21k) Soft MoE [6] MoE Jetpack |              |      |      | Dense Dense (21k) Soft MoE [6] MoE Jetpack |              |
|-------------|------|------|--------------------------------------------|--------------|------|------|--------------------------------------------|--------------|
| ImgNet-1k   | 73.9 | 75.6 | 77.1                                       | 79.9 (+2.8)  | 76.1 | 76.4 | 79.1                                       | 80.5 (+1.4)  |
| Food-101    | 79.6 | 86.9 | 82.0                                       | 89.5 (+7.5)  | 86.9 | 89.0 | 88.7                                       | 90.7 (+2.0)  |
| CIFAR-10    | 92.4 | 97.0 | 92.9                                       | 97.9 (+5.0)  | 96.6 | 97.4 | 97.3                                       | 98.2 (+0.9)  |
| CIFAR-100   | 72.3 | 81.4 | 75.9                                       | 88.4 (+12.5) | 81.4 | 84.4 | 82.8                                       | 88.5 (+5.7)  |
| STL-10      | 61.5 | 83.4 | 67.7                                       | 95.3 (+27.6) | 81.4 | 92.3 | 79.4                                       | 98.7 (+19.3) |
| Flowers     | 62.4 | 81.9 | 70.8                                       | 95.4 (+24.6) | 80.3 | 94.5 | 83.3                                       | 98.6 (+15.3) |
| Pets        | 25.0 | 68.6 | 45.5                                       | 84.3 (+38.8) | 72.9 | 87.3 | 77.4                                       | 94.9 (+17.5) |
| DTD         | 49.4 | 62.5 | 51.3                                       | 69.1 (+17.8) | 63.7 | 68.8 | 64.7                                       | 79.5 (+14.8) |

(a) ViT-T (b) ConvNeXt-F

#### <span id="page-6-3"></span>4.3 Ablations

We perform ablation studies to assess the impact of various components and hyperparameters within the MoE Jetpack. By default, we use a ViT-T model with the SpheroMoE layer integrated from layers 7 to 12, comprising 98 core experts and 196 universal experts (detailed in Appendix A). The Checkpoint Recycling method transforms dense checkpoints of ViT-S and ViT-T, pre-trained on ImageNet-21k, into initial weights for our V-JetMoE-T model.

Effect of MoE Jetpack Components. We conducted the ablation of two key components of the MoE Jetpack on three datasets. As shown in Tab. 2, integrating Checkpoint Recycling with the Soft MoE baseline significantly improves performance across all datasets, with a mean accuracy increment of 9.8%. The SpheroMoE layer further enhances performance, achieving a mean accuracy of 87.9%. These results demonstrate the efficacy of both components, especially when used together, highlighting their synergistic effect in boosting performance.

Checkpoint Recycling vs. Sparse Upcycling. To compare the four checkpoint recycling strategies mentioned in Sec. 3.1 and the method of using duplicated MLPs to construct experts in Sparse Upcycling [16], we conducted experiments on ImageNet. For fairness, we also employed our SpheroMoE layer in the Sparse Upcycling. The results, summarized in

Table 3: Checkpoint Recycling vs. Sparse Upcycling

<span id="page-6-0"></span>

| Method                  | Construction                                                                            | ImageNet                            |
|-------------------------|-----------------------------------------------------------------------------------------|-------------------------------------|
| Sparse Upcycling [16]   | Сору                                                                                    | 79.1                                |
| Checkpoint<br>Recycling | Random Sampling<br>Uniform Selection<br>Graph Partitioning<br>Importance-based Sampling | 79.5<br>79.6<br>79.8<br><b>79.9</b> |

Tab. 3, show that Importance-Based Sampling achieves the highest performance, demonstrating its effectiveness in leveraging critical weights to enhance model performance and convergence speed. Additionally, Checkpoint Recycling is highly flexible, allowing the construction of experts of varying sizes to meet different needs, a feature not provided by sparse upcycling.

Core Experts Ratio. To assess the impact of the Adaptive Dual-path MoE structure introduced in Sec. 3.2 on the accuracy of MoE models, we aimed to determine the ideal balance between performance and resource allocation. We conducted experiments on the Cifar-100 dataset with a constant number of total experts, varying the ratio of core experts. The results, illustrated in Fig. 4, indicate that optimal accuracy is achieved when the proportion of core experts is set at 1/3.

<span id="page-6-2"></span>![](_page_6_Figure_8.jpeg)

Figure 4: This chart shows CIFAR-100 accuracy across different ratios of core (dark) to universal (light) experts, highlighting optimal performance at a 1/3 core ratio.

## Different MoE Jetpack Configurations.

This part evaluates the impact of various MoE Jetpack configurations on model performance, as summarized in Tab. 4. The experiments focus on the placement of SpheroMoE layers, the number of experts per layer, and the base size of converted dense checkpoints. Results indicate that more SpheroMoE layers generally enhance performance, though placing it before layer 7 slightly hurt the performance. Consequently, SpheroMoE layers were incorporated into layers 7–12. Additionally, models with more experts exhibit improved accuracy, highlighting the benefits of increased expert specialization and diversity. Models converted from larger dense checkpoints demonstrate superior performance. These findings suggest that MoE network performance can be improved by increasing the number of MoE layers, incorporating more experts, and utilizing larger base models.

Table 2: Ablation Study on MoE Jetpack Components.

<span id="page-6-1"></span>

| Soft MoE [6] | Checkpoints Recyclin | ng   SpheroMoE | ImageNet             | CIFAR-100            | Flowers | Mean Acc.                                   |
|--------------|----------------------|----------------|----------------------|----------------------|---------|---------------------------------------------|
|              | Baseline ViT-T       |                | 73.9                 | 72.3                 | 62.4    | 69.5                                        |
| <b>√</b> ✓   | <b>√</b> √           |                | 77.1<br>78.4<br>79.9 | 75.9<br>84.7<br>88.4 | 91.2    | 74.6 (+5.1)<br>84.8 (+15.3)<br>87.9 (+18.4) |

Table 4: Comparison of Model Variants with Different Configurations

<span id="page-7-2"></span><span id="page-7-0"></span>

| model          | Weight Init. | MoE Layers | Expert Number       | Param (M) | FLOPs (G) | CIFAR-100 | ImageNet |
|----------------|--------------|------------|---------------------|-----------|-----------|-----------|----------|
| ViT-T          | -            | -          | -                   | 6         | 1.1       | 72.3      | 73.9     |
| Soft MoE-T [6] | -            | 7:12       | 197                 | 354       | 1.2       | 75.9      | 77.1     |
| Soft MoE-S [6] | -            | 7:12       | 197                 | 1412      | 4.5       | 77.5      | 80.3     |
| ViT-T          | <b> </b>     | -          | -                   | 6         | 1.1       | 81.4      | 75.5     |
| V-JetMoE-T     | <b>√</b>     | 11:12      | core: 98, univ: 196 | 92        | 1.1       | 87.4      | -        |
| V-JetMoE-T     | <b>√</b>     | 9:12       | core: 98, univ: 196 | 179       | 1.1       | 87.8      | -        |
| V-JetMoE-T     | ✓            | 5:12       | core: 98, univ: 196 | 352       | 1.2       | 86.7      | -        |
| V-JetMoE-T     | <b> </b>     | 7:12       | core: 32, univ: 64  | 89        | 0.8       | 87.8      | -        |
| V-JetMoE-T     | ✓            | 7:12       | core: 64, univ: 128 | 175       | 1.0       | 88.0      | -        |
| V-JetMoE-T     | <b> </b>     | 7:12       | core: 98, univ: 196 | 265       | 1.1       | 88.4      | 79.9     |
| V-JetMoE-S     | ✓            | 7:12       | core: 98, univ: 196 | 1058      | 4.3       | 89.9      | 82.4     |

#### 4.4 Analysis

In this section, we investigate the influence of the MoE Jetpack on enhancing the convergence speed of MoE models when trained on the ImageNet and CIFAR-100 datasets. Additionally, we provide some intuition regarding the attention patterns of the experts and the contribution of each expert to the final results.

Accelerating MoE Convergence with MoE Jetpack. The impact of MoE Jetpack on convergence speed is evident in Fig. 5 for ImageNet (left) and CIFAR-100 (right). In both cases, models with MoE Jetpack reach the target accuracy significantly faster. For ImageNet, the model with MoE Jetpack achieves approximately 77% top.1 accuracy before 150 epochs, 2 times faster than training from scratch. Notably, for smaller datasets like CIFAR-100, the acceleration effect of MoE Jetpack is more pronounced: The model with MoE Jetpack reaches the 76% top.1 accuracy at around 40 epochs, 8 times faster than the model without it. These results demonstrate that MoE Jetpack substantially accelerates convergence speed, enhancing fine-tuning efficiency and reducing computational resources.

Intuition of Expert Attention Patterns. We visualize the attention maps of experts in Fig. 6(a), which illustrates that different experts focus on different parts of the input image. This diversity in attention suggests that each expert specializes in capturing unique aspects of the input, enhancing the model's ability to represent features comprehensively. The specialization allows the MoE model to combine multiple perspectives, resulting in a more robust and detailed understanding of the input.

Contribution of Each Expert to Final Results. Fig. 6(b) demonstrates the varying contributions of core and universal experts across different layers of the MoE model. Core experts show an increasing influence in the later layers, emphasizing their role in refining specific and highly relevant features. Additionally, the contributions among core experts are markedly uneven, some experts can impact output tokens  $17 \times$  more than others, reflecting greater specialization and diversity in their focus areas. In contrast, universal experts maintain a relatively consistent contribution level, indicating a more uniform integration of broader contextual information throughout the network. This hierarchical structure, balancing the specialized refinement by core experts with the generalized understanding provided by universal experts, enhances the model's overall performance and robustness.

<span id="page-7-1"></span>![](_page_7_Figure_7.jpeg)

Figure 5: Comparison of convergence speeds using MoE Jetpack versus training from scratch on ImageNet (left) and CIFAR-100 (right). MoE Jetpack achieves target accuracies significantly faster, demonstrating a 2x speed increase on ImageNet and an 8x increase on CIFAR-100.

<span id="page-8-1"></span><span id="page-8-0"></span>![](_page_8_Figure_0.jpeg)

Figure 6: (a) This figure illustrates the attention maps generated by five experts in response to an input image, highlighting the experts' specialization. (b) These line charts show varying contributions of core and universal experts, with core experts' influence peaking in later layers, emphasizing their detailed feature refinement, contrasted with the consistent input of universal experts.

