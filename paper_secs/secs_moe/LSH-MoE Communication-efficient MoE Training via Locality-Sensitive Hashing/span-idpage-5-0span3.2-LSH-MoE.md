# <span id="page-5-0"></span>3.2 LSH-MoE

Motivated by the *Token Similarity* observed in Section [3.1,](#page-4-1) we introduce LSH-MoE, a novel MoE training framework that integrates locality-sensitive hashing (LSH) for rapid clustering of input tokens. Our method transmits only the clustering centroids, significantly reducing communication volumes. To counteract the negative effects of compression, we also implement a residual-based error compensation scheme.

As depicted in Figure [5,](#page-4-2) LSH-MoE initially employs (1) an LSH-based clustering method to compress *tokens* into *centriods* for subsequent processing, effectively reducing communication overhead. It then sequentially executes (2) all-to-all communication, expert computation, and another (3) allto-all communication to produce the processed outputs *E(centriods)*. Finally, it introduces (4) a residual-based error compensation method to approximate the expert-processed results *E(tokens)*, by integrating *E(centriods)* with *residuals*. Meanwhile, we also outline the workflow of our LSH-MoE framework in the Algorithm [1](#page-13-0) of Appendix [A.1.](#page-13-1) The key components of our LSH-MoE framework includes an efficient LSH-based clustering algorithm for rapid processing and an residual-based error compensation scheme to minimize quality degradation.

Efficient LSH-based Clustering Algorithm. Since the data to be compressed (the input data for all-to-all communication) is generated dynamically and in real time, pre-compressing it or overlapping compression time with other processing tasks is not feasible. Consequently, selecting an efficient online compression algorithm is crucial. Traditional clustering algorithms, such as K-Means, often encounter computational challenges and efficiency limitations. Locality-sensitive hashing (LSH) address these issues by hashing similar data points into the same buckets, enabling faster similarity detection in high-dimensional spaces.

Numerous LSH algorithms have been developed, each employing a unique hashing approach for mapping data onto buckets. We conducted experiments to evaluate several popular hashing algorithms, including *cross-polytope hashing* and *spherical hashing*. Based on our evaluations in Section [4.5,](#page-8-0) we selected *cross-polytope hashing* as the optimal algorithm for our application. *Cross-polytope hashing* stands out for its method of mapping input vectors to the nearest vertex on a cross-polytope. This process is facilitated by applying randomly rotated cross-polytopes, which effectively segment the surface of the unit sphere. The algorithm can be mathematically represented as follows:

$$LSH(\mathbf{x}) = \operatorname{argmax}_{i \in \{\pm 1, \pm 2, \dots, \pm d\}} |\mathbf{R}\mathbf{x}|_{i}$$
(3)

where R is a random rotation matrix, d is the dimensionality of the space, and |Rx|<sup>i</sup> denotes the absolute value of the i-th component of the rotated vector Rx.

This formula encapsulates how the input vector x is transformed by the rotation matrix R and then mapped to the nearest vertex of the cross-polytope by selecting the dimension i that maximizes the absolute value of the components of Rx. This method effectively segments the high-dimensional space and enhances the clustering efficiency by rapidly identifying similar data points.

Residual-based Error Compensation Scheme. In our LSH-MoE framework, we compress the intermediate activation values within the model network. Unlike gradient compression, this process does not tolerate errors well. Therefore, it is essential to minimize compression-induced errors to ensure minimal impact on model performance. To address this, we implement a novel residual-based gradient compensation strategy, outlined as follows:

1. We first capture the residual for each data point relative to its cluster centroid, defined by the equation:

$$\Delta \text{cluster}_j \leftarrow \{x - \overline{\text{cluster}}_j \mid x \in \text{cluster}_j\}. \tag{4}$$

2. After the expert network computes outputs for the cluster centers, the final step is to restore the processed result for each token by adding back the previously recorded residual:

$$Y_{ij} \leftarrow \{E(\overline{\text{cluster}}_j) + \Delta \text{Cluster}_{jk} \mid k = 1, 2, \dots, N_j\}.$$
 (5)

This error compensation scheme effectively mitigates potential accuracy loss caused by data compression in all-to-all communication, ensuring the fidelity and robustness of the LSH-MoE framework. The experimental results in Section [4](#page-6-1) show that implementing this compensation mechanism enables

<span id="page-6-0"></span>

| Model         | #Layer | $d_{\mathrm{model}}$ | $\mathbf{d}_{\mathbf{ffn}}$ | #Experts | #Params. (MoE) | #Params. (Total) |
|---------------|--------|----------------------|-----------------------------|----------|----------------|------------------|
| RoBERTa-MoE   | 12     | 768                  | 3072                        | 16       | 302M           | 394M             |
| T5-MoE        | 16     | 1024                 | 16384                       | 16       | 8594M          | 9288M            |
| GPT-MoE (15B) | 12     | 768                  | 3072                        | 512      | 14507M         | 14629M           |
| GPT-MoE (52B) | 24     | 1024                 | 4096                        | 512      | 51539M         | 51740M           |
| Swin-MoE-L    | 24     | -                    | -                           | 32       | -              | 946M             |

the model trained with LSH-MoE to achieve an accuracy comparable to that of a model trained without compression. This outcome highlights the effectiveness of our proposed error compensation strategy in preserving model performance despite the challenges posed by data compression in all-to-all communication.

#### 3.3 Scalability Analysis of LSH-MoE

To effectively demonstrate the scalability of our approach, particularly in terms of its applicability to both larger models and larger computational clusters, we conducted a theoretical analysis. This analysis primarily focuses on the **computation overhead** and the communication costs associated with Mixture of Experts (MoE), specifically considering **all-to-all communication overhead**. We derived the ratio of communication time to computation time, highlighting how this ratio evolves as both the scale of the servers and the model size increase. This relationship is crucial for understanding scalability and can be formally expressed as follows:

$$\frac{T_{all\_to\_all}}{T_{compute}} = \frac{\text{FLOPs}}{6B_{inter}} \times \frac{k}{1+2k} \times \frac{w-1}{wh}$$
 (6)

where k represents the number of experts activated per token, FLOPs and  $B_{inter}$  denote the GPU's computation ability and the network performance, w is the number of GPU servers, and h is the hidden size of model. Notably, the first term,  $\frac{\text{FLOPs}}{6B_{inter}}$ , remains constant under fixed hardware conditions. Additionally, scaling MoE models typically emphasizes increasing the number of layers and experts, while the growth in hidden size (h) tends to be gradual, as seen in models like Switch-Transformer [9]. Consequently, when both the model scale and the number of training servers grow, the proportion of all-to-all communication time remains nearly unchanged. This insight underpins the scalability of the LSH-MoE method, demonstrating its robustness in larger-scale settings and supporting its potential in future large-scale applications. For a detailed derivation, please refer to Appendix A.2.

