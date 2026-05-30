# *E. Ablation Study*

Impacts by K, E, and G. We evaluate the performance with configured different K (the number of top experts se-

<span id="page-8-0"></span>TABLE IV: The AlltoAll communication speedup (×) of HD2-MoE, HD-MoE and HierMoE over Megatron-LM with varied K, E, and G.

| Method  | K    |      |      | E    |      |      | G    |      |      |
|---------|------|------|------|------|------|------|------|------|------|
|         | 6    | 8    | 10   | 64   | 128  | 256  | 8    | 16   | 32   |
| HD2-MoE | 0.95 | 1.26 | 1.30 | 1.12 | 1.26 | 1.18 | 1.24 | 1.71 | 1.26 |
| HD-MoE  | 1.37 | 1.72 | 1.82 | 1.57 | 1.72 | 1.61 | 2.36 | 2.50 | 1.72 |
| HierMoE | 1.56 | 1.99 | 2.10 | 1.83 | 1.99 | 1.84 | 2.86 | 2.62 | 1.99 |

lected for each token), E (the number of experts per MoE layer), and G (the total number of GPUs of the cluster) and measure the speedups of our proposed methods over Megatron-LM as shown in Table [IV.](#page-8-0) Results indicate that HierMoE achieves speedups ranging from 1.83× to 1.99× with different E. And the speedup of HierMoE improves from 1.56× to 2.10× as K increases, highlighting a worsened token duplication issue with higher K. As G rises, the speedup of HierMoE drops. At G = 8, without internode communication, its speedup distinctly differs from the others. For G ∈ {16, 32}, the increase in nodes leads to reduced duplicate rates at the first hierarchical level, resulting in a decrease in HD2-MoE's speedup. Nevertheless, HD-MoE achieves a comparable speedup with HD2-MoE with 1.38× when G = 16 and 1.37× when G = 32, demonstrating the validity of our approach in determining the dimensions for HierD-AlltoAll. Meanwhile, compared to HD2-MoE, the speedup of HierMoE improves from 1.53× to 1.58× with increasing G from 16 to 32, highlighting the effectiveness of our HierD-ES strategy.

Performance with different dimensions. We assess the AlltoAll time cost for nine configurations to determine the influence of varying dimensions on 4 nodes and 1 node, as depicted in Fig. [13.](#page-8-2) H1-MoE, H2-MoE, H3-MoE, and H4-MoE denote the MoE layer using hierarchical AlltoAll without deduplication, whereas HD1-MoE, HD2-MoE, HD3- MoE, and HD4-MoE represent the MoE layer with hierarchical deduplication AlltoAll. Notably, experiments on 1 node only have 3 dimensions. HD-MoE corresponds to the MoE layer employing our HierD-AlltoAll. Results indicate that while hierarchical AlltoAll does not reduce communication overhead, our deduplication approach does, and our HierD-AlltoAll optimally selects the dimension.

Performance with different kinds of max functions. We also conduct a set of experiments over three kinds of max function on Eq. [\(9\)](#page-5-3), including a smooth max function on Eq. [\(11\)](#page-6-2), a standard max function and a Log-Sum-Exp function (lnP i (exp x[i])). Results show that the standard max function, smooth max function and Log-Sum-Exp achieve speed up HierMoE over HD-MoE by 1.13×, 1.17× and 1.16×. Smooth max functions improve little to the performance. We thus simply choose the best one. We also evaluate HierMoE against HD-MoE by varying γ within [5, 7, 9, 11, 13, 15, 17, 19] to assess sensitivity to the max function's smoothness in Eq. [\(11\)](#page-6-2). Results indicate a speed up of HierMoE over HD-MoE between 1.16× and 1.17×, suggesting low sensitivity to γ.

Performance with varied expert placements updating frequency. In practice, swapping two experts takes just 1% of the total end-to-end time. We find that HierMoE achieves 1.17×, 1.17×, 1.15×, and 1.13× faster than HD-MoE with an HierD-ES update frequency of every 1, 2, 4, and 8 iterations, respectively. Higher frequencies are seen to have better performance, so we choose to update HierD-ES every iteration.

