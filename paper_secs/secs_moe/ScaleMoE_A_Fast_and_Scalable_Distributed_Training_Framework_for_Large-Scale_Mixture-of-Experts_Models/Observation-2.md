# Observation 2

In Transformer MoE models, there is a huge load imbalance in expert selection. The expert selection becomes more imbalanced as both the number of experts and the ratio of MoE layers increase. Also, the expert selection is quickly skewed as training progresses.

Performance implications of expert imbalance. We identify that the expert selection imbalance degrades the end-toend performance of the existing distributed training frameworks (Figure 5). It mainly stems from two factors: increased all-to-all communication overhead and underutilization of computational resources. Figure 5a shows that the all-to-all communication overhead increases as training progresses. We observe that the expert selection imbalance correlates with allto-all communication overhead. This occurs because all-to-all communication requires equal message sizes across GPUs. When imbalance arises, the framework adds zero padding to make message sizes uniform. As the imbalance grows, the amount of zero padding increases, resulting in larger communication volumes and higher all-to-all communication overhead. Moreover, the framework also suffer from GPU resource underutilization due to the expert selection imbalance. Figure 5b illustrates how the expert selection imbalance leads to GPU underutilization, degrading the overall system performance. For example, given 9 input tokens, the gating network selects expert-1 for 3 tokens and expert-3 for 6 tokens. Since all expert outputs should be ready before the second all-to-all communication, GPUs handling fewer computations must wait till the most heavily-loaded GPU is finished.

## Observation 3

The expert selection imbalance adversely affects the overall system performance, leading to increased all-to-all communication latency and reduced GPU resource utilization.

