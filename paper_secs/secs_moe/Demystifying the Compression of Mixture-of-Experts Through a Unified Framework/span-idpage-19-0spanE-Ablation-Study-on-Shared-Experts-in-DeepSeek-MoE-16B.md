# <span id="page-19-0"></span>E Ablation Study on Shared Experts in DeepSeek-MoE-16B

Although most MoE models follow Equation 2 to implement the experts, models like DeepSeek-MoE-16B adopt a residual (Rajbhandari et al., 2022) form of experts, which brings a special scenario to discuss. In the residual MoE, an additional set of m shared experts  $\{\bar{E}_1, \bar{E}_2, \dots, \bar{E}_m\}$  is always selected by the router G and activated for all inputs. Given an input x, the output can be represented as a degenerated form of Equation 2, where the scores of shared experts are fixed to 1:

$$y = \sum_{i \in \mathcal{K}} G(x)_i \cdot E_i(x) + \sum_{j=1}^m \bar{E}_j(x).$$
(12)

This special form of expert routing may bring a difference in the redundancy distribution of MoE. Here we discuss the influence of shared experts through pruning and present the results in Table 8. We find that pruning without the shared experts will boost the performance at a considerable scale, i.e., +3.6% and +1.5% of the averaged accuracy for unstructured pruning with Wanda and SparseGPT, respectively. This finding reveals a different pattern of inner redundancy in which the shared experts are less compressible compared to the others in residual MoE models, which may inform future work.

<span id="page-19-1"></span>Table 8: Ablation Study of Pruning Shared Experts on DeepSeek-MoE-16B. We consider two scenarios, i.e., pruning both shared experts and normal experts ("w/Pruning Shared Experts") and pruning normal experts only ("w/o Pruning Shared Experts"). We use two mainstream pruning methods (i.e., Wanda (Sun et al., 2023) and SparseGPT (Frantar & Alistarh, 2023)) under both unstructured sparsity (50%) and semi-structured sparsity (2:4).

|                           | ${\bf Deep Seek\text{-}MoE\text{-}16B}$ |       |       |             |           |       |               |      |            |      |  |  |
|---------------------------|-----------------------------------------|-------|-------|-------------|-----------|-------|---------------|------|------------|------|--|--|
| Method                    | Sparsity                                | ARC-C | BoolQ | HellaSwag   | MMLU      | OBQA  | PIQA          | RTE  | WinoGrande | Avg. |  |  |
| Baseline                  | 0%                                      | 48.1  | 72.4  | 77.3        | 37.9      | 44.0  | 80.4          | 63.9 | 70.3       | 61.8 |  |  |
| w/ Pruning Shared Experts |                                         |       |       |             |           |       |               |      |            |      |  |  |
| Wanda                     | 50%                                     | 43.6  | 74.3  | 72.6        | 31.1      | 43.0  | 79.5          | 58.1 | 69.4       | 59.0 |  |  |
| SparseGPT                 | 3070                                    | 43.9  | 73.5  | 74.0        | 33.8      | 41.4  | 79.0          | 61.0 | 68.3       | 59.4 |  |  |
| Wanda                     | 2:4                                     | 38.2  | 66.1  | 67.5        | -7.6      | 39.4  | $-77.\bar{0}$ | 53.8 | 66.7       | 54.5 |  |  |
| ${\bf SparseGPT}$         | 2:4                                     | 43.1  | 68.9  | 71.6        | 27.6      | 41.6  | 78.3          | 57.4 | 66.6       | 56.9 |  |  |
|                           |                                         |       | ,     | w/o Pruning | Shared Ex | perts |               |      |            |      |  |  |
| Wanda                     | F007                                    | 44.0  | 76.3  | 73.5        | 36.2      | 41.0  | 79.3          | 59.9 | 70.2       | 60.0 |  |  |
| SparseGPT                 | 50%                                     | 45.0  | 75.5  | 74.4        | 36.3      | 41.0  | 79.4          | 64.3 | 69.3       | 60.7 |  |  |
| Wanda                     | 9.4                                     | 40.1  | 75.7  | 69.9        | 33.5      | -40.0 | -77.9         | 58.8 | 68.6       | 58.1 |  |  |
| ${\bf SparseGPT}$         | 2:4                                     | 40.7  | 75.7  | 69.9        | 33.3      | 39.0  | 77.7          | 61.4 | 69.4       | 58.4 |  |  |

