# <span id="page-20-0"></span>F Open Discussion and Future Works

Performance of the gating model. The performance of GraphMETRO depends in part on how effectively the gating model can identify distribution shifts from the transform functions. Some functions,

<span id="page-21-0"></span>

|                     | DBLP  |         |            | CiteSeer |         |            |
|---------------------|-------|---------|------------|----------|---------|------------|
|                     | ERM   | ERM-Aug | GraphMETRO | ERM      | ERM-Aug | GraphMETRO |
| i.i.d. (0)          | 85.71 | 85.66   | 85.92      | 75.80    | 76.00   | 78.01      |
| random subgraph (1) | 84.48 | 85.29   | 85.78      | 75.47    | 75.82   | 77.01      |
| drop node (2)       | 71.08 | 74.85   | 76.61      | 62.21    | 63.89   | 66.22      |
| drop edge (3)       | 79.69 | 82.34   | 82.95      | 71.48    | 73.24   | 77.00      |
| add edge (4)        | 83.41 | 84.44   | 84.98      | 74.29    | 74.87   | 77.26      |
| noisy features (5)  | 76.90 | 72.81   | 81.32      | 85.28    | 82.97   | 88.43      |
| (1, 3)              | 77.63 | 81.04   | 81.71      | 70.37    | 71.42   | 74.97      |
| (2, 3)              | 81.99 | 83.65   | 84.26      | 73.60    | 74.06   | 76.11      |
| (1, 4)              | 79.69 | 68.62   | 80.31      | 84.47    | 86.36   | 88.56      |
| (2,4)               | 70.55 | 74.01   | 75.10      | 62.13    | 63.53   | 65.73      |
| (1, 5)              | 71.52 | 68.27   | 71.05      | 66.89    | 62.59   | 67.32      |
| (2, 5)              | 77.73 | 81.13   | 81.85      | 70.19    | 72.21   | 76.77      |
| (3, 5)              | 79.59 | 84.49   | 87.14      | 78.24    | 73.29   | 89.18      |
| (4, 5)              | 70.40 | 74.16   | 76.18      | 61.64    | 63.53   | 66.42      |
| Average             | 77.88 | 78.63   | 81.08      | 72.29    | 72.41   | 76.36      |

Table 6: Numerical results on synthetic node classification datasets

<span id="page-21-1"></span>

|                     | IMDB-MULTI |         |            | REDDIT-BINARY |         |            |
|---------------------|------------|---------|------------|---------------|---------|------------|
|                     | ERM        | ERM-Aug | GraphMETRO | ERM           | ERM-Aug | GraphMETRO |
| i.i.d. (0)          | 50.17      | 49.28   | 49.16      | 72.93         | 73.02   | 75.94      |
| random subgraph (1) | 34.30      | 39.94   | 45.86      | 62.59         | 69.03   | 71.22      |
| drop node (2)       | 50.42      | 48.73   | 48.83      | 70.01         | 72.27   | 72.26      |
| drop edge (3)       | 49.66      | 48.94   | 48.83      | 59.13         | 70.55   | 72.51      |
| add edge (4)        | 49.64      | 48.14   | 48.90      | 65.18         | 67.28   | 69.34      |
| noisy features (5)  | 50.17      | 49.28   | 49.16      | 68.66         | 68.50   | 66.79      |
| (2, 3)              | 34.55      | 40.32   | 45.11      | 58.72         | 64.06   | 66.50      |
| (1, 4)              | 34.32      | 40.28   | 46.01      | 59.40         | 62.81   | 65.29      |
| (2, 4)              | 34.57      | 40.17   | 46.79      | 61.34         | 66.02   | 66.71      |
| (1, 5)              | 49.31      | 48.36   | 48.68      | 65.89         | 66.88   | 68.09      |
| (2, 5)              | 50.51      | 48.78   | 48.79      | 68.72         | 69.77   | 68.76      |
| (3, 5)              | 49.38      | 47.72   | 48.35      | 55.36         | 65.21   | 64.87      |
| (1, 3)              | 48.72      | 48.36   | 48.76      | 61.08         | 61.71   | 62.57      |
| (4, 5)              | 34.62      | 39.88   | 46.15      | 62.99         | 68.68   | 68.34      |
| Average             | 44.31      | 45.58   | 47.82      | 63.71         | 67.56   | 68.51      |

Table 7: Numerical results on synthetic graph classification datasets

like adding node feature noise and extracting random subgraphs, are inherently disentangled, making it easy for the gating model to differentiate between these distributions. Other functions, such as dropping paths and dropping edges, may be more similar, but the method remains robust as long as each expert produces the corresponding invariant representation. More complex combinations of transforms pose a greater challenge for the gating model's expressiveness. To address this, initializing the gating model with a pre-trained model from a diverse dataset may enhance its ability to predict mixtures, improving performance on unseen graphs.

Comparison with invariant learning methods. GraphMETRO differs from traditional invariant learning, where environments are constructed using environment variables. Instead, GraphMETRO views distribution shifts on an instance as a mixture, represented by the score vector from the gating function. This approach enables the creation of infinite environments, as the score vector is continuous. When restricting the gating function to binary outputs, GraphMETRO can simulate finite environments, akin to the environment construction in invariant learning. Additionally, the concept of referential invariant representation using the base model  $\xi_0$  sets GraphMETRO apart from previous invariant learning approaches.

**Applicability of GraphMETRO** . A key question is how well the predefined transform functions capture complex distribution shifts.

• General domain: In our experiments, we primarily use five universal graph augmentations (as listed in [82]). Our code also includes additional transforms (Appendix C). While these transforms are not exhaustive, they cover a wide range of shifts observed in our results. However, real-world

distribution shifts may go beyond the predefined transforms, and in such cases, GraphMETRO might struggle to capture and mitigate unknown shifts. This is a limitation when the test distribution or domain knowledge is insufficient.

• Specific domains: In certain domains, additional knowledge can help infer distribution shifts, such as an increase in malicious users in a trading system. This knowledge can guide the construction of transform functions to better cover the target distribution shifts. Specifically, two sources of knowledge can be used: i) Domain knowledge, *e.g.,* in molecular datasets, transform functions could add carbon structures to molecules while preserving functional groups, or in social networks, known user behaviors can guide transformations. ii) Leveraging samples from the target distribution (*i.e.,* domain adaptation), where samples from the target can inform the selection of relevant transforms. For example, by measuring the distance between the extrapolated datasets under specific transforms and the target samples in the embedding space, more relevant transform functions can be selected. This presents an interesting direction for future work.

Label distribution shifts. In this work, we focus on distribution shifts in graph structures and features. Extending GraphMETRO to handle label distribution shifts would be a complementary and interesting direction. Label shifts affect various modalities, including graphs and images, and existing methods [\[44,](#page-12-18) [5\]](#page-10-17) designed for label shifts could be integrated into our framework with minimal adjustments, such as modifying the loss function or training pipeline.

