# <span id="page-14-0"></span>**C.1 Component Ablation**

To clearly show the contribution of each component of PreMoE, we conduct an ablation study on DeepSeek-R1 at 50% sparsity. Table [C.1](#page-14-1) progressively adds each component to the baseline.

<span id="page-14-1"></span>Table C.1: Component ablation on DeepSeek-R1 at 50% sparsity. Each row adds one component to the previous configuration.

| Method                                                                           | MATH-500                | GPQA                    | LCB                     |
|----------------------------------------------------------------------------------|-------------------------|-------------------------|-------------------------|
| All Logits (baseline)                                                            | 3.60                    | 28.79                   | 0.00                    |
| + TopK filtering (Act-Logits)<br>+ Threshold Filtering<br>+ Logit Transformation | 88.20<br>97.00<br>95.00 | 48.48<br>68.18<br>58.08 | 52.94<br>67.28<br>58.09 |
| + All (PreMoE)                                                                   | 97.60                   | 72.22                   | 66.36                   |

**Key observations:** (1) TopK filtering dramatically improves over using all logits (+84.6 on MATH-500). (2) Threshold filtering alone provides a large boost (+8.8 on MATH-500, +19.7 on GPQA). (3) Logit transformation alone also helps but less than filtering. (4) Combining all components yields the best overall performance, demonstrating their complementary benefits.

