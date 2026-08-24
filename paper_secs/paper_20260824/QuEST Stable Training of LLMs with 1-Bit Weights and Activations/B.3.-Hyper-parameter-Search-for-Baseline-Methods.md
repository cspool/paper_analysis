# B.3. Hyper-parameter Search for Baseline Methods

> **[图片提取文字 (无描述)]:**
> LR Scale Alpha Weight Decay Validation Loss 4.15 4e-1 4e-1 3e-1 4.10 3e-1 -2e-1 -2e-1 -4.05 4.00 1e-1 1e-1 8e-2 7e-2 6e-2 8e-2 7e-2 3.95 6e-2 5e-2 3.90 4e-2 -3e-2 3.85 3e-2 2e-2 2e-2 3.75
![](_page_14_Figure_7.jpeg)

<span id="page-14-1"></span>Figure 11. Hyperparameter search for PACT on a 30M parameter model with 4-bit weights and activations, trained on 10% of the dataset. The search explores different values for learning rate scaling (LR Scale) and alpha weight decay, with validation loss indicated by the color gradient. Lower validation loss (darker colors) corresponds to better configurations.

To ensure fair comparisons between QuEST and prior QAT methods, we conducted hyperparameter searches for both PACT and LSQ. Given PACT's instability at lower bitwidths, we extensively tuned two key hyperparameters: weight decay and learning rate scaling s for the quantization parameter α (i.e., η<sup>α</sup> = s × η). Figure [11](#page-14-1) shows the loss achieved across different weight decay and LR scale values.

For LSQ, we only tuned weight decay, as the LSQ formulation already applies scaling internally to the gradient of α, making additional learning rate adjustments unnecessary. Table [5](#page-15-2) summarizes the results of the weight decay search across 2-bit, 3-bit, and 4-bit LSQ models, where the best-performing configuration (highlighted in bold) was used for final model comparisons.

| Weight Decay | 2-bit PPL↓ | 3-bit PPL↓ | 4-bit PPL↓ |
|--------------|------------|------------|------------|
| 0.001        | 37.02      | 31.10      | 27.93      |
| 0.01         | 36.91      | 30.89      | 27.72      |
| 0.1          | 36.54      | 30.26      | 27.51      |
| 1.0          | 38.12      | 31.16      | 28.67      |

Table 5. Weight decay hyperparameter search results for LSQ across different bitwidths of 30M model. The best-performing setting is highlighted in bold.

Our hyperparameter search ensured that LSQ and PACT were tuned optimally before comparing against QuEST, leading to a fair evaluation of performance across all tested quantization methods.

