# <span id="page-18-0"></span>A.11 Task-Preference and Sparsity in Experts-Selection

In Section 3.3, we calculate the expert selection frequency and pairwise cosine similarity of MoE models across different datasets for four types of reasoning tasks, providing a macroscopic view of the task preferences in expert selection. In this section, we delve deeper into the microscopic details, specifically discussing the expert selection preferences of the Phi3.5-moe and Deepseek-moe-16b-base models across 8 datasets spanning 4 task types. This analysis also aligns with previous work (Li et al., 2024a), further demonstrating their sparsity. As shown in Figure 10, the four rows, from top to bottom, correspond to four different task

types ((QA/CR), Math, Code, Specific Language). For each row, the left and right plots respectively show the expert selection frequencies of Phi3.5moe on two different datasets of the same task type. From this, a clear pattern can be observed, indicating that the MoE model exhibits remarkably similar expert selection preferences across datasets within the same task category. For example, in the first row, as illustrated in the left and right subfigures, certain experts such as Expert13 in Layer2, Expert9 in Layer8, and Expert12 in Layer11 are frequently selected for the openbookga and arcchallenge datasets, with average selection frequencies exceeding 30% (note that each layer has 16 experts, and a completely balanced selection would result in a frequency of 6.25% per expert). Conversely, some experts, such as Expert1 in Layer5, Expert7 in Layer14, and Expert8 in Layer27, are rarely selected, with frequencies below 1%.

Similarly, in the second row, for the Math

Table 21: Detailed results of average accuracy of Deepseek-moe-16b-base on 8 Zero-Shot tasks under quantization and pruning.

| Bits  | Method         | PIQA  | ARC-E | ARC-C | BOOLQ | HS    | WG    | MATHQA | MMLU  | Avg   |
|-------|----------------|-------|-------|-------|-------|-------|-------|--------|-------|-------|
| 16.00 | Full Precision | 80.52 | 73.19 | 47.53 | 72.57 | 77.43 | 69.93 | 31.66  | 38.18 | 61.37 |
|       | EAC-MoE        | 77.04 | 69.11 | 41.30 | 72.55 | 69.85 | 67.48 | 28.61  | 28.07 | 56.75 |
| 2.06  | EAC-MoE∗       | 75.19 | 67.93 | 39.93 | 70.45 | 66.55 | 63.85 | 27.77  | 26.81 | 54.81 |
|       | EAC-MoE        | 78.29 | 69.15 | 42.32 | 74.50 | 66.55 | 63.85 | 27.77  | 26.81 | 54.81 |
| 2.54  | EAC-MoE∗       | 76.17 | 71.74 | 41.13 | 71.74 | 68.31 | 66.85 | 28.11  | 30.57 | 56.83 |
|       | EAC-MoE        | 79.54 | 73.15 | 46.16 | 75.02 | 75.55 | 70.24 | 31.59  | 37.45 | 61.09 |
| 3.03  | EAC-MoE∗       | 77.20 | 71.55 | 45.05 | 72.51 | 72.86 | 66.46 | 31.56  | 33.33 | 58.82 |

<span id="page-19-1"></span>Table 22: Detailed results of average accuracy of Qwen1.5-MoE-A2.7B on 8 Zero-Shot tasks under quantization and pruning.

| Bits  | Method         | PIQA  | ARC-E | ARC-C | BOOLQ | HS    | WG    | MATHQA | MMLU  | Avg   |
|-------|----------------|-------|-------|-------|-------|-------|-------|--------|-------|-------|
| 16.00 | Full Precision | 80.79 | 69.44 | 44.37 | 79.57 | 77.17 | 69.77 | 35.57  | 61.08 | 64.72 |
|       | EAC-MoE        | 79.16 | 65.07 | 42.83 | 75.01 | 72.44 | 68.19 | 32.63  | 54.12 | 61.18 |
| 2.06  | EAC-MoE∗       | 77.91 | 64.52 | 42.15 | 74.86 | 70.94 | 67.09 | 32.26  | 50.46 | 60.02 |
|       | EAC-MoE        | 78.78 | 66.75 | 43.17 | 72.57 | 73.53 | 68.19 | 32.29  | 56.01 | 61.41 |
| 2.54  | EAC-MoE∗       | 77.53 | 65.24 | 40.44 | 74.89 | 72.49 | 68.27 | 32.16  | 53.52 | 60.57 |
|       | EAC-MoE        | 80.41 | 66.92 | 42.15 | 76.88 | 75.81 | 69.22 | 31.72  | 58.72 | 62.73 |
| 3.03  | EAC-MoE∗       | 79.16 | 66.71 | 41.30 | 75.35 | 74.97 | 68.75 | 31.99  | 55.96 | 61.77 |

task across two datasets, experts such as Expert7 in Layer5, Expert9 in Layer8, and Expert11 in Layer15 are selected with an average frequency exceeding 40%, while others, such as Expert0 in Layer0 and Expert8 in Layer2, are seldom chosen. These results provide detailed evidence that Phi3.5 moe demonstrates a high degree of similarity in expert selection frequencies within task categories while also exhibiting significant sparsity. From another perspective, Phi3.5-moe demonstrates entirely distinct expert selection preferences across different reasoning tasks. For instance, Expert13 in Layer2 is frequently selected for (QA/CR) tasks but is neither prominent nor frequently chosen in the other three tasks. Similarly, Expert7 in Layer14 is heavily utilized in Code tasks but is rarely selected in the other three task categories.

A similar pattern is observed in Deepseek-moe-16b-base, which has 64 experts per layer, as shown in Figure [11.](#page-21-0) While displaying clear intra-category similarities and inter-category differences in expert selection, the larger number of experts results in an even greater degree of sparsity in expert selection for Deepseek-moe-16b-base.

## <span id="page-19-0"></span>A.12 Pruning on Mixtral-8x7B

In Section [6.3,](#page-6-2) when employing a more aggressive pruning strategy, unlike the other three models which maintain relatively stable average accuracy under significant inference speedup, Mixtral-8x7B

exhibits notable performance degradation. This section delves into this phenomenon and analyzes its underlying causes.

Similar to Figure [12,](#page-21-1) we plot the changes in Mixtral-8x7B's average accuracy, expert pruning rate, and inference speedup as the pruning threshold varied. As shown in the figure, unlike Phi3.5 moe and Deepseek-moe-16b-base, where a significant drop in accuracy only occurs when the pruning threshold exceeded 0.7 and the expert pruning rate approached 40%, Mixtral-8x7B begins to show a noticeable decline in accuracy once the pruning threshold surpassed 0.3.

We further analyze the expert selection frequency of Mixtral-8x7B across two different datasets. As illustrated in Figure [13,](#page-21-2) compared to Phi3.5-moe and Deepseek-moe-16b-base, Mixtral-8x7B exhibits weaker sparsity in expert selection. Apart from a few experts, such as Expert6 in Layer3 (top) and Expert2 in Layer25 (bottom), whose average selection frequencies exceed the mean (0.125), the selection frequencies of the remaining experts are relatively balanced. This phenomenon has also been noted in [\(Jiang et al.,](#page-10-0) [2024;](#page-10-0) [Li et al.,](#page-10-3) [2024a\)](#page-10-3). Consequently, Mixtral-8x7B is more sensitive to dynamic expert pruning compared to the other three models, making it less suitable for the aggressive pruning settings (α = 0.7) proposed in our PESF method. Nevertheless, our approach achieved commend-

<span id="page-20-0"></span>![](_page_20_Figure_0.jpeg)

Figure 10: The frequency of expert selection across 8 datasets spanning 4 task types for Phi3.5-moe.

able inference speedup and accuracy retention on Mixtral-8x7B under conservative pruning settings ( $\alpha=0.3$ ). In the future, we aim to explore methods to achieve higher pruning rates and speedup while maintaining accuracy on Mixtral-8x7B.

