# I. Ablation on Quantization Steps

We conducted an ablation study on the PACS dataset to identify the optimal number of steps after which quantization should be applied. We perform 7-bit quantization and the results are summarized below:

| Quantization Step | OOD Accuracy |
|-------------------|--------------|
| No quantization   | 84.7 ± 0.5   |
| 1000              | 86.2 ± 0.4   |
| 2000              | 87.8 ± 0.3   |
| 3000              | 86.9 ± 0.4   |
| 4000              | 85.1 ± 0.3   |

Table 14. OOD Accuracy across different quantization steps.

