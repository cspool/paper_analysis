# <span id="page-12-2"></span>C. More Ablation Results

In our efficiency evaluation, we observe that with a residual length of 32, KIVI achieves a significantly higher memory compression rate, which in turn leads to increased throughput. Additionally, our ablation study reveals that changing the residual length from 128 to 32 does not result in a substantial performance gap. We demonstrate KIVI with a residual length of 32 across all benchmark datasets. As shown in Tables [6](#page-12-4) and [7,](#page-13-1) KIVI with a residual length of 32 also delivers performance comparable to that of the 16-bit full model.

<span id="page-12-4"></span>Table 6: Performance comparison between 16bit, KIVI-2 (2bit) / KIVI-4 (4bit) with residual length 128 and 32 across various models. R32 stands for residual length 32.

| Model       |                | CoQA  | TruthfulQA | GSM8K |
|-------------|----------------|-------|------------|-------|
|             | 16bit          | 63.88 | 30.76      | 13.50 |
| Llama-2-7B  | KIVI-2<br>R128 | 63.05 | 33.95      | 12.74 |
|             | KIVI-2<br>R32  | 62.85 | 33.01      | 13.57 |
|             | 16bit          | 66.37 | 29.53      | 22.67 |
| Llama-2-13B | KIVI-2<br>R128 | 66.23 | 29.84      | 20.77 |
|             | KIVI-2<br>R32  | 66.57 | 29.35      | 20.62 |
|             | 16bit          | 59.83 | 23.20      | 4.55  |
|             | KIVI-4<br>R128 | 59.67 | 22.58      | 4.47  |
| Falcon-7B   | KIVI-4<br>R32  | 59.73 | 22.96      | 3.94  |
|             | KIVI-2<br>R128 | 57.48 | 24.98      | 3.41  |
|             | KIVI-2<br>R32  | 57.50 | 25.70      | 2.20  |
|             | 16bit          | 67.40 | 30.45      | 38.36 |
| Mistral-7B  | KIVI-2<br>R128 | 66.35 | 32.17      | 36.01 |
|             | KIVI-2<br>R32  | 65.90 | 31.21      | 34.34 |

