# <span id="page-21-0"></span>D Sample Size Analysis

Table [16](#page-22-0) investigates the impact of sample size on DartQuant's performance. All experiments are performed with a token sampling ratio of 10%. The results show that DartQuant's calibration performance remains robust even with a small dataset.

<span id="page-22-0"></span>Table 16: Comparison of DartQuant Calibration Results with Different Sample Sizes.

| Model | Sample | WikiText2 | PTB   | C4    | Avg   |
|-------|--------|-----------|-------|-------|-------|
|       | 32     | 5.91      | 42.61 | 8.03  | 18.85 |
|       | 64     | 5.88      | 43.33 | 8.00  | 19.07 |
| 2 7b  | 128    | 5.92      | 42.63 | 7.99  | 18.85 |
|       | 256    | 5.92      | 42.41 | 8.04  | 18.79 |
|       | 32     | 7.30      | 12.66 | 11.79 | 10.58 |
|       | 64     | 7.30      | 12.77 | 11.79 | 10.62 |
| 3 8b  | 128    | 7.29      | 12.71 | 11.85 | 10.62 |
|       | 256    | 7.41      | 12.83 | 11.99 | 10.74 |

