# <span id="page-18-0"></span>**D Ablation Study on Compression Orders**

In Section [7,](#page-9-2) we discussed the combination of Expert Trimming and Expert Slimming. Here we ablate on the orders of compression when combining these two techniques. Results in Table [7](#page-18-1) show that the order of Expert Trimming and Expert Slimming doesn't have a significant influence on the performance, where applying Expert Slimming then Expert Trimming ("S+T") performs slightly better for Mixtral-8×7B (e.g. +0.5, +0.4 and +0.1 for Expert Drop, Layer Drop and Block Drop, respectively). To this end, we choose "S+T" as the final implementation in our experiments.

<span id="page-18-1"></span>Table 7: **Ablation results on different orders of Expert Slimming and Expert Trimming.** "S+T" denotes first applying Expert Slimming then Expert Trimming, and "T+S" denotes the reversed order.

| Mixtral-8×7B        |       |       |           |      |      |      |      |            |      |  |  |
|---------------------|-------|-------|-----------|------|------|------|------|------------|------|--|--|
| Method              | ARC-C | BoolQ | HellaSwag | MMLU | OBQA | PIQA | RTE  | WinoGrande | Avg. |  |  |
| Baseline            | 59.4  | 84.2  | 84.0      | 67.9 | 46.8 | 83.8 | 70.4 | 75.6       | 71.5 |  |  |
| + E2/8, AWQ (S+T)   | 50.7  | 79.1  | 78.9      | 52.4 | 44.2 | 81.2 | 55.6 | 75.9       | 64.8 |  |  |
| + E2/8, AWQ (T+S)   | 50.8  | 79.9  | 78.7      | 49.2 | 44.4 | 80.9 | 55.2 | 75.4       | 64.3 |  |  |
| + L8/32, AWQ (S+T)  | 46.2  | 84.2  | 74.2      | 66.2 | 39.0 | 75.5 | 69.3 | 74.2       | 66.1 |  |  |
| + L8/32, AWQ (T+S)  | 46.8  | 84.4  | 74.0      | 65.3 | 39.8 | 75.0 | 66.8 | 73.2       | 65.7 |  |  |
| + B5/32, AWQ (S+T)  | 50.6  | 85.1  | 77.5      | 66.9 | 41.4 | 76.1 | 71.8 | 74.5       | 68.0 |  |  |
| + B5/32, AWQ (T+S)  | 50.3  | 84.7  | 77.4      | 65.8 | 42.0 | 78.8 | 70.4 | 74.0       | 67.9 |  |  |
| DeepSeek-MoE-16B    |       |       |           |      |      |      |      |            |      |  |  |
|                     |       |       |           |      |      |      |      |            |      |  |  |
| Method              | ARC-C | BoolQ | HellaSwag | MMLU | OBQA | PIQA | RTE  | WinoGrande | Avg. |  |  |
| Baseline            | 48.1  | 72.4  | 77.3      | 37.9 | 44.0 | 80.4 | 63.9 | 70.3       | 61.8 |  |  |
| + E16/64, AWQ (S+T) | 44.0  | 66.0  | 74.5      | 27.9 | 42.6 | 78.5 | 56.3 | 67.3       | 57.1 |  |  |
| + E16/64, AWQ (T+S) | 44.7  | 64.1  | 74.0      | 29.0 | 42.6 | 79.9 | 54.2 | 68.4       | 57.1 |  |  |
| + L4/28, AWQ (S+T)  | 42.1  | 72.0  | 69.2      | 33.7 | 39.8 | 75.1 | 47.7 | 66.5       | 55.8 |  |  |
| + L4/28, AWQ (T+S)  | 42.4  | 71.7  | 69.1      | 33.4 | 40.1 | 74.8 | 47.6 | 66.2       | 55.7 |  |  |
| + B4/28, AWQ (S+T)  | 40.1  | 70.2  | 68.6      | 36.1 | 38.4 | 76.2 | 51.6 | 66.4       | 56.0 |  |  |

