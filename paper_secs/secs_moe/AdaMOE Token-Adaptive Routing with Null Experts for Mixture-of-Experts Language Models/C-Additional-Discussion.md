# C Additional Discussion

The top-p router can also implement token-adaptive expert selection. It selects experts based on the sum of routing probabilities exceeding a threshold p. This allows for a variable number of experts to be chosen for different tokens. However, compared to our A*da*MOE, this approach has the following drawbacks:

- 1. The value of p cannot be predefined according to the compute budget, and finding an appropriate p often requires multiple attempts.
- 2. It cannot enable tokens to bypass some layers.

Moreover, our method is actually compatible with the top-p approach. We can incorporate null experts and simultaneously use top-p. This compatibility opens up avenues for further exploration in the future.

<span id="page-11-1"></span>

|          |      |       | Epoch |       | Rank of LoRAs |
|----------|------|-------|-------|-------|---------------|
|          |      | 1     | 10    | 8     | 32            |
| Baseline | Acc. | 45.95 | 87.19 | 45.95 | 46.72         |
|          | Load | 2.00  | 2.00  | 2.00  | 2.00          |
| AdaMOE   | Acc. | 48.88 | 88.54 | 48.88 | 49.01         |
|          | Load | 1.92  | 1.88  | 1.92  | 1.89          |

Table 6: Robustness of our method under different epochs and ranks of LoRAs.

<span id="page-11-0"></span>

|          |      | Metric       | RTE           | COLA          | SQA           | CQA           | OQA           |
|----------|------|--------------|---------------|---------------|---------------|---------------|---------------|
| Baseline | k1   | Acc.<br>Load | 65.06<br>1.00 | 85.39<br>1.00 | 63.35<br>1.00 | 76.09<br>1.00 | 65.81<br>1.00 |
|          | k2   | Acc.<br>Load | 63.66<br>2.00 | 84.69<br>2.00 | 58.41<br>2.00 | 76.77<br>2.00 | 65.57<br>2.00 |
|          | m5k4 | Acc.<br>Load | 66.64<br>1.78 | 85.01<br>1.77 | 66.64<br>1.80 | 76.96<br>1.77 | 66.48<br>1.78 |
|          | m7k4 | Acc.<br>Load | 67.19<br>1.47 | 85.83<br>1.45 | 68.17<br>1.47 | 77.64<br>1.44 | 66.79<br>1.47 |
| AdaMOE   | m9k4 | Acc.<br>Load | 68.38<br>1.22 | 84.95<br>1.23 | 63.62<br>1.22 | 76.47<br>1.22 | 67.98<br>1.22 |
|          | m5k2 | Acc.<br>Load | 67.01<br>0.44 | 84.61<br>0.44 | 66.82<br>0.44 | 76.52<br>0.44 | 67.89<br>0.44 |

Table 7: Exact values for Figure [5,](#page-5-1) averaged from results with 3 random seeds.

<span id="page-12-0"></span>

| ARC-C | Baseline |       |       | AdaMOE |       |       |       |
|-------|----------|-------|-------|--------|-------|-------|-------|
| m, k  | 0,2      | 8,3   | 16,4  | 24,5   | 32,6  | 40,7  | 40,8  |
| Acc.  | 87.46    | 89.15 | 87.12 | 86.10  | 85.08 | 86.10 | 85.76 |
| Load  | 2.00     | 1.67  | 1.70  | 1.56   | 1.49  | 1.59  | 1.34  |
| HELLA | Baseline |       |       | AdaMOE |       |       |       |
| m, k  | 0,2      | 8,3   | 16,4  | 24,5   | 32,6  | 40,7  | 40,8  |
| Acc.  | 84.10    | 85.50 | 83.10 | 81.30  | 80.40 | 82.50 | 79.20 |
| Load  | 2.00     | 1.68  | 1.64  | 1.45   | 1.39  | 1.37  | 1.44  |
| OQA   | Baseline |       |       | AdaMOE |       |       |       |
| m, k  | 0,2      | 8,3   | 16,4  | 24,5   | 32,6  | 40,7  | 40,8  |
| Acc.  | 89 94    | 88.2  | 89.2  | 86.6   | 86.8  | 85    | 82.6  |
| Load  | 2.00     | 1.70  | 1.71  | 1.49   | 1.54  | 1.56  | 1.50  |
| PIQA  | Baseline |       |       | AdaMOE |       |       |       |
| m, k  | 0,2      | 8,3   | 16,4  | 24,5   | 32,6  | 40,7  | 40,8  |
| Acc.  | 90.48    | 90.32 | 89.99 | 88.30  | 86.67 | 86.78 | 85.42 |
| Load  | 2.00     | 1.59  | 1.53  | 1.46   | 1.39  | 1.32  | 1.33  |
| WINO  | Baseline |       |       | AdaMOE |       |       |       |
| m, k  | 0,2      | 8,3   | 16,4  | 24,5   | 32,6  | 40,7  | 40,8  |
| Acc.  | 80.43    | 81.93 | 79.32 | 78.17  | 77.66 | 71.43 | 79.16 |
| Load  | 2.00     | 1.66  | 1.72  | 1.71   | 1.73  | 1.59  | 1.45  |

Table 8: Performance of more m and k combinations on various datasets. As a supplement to the experimental results in Section [4.2.](#page-6-1)