# <span id="page-16-0"></span>E DOWNSTREAM EVALUATION RESULTS

This section provides the detailed downstream evaluation results for the main experiments of scalability of ReMoE in Section [4.3](#page-6-1) and ablations on load balancing in Section [5.2.](#page-7-1)

#### E.1 SCALING IN ACTIVE PARAMETERS N

The downstream evaluation results for scaling with respect to the parameter count N, as discussed in Section [4.3,](#page-6-1) are presented in Table [8.](#page-16-1) These results highlight the performance comparison with increasing model parameters.

<span id="page-16-1"></span>

| Model | N    | ARC-c | ARC-e | BoolQ | HellaSwag | LAMBADA | PIQA  | RACE  | Avg.  |
|-------|------|-------|-------|-------|-----------|---------|-------|-------|-------|
|       | 182M | 19.45 | 43.35 | 54.40 | 28.61     | 31.09   | 61.97 | 28.52 | 38.20 |
| Dense | 469M | 21.50 | 49.12 | 56.88 | 31.12     | 36.74   | 64.47 | 30.53 | 41.48 |
|       | 978M | 21.93 | 50.88 | 60.24 | 32.42     | 41.06   | 67.46 | 31.77 | 43.68 |
|       | 182M | 20.82 | 45.03 | 57.55 | 29.84     | 31.81   | 63.28 | 28.42 | 39.53 |
| MoE   | 469M | 23.63 | 52.40 | 53.94 | 32.43     | 43.64   | 68.34 | 31.48 | 43.69 |
|       | 978M | 23.81 | 52.90 | 58.90 | 35.01     | 44.42   | 67.90 | 31.48 | 44.91 |
|       | 182M | 20.22 | 46.68 | 54.16 | 30.26     | 35.94   | 63.55 | 29.38 | 40.03 |
| ReMoE | 469M | 21.67 | 53.16 | 58.75 | 33.80     | 40.66   | 67.95 | 31.20 | 43.88 |
|       | 978M | 24.06 | 55.26 | 57.28 | 35.93     | 44.42   | 68.99 | 30.43 | 45.20 |

Table 8: Downstream results of scaling in active parameters N.

#### E.2 SCALING IN EXPERT COUNT E

Table [9](#page-16-2) contains the downstream evaluation results for scaling with respect to the expert count E, as examined in Section [4.3.](#page-6-1) This analysis illustrates how varying the number of experts influences the overall model effectiveness of MoE and ReMoE.

<span id="page-16-2"></span>

| Model | E   | ARC-c | ARC-e | BoolQ | HellaSwag | LAMBADA | PIQA  | RACE  | Avg.  |
|-------|-----|-------|-------|-------|-----------|---------|-------|-------|-------|
| Dense | -   | 19.45 | 43.35 | 54.40 | 28.61     | 31.09   | 61.97 | 28.52 | 38.20 |
|       | 4   | 20.73 | 44.49 | 59.63 | 29.14     | 31.40   | 63.33 | 29.19 | 39.70 |
|       | 8   | 20.82 | 45.03 | 57.55 | 29.84     | 31.81   | 63.28 | 28.42 | 39.53 |
|       | 16  | 20.90 | 45.29 | 46.36 | 30.50     | 33.22   | 64.96 | 28.33 | 38.50 |
| MoE   | 32  | 19.54 | 47.35 | 52.29 | 31.12     | 35.63   | 64.25 | 28.23 | 39.77 |
|       | 64  | 19.88 | 46.63 | 60.06 | 31.47     | 36.33   | 65.07 | 28.04 | 41.06 |
|       | 128 | 20.99 | 47.69 | 56.73 | 32.00     | 36.62   | 65.67 | 28.04 | 41.10 |
|       | 4   | 19.88 | 46.46 | 57.43 | 29.64     | 33.57   | 62.95 | 27.66 | 39.66 |
|       | 8   | 20.22 | 46.68 | 54.16 | 30.26     | 35.94   | 63.55 | 29.38 | 40.03 |
|       | 16  | 20.90 | 49.28 | 53.36 | 30.85     | 37.09   | 65.83 | 30.05 | 41.05 |
| ReMoE | 32  | 20.56 | 48.11 | 59.54 | 31.42     | 37.84   | 65.18 | 28.42 | 41.58 |
|       | 64  | 20.82 | 50.51 | 57.80 | 32.17     | 36.74   | 65.78 | 27.46 | 41.61 |
|       | 128 | 19.97 | 51.05 | 56.97 | 32.40     | 37.92   | 66.70 | 29.86 | 42.12 |

Table 9: Downstream results of scaling in expert count E.

### E.3 SCALING IN GRANULARITY G

The downstream evaluation results for scaling with respect to the granularity G are shown in Table [10,](#page-17-1) based on the experiments in Section [4.3.](#page-6-1) These results demonstrate the superiority of finegrained ReMoE over fine-grained MoE.

<span id="page-17-1"></span>

| Model   | G  | ARC-c | ARC-e | BoolQ | HellaSwag | LAMBADA | PIQA  | RACE  | Avg.  |
|---------|----|-------|-------|-------|-----------|---------|-------|-------|-------|
| Dense   | -  | 19.45 | 43.35 | 54.40 | 28.61     | 31.09   | 61.97 | 28.52 | 38.20 |
| Dense×8 | -  | 22.78 | 48.11 | 59.66 | 31.11     | 35.65   | 65.02 | 29.57 | 41.70 |
|         | 1  | 20.82 | 45.03 | 57.55 | 29.84     | 31.81   | 63.28 | 28.42 | 39.53 |
|         | 2  | 21.42 | 46.55 | 54.25 | 29.95     | 32.52   | 64.09 | 28.61 | 39.62 |
|         | 4  | 20.99 | 46.09 | 55.90 | 30.52     | 35.16   | 63.98 | 29.28 | 40.27 |
| MoE     | 8  | 21.59 | 47.73 | 60.70 | 30.83     | 36.41   | 64.69 | 28.04 | 41.42 |
|         | 16 | 19.80 | 48.82 | 57.34 | 30.64     | 36.00   | 64.74 | 28.71 | 40.86 |
|         | 32 | 21.67 | 48.78 | 57.85 | 31.27     | 37.10   | 64.69 | 28.52 | 41.41 |
|         | 64 | 20.14 | 48.74 | 61.50 | 31.03     | 36.31   | 63.93 | 27.85 | 41.35 |
|         | 1  | 20.22 | 46.68 | 54.16 | 30.26     | 35.94   | 63.55 | 29.38 | 40.03 |
|         | 2  | 20.14 | 47.39 | 57.95 | 30.60     | 34.52   | 63.71 | 28.52 | 40.40 |
|         | 4  | 20.39 | 47.94 | 55.35 | 31.04     | 36.11   | 64.64 | 29.00 | 40.64 |
| ReMoE   | 8  | 20.82 | 48.36 | 60.49 | 30.90     | 36.06   | 63.87 | 28.90 | 41.34 |
|         | 16 | 21.25 | 49.41 | 56.06 | 30.91     | 36.23   | 64.91 | 29.95 | 41.25 |
|         | 32 | 20.90 | 48.86 | 55.81 | 31.14     | 36.58   | 64.69 | 30.05 | 41.15 |
|         | 64 | 20.65 | 48.74 | 60.06 | 31.56     | 36.43   | 65.40 | 29.00 | 41.69 |

Table 10: Downstream results of scaling in granularity G.

#### E.4 LOAD BALANCING ABLATIONS

Table [11](#page-17-2) presents the downstream evaluation results for the load balancing ablations, as discussed in Section [5.2.](#page-7-1) These results compare performance with and without load balancing, offering insights into the different roles of load balancing in MoE and ReMoE.

<span id="page-17-2"></span>

| Model | LB | ARC-c | ARC-e | BoolQ | HellaSwag | LAMBADA | PIQA  | RACE  | Avg.  |
|-------|----|-------|-------|-------|-----------|---------|-------|-------|-------|
| Dense | -  | 19.45 | 43.35 | 54.40 | 28.61     | 31.09   | 61.97 | 28.52 | 38.20 |
| MoE   | ×  | 19.20 | 44.74 | 50.80 | 28.60     | 30.18   | 62.24 | 27.94 | 37.67 |
| MoE   | ✓  | 20.05 | 45.16 | 57.83 | 29.83     | 32.97   | 63.55 | 28.33 | 39.67 |
| ReMoE | ×  | 19.45 | 46.34 | 56.94 | 30.19     | 31.79   | 63.33 | 28.61 | 39.52 |
| ReMoE | ✓  | 20.22 | 46.68 | 54.16 | 30.26     | 35.94   | 63.55 | 29.38 | 40.03 |

Table 11: Downstream results of training with or without load balancing.

