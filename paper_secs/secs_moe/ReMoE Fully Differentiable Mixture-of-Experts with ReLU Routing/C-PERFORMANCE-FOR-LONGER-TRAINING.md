# C PERFORMANCE FOR LONGER TRAINING

We conduct experiments of training MoE and ReMoE for a longer duration. We experiment with N =469M, E = 8, k = 1 and train the models with a batch size of 4M tokens and training over 120B tokens. The results, as shown in Table [5,](#page-15-1) indicate that the superiority of ReMoE persists in longer training.

<span id="page-15-1"></span>

| Model | Valid Loss | ARC-c | ARC-e | BoolQ | HellaSwag | LAMBADA | PIQA  | RACE  | Avg.  |
|-------|------------|-------|-------|-------|-----------|---------|-------|-------|-------|
| MoE   | 1.716      | 23.62 | 52.40 | 53.94 | 35.43     | 43.64   | 68.34 | 31.48 | 44.12 |
| ReMoE | 1.689      | 25.34 | 55.22 | 55.96 | 36.76     | 45.82   | 68.93 | 30.43 | 45.49 |

Table 5: Performance of training N =469M, E = 8, k = 1 models for 120B tokens.

