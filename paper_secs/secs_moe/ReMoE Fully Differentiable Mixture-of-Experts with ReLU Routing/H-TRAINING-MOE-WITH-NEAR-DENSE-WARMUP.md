# H TRAINING MOE WITH NEAR-DENSE WARMUP

In ReMoE, the training process naturally progresses through three stages, with the first two involving near-dense training where the majority of experts are active. To facilitate a fairer comparison, in Section [4.3,](#page-6-1) we train the MoE model for additional tokens to match the overall computational cost. In this section, we explore an alternative approach by introducing a similar near-dense warmup phase for MoE, referred to as "MoE with warmup," to align its computational footprint with ReMoE across each stage. Specifically, we train the MoE with N = 182M, E = 8, and k = 6—approximately matching the average sparsity of ReMoE during Stages I and II, as depicted in Figure [4a—](#page-4-1)for the first 100 steps, before transitioning to k = 1 for the remainder of the training process.

Table [13](#page-20-0) compares this warmup variant to both standard MoE and ReMoE. The results indicate that the warmup phase provides a modest improvement in validation loss compared to standard MoE, despite matching the overall computational cost. Nonetheless, ReMoE consistently outperforms both variants. This suggests that the three-stage training pipeline learned by ReMoE, with Stages I and II comprising only the first 100 steps, is beneficial to overall performance.

<span id="page-20-0"></span>

| Model                 | Valid<br>Loss | ARC<br>c | ARC<br>e | BoolQ | Hella<br>Swag | LAM<br>BADA | PIQA  | RACE  | Avg.  |
|-----------------------|---------------|----------|----------|-------|---------------|-------------|-------|-------|-------|
| MoE                   | 1.936         | 20.82    | 45.03    | 57.55 | 29.84         | 31.81       | 63.28 | 28.42 | 39.53 |
| MoE<br>with<br>warmup | 1.928         | 20.73    | 46.38    | 52.35 | 30.28         | 33.90       | 63.76 | 27.66 | 39.29 |
| ReMoE                 | 1.921         | 20.22    | 46.68    | 54.16 | 30.26         | 35.94       | 63.55 | 29.38 | 40.03 |

Table 13: Performance of MoE with near-dense warmup

We further extend our experiments with MoE using warmup to configurations with larger E, which increases the computational cost of near-dense training. The results, summarized in Table [14,](#page-20-1) show that as E increases, the warmup setting consistently improves performance. However, ReMoE still outperforms both variants, maintaining a steeper performance scaling with respect to E.

<span id="page-20-1"></span>

| Model,<br>E =8        | Valid<br>Loss | Avg.<br>Acc. | Model,<br>E =32       | Valid<br>Loss | Avg.<br>Acc. | Model,<br>E =128      | Valid<br>Loss | Avg.<br>Acc. |
|-----------------------|---------------|--------------|-----------------------|---------------|--------------|-----------------------|---------------|--------------|
| MoE                   | 1.936         | 39.53        | MoE                   | 1.874         | 39.77        | MoE                   | 1.852         | 41.10        |
| MoE<br>with<br>warmup | 1.928         | 39.29        | MoE<br>with<br>warmup | 1.869         | 40.06        | MoE<br>with<br>warmup | 1.841         | 41.34        |
| ReMoE                 | 1.921         | 40.03        | ReMoE                 | 1.852         | 41.58        | ReMoE                 | 1.815         | 42.12        |

Table 14: Results for MoE with warmup under different expert count E

To further investigate the impact of warmup steps on MoE performance, we vary the number of warmup steps for the E = 8 MoE configuration among 50, 100, 500, and 1000. The training curves of these models, along with standard MoE and ReMoE, are shown in Figure [13,](#page-21-0) and the final validation losses are summarized in Table [15.](#page-21-0)

Our results reveal that performance does not improve monotonically with an increasing number of warmup steps, despite the additional computation. This behavior arises due to the discrepancy between the training objectives of k = 6 (warmup phase) and k = 1 (post-warmup phase). For instance, when warmup concludes after 100 steps, the transition between phases is smooth, with the loss changing minimally from 6.491 → 6.751. However, extending warmup to 500 or 1000 steps leads to a more pronounced loss gap of 3.101 → 5.827 and 2.695 → 4.428, respectively.

<span id="page-21-0"></span>![](_page_21_Figure_1.jpeg)

| Model | Warmup Steps | Valid Loss |  |  |
|-------|--------------|------------|--|--|
|       | 0            | 1.937      |  |  |
| MoE   | 50           | 1.930      |  |  |
|       | 100          | 1.928      |  |  |
|       | 500          | 1.930      |  |  |
|       | 1000         | 1.931      |  |  |
| ReMoE | -            | 1.921      |  |  |

Table 15: Final validation loss of MoE with different warmup steps

Figure 13: Training curves of MoE with different warmup steps

In summary, near-dense warmup can enhance the performance of TopK MoE when training from scratch by providing a better initialization for the experts. However, the warmup phase should conclude while the language model loss is still decreasing rapidly. Prolonging the warmup can exacerbate the gap between the warmup and subsequent training phases, ultimately degrading performance. In contrast, ReMoE naturally determines the appropriate warmup steps and sparsity levels due to its continuous and differentiable training dynamics.

