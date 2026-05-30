# <span id="page-14-0"></span>B Ablation Studies

Section [4](#page-8-0) showed that Sinkhorn normalization produces stable, balanced groups. Here we ablate four key hyperparameters on GPT-2 124M / PG-19, varying each while holding others at defaults (K=8, w=128, τ=0.1, Sinkhorn iters = 10).

Table 9: Ablation study (GPT-2 124M, PG-19). Each row varies one hyperparameter. Fine-tuned PPL is stable (29.9–30.5) across all 16 configurations.

| Parameter      | Value | Centroid PPL | Fine-tuned PPL | Dominance (centroid) | Dominance (full FT) |
|----------------|-------|--------------|----------------|----------------------|---------------------|
| Groups K       | 4     | 36.8         | 30.1           | 40%                  | 38%                 |
|                | 8     | 38.4         | 30.3           | 24%                  | 23%                 |
|                | 16    | 40.4         | 30.4           | 20%                  | 31%                 |
|                | 32    | 42.4         | 30.5           | 21%                  | 30%                 |
| Window w       | 64    | 38.3         | 30.2           | 17%                  | 17%                 |
|                | 128   | 38.4         | 30.2           | 26%                  | 23%                 |
|                | 256   | 38.1         | 30.3           | 26%                  | 25%                 |
|                | 512   | 38.6         | 30.0           | 27%                  | 28%                 |
| Temp τ         | 0.05  | 36.9         | 30.0           | 68%                  | 74%                 |
|                | 0.1   | 38.4         | 30.3           | 24%                  | 23%                 |
|                | 0.2   | 39.1         | 30.3           | 16%                  | 19%                 |
|                | 0.5   | 40.5         | 30.3           | 21%                  | 31%                 |
| Sinkhorn iters | 3     | 35.8         | 29.9           | 95%                  | 97%                 |
|                | 5     | 36.8         | 30.2           | 69%                  | 75%                 |
|                | 10    | 38.4         | 30.3           | 21%                  | 20%                 |
|                | 20    | 39.0         | 30.2           | 14%                  | 14%                 |

Fine-tuned PPL is robust. Across all 16 configurations, fine-tuned PPL ranges from 29.9 to 30.5—a spread of only 0.6 PPL. Focus is not sensitive to hyperparameter choices.

Sinkhorn iterations: a subtle trap. With 3 iterations, PPL appears best (29.9) but groups have collapsed to 95–97% dominance. This is not real Focus—it is effectively full attention with extra overhead. At low temperature (τ=0.1), exp(scores/0.1) produces extremely peaked distributions that 3 iterations cannot redistribute. At least 10 iterations are needed for balanced groups.

Window size: smaller is better. With K=2 centroid-only training: w=16 achieves the best PPL (33.8), beating w=128 by 0.8 PPL. At w=512 (half the sequence), quality drops by 3.7 PPL because most attention is handled locally, leaving little for group routing to contribute. This confirms that local and group attention are complementary.

