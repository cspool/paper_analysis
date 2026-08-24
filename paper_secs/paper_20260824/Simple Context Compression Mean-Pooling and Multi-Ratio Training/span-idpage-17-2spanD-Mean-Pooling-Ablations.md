# <span id="page-17-2"></span>**D Mean-Pooling Ablations**

We ablate the mean-pooling baseline using Gemma2-2B [\(Table 8\)](#page-18-1), testing: (1) Fixed Decoder (only encoder trained); (2) Fixed Encoder (only decoder trained); (3) No Encoder (context represented via decoder token embeddings only); (4) w/o Linear Layer (pooling output passed directly); (5) Ratio Sampling (one ratio sampled per instance rather than training on all ratios). Freezing the decoder causes considerable but not catastrophic degradation, consistent with prior work [\(Louis et al.,](#page-11-0) [2025\)](#page-11-0). Freezing or removing the encoder is more detrimental (>12% drop). The linear layer has minimal impact (0.7% reduction when removed). Ratio sampling speeds up training at a small cost (1.4% drop).

<span id="page-18-1"></span>

| Ablation (GEMMA2-2B) | $4 \times$ | $8 \times$ | $16 \times$ | $32 \times$ | $64 \times$ | $128 \times$ | Δ       |
|----------------------|------------|------------|-------------|-------------|-------------|--------------|---------|
| Default              | 68.1       | 65.4       | 61.0        | 54.9        | 48.8        | 43.7         | (+0.0)  |
| Fixed Decoder        | 64.9       | 61.9       | 57.0        | 51.5        | 45.0        | 39.8         | (-3.6)  |
| Fixed Encoder        | 57.4       | 49.9       | 44.1        | 39.8        | 36.2        | 34.8         | (-13.3) |
| No Encoder           | 58.7       | 51.9       | 44.9        | 40.2        | 36.2        | 34.1         | (-12.6) |
| w/o Linear Layer     | 67.7       | 64.5       | 60.0        | 54.1        | 48.1        | 43.2         | (-0.7)  |
| Ratio Sampling       | 67.1       | 64.0       | 59.3        | 53.5        | 47.5        | 42.2         | (-1.4)  |

Table 8: Ablation study for mean pooling using GEMMA2-2B as the teacher LLM. Numbers are macro-averaged  $F_1$  scores.  $\Delta$ : mean change vs. Default across ratios; **bold** = best per column.

