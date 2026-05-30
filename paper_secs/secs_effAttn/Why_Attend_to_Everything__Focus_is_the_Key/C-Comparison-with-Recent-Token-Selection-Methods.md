# C Comparison with Recent Token-Selection Methods

We compare Focus against recent token-selection methods (SparQ [\[Ribar et al.,](#page-13-2) [2024\]](#page-13-2), MagicPIG [\[Chen et al.,](#page-13-3) [2024\]](#page-13-3)) on GPT-2 124M / PG-19. These methods select top-k=32 tokens per query at inference without modifying weights. Note that they operate at a different sparsity level than Focus: token selection at k=32 retains 3% of tokens per query, while Focus with K=4, top-k=2 retains ∼50% of distant pairs.

Table 10: Token-selection methods vs Focus on GPT-2 124M / PG-19 (k=32). Token-selection methods preserve downstream benchmarks but degrade PPL by 5–10 points. Focus improves PPL with zero benchmark degradation.

| Method                                              | PPL ↓        | HellaSwag    | ARC-E        | PIQA         | LAMBADA      |
|-----------------------------------------------------|--------------|--------------|--------------|--------------|--------------|
| Pretrained                                          | 42.8         | 31.1         | 39.5         | 62.5         | 32.6         |
| SparQ [Ribar et al., 2024]<br>SparQ (mean realloc.) | 52.8<br>48.3 | 31.3<br>31.2 | 39.4<br>39.3 | 62.4<br>62.3 | 34.3<br>33.1 |
| MagicPIG [Chen et al., 2024]                        | 52.8         | 31.3         | 39.4         | 62.5         | 34.0         |
| Focus (ours)                                        | 36.2         | 31.1         | 39.5         | 62.5         | 32.6         |

Token-selection methods preserve downstream benchmarks but degrade PPL by 5–10 points. Focus improves PPL (42.8→36.2) with exactly zero benchmark change. The methods achieve speedup through different mechanisms and operate at different sparsity levels, making direct comparison nuanced; we include this for completeness.

Focus exactly matches pretrained on all four benchmarks. SparQ and MagicPIG show minor fluctuations (±0.2–1.7 points) but no systematic degradation, indicating that downstream classification tasks are robust to token-level sparsity at this level. The critical distinction is perplexity: Focus improves PPL by 6.6 points while training-free methods degrade it by 5–10 points.

