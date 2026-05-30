# B Impact of Continued Pretraining Token Count on DSMoE Performance

To evaluate how the number of tokens used in continued pretraining affects DSMoE performance, we conducted a series of controlled experiments on both LLaMA-7B and LLaMA-1B models. Tables [5](#page-11-0) and [6](#page-11-1) show the perplexity changes for both models across different token counts. Our approach achieves relatively favorable performance even with fewer tokens, illustrating the relationship between training tokens and complexity (PPL). Performance tends to stabilize after approximately 8 billion training tokens.

<span id="page-11-0"></span>

| Tokens (B) | 2.2   | 3.8   | 5.4   | 7.0   | 7.8   | 8.6   |
|------------|-------|-------|-------|-------|-------|-------|
| PPL        | 7.384 | 7.323 | 7.481 | 7.488 | 7.445 | 7.422 |

Table 5: Effect of token count on LLaMA-1B DSMoE model performance

<span id="page-11-1"></span>

| Tokens (B) | 2.4   | 3.2   | 4.8   | 6.4   | 8.0   | 9.6   |
|------------|-------|-------|-------|-------|-------|-------|
| PPL        | 4.091 | 4.029 | 3.994 | 3.975 | 3.929 | 3.916 |

Table 6: Effect of token count on LLaMA-7B DSMoE model performance