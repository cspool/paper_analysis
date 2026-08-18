# *2) Token-Level Correlation: (Ob2)*

We examine expert selection relations for the same layer between adjacent tokens in [Figure 5.](#page-3-1) Each pixel in the heatmap displays the conditional probability of selecting expert j in the next token given that expert i was activated in the previous token, with bright colors indicating higher probabilities.

Similar to layer-level patterns, cross-token heatmaps exhibit white dots, bright vertical lines, and variation across layers and models, indicating correlations between adjacent tokens. However, token-level relations reveal a common pattern appearing across all models: the bright diagonal line that indicates the tendency to select the same expert across adjacent tokens. This diagonal pattern emerges predominantly in higher layers (17 and 43) but not in lower layers (1 and 3), regardless of models.

We apply the same conditional-CDF analysis to the tokenlevel relation. As shown in [Figure 5\(](#page-3-1)d), the top 20% of nexttoken expert candidates cover 47%, 62%, 80%, and 53% of the cumulative conditional probability in DeepSeek-V3, Qwen3, Llama 4, and Kimi K2, respectively, averaged across all MoE layers. The correlation is again strongest in Llama 4 and weakest in DeepSeek-V3.

