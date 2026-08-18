# <span id="page-2-2"></span>1) Layer-Level Correlation: (Ob1)

As shown in Figure 4 (a) and (b), we present heatmaps for

![](_page_3_Figure_0.jpeg)

<span id="page-3-1"></span>Figure 5. Cross-token expert correlation. (a, b, c) Joint co-activation heatmaps between tokens t and t+1 in DeepSeek-V3, Llama 4, and Qwen3. (d) Conditional CDF for each layer's top-1 expert: top 20% of the next-token expertalready covers most of probability mass.

![](_page_3_Figure_2.jpeg)

<span id="page-3-2"></span>Figure 6. Expert activation patterns remain consistent across prefill and decode stages for both (a, b) cross-layer heatmap and (c, d) cross-token heatmaps. Spearman's ratio quantified in (e, f) shows a strong relation (≥ 0.7).

Deepseek and Qwen illustrating expert selection relationships across adjacent layers. Each pixel in the heatmap displays the conditional probability of selecting expert j in the next layer given that expert i was activated in the previous layer, with bright colors indicating higher probabilities.

The heatmaps reveal clear cross-layer correlations with white dots highlighting specific expert pairs with significantly higher selection probabilities across adjacent layers. However, correlation patterns vary across layers within the same model and differ between models due to architectural variations. For instance, patterns between layers 3-4 differ from those between layers 30-31. Qwen3's notably brighter heatmap indicates stronger cross-layer correlations than Deepseek. Beyond the white dots, there are also consistent bright vertical lines, suggesting certain experts are frequently chosen regardless of previous layer selections. These patterns indicate generally popular experts, analyzed further in Sec. [III-C1.](#page-4-0)

To quantify these relationships, we analyze the conditional CDF P(e<sup>j</sup> | ei) in [Figure 4\(](#page-2-1)c): the top 20% of nextlayer candidates already cover 50%, 65%, 77%, and 56% of the conditional probability mass for DeepSeek-V3, Qwen3, Llama 4[1](#page-3-0) , and Kimi K2, respectively. This reveals strong, model-dependent cross-layer correlations, with Llama4 showing the strongest effect and Deepseek the weakest.

