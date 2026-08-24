# <span id="page-7-0"></span>6.1 MAIN RESULTS

As shown in Table [1,](#page-7-1) LAPO achieves a superior balance of reasoning accuracy and computational efficiency, consistently outperforming its base models and establishing a new state-of-the-art frontier among methods that do not rely on external length controls.

LAPO simultaneously enhances reasoning performance and reduces test-time computes. Compared to its base models, LAPO delivers substantial gains. On DeepScaleR-1.5B-Preview, it reduces tokens by 38.5% while boosting average accuracy by 2.3 points; a similar trend holds for DeepSeek-R1-1.5B (41.0% token cut and 1.2 point accuracy gain). This validates that LAPO learns to produce more concise yet effective reasoning.

LAPO surpasses existing efficient reasoning optimization approaches. When compared with leading efficiency methods, LAPO consistently demonstrates a superior accuracy-efficiency tradeoff. On the more capable DeepScaleR-1.5B base model, LAPO-I achieves the highest average accuracy among all tested methods. This advantage holds across different baseline paradigms. It surpasses budget-driven methods like ThinkPrune-4k and L1-Max under a fair 4k training context. Compared to implicit regularization methods like HAPO, LAPO shows a clear advantage in preserving accuracy. Furthermore, while adaptive activation methods like AutoThink can be highly token-efficient, they do not reach the same level of reasoning quality. This comprehensive comparison highlights that LAPO's "Discover-Internalize" process, which fosters an autonomous

<span id="page-8-2"></span>Table 2: Experimental results with different length guidance for LAPO-I. Bold and underline indicate the best and second-best Pass@1 scores. w/ indicates the length guidance used in LAPO-I.

| Method                              | MATH-500    |      | AIME2024    |      | AMC-23      |      | OlympiadBench |      | Average     |      |
|-------------------------------------|-------------|------|-------------|------|-------------|------|---------------|------|-------------|------|
|                                     | Pass@1 #Tok |      | Pass@1 #Tok |      | Pass@1 #Tok |      | Pass@1 #Tok   |      | Pass@1 #Tok |      |
| Base model: DeepScaleR-1.5B-Preview |             |      |             |      |             |      |               |      |             |      |
| Base                                | 85.8        | 3280 | 35.5        | 9246 | 74.2        | 6416 | 54.6          | 5974 | 62.5        | 6229 |
| LAPO-D                              | 86.4        | 2365 | 37.6        | 5945 | 77.6        | 3655 | 56.1          | 4499 | 64.4        | 4116 |
| w/ Exact                            | 86.3        | 2168 | 38.1        | 5371 | 78.3        | 3765 | 56.3          | 4024 | 64.8        | 3832 |
| w/ Range                            | 86.6        | 2153 | 36.5        | 6095 | 76.9        | 3600 | 56.2          | 4011 | 64.1        | 3964 |
| w/ Outside                          | 86.5        | 2251 | 36.4        | 5882 | 76.3        | 3850 | 55.4          | 4105 | 63.9        | 4022 |
| w/ Implicit                         | 86.9        | 2181 | 36.2        | 5963 | 76.1        | 4002 | 55.1          | 4206 | 63.6        | 4088 |

and continuous length adaptation, leads to a more robust and effective reasoning policy than methods relying on external budgets, progressive compression, or binary mode-switching.

Both Discovery and Internalization stages contribute to the final performance. LAPO-D first establishes a strong foundation, achieving a significant 36.0% token reduction on its own by learning natural reasoning length distributions. This is highlighted by comparing it to the Acc-Only baseline. While simply finetuning for accuracy yields some token reduction, LAPO-D's length-aware reward achieves substantially greater efficiency while also improving average accuracy by 0.5 points. This demonstrates that encouraging conciseness via our reward not only prunes redundant thoughts but also helps the model find more robust reasoning patterns. Building on this superior foundation, LAPO-I achieves an additional 6.9% efficiency gain by internalizing these patterns through incontext guidance. This progressive refinement indicates that our framework learns a generalizable principle of adaptive reasoning.

#### <span id="page-8-0"></span>6.2 ABLATION STUDY ON IN-CONTEXT GUIDANCE

To validate that our method's success stems from internalizing a self-proposed plan, we ablate the two key factors of our in-context guidance: its form (how precise the guidance is) and its position (whether it's part of the model's internal thought process). We compare our default approach (w/ Exact) against three variants: w/ Range (less precise guidance), w/ Outside (placing the guidance before <think>), and w/ Implicit (no guidance, relying only on the reward). As shown in Table [2,](#page-8-2) the results demonstrate that both form and position are critical for effective internalization.

Our default method outperforms the less precise Range variant, indicating that specific targets discovered in Discovery stage provide a stronger learning signal. More critically, the guidance's position determines whether the model internalizes a plan or merely follows instructions. Moving the guidance outside the <think> block transforms it into an external command and causes accuracy to drop significantly to 63.9%. This illustrates that the model performs best when the budget is framed as part of its own cognitive plan. Finally, removing the guidance entirely results in the worst performance, with accuracy dropping to 63.6% and token count reverting to the LAPO-D baseline. This indicates that our explicit, properly-positioned, self-declarative guidance is the critical mechanism for internalization.

#### <span id="page-8-1"></span>6.3 ABLATION ON STATISTICAL METRICS FOR TARGET LENGTH

The choice of a statistical measure to derive the target length n from the distribution of successful solutions is critical. We conduct an ablation study comparing three strategies for this selection: using the median (our default), the mean, and the minimum length.

As shown in Table [3,](#page-9-2) the median proves to be the most effective choice, achieving the best balance between accuracy and efficiency. Using the median as the target yields the highest average accuracy (64.8%) with an efficient token count of 3,832. This validates our hypothesis that the median, due to its robustness to outliers, provides the most representative signal of a "typically" effective reasoning depth. In contrast, the mean is susceptible to a few excessively long, successful solutions, leading it to set overly generous budgets, resulting in higher token usage (4,040) and slightly lower accuracy.

<span id="page-9-2"></span>Table 3: Experimental results within different statistical metrics used for target length selection in LAPO-I. Bold and underline indicate the best and second-best Pass@1 scores. w/ indicates statistical metrics used for target length selection in LAPO-I.

| Method                              | MATH-500    |      | AIME2024    |      | AMC-23      |      | OlympiadBench |      | Average     |      |
|-------------------------------------|-------------|------|-------------|------|-------------|------|---------------|------|-------------|------|
|                                     | Pass@1 #Tok |      | Pass@1 #Tok |      | Pass@1 #Tok |      | Pass@1 #Tok   |      | Pass@1 #Tok |      |
| Base model: DeepScaleR-1.5B-Preview |             |      |             |      |             |      |               |      |             |      |
| Base                                | 85.8        | 3280 | 35.5        | 9246 | 74.2        | 6416 | 54.6          | 5974 | 62.5        | 6229 |
| LAPO-D                              | 86.4        | 2365 | 37.6        | 5945 | 77.6        | 3655 | 56.1          | 4499 | 64.4        | 4116 |
| w/ Median                           | 86.3        | 2168 | 38.1        | 5371 | 78.3        | 3765 | 56.3          | 4024 | 64.8        | 3832 |
| w/ Mean                             | 85.6        | 2308 | 36.8        | 6030 | 77.4        | 3658 | 56.6          | 4164 | 64.1        | 4040 |
| w/ Minimum                          | 85.9        | 2031 | 36.3        | 6080 | 76.7        | 3324 | 55.0          | 3851 | 63.5        | 3821 |

The minimum, while achieving the most aggressive compression (3,821 tokens), suffers a significant accuracy drop (to 63.5%), suggesting it promotes an over-shortening strategy that discards necessary reasoning steps. These findings underscore the importance of robust statistical measures for learning a well-calibrated reasoning-efficiency trade-off.

### <span id="page-9-0"></span>6.4 ANALYSIS OF INTERNALIZATION

To validate that LAPO fosters genuine internalization, we stress-tested our default LAPO-I model against the w/ Outside ablation variant using adversarial Short (500 tokens) and Long (3500 tokens) length prompts. The results in Table [4](#page-9-3) reveal a stark behavioral divergence. Our default LAPO-I remains robust, its output length staying stable around its 2200-token baseline, thus ignoring the conflicting external instructions. In contrast, the w/ Outside model is clearly influenced: its token count drops to 1247 under the Short constraint and rises to 2821 under the Long one. This comparison indicates that the placement of guidance is critical. Framing the budget as part of the model's internal plan (inside <think>) builds a robust, internalized behavior. Framing it externally teaches superficial instruction-following. This

<span id="page-9-3"></span>Table 4: Robustness of LAPO-I to conflicting length instructions on MATH-500.

| Method            | Length     | MATH-500   |      |  |  |  |  |  |
|-------------------|------------|------------|------|--|--|--|--|--|
|                   | Constraint | Pass@1 (%) | #Tok |  |  |  |  |  |
| LAPO-I            |            |            |      |  |  |  |  |  |
| Base              | N/A        | 86.3       | 2168 |  |  |  |  |  |
| +Short            | 500        | 86.0       | 2279 |  |  |  |  |  |
| +Long             | 3500       | 85.9       | 2300 |  |  |  |  |  |
| LAPO-I w/ Outside |            |            |      |  |  |  |  |  |
| Base              | N/A        | 86.2       | 2251 |  |  |  |  |  |
| +Short            | 500        | 85.1       | 1247 |  |  |  |  |  |
| +Long             | 3500       | 86.1       | 2821 |  |  |  |  |  |

indicates the observed robustness of LAPO-I is a direct result of our internalization mechanism.

