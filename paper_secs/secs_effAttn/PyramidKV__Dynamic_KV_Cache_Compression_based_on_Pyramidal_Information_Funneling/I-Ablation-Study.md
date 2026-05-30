# I Ablation Study

In this section, we present an ablation study for hyperparameters and allocation strategies.

Based on our observations of the attention pattern, we find that a relatively stable, linear arithmetic decrease aligns more closely with the underlying structure of the pattern. We conduct experiments comparing various allocation strategies.

We conducted hyperparameter testing on the original development sets of 16 datasets in LongBench. The parameter  $\beta$  demonstrated remarkable stability, showing minimal sensitivity to varying hyperparameter settings, which highlights its robustness. Conversely,  $\alpha$  consistently produced superior results when set to 8 or 16. Consequently, these values were adopted for subsequent experiments. In Appendix H.2 and H.3, we further analyzed the impact of hyperparameter selection on KV cache budget allocation across different layers. The experiments reaffirmed that  $\beta$  had negligible influence on the outcomes, underscoring its stability. Meanwhile,  $\alpha$  continued to deliver optimal results at values of 8 and 16.

### I.1 Allocation Srategies

Based on our observations of the attention pattern, we find that a relatively stable, linear arithmetic decrease aligns more closely with the underlying structure of the pattern.

We conduct experiments comparing various pyramidal allocation strategies (i.e., linear decay strategy, geometric decay strategy and exponential decay strategy) with a cache size of 64 as Table 4 to confirm that a linear strategy is indeed optimal or preferable.

We also propose three adaptive allocation baselines, which are based on the entropy, Gini coefficient, and sparsity of the attention values at each layer. The weight of each layer is calculated based on its corresponding metric (entropy, Gini coefficient, or sparsity), and the budget is allocated accordingly. Specifically:

- Entropy-based allocation: Layers with higher entropy receive higher weights. Each layer's entropy is calculated based on the the layer's attention.
- Gini coefficient-based allocation: Layers with higher Gini coefficients receive higher weights. Each layer's Gini coefficient is calculated based on the the layer's attention

The empirical results as Table 4 consistently showed that the linear strategy outperformed its counterparts, establishing it as the most effective approach for our use case. The experiment strengthens the rationale for choosing the specific allocation method.

|         | Single  | -Docum | ent QA | Multi-   | Document- | QA      | Sun       | nmariza | tion      | Few   | -shot Le | arning | Syntl  | netic   | Code       | Avg.     |
|---------|---------|--------|--------|----------|-----------|---------|-----------|---------|-----------|-------|----------|--------|--------|---------|------------|----------|
| Stra.   | NrtvQA  | Qasper | MF-en  | HotpotQA | 2WikiMQA  | Musique | GovReport | QMSum   | MultiNews | TREC  | TriviaQA | SAMSum | PCount | PRe     | LCC RB     | 8        |
| Geo.    | 20.51   | 15.04  | 29.4   | 34.93    | 26.41     | 16.6    | 18.32     | 21.68   | 18.81     | 52    | 87.51    | 36.15  | 5.18   | 69.17 5 | 3.11 44.9  | 1 34.36  |
| Exp.    | 20.58   | 14.82  | 28.74  | 34.34    | 26.24     | 16.11   | 18.41     | 21.63   | 18.75     | 52.00 | 87.94    | 36.26  | 5.19   | 69.17 5 | 54.34 43.2 | 21 34.23 |
| Lin.    | 21.13   | 14.18  | 30.26  | 35.12    | 23.76     | 16.17   | 18.33     | 21.65   | 19.23     | 58.00 | 88.31    | 37.07  | 5.23   | 69.50 5 | 2.61 45.7  | 4 34.76  |
| Entropy | . 18.12 | 14.12  | 27.22  | 33.21    | 21.16     | 15.16   | 17.76     | 19.87   | 17.09     | 51    | 87.31    | 34.29  | 5.09   | 68.91 5 | 0.12 42.9  | 8 32.71  |
| Gini.   | 17.92   | 14.61  | 28.21  | 32.67    | 19.98     | 15.98   | 16.20     | 19.29   | 18.21     | 51.00 | 86.21    | 34.97  | 5.11   | 65.51 5 | 51.98 43.3 | 37 32.58 |

<span id="page-18-0"></span>Table 4: Ablation study of allocation strategies.

#### **I.2** Hyper Parameter $\alpha$

We present the study of  $\alpha$  for LlaMa-3-8B-Instruct in 128 KV cache size budget at Table 5.We find that a small alpha value (i.e., 8, 16) leads to better performance than a larger alpha value (i.e., 24, 32, 40, 48).

|    | Single | -Docum | ent QA | Multi-   | Document | QA      |           | nmarizat |           |       | -shot Le |        | Synt  |                   | Avg.    |
|----|--------|--------|--------|----------|----------|---------|-----------|----------|-----------|-------|----------|--------|-------|-------------------|---------|
| α  | NrtvQA | Qasper | MF-en  | HotpotQA | WikiMQA  | Musique | GovRePort | OMSum.   | MultiNews | TREC  | iriviaQA | SAMSum | PCoun | PRE LCC RBS       | ?       |
| 8  | 21.40  | 16.92  | 31.62  | 38.45    | 28.72    | 18.59   | 19.96     | 22.49    | 20.96     | 66.50 | 89.35    | 38.43  | 5.92  | 69.00 57.86 51.8  | 0 37.37 |
| 16 | 23.37  | 16.21  | 33.93  | 38.24    | 27.28    | 20.57   | 19.71     | 21.93    | 20.86     | 60.00 | 88.75    | 38.34  | 5.48  | 69.12 57.84 53.4  | 2 37.19 |
| 24 | 22.85  | 14.51  | 32.26  | 38.38    | 28.36    | 20.33   | 19.55     | 21.72    | 20.72     | 54.50 | 88.71    | 38.46  | 5.48  | 69.50 56.83 53.6  | 5 36.61 |
| 32 | 23.01  | 14.54  | 31.68  | 38.86    | 29.90    | 19.16   | 19.20     | 21.83    | 20.52     | 49.50 | 87.01    | 38.01  | 5.75  | 69.50 57.02 54.5  | 4 36.25 |
| 40 | 21.70  | 13.06  | 30.14  | 36.78    | 27.34    | 18.88   | 18.72     | 21.37    | 19.79     | 44.00 | 87.74    | 38.43  | 6.08  | 69.25 56.11 53.89 | 9 35.21 |
| 48 | 21.51  | 12.30  | 29.77  | 39.04    | 26.76    | 17.97   | 18.65     | 21.20    | 20.29     | 44.50 | 87.73    | 38.44  | 5.51  | 69.25 56.73 53.8  | 8 35.22 |

<span id="page-18-1"></span>Table 5: Ablation on  $\alpha$ .

#### I.3 Hyper Parameter $\beta$

One topic we want to analyze for our ablation study is the selection of  $\beta$ , which can determine the staircase. The smaller  $\beta$  is, the gentler the staircase is; the larger  $\beta$  is, the steeper the staircase is. We want to investigate the effect of  $\beta$  step size on the final result. Results on 128 KV cache size and LlaMa-3-8B-Instruct are shown in Table 6. The results at Table 6 show that using a relatively small value of  $\beta$  yields better outcomes, and PyramidKV is generally robust to the selection of  $\beta$ .

|    | Single | -Docum | ent QA | Multi    | -Documen | t QA    | Su        | mmariza | tion      | Few   | -shot L  | earning | Synt  | hetic | Code       | Avg.    |
|----|--------|--------|--------|----------|----------|---------|-----------|---------|-----------|-------|----------|---------|-------|-------|------------|---------|
| β  | NrtvQA | Qasper | MF-en  | HotpotQA | 2WikiMQA | Musique | GovReport | QMSum   | MultiNews | TREC  | friviaQA | SAMSum  | PCoun | PRE   | LCC RB     | 2       |
| 20 | 21.40  | 16.92  | 33.79  | 39.73    | 28.72    | 18.59   | 19.86     | 22.48   | 20.95     | 66.50 | 89.35    | 38.39   | 5.92  | 69.00 | 56.49 47.9 | 5 37.25 |
| 18 | 21.71  | 16.24  | 33.59  | 39.89    | 27.94    | 18.38   | 19.76     | 22.32   | 21.20     | 66.50 | 88.98    | 38.93   | 5.46  | 69.50 | 56.47 49.2 | 3 37.25 |
| 16 | 21.74  | 14.86  | 33.64  | 39.18    | 28.17    | 18.77   | 19.57     | 22.25   | 21.48     | 66.50 | 89.69    | 38.87   | 5.82  | 69.50 | 57.02 50.1 | 1 37.32 |
| 14 | 22.53  | 16.31  | 33.50  | 40.50    | 28.15    | 19.26   | 19.66     | 22.39   | 21.38     | 65.50 | 90.02    | 38.56   | 5.75  | 69.50 | 57.51 49.7 | 1 37.51 |

<span id="page-18-2"></span>Table 6: Ablation on  $\beta$ .

