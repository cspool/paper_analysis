# <span id="page-18-0"></span>E Comparison of Speedup Aggregation Metrics

> **[图片提取文字 (无描述)]:**
> 40 37.3% 40 25.7% 25.0% 30 30 8 8 8 17.5% Opt@1 Opt@1 මු 20 <u>.</u> 15.8% 9.8% 5.9% 5.9% 6.2% 10 8.5% 10 + 10 2.6% 3.3% 2.6% 3.9% 1.3% 0.0% 0.0% o3-mini o4-mini o4-mini claude-3,6 claude-3,7 claude-4
![](_page_18_Figure_5.jpeg)

Figure 11: Comparison of speedup aggregation metrics with its effect on OPT@1 scores. Left: arithmetic mean, middle: geometric mean, right: risk-adjusted geometric mean (RAGM). Each metric exhibits different sensitivities to outliers and distributional properties.

Arithmetic Mean: Treats every test equally but is highly susceptible to large outliers—a single extreme speedup can disproportionately inflate the average and mask regressions elsewhere.

Geometric Mean: large speedups still exert substantial influence: for example, speedups of [0.1, 1000] yield a GM of 10, despite a 90% slowdown on one test. This again allows dramatic wins to disguise serious regressions.

Risk-Adjusted Geometric Mean (RAGM) Computed as exp µ − 0.5γσ<sup>2</sup> with µ = 1 n Plog s<sup>i</sup> , σ <sup>2</sup> = n P(log s<sup>i</sup> − µ) 2 , and tunable γ. By penalizing distributions with high variance, RAGM ensures that extreme slowdowns and spikes are reflected, offering a symmetric treatment. However, we do not want symmetric treatment—large wins on minor tests shouldn't hurt, only significant regressions matter.

We study several such aggregation metrics and find that Harmonic Mean was the most suitable for our use case. Its asymmetric sensitivity punishes slowdowns heavily, while almost ignoring large speedups. This matches our goal of flagging regressions without overstating trivial wins.

