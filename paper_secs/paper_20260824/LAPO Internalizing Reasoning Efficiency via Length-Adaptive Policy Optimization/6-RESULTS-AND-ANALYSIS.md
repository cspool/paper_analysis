# 6 RESULTS AND ANALYSIS

We present comprehensive experimental results and analysis to validate LAPO's effectiveness and understand its underlying mechanisms. We begin with the main results (Section [6.1\)](#page-7-0), benchmarking

<span id="page-7-1"></span>Table 1: Main results on MATH500, AIME2024, AMC23, and OlympiadBench. We report Pass@1 accuracy (%) and the average number of generated tokens (#Tok). For each metric, bold indicates the best and underline indicates the second-best Pass@1 score within each base model group.

|                                     | MATH-500                     |      | AIME2024    |       | AMC-23      |      | OlympiadBench |      | Average     |      |
|-------------------------------------|------------------------------|------|-------------|-------|-------------|------|---------------|------|-------------|------|
|                                     | Pass@1 #Tok                  |      | Pass@1 #Tok |       | Pass@1 #Tok |      | Pass@1 #Tok   |      | Pass@1 #Tok |      |
|                                     | Base model: DeepSeek-R1-1.5B |      |             |       |             |      |               |      |             |      |
| HAPO                                | 82.2                         | 2288 | 31.3        | 8649  | 67.3        | 4735 | 50.1          | 5024 | 57.7        | 5174 |
| AutoThink                           | 83.5                         | 2017 | 29.7        | 7084  | 70.2        | 3499 | 51.2          | 4606 | 58.6        | 3825 |
| AdaptThink                          | 81.6                         | 1580 | 23.9        | 6432  | 63.2        | 2860 | 48.5          | 4616 | 54.3        | 3871 |
| Base                                | 83.1                         | 4031 | 30.3        | 12150 | 68.3        | 7222 | 50.0          | 8942 | 57.9        | 8086 |
| + Acc-Only                          | 83.3                         | 3061 | 31.6        | 10628 | 70.5        | 5307 | 50.6          | 6402 | 59.0        | 6349 |
| + LAPO-D                            | 84.7                         | 2566 | 28.5        | 8415  | 72.2        | 4132 | 51.3          | 5595 | 59.2        | 5177 |
| + LAPO-I                            | 84.3                         | 2354 | 29.3        | 8318  | 71.2        | 3568 | 51.7          | 4863 | 59.1        | 4775 |
| Base model: DeepScaleR-1.5B-Preview |                              |      |             |       |             |      |               |      |             |      |
| L1-Exact                            | 80.6                         | 1953 | 24.4        | 2625  | 70.9        | 2177 | 48.8          | 2357 | 56.2        | 2278 |
| L1-Max                              | 81.9                         | 1673 | 24.9        | 3638  | 72.7        | 2705 | 50.5          | 2151 | 57.5        | 2541 |
| ThinkPrune-I2k                      | 85.5                         | 1707 | 34.9        | 5095  | 74.3        | 2913 | 54.7          | 3498 | 62.3        | 3303 |
| ThinkPrune-4k                       | 86.6                         | 2042 | 35.5        | 6488  | 76.3        | 3839 | 55.7          | 4010 | 63.5        | 4094 |
| HAPO                                | 84.4                         | 2370 | 31.4        | 7702  | 70.3        | 4301 | 51.4          | 4571 | 59.3        | 4736 |
| AutoThink                           | 84.9                         | 1635 | 36.2        | 7201  | 67.8        | 3658 | 52.5          | 4085 | 60.4        | 4144 |
| Thinkless                           | 81.3                         | 2944 | 28.9        | 9143  | 65.7        | 5276 | 50.2          | 6057 | 56.5        | 5855 |
| Base                                | 85.8                         | 3280 | 35.5        | 9246  | 74.2        | 6416 | 54.6          | 5974 | 62.5        | 6229 |
| + Acc-Only                          | 85.6                         | 2510 | 36.9        | 7319  | 77.6        | 4244 | 55.6          | 4712 | 63.9        | 4696 |
| + LAPO-D                            | 86.4                         | 2365 | 37.6        | 5945  | 77.6        | 3655 | 56.1          | 4499 | 64.4        | 4116 |
| + LAPO-I                            | 86.3                         | 2168 | 38.1        | 5371  | 78.3        | 3765 | 56.3          | 4024 | 64.8        | 3832 |

LAPO against baselines and state-of-the-art methods. We then conduct in-depth ablation studies on key design choices, including the the form of length guidance (Section [6.2\)](#page-8-0) and the statistical metrics for target length selection (Section [6.3\)](#page-8-1). And a targeted experiment demonstrating the model's robust internalization of reasoning efficiency (Section [6.4\)](#page-9-0). Finally, we provide a mechanistic analysis of how LAPO works, examining its emergent ability for difficulty-aware resource allocation (Section [6.5\)](#page-9-1), its qualitative refinement of reasoning patterns (Section [6.6\)](#page-10-1).

