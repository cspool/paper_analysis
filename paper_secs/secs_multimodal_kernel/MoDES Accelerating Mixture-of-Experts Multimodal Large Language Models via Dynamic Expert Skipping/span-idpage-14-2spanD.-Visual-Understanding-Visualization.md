# <span id="page-14-2"></span>**D. Visual Understanding Visualization**

In this section, we present a case study comparing our proposed MoDES with previous SOTA methods [6, 22] for LLMs. As shown in Figs. I and II, MoDES consistently generates text that far outperforms the baselines.

#### <span id="page-14-3"></span>**E.** Ablation for N

We apply MoDES to Kimi-VL-A3B-Instruct [50] using different numbers of data samples from GQA [25] and show the results in Tab. III. The results indicate a clear trend:

<span id="page-14-6"></span>

| Table III. Holation legalts for 11. |         |               |                      |       |       |  |  |  |  |  |  |
|-------------------------------------|---------|---------------|----------------------|-------|-------|--|--|--|--|--|--|
| N                                   | ChartQA | MME           | MMBench              | LVB   | VMMMU |  |  |  |  |  |  |
| Skip 67% Experts ( $\rho = 0.65$ )  |         |               |                      |       |       |  |  |  |  |  |  |
| 2048                                | 88.32   | 2201          | 82.79                | 62.92 | 48.89 |  |  |  |  |  |  |
| 1024 (Ours)                         | 88.24   | 2204          | 82.73                | 62.90 | 48.78 |  |  |  |  |  |  |
| 512                                 | 87.44   | 2122          | 81.27                | 61.95 | 47.68 |  |  |  |  |  |  |
| 256                                 | 85.56   | 2085          | 79.68                | 60.63 | 45.11 |  |  |  |  |  |  |
|                                     | Skip 8  | 3% <i>Exp</i> | erts ( $\rho = 0.80$ | ))    |       |  |  |  |  |  |  |
| 2048                                | 84.84   | 2186          | 81.45                | 62.63 | 46.67 |  |  |  |  |  |  |
| 1024 (Ours)                         | 84.20   | 2162          | <u>81.44</u>         | 62.60 | 47.11 |  |  |  |  |  |  |
| 512                                 | 84.12   | 2118          | 80.27                | 61.88 | 46.85 |  |  |  |  |  |  |
| 256                                 | 83.35   | 2016          | 77.48                | 59.84 | 43.69 |  |  |  |  |  |  |

With more calibration samples, models using *expert skip-ping* perform better. Yet the accuracy gains become smaller as the sample count grows. Moreover, doubling the samples increases both calibration and search time by  $\sim 2 \times$ . To balance accuracy and cost, we use 1024 samples in this paper. This choice provides most of the achievable gains while keeping computation reasonable (Sec. 6.3).

#### <span id="page-14-4"></span>**F.** Ablation for D

Table IV. Ablation results for D.

<span id="page-14-7"></span>

| D                                  | ChartQA | MME     | MMBench              | LVB   | VMMMU        |  |  |  |  |  |
|------------------------------------|---------|---------|----------------------|-------|--------------|--|--|--|--|--|
| Skip 67% Experts ( $\rho = 0.65$ ) |         |         |                      |       |              |  |  |  |  |  |
| 200                                | 88.16   | 2219    | 82.78                | 62.94 | 48.76        |  |  |  |  |  |
| 100 (Ours)                         | 88.24   | 2204    | 82.73                | 62.90 | 48.78        |  |  |  |  |  |
| 50                                 | 87.85   | 2178    | 81.76                | 62.21 | 47.89        |  |  |  |  |  |
|                                    | Skip    | 83% Exp | erts ( $\rho = 0.80$ | ))    |              |  |  |  |  |  |
| 200                                | 84.78   | 2178    | 81.61                | 62.59 | 47.00        |  |  |  |  |  |
| 100 (Ours)                         | 84.20   | 2162    | 81.44                | 62.60 | <u>47.11</u> |  |  |  |  |  |
| 50                                 | 83.96   | 2143    | 80.68                | 62.47 | 47.15        |  |  |  |  |  |

We ablate the number of grid points D in the search space  $\mathcal{B}$ . As shown in Tab. IV, larger D brings diminishing

<span id="page-15-0"></span>accuracy gains, so using a very fine grid (e.g., D>100) is unnecessary. The time cost also grows roughly linearly with D. Based on this trade-off, we set D=100 in this work.

