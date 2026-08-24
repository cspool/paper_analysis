# <span id="page-13-1"></span>F Performance under Different Token Budgets

To evaluate the performance scalability and resource sensitivity of our method, we analyze its behavior under varying inference-time token budgets (i.e., the maximum number of tokens the model is allowed to generate). We compare ASAP with three strong baselines—SPIRIT, Original, and Zero-shot—on HumanEval+, Live-CodeBench v1\_v3, LiveCodeBench v4\_v5, LeetcodeDataset, GSM8K, MATH500, AIME24, and AIME25. For simpler benchmarks (including HumanEval+, GSM8K, and MATH500), we evaluate the performance under four budget settings, ranging from 1K to 6K tokens. For more complex benchmarks (including LiveCodeBench v1\_v3, LiveCodeBench v4\_v5, LeetcodeDataset, AIME24, and AIME25), we evaluate the performance under six budget settings, ranging from 2K to 12K tokens. Results are shown in Table [11,](#page-14-2) Table [12,](#page-14-3) Table [13,](#page-15-0) Table [14,](#page-15-1) Table [15,](#page-15-2) Table [16,](#page-15-3) Table [17,](#page-16-0) and Table [18.](#page-16-1)

## <span id="page-13-2"></span>G Training Efficiency

To quantify the training efficiency gains, we present results of the CodeForces-CoTs dataset in Table [5](#page-7-0) and results of the OpenR1-Math dataset in Table [19.](#page-13-4) We report two key metrics: the *average number of tokens* per sample and the *average training time* measured in seconds per step.

<span id="page-13-4"></span>

| Methods     | Tokens                                         | Time                         |
|-------------|------------------------------------------------|------------------------------|
| Original    | 5807                                           | 47.82                        |
|             | Selective Context 3149 (-45.8%) 25.85 (-45.9%) |                              |
| LLMLingua-2 |                                                | 3478 (-40.1%) 28.75 (-39.9%) |
| TokenSkip   |                                                | 4728 (-18.6%) 39.20 (-18.0%) |
| SPIRIT      |                                                | 2858 (-50.8%) 23.67 (-50.5%) |
| ASAP        |                                                | 1834 (-68.4%) 15.36 (-67.9%) |

Table 19: Training efficiency comparison on OpenR1- Math dataset. We report the average number of tokens per sample and training time measured in seconds per step. Percentages indicate the reduction relative to the Original baseline.

<span id="page-14-0"></span>

| Methods         | HE+   |       |       |       | LCBv1_v3 |       |       | LCBv4_v5 |       | LCD   |       |       |
|-----------------|-------|-------|-------|-------|----------|-------|-------|----------|-------|-------|-------|-------|
|                 | Acc ↑ | Tok ↓ | Lat ↓ | Acc ↑ | Tok ↓    | Lat ↓ | Acc ↑ | Tok ↓    | Lat ↓ | Acc ↑ | Tok ↓ | Lat ↓ |
| Zero-shot 64.02 |       | 3334  | 1.86  | 44.12 | 7162     | 6.92  | 25.00 | 8508     | 8.90  | 27.19 | 8358  | 8.65  |
| Original        | 76.22 | 2978  | 1.63  | 52.61 | 6614     | 6.16  | 31.34 | 8202     | 8.60  | 26.32 | 8413  | 8.85  |
| SPIRIT          | 72.56 | 3159  | 1.74  | 52.61 | 6280     | 5.84  | 30.22 | 7913     | 8.45  | 26.75 | 8449  | 8.73  |
| ASAP            | 76.83 | 2494  | 1.30  | 48.86 | 3605     | 2.18  | 32.84 | 4175     | 2.69  | 27.63 | 3792  | 2.42  |

Table 9: Experimental results of different methods on code generation benchmarks with DeepSeek-R1-Distill-Llama-8B. We report accuracy (Acc), average number of generated tokens (Tok), and average generation latency (Lat) measured in seconds.

<span id="page-14-1"></span>

| Methods         | GSM8K |       |       | MATH500 |       |       | AIME24 |       |       | AIME25      |       |       |
|-----------------|-------|-------|-------|---------|-------|-------|--------|-------|-------|-------------|-------|-------|
|                 | Acc ↑ | Tok ↓ | Lat ↓ | Acc ↑   | Tok ↓ | Lat ↓ | Acc ↑  | Tok ↓ | Lat ↓ | Acc ↑       | Tok ↓ | Lat ↓ |
| Zero-shot 79.15 |       | 1262  | 0.36  | 57.20   | 2612  | 1.08  | 33.33  | 8445  |       | 10.42 26.67 | 8597  | 10.54 |
| Original        | 84.91 | 1310  | 0.37  | 63.00   | 2534  | 1.01  | 36.67  | 8550  |       | 10.04 30.00 | 8268  | 10.05 |
| SPIRIT          | 85.67 | 1256  | 0.35  | 62.60   | 2533  | 1.01  | 36.67  | 8788  |       | 10.04 36.67 | 8094  | 9.57  |
| ASAP            | 87.34 | 768   | 0.20  | 66.00   | 1734  | 0.65  | 36.67  | 5314  | 6.97  | 33.33       | 5348  | 7.05  |

Table 10: Experimental results of different methods on mathematical reasoning benchmarks with DeepSeek-R1- Distill-Llama-8B. We report accuracy (Acc), average number of generated tokens (Tok), and average generation latency (Lat) measured in seconds.

<span id="page-14-2"></span>

| Budget |       | Zero-shot |       |       | Original |       |       | SPIRIT |       |       | ASAP  |       |  |
|--------|-------|-----------|-------|-------|----------|-------|-------|--------|-------|-------|-------|-------|--|
|        | Acc ↑ | Tok ↓     | Lat ↓ | Acc ↑ | Tok ↓    | Lat ↓ | Acc ↑ | Tok ↓  | Lat ↓ | Acc ↑ | Tok ↓ | Lat ↓ |  |
| 1K     | 9.76  | 1007      | 0.28  | 14.63 | 983      | 0.28  | 10.98 | 995    | 0.28  | 23.78 | 946   | 0.27  |  |
| 2K     | 42.68 | 1813      | 0.53  | 43.29 | 1702     | 0.49  | 47.56 | 1690   | 0.49  | 54.88 | 1502  | 0.44  |  |
| 4K     | 66.46 | 2561      | 0.85  | 65.85 | 2511     | 0.82  | 69.51 | 2401   | 0.80  | 71.34 | 2116  | 0.72  |  |
| 6K     | 68.29 | 3051      | 1.16  | 75.61 | 2973     | 1.12  | 75.61 | 2764   | 1.07  | 78.66 | 2464  | 0.98  |  |

Table 11: Results of different methods under different budgets on HumanEval+. We report accuracy (Acc), average number of generated tokens (Tok), and average generation latency (Lat) measured in seconds.

<span id="page-14-3"></span>

| Budget | Zero-shot |       |       | Original |       |       | SPIRIT |       |       | ASAP  |       |       |
|--------|-----------|-------|-------|----------|-------|-------|--------|-------|-------|-------|-------|-------|
|        | Acc ↑     | Tok ↓ | Lat ↓ | Acc ↑    | Tok ↓ | Lat ↓ | Acc ↑  | Tok ↓ | Lat ↓ | Acc ↑ | Tok ↓ | Lat ↓ |
| 2K     | 16.50     | 1966  | 0.52  | 17.16    | 1920  | 0.51  | 18.95  | 1908  | 0.51  | 21.57 | 1833  | 0.49  |
| 4K     | 32.68     | 3499  | 1.06  | 30.72    | 3432  | 1.05  | 34.80  | 3370  | 1.03  | 34.97 | 3244  | 1.00  |
| 6K     | 39.05     | 4806  | 1.70  | 42.65    | 4673  | 1.67  | 43.14  | 4605  | 1.64  | 46.24 | 4358  | 1.54  |
| 8K     | 44.28     | 5903  | 2.46  | 47.71    | 5723  | 2.43  | 51.80  | 5515  | 2.27  | 52.61 | 4919  | 1.90  |
| 10K    | 42.16     | 7088  | 3.59  | 52.12    | 6611  | 3.15  | 50.82  | 6524  | 3.09  | 54.74 | 5177  | 2.09  |
| 12K    | 43.95     | 7988  | 5.10  | 54.41    | 7473  | 4.22  | 51.63  | 7362  | 4.09  | 55.56 | 5322  | 2.27  |

Table 12: Results of different methods under different budgets on LiveCodeBench v1\_v3. We report accuracy (Acc), average number of generated tokens (Tok), and average generation latency (Lat) measured in seconds.

<span id="page-15-0"></span>

| Budget | Zero-shot |       |       |       | Original |       |       | SPIRIT |       |       | ASAP  |       |  |
|--------|-----------|-------|-------|-------|----------|-------|-------|--------|-------|-------|-------|-------|--|
|        | Acc ↑     | Tok ↓ | Lat ↓ | Acc ↑ | Tok ↓    | Lat ↓ | Acc ↑ | Tok ↓  | Lat ↓ | Acc ↑ | Tok ↓ | Lat ↓ |  |
| 2K     | 6.72      | 2021  | 0.59  | 6.34  | 1999     | 0.57  | 8.21  | 1993   | 0.56  | 13.43 | 1930  | 0.54  |  |
| 4K     | 16.79     | 3820  | 1.22  | 15.67 | 3799     | 1.20  | 20.15 | 3712   | 1.18  | 20.90 | 3594  | 1.15  |  |
| 6K     | 23.13     | 5444  | 2.07  | 22.76 | 5397     | 2.00  | 26.49 | 5237   | 1.93  | 30.60 | 4988  | 1.85  |  |
| 8K     | 25.37     | 6927  | 3.27  | 25.74 | 6882     | 3.24  | 30.60 | 6634   | 3.09  | 35.07 | 5793  | 2.38  |  |
| 10K    | 25.37     | 8336  | 5.15  | 30.97 | 8289     | 4.83  | 33.58 | 7892   | 4.62  | 36.19 | 6035  | 2.61  |  |
| 12K    | 25.75     | 9706  | 7.44  | 32.46 | 9567     | 7.10  | 34.33 | 8987   | 6.73  | 36.57 | 6128  | 2.76  |  |

Table 13: Results of different methods under different budgets on LiveCodeBench v4\_v5. We report accuracy (Acc), average number of generated tokens (Tok), and average generation latency (Lat) measured in seconds.

<span id="page-15-1"></span>

| Budget | Zero-shot |             |       |       | Original |       |       | SPIRIT |       |       | ASAP  |       |  |
|--------|-----------|-------------|-------|-------|----------|-------|-------|--------|-------|-------|-------|-------|--|
|        | Acc ↑     | Tok ↓       | Lat ↓ | Acc ↑ | Tok ↓    | Lat ↓ | Acc ↑ | Tok ↓  | Lat ↓ | Acc ↑ | Tok ↓ | Lat ↓ |  |
| 2K     | 7.02      | 2028        | 0.53  | 6.14  | 2020     | 0.53  | 7.02  | 2001   | 0.53  | 10.09 | 1965  | 0.53  |  |
| 4K     | 13.16     | 3848        | 1.21  | 13.16 | 3854     | 1.21  | 16.23 | 3789   | 1.19  | 15.79 | 3758  | 1.19  |  |
| 6K     | 16.23     | 5553        | 2.04  | 16.67 | 5548     | 2.04  | 18.86 | 5407   | 2.00  | 19.30 | 5387  | 2.00  |  |
| 8K     | 19.30     | 7165        | 3.27  | 22.37 | 7104     | 3.18  | 22.37 | 6882   | 3.04  | 23.25 | 6722  | 2.88  |  |
| 10K    | 19.74     | 8680        | 4.95  | 25.00 | 8485     | 4.72  | 25.00 | 8186   | 4.45  | 27.63 | 7541  | 3.48  |  |
| 12K    |           | 21.49 10142 | 7.58  | 28.07 | 9717     | 7.09  | 26.32 | 9354   | 6.86  | 27.63 | 7902  | 3.83  |  |

Table 14: Results of different methods under different budgets on LeetCodeDataset. We report accuracy (Acc), average number of generated tokens (Tok), and average generation latency (Lat) measured in seconds.

<span id="page-15-2"></span>

| Budget | Zero-shot |       |       | Original |       |       | SPIRIT |       |       | ASAP  |       |       |
|--------|-----------|-------|-------|----------|-------|-------|--------|-------|-------|-------|-------|-------|
|        | Acc ↑     | Tok ↓ | Lat ↓ | Acc ↑    | Tok ↓ | Lat ↓ | Acc ↑  | Tok ↓ | Lat ↓ | Acc ↑ | Tok ↓ | Lat ↓ |
| 1K     | 48.75     | 963   | 0.19  | 59.29    | 942   | 0.19  | 54.66  | 925   | 0.18  | 83.93 | 693   | 0.14  |
| 2K     | 83.55     | 1301  | 0.27  | 86.35    | 1250  | 0.26  | 88.55  | 1118  | 0.23  | 90.75 | 753   | 0.16  |
| 4K     | 88.65     | 1553  | 0.37  | 90.37    | 1432  | 0.34  | 90.52  | 1227  | 0.28  | 91.28 | 778   | 0.18  |
| 6K     | 89.23     | 1714  | 0.46  | 91.05    | 1513  | 0.39  | 91.28  | 1297  | 0.33  | 91.81 | 790   | 0.20  |

Table 15: Results of different methods under different budgets on GSM8K. We report accuracy (Acc), average number of generated tokens (Tok), and average generation latency (Lat) measured in seconds.

<span id="page-15-3"></span>

| Budget | Zero-shot |       |       | Original |       |       | SPIRIT |       |       | ASAP  |       |       |
|--------|-----------|-------|-------|----------|-------|-------|--------|-------|-------|-------|-------|-------|
|        | Acc ↑     | Tok ↓ | Lat ↓ | Acc ↑    | Tok ↓ | Lat ↓ | Acc ↑  | Tok ↓ | Lat ↓ | Acc ↑ | Tok ↓ | Lat ↓ |
| 1K     | 19.00     | 1020  | 0.19  | 28.00    | 1017  | 0.19  | 18.40  | 1012  | 0.19  | 36.40 | 935   | 0.19  |
| 2K     | 42.20     | 1804  | 0.39  | 52.00    | 1767  | 0.39  | 54.40  | 1592  | 0.36  | 59.80 | 1347  | 0.31  |
| 4K     | 60.40     | 2629  | 0.70  | 63.80    | 2511  | 0.66  | 64.20  | 2144  | 0.57  | 70.80 | 1649  | 0.43  |
| 6K     | 66.60     | 3100  | 0.94  | 70.60    | 2843  | 0.84  | 69.60  | 2460  | 0.74  | 71.00 | 1758  | 0.52  |

Table 16: Results of different methods under different budgets on MATH500. We report accuracy (Acc), average number of generated tokens (Tok), and average generation latency (Lat) measured in seconds.

<span id="page-16-0"></span>

| Budget | Zero-shot |       |       | Original |       |       | SPIRIT |       |       | ASAP  |       |       |
|--------|-----------|-------|-------|----------|-------|-------|--------|-------|-------|-------|-------|-------|
|        | Acc ↑     | Tok ↓ | Lat ↓ | Acc ↑    | Tok ↓ | Lat ↓ | Acc ↑  | Tok ↓ | Lat ↓ | Acc ↑ | Tok ↓ | Lat ↓ |
| 2K     | 0.00      | 2048  | 0.93  | 16.67    | 2048  | 0.93  | 6.67   | 2048  | 0.93  | 10.00 | 1978  | 0.93  |
| 4K     | 20.00     | 4003  | 2.12  | 20.00    | 3984  | 2.11  | 23.33  | 3877  | 2.10  | 36.67 | 3415  | 1.99  |
| 6K     | 33.33     | 5682  | 3.58  | 36.67    | 5695  | 3.55  | 40.00  | 5216  | 3.32  | 36.67 | 4410  | 3.03  |
| 8K     | 30.00     | 7073  | 5.30  | 40.00    | 7093  | 5.14  | 46.67  | 6243  | 4.47  | 40.00 | 5159  | 4.10  |
| 10K    | 36.67     | 8352  | 6.76  | 46.67    | 8034  | 6.43  | 46.67  | 7198  | 5.78  | 46.67 | 5552  | 5.04  |
| 12K    | 40.00     | 9318  | 7.84  | 46.67    | 8990  | 7.75  | 46.67  | 8363  | 7.21  | 46.67 | 5767  | 5.88  |

Table 17: Results of different methods under different budgets on AIME24. We report accuracy (Acc), average number of generated tokens (Tok), and average generation latency (Lat) measured in seconds.

<span id="page-16-1"></span>

| Budget |       | Zero-shot |       |       | Original |       |       | SPIRIT |       |       | ASAP  |       |  |
|--------|-------|-----------|-------|-------|----------|-------|-------|--------|-------|-------|-------|-------|--|
|        | Acc ↑ | Tok ↓     | Lat ↓ | Acc ↑ | Tok ↓    | Lat ↓ | Acc ↑ | Tok ↓  | Lat ↓ | Acc ↑ | Tok ↓ | Lat ↓ |  |
| 2K     | 6.67  | 2046      | 0.95  | 13.33 | 2048     | 0.94  | 3.33  | 2044   | 0.94  | 10.00 | 2020  | 0.95  |  |
| 4K     | 20.00 | 3851      | 2.17  | 26.67 | 3834     | 2.16  | 16.67 | 3792   | 2.15  | 20.00 | 3511  | 2.15  |  |
| 6K     | 30.00 | 5369      | 3.62  | 33.33 | 5452     | 3.63  | 36.67 | 5360   | 3.63  | 30.00 | 4484  | 3.14  |  |
| 8K     | 36.67 | 6848      | 5.30  | 36.67 | 6798     | 5.32  | 36.67 | 6611   | 5.16  | 33.33 | 5002  | 4.10  |  |
| 10K    | 40.00 | 8145      | 6.67  | 43.33 | 8026     | 6.75  | 43.33 | 7817   | 6.57  | 36.67 | 5434  | 5.10  |  |
| 12K    | 36.67 | 9442      | 8.25  | 40.00 | 9461     | 8.32  | 46.67 | 8598   | 7.78  | 36.67 | 5720  | 6.06  |  |

.

Table 18: Results of different methods under different budgets on AIME25. We report accuracy (Acc), average number of generated tokens (Tok), and average generation latency (Lat) measured in seconds.