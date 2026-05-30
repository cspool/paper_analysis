# <span id="page-12-1"></span>D. More Experimental Results

We present additional results using Llama3-8B, Mistral-7B-v0.2, and LongChat-7B-v1.5 in LongBench, which can be found in Table [8,](#page-13-2) Table [9](#page-13-0) and Table [10,](#page-14-0) respectively.

We also show result of Needle-in-a-Haystack Test in Figure [4.](#page-6-0) The settings largely follow the format of the original passkey retrieval task [\(Mohtashami and Jaggi,](#page-9-16) [2023\)](#page-9-16) while including some modern modifications set forward by [Arize-ai](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) and the technical report of Gemini 1.5 [\(Reid et al.,](#page-10-14) [2024\)](#page-10-14).

<span id="page-12-3"></span><sup>2</sup><https://paulgraham.com/articles.html>

<span id="page-13-1"></span>Table 7: Performance evaluation of KIVI with residual length 128 and 32 on various models across a range of benchmarks in LongBench. R32 stands for residual length 32.

| Model           |                         | Qasper | QMSum | MultiNews | TREC  | TriviaQA | SAMSum | LCC   | RepoBench-P | Average |
|-----------------|-------------------------|--------|-------|-----------|-------|----------|--------|-------|-------------|---------|
|                 | 16bit                   | 9.52   | 21.28 | 3.51      | 66.00 | 87.72    | 41.69  | 66.66 | 59.82       | 44.52   |
| Llama2-7B       | KIVI-2 $R128$           | 9.31   | 20.50 | 1.14      | 66.00 | 87.42    | 42.71  | 66.88 | 60.23       | 44.27   |
|                 | KIVI-2 $R32$            | 9.26   | 20.53 | 0.97      | 66.00 | 87.42    | 42.61  | 66.22 | 59.67       | 44.08   |
|                 | 16bit                   | 9.32   | 21.38 | 3.71      | 70.00 | 87.87    | 43.55  | 66.61 | 56.42       | 44.85   |
| Llama2-13B      | KIVI-2 $R128$           | 8.58   | 20.69 | 6.19      | 69.50 | 87.78    | 44.30  | 65.08 | 55.46       | 44.69   |
|                 | KIVI-2 $R32$            | 8.38   | 20.74 | 7.01      | 69.50 | 87.78    | 44.43  | 64.89 | 55.31       | 44.75   |
|                 | 16bit                   | 19.65  | 20.54 | 26.36     | 63.00 | 84.28    | 41.12  | 59.75 | 52.93       | 45.95   |
| Llama2-7B-Chat  | $\mathtt{KIVI-2}\ R128$ | 19.32  | 20.46 | 25.48     | 63.00 | 84.84    | 40.60  | 58.71 | 52.97       | 45.67   |
|                 | KIVI-2 $R32$            | 19.10  | 20.08 | 25.33     | 63.00 | 85.04    | 39.80  | 57.91 | 52.38       | 45.33   |
|                 | 16bit                   | 24.18  | 20.37 | 25.69     | 67.50 | 86.90    | 42.18  | 50.23 | 50.64       | 45.96   |
| Llama2-13B-Chat | $\mathtt{KIVI-2}\ R128$ | 23.59  | 20.76 | 25.25     | 67.50 | 87.17    | 41.56  | 49.93 | 48.45       | 45.52   |
|                 | KIVI-2 $R32$            | 23.56  | 20.90 | 25.45     | 67.50 | 87.42    | 41.40  | 48.93 | 48.81       | 45.49   |
|                 | 16bit                   | 1.48   | 2.35  | 11.09     | 13.00 | 5.84     | 2.44   | 23.86 | 9.69        | 8.71    |
|                 | $\mathtt{KIVI-4}\ R128$ | 1.04   | 2.41  | 11.98     | 13.00 | 5.84     | 2.36   | 23.72 | 9.92        | 8.78    |
| Falcon-7B       | $\mathtt{KIVI-4}\ R32$  | 1.03   | 2.45  | 11.99     | 13.50 | 5.84     | 2.46   | 23.88 | 9.95        | 8.88    |
|                 | KIVI-2 $R128$           | 1.98   | 3.61  | 6.78      | 10.00 | 6.24     | 2.73   | 22.18 | 10.12       | 7.95    |
|                 | KIVI-2 $R32$            | 2.28   | 3.23  | 6.73      | 10.00 | 6.31     | 2.88   | 22.71 | 10.45       | 8.07    |
|                 | 16bit                   | 8.12   | 19.98 | 19.99     | 67.50 | 89.80    | 41.69  | 66.59 | 58.99       | 46.58   |
| Mistral-7B      | $\mathtt{KIVI-2}\ R128$ | 6.92   | 19.71 | 17.92     | 66.50 | 89.63    | 41.66  | 65.52 | 58.99       | 45.85   |
|                 | KIVI-2 $R32$            | 6.84   | 19.81 | 17.20     | 66.50 | 89.63    | 42.82  | 65.13 | 58.06       | 45.74   |

<span id="page-13-2"></span>Table 8: The results of Llama-3-8B-Instruct with KIVI on LongBench. The model has 8K context length and applies group query attention, which uses 8 heads for KV cache instead of the full 32 heads. We use a 32 group size and 128 residual length for both KIVI-2 and KIVI-4. The baseline is of full precision.

|           | NarrativeQA | Qasper | MultiFieldQA | HotpotQA | MuSiQue | 2WikiMQA | GovReport | QMSum |
|-----------|-------------|--------|--------------|----------|---------|----------|-----------|-------|
| Baseline  | 21.71       | 44.24  | 44.54        | 46.82    | 21.49   | 36.42    | 30.03     | 22.67 |
| w./KIVI-2 | 21.35       | 43.17  | 44.49        | 46.79    | 20.56   | 37.05    | 29.98     | 22.07 |
| w./KIVI-4 | 21.01       | 44.83  | 44.60        | 46.96    | 21.43   | 36.48    | 30.22     | 22.44 |
|           | MultiNews   | LCC    | RepoBench-P  | TriviaQA | SAMSum  | TRec     | PR        | Avg   |
| Baseline  | 27.79       | 57.00  | 51.22        | 90.23    | 42.53   | 74.50    | 67.00     | 45.21 |
| w./KIVI-2 | 27.77       | 50.84  | 46.65        | 90.54    | 42.26   | 74.50    | 67.50     | 44.37 |
| w./KIVI-4 | 27.97       | 57.36  | 52.03        | 90.33    | 42.97   | 74.50    | 66.50     | 45.31 |

<span id="page-13-0"></span>Table 9: The results of Mistral-7B-Instruct-v0.2 with KIVI on LongBench. The model has 32K context length and applies group query attention, which uses 8 heads for KV cache instead of the full 32 heads. We use a 32 group size and 128 residual length for both KIVI-2 and KIVI-4. The baseline is of full precision.

|           | NarrativeQA | Qasper | MultiFieldQA | HotpotQA | MuSiQue | 2WikiMQA | GovReport | QMSum |
|-----------|-------------|--------|--------------|----------|---------|----------|-----------|-------|
| Baseline  | 21.02       | 29.41  | 47.13        | 36.53    | 19.13   | 21.76    | 32.59     | 23.99 |
| w./KIVI-2 | 20.61       | 28.73  | 44.88        | 35.47    | 17.95   | 20.68    | 32.55     | 23.65 |
| w./KIVI-4 | 20.97       | 29.41  | 46.52        | 36.25    | 19.53   | 21.66    | 32.97     | 24.06 |
|           | MultiNews   | LCC    | RepoBench-P  | TriviaQA | SAMSum  | TRec     | PR        | Avg   |
| Baseline  | 27.09       | 53.49  | 51.40        | 86.23    | 43.04   | 71.00    | 89.33     | 43.54 |
| w./KIVI-2 | 26.54       | 53.03  | 51.16        | 86.00    | 43.34   | 71.00    | 80.83     | 42.43 |
| w./KIVI-4 | 26.89       | 53.33  | 51.41        | 86.23    | 43.34   | 71.00    | 89.42     | 43.53 |

<span id="page-14-0"></span>Table 10: The results of LongChat-7B-v1.5-32K with KIVI on LongBench. The model has 32K context length. We use a 32 group size and 128 residual length for both KIVI-2 and KIVI-4. The baseline is of full precision.

|            | NarrativeQA | Qasper | MultiFieldQA | HotpotQA | MuSiQue | 2WikiMQA | GovReport | QMSum |
|------------|-------------|--------|--------------|----------|---------|----------|-----------|-------|
| Baseline   | 20.65       | 29.42  | 43.15        | 33.05    | 14.66   | 24.14    | 30.85     | 22.84 |
| w./ KIVI-2 | 20.79       | 28.69  | 41.02        | 32.91    | 13.82   | 23.00    | 30.47     | 22.59 |
| w./ KIVI-4 | 20.49       | 28.90  | 43.24        | 33.07    | 14.66   | 24.86    | 31.40     | 22.84 |
|            | MultiNews   | LCC    | RepoBench-P  | TriviaQA | SAMSum  | TRec     | PR        | Avg   |
| Baseline   | 26.55       | 54.83  | 58.94        | 83.99    | 40.75   | 66.50    | 30.50     | 38.72 |
| w./ KIVI-2 | 26.28       | 54.11  | 57.62        | 83.19    | 41.28   | 66.50    | 32.25     | 38.30 |
| w./ KIVI-4 | 26.52       | 54.06  | 58.77        | 83.88    | 40.62   | 67.00    | 31.50     | 38.79 |