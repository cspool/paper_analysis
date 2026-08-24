# <span id="page-11-1"></span>F Ablation Details

Effect of Probing Data Size. We evaluate how training size affects probing quality. As shown in Table [6,](#page-12-2) performance remains stable across 500–3000 training examples, with only marginal gains. This suggests that even a small probing set can support effective compression.

Feature Selection Details To construct a compact attention-based feature set, we use the Minimum Redundancy Maximum Relevance (mRMR) algorithm. We first compute mutual information between each feature (i.e., attention head statistics) and the binary relevance label, selecting the most informative one. We then iteratively add features that maximize relevance while minimizing redundancy, measured via Pearson correlation with already selected features. The number of features is

<span id="page-12-1"></span>

| Method                           |           | LongBench-En (2K Constraint) |       |        | LongBench-Zh (2K Constraint) | Overall  |        |       |
|----------------------------------|-----------|------------------------------|-------|--------|------------------------------|----------|--------|-------|
|                                  | SingleDoc | MultiDoc                     | Summ. | En-AVG | SingleDoc                    | MultiDoc | Zh-AVG | AVG   |
| Sentinel (Qwen2.5-0.5B-Instruct) | 37.73     | 46.16                        | 23.03 | 35.64  | 62.24                        | 18.57    | 40.41  | 38.02 |
| Sentinel (Qwen2.5-1.5B-Instruct) | 39.48     | 46.07                        | 23.10 | 36.22  | 62.02                        | 18.91    | 40.47  | 38.34 |
| Sentinel (Qwen2.5-3B-Instruct)   | 39.53     | 47.97                        | 23.06 | 36.85  | 62.04                        | 19.23    | 40.63  | 38.74 |
| Sentinel (Qwen2.5-7B-Instruct)   | 38.79     | 45.56                        | 22.52 | 35.62  | 60.88                        | 18.43    | 39.66  | 37.64 |
| Sentinel (Llama-3.2-1B-Instruct) | 39.43     | 44.96                        | 21.90 | 35.43  | 60.64                        | 19.18    | 39.91  | 37.67 |
| Sentinel (Llama-3.2-3B-Instruct) | 36.03     | 44.46                        | 22.00 | 34.17  | 59.24                        | 18.89    | 39.06  | 36.62 |
| Sentinel (Llama-3.1-8B-Instruct) | 36.58     | 45.15                        | 22.90 | 34.87  | 60.84                        | 19.07    | 39.95  | 37.41 |
| Sentinel (Qwen3-0.6B)            | 38.12     | 42.55                        | 22.77 | 34.48  | 60.04                        | 18.51    | 39.27  | 36.88 |
| Sentinel (Qwen3-1.7B)            | 36.52     | 42.06                        | 22.29 | 33.62  | 60.79                        | 17.96    | 39.38  | 36.50 |
| Sentinel (Qwen3-4B)              | 37.15     | 43.17                        | 22.67 | 34.33  | 59.68                        | 17.74    | 38.71  | 36.52 |
| Sentinel (Qwen3-8B)              | 36.31     | 42.19                        | 22.15 | 33.55  | 60.74                        | 17.77    | 39.26  | 36.40 |
| Original Prompt                  | 38.84     | 44.74                        | 22.76 | 35.45  | 60.06                        | 18.21    | 39.14  | 37.30 |

Table 5: Detailed Sentinel performance across different proxy model families and scales under a 2K-token context constraint. The Summ. column corresponds to query-conditioned summarization tasks (QMSum).

<span id="page-12-2"></span>

| Method                                                       |                | LongBench-En (2K Constraint) |                |                | LongBench-Zh (2K Constraint) | Overall        |                |                |
|--------------------------------------------------------------|----------------|------------------------------|----------------|----------------|------------------------------|----------------|----------------|----------------|
|                                                              | SingleDoc      | MultiDoc                     | Summ.          | En-AVG         | SingleDoc                    | MultiDoc       | Zh-AVG         | AVG            |
| Qwen2.5-0.5B-Instruct (500)                                  | 37.29          | 46.94                        | 23.25          | 35.83          | 62.04                        | 18.42          | 40.23          | 38.03          |
| Qwen2.5-0.5B-Instruct (1000)                                 | 38.35          | 47.43                        | 23.66          | 36.48          | 61.43                        | 18.57          | 40.00          | 38.24          |
| Qwen2.5-0.5B-Instruct (2000)<br>Qwen2.5-0.5B-Instruct (3000) | 36.70<br>37.73 | 47.48<br>46.16               | 22.89<br>23.03 | 35.69<br>35.64 | 61.57<br>62.24               | 18.76<br>18.57 | 40.16<br>40.41 | 37.92<br>38.02 |

Table 6: Performance of Qwen2.5-0.5B-Instruct with different probing sizes on LongBench under a 2K-token context constraint. The Summ. column corresponds to query-conditioned summarization tasks (QMSum).

capped at the number of heads in a single decoder layer to ensure compactness and interpretability.

Compression Ratio. Table [7](#page-13-0) reports results with varying compression ratios (τ ∈ {0.1, 0.2, 0.3, 0.4, 0.5}), under a fixed chunk size of 1024. Sentinel remains robust even at high compression, while Raw attention deteriorates significantly.

