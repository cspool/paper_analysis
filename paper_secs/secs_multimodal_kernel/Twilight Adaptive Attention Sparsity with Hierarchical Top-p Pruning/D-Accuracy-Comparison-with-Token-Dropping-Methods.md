# D Accuracy Comparison with Token Dropping Methods

As discussed in [Section 2,](#page-2-2) top-k sparse attention methods can be broadly categorized into two types: token dropping and token selecting. Prior research [\[9\]](#page-14-2) has established that token selecting generally outperforms token dropping, as the latter inevitably incurs irreversible information loss. To further validate this observation, we conduct comparative experiments between Twilight and two representative token-dropping methods: StreamingLLM [\[17\]](#page-14-10) and SnapKV [\[18\]](#page-14-11). As demonstrated in [Table 6,](#page-20-1) DS-Twilight achieves notably better performance over both baseline methods.

<span id="page-20-1"></span>Table 6: Comparison of StreamingLLM, SnapKV, and Twilight on the Longbench benchmark with the Longchat-7B-v1.5-32k model.

| Dataset     | StreamingLLM (Budget=4096) | SnapKV (Budget=4096) | DS-Twilight |
|-------------|----------------------------|----------------------|-------------|
| Qasper      | 26.39                      | 29.44                | 32.34       |
| MulQA-en    | 33.2                       | 40.03                | 43.89       |
| HotpotQA    | 24.29                      | 33.67                | 34.67       |
| 2WikiMQA    | 20.1                       | 24.13                | 25.43       |
| Musique     | 10.87                      | 13.45                | 13.84       |
| GovReport   | 26.92                      | 26.09                | 31.88       |
| QMSum       | 20.8                       | 22.53                | 23.01       |
| MultiNews   | 26.46                      | 25.61                | 26.32       |
| TrivialQA   | 75.6                       | 80.82                | 85.29       |
| PR-en       | 24.17                      | 30.25                | 35.50       |
| LCC         | 52.47                      | 52.62                | 55.03       |
| Repobench-P | 51.02                      | 55.99                | 57.27       |
| Avg.        | 32.69                      | 36.22                | 38.71       |

