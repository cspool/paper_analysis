# <span id="page-18-0"></span>**F.4 Scaling Anaylsis**

In this section, we conduct empirical studies to evaluate the throughput of MHLA across different tasks under varying

<span id="page-18-7"></span>**Table 13** High-resolution classification accuracy of DeiT-T with and without MHLA.

| Model         | Resolution | ACC  |
|---------------|------------|------|
| DeiT-T        | 384×384    | 74.4 |
| DeiT-T + MHLA | 384×384    | 77.5 |
| DeiT-T        | 512×512    | 75.3 |
| DeiT-T + MHLA | 512×512    | 78.3 |

sequence lengths N and token-level head numbers M. The results in Tab. [14](#page-20-2) show that when M<sup>2</sup> < N is satisfied, MHLA introduces only negligible overhead, whereas larger M leads to more noticeable overhead.

**Table 9 Fast adaptation results on DiT-XL/2 with MHLA, with and without guidance.**

<span id="page-18-5"></span>

| Model       | Attention Type | Resolution | FID ↓ | IS ↑   | sFID ↓ | Precision ↑ | Recall ↑ |
|-------------|----------------|------------|-------|--------|--------|-------------|----------|
| DiT-XL/2    | Self Attention | 256        | 9.62  | 121.50 | 6.85   | 0.67        | 0.67     |
|             | MHLA (Ours)    | 256        | 8.34  | 121.27 | 5.52   | 0.69        | 0.65     |
| DiT-XL/2(G) | Self Attention | 256        | 2.27  | 278.24 | 4.60   | 0.83        | 0.57     |
|             | MHLA (Ours)    | 256        | 2.54  | 252.07 | 4.67   | 0.83        | 0.56     |

**Table 10 Comparison of different attention types across models.**

<span id="page-19-3"></span>

| Model    | Attention Type             | Resolution | FID ↓  | IS ↑  | sFID ↓ | Precision ↑ | Recall ↑ |
|----------|----------------------------|------------|--------|-------|--------|-------------|----------|
| DiT-S/2  | Self Attention             | 256        | 68.40  | –     | –      | –           | –        |
|          | Linear Attention           | 256        | 89.72  | 15.24 | 21.87  | 0.28        | 0.41     |
|          | MHLA (Ours)                | 256        | 59.80  | 23.49 | 10.16  | 0.39        | 0.56     |
|          | Self Attention             | 512        | 84.54  | 15.53 | 17.02  | 0.36        | 0.49     |
|          | Linear Attention           | 512        | 125.33 | 33.11 | 11.64  | 0.22        | 0.29     |
|          | MHLA (Ours)                | 512        | 78.63  | 13.11 | 18.50  | 0.40        | 0.49     |
|          | GLA [54]                   | 256        | 62.06  | –     | –      | –           | –        |
| DiG-S/2  | GLA                        | 512        | 99.04  | –     | –      | –           | –        |
|          | MHLA (Ours)                | 256        | 59.49  | 24.04 | 11.51  | 0.40        | 0.57     |
|          | Self Attention             | 256        | 43.47  | –     | –      | –           | –        |
| DiT-B/2  | Linear Attention           | 256        | 60.47  | 24.27 | 13.69  | 0.39        | 0.57     |
|          | MHLA (Ours)                | 256        | 37.47  | 38.79 | 7.35   | 0.51        | 0.63     |
| DiT-L/2  | Self Attention             | 256        | 23.33  | –     | –      | –           | –        |
|          | Linear Attention           | 256        | 32.35  | 45.57 | 8.55   | 0.54        | 0.62     |
|          | MHLA (Ours, w/None)        | 256        | 25.37  | 54.38 | 6.06   | 0.59        | 0.61     |
|          | MHLA (Ours, w/ CPE)        | 256        | 24.21  | 57.62 | 6.12   | 0.59        | 0.62     |
|          | MHLA (Ours, w/ CPE+Gating) | 256        | 21.37  | 63.47 | 5.80   | 0.61        | 0.62     |
|          | Self Attention             | 256        | 19.47  | –     | –      | –           | –        |
|          | Linear Attention           | 256        | 28.63  | 51.15 | 8.23   | 0.57        | 0.62     |
| DiT-XL/2 | MHLA (Ours, w/ None)       | 256        | 20.32  | 65.95 | 6.01   | 0.61        | 0.62     |
|          | MHLA (Ours, w/ CPE)        | 256        | 22.79  | 61.80 | 5.53   | 0.60        | 0.62     |
|          | MHLA (Ours, w/ CPE+Gating) | 256        | 19.17  | 68.97 | 5.70   | 0.63        | 0.62     |

However, our ablation studies in Tab. [7b](#page-10-2) have already demonstrated that choosing M such that M<sup>2</sup> < N is sufficient to achieve strong performance.

