# D. Per-Domain Performance Improvement

<span id="page-14-0"></span>We also report per-domain performance improvement for PACS [\(Li et al.,](#page-10-27) [2017\)](#page-10-27) and Terra Incognito [\(Beery et al.,](#page-9-22) [2018\)](#page-9-22) dataset. We choose the best model based on the validation set and report the results in [7](#page-14-0) and [8.](#page-14-1) The results with quantization correspond to 7 bit-precision and we perform quantization after 2000 steps. Table [7](#page-14-0) and [8](#page-14-1) show that EoQ is consistently better than the current state-of-the-art methods across domains for different datasets.

| Algorithm      | Art  | Cartoon | Painting | Sketch | Avg. |
|----------------|------|---------|----------|--------|------|
| ERM (our runs) | 89.8 | 79.7    | 96.8     | 72.5   | 84.7 |
| SWAD           | 89.3 | 83.4    | 97.3     | 82.5   | 88.1 |
| EoA            | 90.5 | 83.4    | 98.0     | 82.5   | 88.6 |
| DiWA           | 90.6 | 83.4    | 98.2     | 83.8   | 89.0 |
| QT-DoG         | 89.1 | 82.4    | 96.9     | 82.3   | 87.8 |
| EoQ            | 90.7 | 83.7    | 98.2     | 84.8   | 89.3 |

<span id="page-14-1"></span>Table 7. Per-Domain Accuracy Comparison for PACS. We report the accuracy for each domain of the PACS dataset along with the average across all domains. Our proposed quantization is shaded in Gray.

| Algorithm      | L100 | L38  | L43  | L46  | Avg. |
|----------------|------|------|------|------|------|
| ERM (our runs) | 58.2 | 38.3 | 57.1 | 35.1 | 47.2 |
| SWAD           | 55.4 | 44.9 | 59.7 | 39.9 | 50.0 |
| DiWA           | 57.2 | 50.1 | 60.3 | 39.8 | 51.9 |
| EoA            | 57.8 | 46.5 | 61.3 | 43.5 | 52.3 |
| QT-DoG         | 60.2 | 46.4 | 55.2 | 41.4 | 50.8 |
| EoQ            | 61.8 | 48.2 | 59.2 | 43.7 | 53.2 |

Table 8. Per-Domain Accuracy Comparison for Terra Incognito. We report the accuracy for each domain of the Terra Incognito dataset along with the average across all domains. Our proposed quantization is shaded in Gray.

