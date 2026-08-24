# <span id="page-16-0"></span>C. Hyperparameter

The Dynamic-PI method interpolates dynamically according to the length of the input token. NTK-Aware refer to [\(LocalL-](#page-10-4)[LaMA, 2023b\)](#page-10-4) and the maximum length is set to 128. ChunkLlama, InfLLM and AttenCalibration-NTK use hyperparameters from open source repositories. About our method, when performing parallel KV Cache compression, we use last 8 token's cumulative attention scores to compress the KV cache size within each chunk to 2000/4000 for llama2/llama3. For the hyperparameter τ , on Longbench, we retain 3 chunks from the priority queue except for PRe, in which dataset we retain only 1 chunk. On InfiniteBench, we retain 1 chunk for retrieval tasks and 3 chunks for other tasks from the priority queue. In all datasets, the context length of each chunk, including the query, is the maximum pre-training length of the model. R<sup>s</sup> is obtained from the first 100 tokens of the chunk, R<sup>r</sup> is obtained from the last 100 tokens of the chunk, and the remaining part of the chunk obtains Rm. All experiments are performed on 8 64G AMD Instinct MI210.

<span id="page-17-0"></span>

|                              | Single  | -Docume | nt QA | Mult     | i-Docume | nt QA   | Su       | mmariza   | tion       | Few   | -shot Lea | rning  | Synt           | hetic | Co    | ode   |       |
|------------------------------|---------|---------|-------|----------|----------|---------|----------|-----------|------------|-------|-----------|--------|----------------|-------|-------|-------|-------|
| Methods                      | NITY QA | Qasper  | Mr-en | HotpotQ! | 2WikiMQ! | Musique | GovRepor | QMSum     | MultiNews  | TREC  | TriviaQA  | SAMSum | <b>P</b> Count | pRe   | Lec   | RR.P  | AVG   |
| Max Length                   | 84123   | 24204   | 17727 | 20325    | 19001    | 20520   | 60515    | 34477     | 16271      | 13049 | 26756     | 21884  | 32699          | 17158 | 37628 | 58822 | 30657 |
|                              |         |         |       |          |          | Lla     | ma2-7B-c | hat-hf(4l | <b>(</b> ) |       |           |        |                |       |       |       |       |
| No-eviction                  | 23.20   | 17.50   | 37.07 | 38.67    | 32.68    | 20.22   | 25.00    | 22.79     | 25.84      | 64.00 | 84.63     | 40.67  | 4.00           | 31.50 | 59.37 | 58.53 | 36.60 |
| Sink-eviction-layer-1-8      | 23.75   | 18.69   | 38.41 | 39.86    | 32.91    | 20.75   | 24.86    | 22.10     | 25.56      | 63.00 | 84.42     | 40.78  | 4.50           | 30.00 | 54.67 | 59.30 | 36.47 |
| Sink-eviction-layer-9-16     | 23.34   | 19.10   | 38.21 | 38.73    | 30.42    | 21.04   | 25.31    | 21.86     | 25.16      | 62.00 | 85.53     | 41.26  | 3.00           | 29.50 | 56.75 | 58.31 | 36.22 |
| Sink-eviction-layer-17-24    | 24.46   | 17.75   | 36.84 | 38.79    | 30.59    | 19.67   | 25.42    | 22.20     | 25.58      | 62.00 | 85.35     | 40.24  | 4.00           | 28.50 | 58.41 | 58.17 | 36.12 |
| Sink-eviction-layer-25-32    | 23.87   | 18.40   | 35.91 | 38.96    | 31.02    | 20.21   | 25.32    | 22.00     | 25.81      | 64.00 | 84.19     | 39.77  | 3.50           | 30.00 | 58.58 | 58.21 | 36.23 |
| Recency-eviction-layer-1-8   | 22.71   | 16.95   | 35.24 | 36.14    | 30.60    | 17.19   | 25.21    | 22.11     | 26.22      | 59.00 | 68.00     | 40.03  | 2.50           | 31.00 | 58.07 | 51.47 | 33.90 |
| Recency-eviction-layer-9-16  | 24.95   | 13.54   | 35.67 | 34.13    | 30.69    | 17.77   | 25.14    | 22.85     | 25.43      | 54.50 | 79.23     | 39.16  | 5.00           | 27.50 | 57.73 | 57.57 | 34.43 |
| Recency-eviction-layer-17-24 | 21.68   | 15.17   | 34.97 | 32.79    | 26.93    | 13.95   | 25.29    | 22.10     | 25.42      | 62.50 | 80.47     | 38.60  | 6.00           | 34.00 | 58.31 | 57.76 | 34.75 |
| Recency-eviction-layer-25-32 | 24.15   | 17.32   | 37.82 | 36.76    | 29.86    | 18.48   | 25.21    | 22.06     | 25.67      | 64.00 | 83.20     | 36.67  | 5.00           | 30.50 | 56.13 | 56.52 | 35.58 |
| Middle-eviction-layer-1-8    | 22.41   | 16.84   | 37.94 | 39.99    | 33.24    | 19.62   | 24.74    | 22.02     | 25.80      | 63.50 | 83.67     | 40.00  | 5.00           | 32.50 | 59.22 | 56.71 | 36.45 |
| Middle-eviction-layer-9-16   | 22.96   | 17.63   | 37.39 | 40.51    | 30.68    | 21.09   | 25.14    | 21.94     | 25.66      | 62.00 | 85.02     | 40.57  | 4.00           | 35.50 | 59.28 | 57.88 | 36.70 |
| Middle-eviction-layer-17-24  | 21.72   | 17.06   | 35.88 | 39.99    | 31.76    | 19.06   | 25.00    | 22.20     | 25.77      | 58.00 | 83.62     | 40.02  | 4.50           | 38.00 | 59.38 | 57.98 | 36.25 |
| Middle-eviction-layer-25-32  | 21.64   | 17.04   | 36.32 | 40.51    | 32.80    | 19.14   | 25.07    | 22.14     | 25.86      | 63.50 | 84.39     | 40.50  | 3.00           | 30.00 | 59.15 | 57.86 | 36.18 |
| All-eviction-layer-1-8       | 0.33    | 0.05    | 1.24  | 0.32     | 0.68     | 0.38    | 1.76     | 3.25      | 1.59       | 1.50  | 5.40      | 1.75   | 0.41           | 0.50  | 23.61 | 12.17 | 3.43  |
| All-eviction-layer-9-16      | 1.60    | 1.97    | 5.06  | 0.65     | 1.44     | 0.81    | 21.75    | 36.22     | 1.81       | 35.00 | 30.77     | 10.54  | 3.00           | 0.50  | 33.25 | 22.89 | 12.95 |
| All-eviction-layer-17-24     | 12.20   | 10.26   | 16.30 | 20.30    | 18.10    | 8.54    | 13.63    | 20.43     | 17.61      | 49.00 | 11.93     | 29.74  | 5.50           | 26.00 | 41.92 | 23.67 | 20.32 |
| All-eviction-layer-25-32     | 11.19   | 7.62    | 11.11 | 12.49    | 8.83     | 1.79    | 11.45    | 16.21     | 12.87      | 43.00 | 30.77     | 9.94   | 3.50           | 22.00 | 24.42 | 22.19 | 15.59 |
| Ours-calibration             | 24.95   | 19.07   | 38.16 | 39.53    | 32.62    | 22.64   | 25.42    | 22.82     | 26.01      | 63.00 | 85.41     | 40.36  | 5.00           | 32.50 | 59.04 | 58.84 | 37.21 |

Table 8: Bias Token Eviction Ablation. Sink-eviction-layer-1-8 typically means evicting sink bias tokens in layers 1 to 8, and other naming conventions follow the same pattern. Ours-calibration refers to the approach where layers 9-16 adopt the recency bias token eviction method, while layers 1-8 evict middle bias tokens, and layers 25-32 evict sink bias tokens.

