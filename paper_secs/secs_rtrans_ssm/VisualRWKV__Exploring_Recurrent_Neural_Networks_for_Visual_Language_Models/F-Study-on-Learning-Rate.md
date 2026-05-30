# F Study on Learning Rate

In this section, We will explore the effect of learning rates on VisualRWKV. Setting different initial learning rates and using a cosine learning rate scheduler, the performance of the model on multiple <span id="page-16-0"></span>benchmarks is shown in the Table [10.](#page-16-0)

| Method     | Size | Learning Rate    | VQAv2 | ScienceQA | TextVQA | GQA   | MME     |
|------------|------|------------------|-------|-----------|---------|-------|---------|
| VisualRWKV | 1.6B | 2e-5 to 2e-5     | 66.85 | 57.51     | 41.85   | 52.07 | 1080.77 |
| VisualRWKV | 1.6B | 3e-5 to 1e-5     | 67.25 | 53.40     | 41.84   | 52.49 | 1115.70 |
| VisualRWKV | 1.6B | 3e-5 to 1.5e-5   | 67.54 | 56.62     | 42.18   | 52.82 | 1111.66 |
| VisualRWKV | 1.6B | 4e-5 to 1.5e-5   | 68.51 | 55.68     | 43.73   | 54.31 | 1151.20 |
| VisualRWKV | 1.6B | 5e-5 to 1.5e-5   | 69.26 | 57.61     | 43.17   | 54.85 | 1208.96 |
| VisualRWKV | 1.6B | 6e-5 to 1.5e-5   | 69.42 | 59.05     | 43.57   | 55.23 | 1204.90 |
| VisualRWKV | 1.6B | 1e-4 to 1.5e-5   | 70.02 | 55.58     | 42.24   | 55.72 | 1212.52 |
| VisualRWKV | 1.6B | 1.5e-4 to 1.5e-5 | 68.89 | 55.63     | 41.90   | 54.09 | 1249.51 |
| VisualRWKV | 3B   | 4e-5 to 1e-5     | 68.65 | 65.99     | 48.46   | 54.40 | 1323.18 |
| VisualRWKV | 3B   | 5e-5 to 1.25e-5  | 71.52 | 65.34     | 48.68   | 59.56 | 1369.19 |
| VisualRWKV | 7B   | 2e-5 to 2e-5     | 68.31 | 68.91     | 50.09   | 52.80 | 1340.44 |
| VisualRWKV | 7B   | 4e-5 to 1e-5     | 75.82 | 68.22     | 51.01   | 64.27 | 1387.75 |

Table 10: Impact of Learning Rate on the Performance of the VisualRWKV on 5 benchmarks.

