# <span id="page-10-3"></span>D Ablation Studies on Retrieval Parameters

We ablate the parameters of Cross-model Retrieval using Vicuna-7B as the target model and Vicuna-68M/EAGLE as draft models on 8K-token Gov-Report inputs (Table [8\)](#page-11-0). The optimal working KV cache size is around 1K for Vicuna-68M and 2K for EAGLE, which we adopt for the ablation. Under these settings, the best results are obtained with a chunk size of 32, top-k values of 32 and 64, and retrieval frequencies of 4 and 8 steps for Vicuna-68M/EAGLE, respectively.

<span id="page-10-4"></span><sup>1</sup>EAGLE models are publicly available under the Apache 2.0 license.

<span id="page-11-0"></span>

| Working<br>Cache Size | Vicuna-68M | EAGLE | Chunk<br>Size | Vicuna-68M | EAGLE | Top-k | Vicuna-68M | EAGLE | Retrieval<br>Frequency | Vicuna-68M | EAGLE |
|-----------------------|------------|-------|---------------|------------|-------|-------|------------|-------|------------------------|------------|-------|
| 64                    | 32.52      | 39.10 | 1             | 31.05      | 48.05 | 2     | 30.72      | 38.22 | 1                      | 33.05      | 47.78 |
| 128                   | 32.91      | 39.95 | 2             | 32.27      | 49.49 | 4     | 32.65      | 40.36 | 2                      | 33.54      | 46.78 |
| 256                   | 33.65      | 41.53 | 4             | 32.97      | 49.55 | 8     | 32.76      | 41.49 | 4                      | 33.59      | 48.17 |
| 512                   | 33.53      | 42.77 | 8             | 33.39      | 49.18 | 16    | 33.19      | 43.90 | 8                      | 33.11      | 48.52 |
| 1024                  | 33.69      | 44.19 | 16            | 33.41      | 48.92 | 32    | 33.28      | 47.21 | 16                     | 33.16      | 48.36 |
| 2048                  | 32.36      | 45.33 | 32            | 33.52      | 49.68 | 64    | 32.50      | 48.09 | 32                     | 33.28      | 48.11 |
| 4096                  | 25.84      | 43.68 | 64            | 33.23      | 48.25 | 128   | 25.20      | 45.14 | 64                     | 33.29      | 48.13 |
| 8192                  | 24.32      | 33.10 | 128           | 33.20      | 47.48 | 256   | 23.95      | 32.48 | 128                    | 33.21      | 48.20 |

Table 8: Ablation study of Cross-model Retrieval parameters. The table reports decoding speed (tokens/s) using Vicuna-7B as the target model on 8K-token GovReport inputs.