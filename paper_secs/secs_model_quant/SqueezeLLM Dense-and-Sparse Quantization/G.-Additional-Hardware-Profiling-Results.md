# G. Additional Hardware Profiling Results

In Tab. [G.11,](#page-19-0) we provide additional hardware profiling results using a sequence length of 1024. All the experimental setups and details are identical to Sec. [5.4](#page-7-1) and Tab. [3.](#page-8-1)

<span id="page-20-3"></span>Table G.12. Matrix-vector kernel runtime (in seconds) for generating 128 tokens, benchmarked on an A100 GPU. Our kernel implementation attains 1.5-2.5× performance speedups relative to the fp16 matrix-vector multiply kernel across different model sizes without any additional optimizations or tuning. We include GPTQ (with group size 128) without reordering for comparison against the latency of uniform quantization with grouping.

| Method             | Bit Width | 7B   | 30B  |      |
|--------------------|-----------|------|------|------|
| Baseline           | 16        | 1.21 | 2.32 | 5.56 |
| GPTQ (g128)        | 4         | 0.92 | 1.51 | 3.24 |
| SqueezeLLM         | 4         | 0.83 | 1.52 | 3.66 |
| SqueezeLLM (0.45%) | 4         | 1.09 | 1.87 | 4.25 |
| GPTQ (g128)        | 3         | 0.62 | 1.03 | 2.39 |
| SqueezeLLM         | 3         | 0.56 | 0.97 | 2.26 |
| SqueezeLLM (0.45%) | 3         | 0.83 | 1.32 | 2.86 |

<span id="page-20-0"></span>Table H.13. Perplexity comparison of LLaMA-30B and 65B models quantized into 4 and 3 bits using different methods including RTN, GPTQ, AWQ and SpQR on C4 and WikiText-2. We compare the performance of GPTQ, AWQ, and SqueezeLLM in groups based on similar model sizes. In the first group, we compare dense-only SqueezeLLM with non-grouped GPTQ. In the subsequent groups, we compare SqueezeLLM with different levels of sparsity to GPTQ and AWQ with different group sizes.

| LLaMA-30B          |                           | 3-bit |                 | 4-bit                     |    |                 |  |  |
|--------------------|---------------------------|-------|-----------------|---------------------------|----|-----------------|--|--|
| Method             | Avg. Bits<br>(comp. rate) | C4    | PPL (↓)<br>Wiki | Avg. Bits<br>(comp. rate) | C4 | PPL (↓)<br>Wiki |  |  |
| Baseline           | 16                        | 5.98  | 4.10            | 16                        |    | 5.98 4.10       |  |  |
| RTN                | 3 (5.33)                  |       | 28.53 14.89     | 4 (4.00)                  |    | 6.33 4.54       |  |  |
| GPTQ               | 3 (5.33)                  | 7.31  | 5.76            | 4 (4.00)                  |    | 6.20 4.43       |  |  |
| SpQR               | -                         | -     | -               | 3.89 (4.11)               |    | 6.08 4.25       |  |  |
| SqueezeLLM         | 3.02 (5.31)               | 6.37  | 4.66            | 4.03 (3.97)               |    | 6.06 4.22       |  |  |
| GPTQ (g128)        | 3.25 (4.92)               | 6.47  | 4.83            | 4.25 (3.77)               |    | 6.07 4.24       |  |  |
| AWQ (g128)         | 3.25 (4.92)               | 6.38  | 4.63            | 4.25 (3.77)               |    | 6.05 4.21       |  |  |
| SqueezeLLM (0.45%) | 3.25 (4.92)               | 6.23  | 4.44            | 4.25 (3.77)               |    | 6.04 4.18       |  |  |

| LLaMA-65B          |                           | 3-bit         |                                                    | 4-bit       |  |           |  |
|--------------------|---------------------------|---------------|----------------------------------------------------|-------------|--|-----------|--|
| Method             | Avg. Bits<br>(comp. rate) | PPL (↓)<br>C4 | Avg. Bits<br>PPL (↓)<br>(comp. rate)<br>C4<br>Wiki |             |  |           |  |
| Baseline           | 16                        | 5.62          | 3.53                                               | 16          |  | 5.62 3.53 |  |
| RTN                | 3 (5.33)                  |               | 12.77 10.59                                        | 4 (4.00)    |  | 5.86 3.92 |  |
| GPTQ               | 3 (5.33)                  | 6.70          | 5.58                                               | 4 (4.00)    |  | 5.81 4.11 |  |
| SpQR               | 3 (5.33)                  | -             | 4.2†                                               | 3.90 (4.10) |  | 5.70 3.68 |  |
| SqueezeLLM         | 3.02 (5.30)               | 5.99          | 4.05                                               | 4.04 (3.96) |  | 5.69 3.76 |  |
| GPTQ (g128)        | 3.25 (4.92)               | 6.01          | 4.55                                               | 4.25 (3.77) |  | 5.69 3.76 |  |
| AWQ (g128)         | 3.25 (4.92)               | 5.94          | 4.00                                               | 4.25 (3.77) |  | 5.68 3.67 |  |
| SqueezeLLM (0.45%) | 3.24 (4.94)               | 5.84          | 3.88                                               | 4.26 (3.76) |  | 5.67 3.63 |  |

Additionally, in Tab. [G.12,](#page-20-3) we demonstrate that our custom CUDA kernels (both including and without including outliers) attain significant speedups of 1.5-2.5× relative to the fp16 baseline. These results were obtained without any additional optimizations or tuning specifically for the A100, demonstrating how our kernels are easily portable across different GPUs and do not introduce complexity.

