# <span id="page-14-0"></span>E. Efficient Deployment of Elastic Attention

A critical system bottleneck in hybrid attention pertains to the efficient scheduling of heterogeneous attention workloads within a single batch. Algorithm [1\(](#page-14-6)a) illustrates the conventional paradigm, termed *Serial Dispatch*, as adopted by libraries like Flash-Attn[11](#page-14-7) [\(Dao,](#page-8-20) [2024\)](#page-8-20). This method entails explicit data rearrangement (highlighted in red), requiring the materialization of input tensors according to routing decisions r.

```
6https://modelscope.cn/models/OpenBMB/MiniCPM4-8B/file/view/master/modeling_minicpm.py
7https://github.com/MoonshotAI/MoBA
8https://github.com/XunhaoLai/native-sparse-attention-triton
9https://github.com/princeton-pli/PruLong
10https://github.com/mit-han-lab/duo-attention
11https://github.com/Dao-AILab/flash-attention
```

<span id="page-15-2"></span>*Table 7.* LongBench-E results comparison. The 1st and the 2nd performance in each comparison group are highlighted with **bold font** and <u>underlined</u>, respectively.

| Method                       | Single-D | ocument QA   | Multi-Do     | cument QA | Summa         | rization     | F     | ew-shot Lea | rning  | Syntl  | netic | Code  |       | Avg.  |
|------------------------------|----------|--------------|--------------|-----------|---------------|--------------|-------|-------------|--------|--------|-------|-------|-------|-------|
| Method                       | MF-en    | Qasper       | HotpotQA     | 2WikiMQA  | GovReport     | MultiNews    | TREC  | TriviaQA    | SAMSum | PCount | PRe   | Lcc   | RB-P  | Avg.  |
|                              |          |              |              | Qw        | en3-4B backb  | one model    |       |             |        |        |       |       |       |       |
| Qwen3-4B                     | 52.16    | 35.21        | 44.81        | 32.15     | 33.47         | 23.45        | 70.67 | 88.22       | 39.74  | 2.33   | 96.84 | 57.93 | 50.84 | 48.45 |
| + InfLLM-V2                  | 49.13    | 37.51        | 40.26        | 29.68     | 33.99         | 25.92        | 67.67 | 86.14       | 38.76  | 4.30   | 72.78 | 62.56 | 56.59 | 46.68 |
| + DuoAttention               | 48.85    | 34.60        | 42.25        | 29.18     | 33.26         | 23.68        | 66.33 | 87.70       | 39.73  | 2.11   | 92.24 | 57.57 | 50.25 | 46.95 |
| + PruLong                    | 48.88    | 34.28        | 45.14        | 30.01     | 33.36         | 23.70        | 66.33 | 88.31       | 39.77  | 4.79   | 89.67 | 55.44 | 50.95 | 47.19 |
| + MoBA                       | 42.41    | 35.80        | 38.76        | 29.84     | 33.82         | 25.11        | 68.67 | 85.69       | 39.34  | 2.33   | 72.52 | 58.61 | 50.63 | 45.09 |
| + NSA                        | 42.46    | 33.90        | 38.10        | 31.53     | 31.69         | 23.07        | 68.00 | 82.49       | 39.82  | 5.67   | 41.93 | 62.44 | 53.77 | 43.02 |
| + XAttention                 | 50.36    | 32.80        | 44.54        | 33.17     | 33.79         | 23.73        | 69.00 | 87.44       | 39.90  | 3.42   | 74.59 | 59.62 | 49.42 | 46.44 |
| + Elastic Attention (FA-SSA) | 49.13    | 35.27        | 47.87        | 29.85     | 33.35         | 23.65        | 69.33 | 87.23       | 40.62  | 2.00   | 94.86 | 57.80 | 50.87 | 48.08 |
| + Elastic Attention (FA-XA)  | 52.06    | <u>36.74</u> | 45.65        | 33.18     | 33.45         | 23.55        | 68.67 | 87.33       | 39.79  | 4.39   | 84.32 | 50.47 | 58.11 | 47.59 |
| + Elastic Attention (XA-SSA) | 49.21    | 34.63        | 47.32        | 30.02     | 33.14         | 23.75        | 68.00 | 87.64       | 40.12  | 5.50   | 93.00 | 59.15 | 50.98 | 48.14 |
|                              |          |              |              | Qw        | en3-8B backb  | one model    |       |             |        |        |       |       |       |       |
| Qwen3-8B                     | 49.92    | 41.22        | 58.98        | 44.21     | 33.27         | 23.42        | 71.33 | 86.77       | 41.83  | 2.00   | 98.33 | 66.31 | 56.08 | 52.16 |
| + InfLLM-V2                  | 46.35    | 38.05        | 49.51        | 35.14     | 34.49         | 24.67        | 67.00 | 87.03       | 39.61  | 12.07  | 79.67 | 66.86 | 52.30 | 49.03 |
| + DuoAttention               | 49.59    | 41.32        | 51.82        | 37.22     | 33.14         | 23.19        | 70.33 | 86.83       | 42.11  | 0.00   | 95.00 | 67.43 | 57.32 | 50.63 |
| + PruLong                    | 51.06    | 41.04        | 53.87        | 39.90     | 33.20         | 23.39        | 70.33 | 87.68       | 41.83  | 0.33   | 97.00 | 67.30 | 57.18 | 51.34 |
| + MoBA                       | 48.81    | 40.65        | 48.26        | 38.29     | 35.19         | 25.47        | 67.67 | 84.13       | 40.85  | 14.00  | 83.89 | 68.21 | 56.89 | 50.47 |
| + NSA                        | 44.29    | 36.96        | 42.26        | 32.63     | 32.93         | 23.00        | 74.33 | 86.74       | 40.36  | 2.67   | 55.33 | 69.81 | 54.57 | 46.12 |
| + XAttention                 | 48.93    | 38.03        | <u>57.32</u> | 40.65     | 33.41         | 23.37        | 69.33 | 87.69       | 41.77  | 2.00   | 83.72 | 65.71 | 55.53 | 50.13 |
| + Elastic Attention (FA-SSA) | 51.45    | 40.84        | 53.34        | 39.74     | 33.14         | 23.24        | 72.33 | 88.67       | 41.56  | 0.00   | 96.14 | 68.44 | 57.47 | 51.51 |
| + Elastic Attention (FA-XA)  | 48.18    | 39.83        | 59.38        | 40.60     | 33.30         | 23.29        | 69.00 | 87.78       | 41.92  | 2.33   | 99.56 | 65.40 | 55.74 | 51.66 |
| + Elastic Attention (XA-SSA) | 45.73    | 33.13        | 51.80        | 30.95     | 33.21         | 23.32        | 70.00 | 87.83       | 40.73  | 0.67   | 92.78 | 68.56 | 55.13 | 49.25 |
|                              |          |              |              | Llama-3.1 | 1-8B-Instruct | backbone mo  | odel  |             |        |        |       |       |       |       |
| Llama-3.1-8B-Instruct        | 53.44    | 44.06        | 59.62        | 44.08     | 34.50         | 26.02        | 71.00 | 90.54       | 42.94  | 12.67  | 99.33 | 63.85 | 47.78 | 53.28 |
| + InfLLM-V2                  | 48.60    | 38.93        | 50.74        | 41.85     | 34.40         | 25.76        | 69.00 | 89.92       | 43.04  | 6.33   | 77.73 | 59.78 | 68.81 | 50.73 |
| + NSA                        | 42.86    | 41.80        | 40.79        | 39.94     | 34.57         | 25.29        | 68.00 | 89.69       | 42.27  | 2.33   | 28.00 | 65.18 | 50.12 | 44.03 |
| + DuoAttention               | 52.68    | 44.62        | 52.01        | 38.41     | 34.01         | 25.84        | 69.67 | 90.33       | 42.06  | 10.13  | 99.00 | 64.69 | 48.24 | 51.82 |
| + PruLong                    | 50.74    | 44.63        | 49.70        | 36.41     | 34.25         | 25.78        | 70.00 | 91.45       | 42.13  | 9.80   | 97.33 | 68.55 | 53.59 | 52.11 |
| + MoBA                       | 48.92    | 44.33        | 46.36        | 41.45     | 34.79         | 26.65        | 69.67 | 89.91       | 40.76  | 7.33   | 64.67 | 69.47 | 59.48 | 49.69 |
| + XAttention                 | 53.80    | 43.84        | <u>59.98</u> | 43.47     | 34.47         | <u>26.05</u> | 72.33 | 90.39       | 42.99  | 8.33   | 73.33 | 62.92 | 50.13 | 51.00 |
| + Elastic Attention (FA-SSA) | 53.48    | 46.35        | 55.50        | 42.33     | 34.50         | 25.77        | 69.67 | 91.70       | 42.60  | 9.33   | 98.67 | 67.72 | 53.69 | 53.35 |
| + Elastic Attention (FA-XA)  | 53.65    | 45.14        | 60.21        | 45.67     | 34.56         | 26.03        | 71.67 | 91.37       | 42.62  | 8.31   | 91.00 | 63.22 | 49.75 | 52.7  |
| + Elastic Attention (XA-SSA) | 51.81    | 44.79        | 53.53        | 39.27     | 34.56         | 25.98        | 70.67 | 90.59       | 42.24  | 9.07   | 73.67 | 68.12 | 53.49 | 50.71 |

Consequently, it introduces two major sources of inefficiency that contradict the high-throughput requirements of long-context inference: (1) **Memory Overhead**, incurred by allocating and copying non-contiguous tensor fragments (e.g., separating retrieval heads from sparse heads); and (2) **Kernel Launch & Scheduling Overhead**. As noted in the main text, parallelism along the sequence dimension dominates execution in long-context scenarios. Launching separate kernels for different head groups fragments this workload, incurring latency from multiple kernel invocations and disrupting the GPU's ability to globally schedule thread blocks across available streaming multiprocessors.

To overcome these limitations, we employ the Block Sparse Attention (BSA) Kernel<sup>12</sup> (Guo et al., 2024) (Algorithm 1(b)). Obviating the need for tensor splitting, we pass routing decisions r directly to the kernel as lightweight metadata  $\mathbf{m}$ . As depicted in the green block, the kernel leverages *thread-block level branching*: each thread block dynamically retrieves its assigned head's type from  $\mathbf{m}$  and executes the corresponding attention logic. This design enables a **unified kernel launch** for all heads. By keeping the grid dimensions intact (Batch  $\times$  Heads  $\times$  Sequence Blocks), we effectively eliminate redundant memory copies and avoid workload fragmentation, allowing the GPU hardware scheduler to optimally distribute sequence blocks. One can refer to the anonymous code provided in Appendix ?? for more details.

