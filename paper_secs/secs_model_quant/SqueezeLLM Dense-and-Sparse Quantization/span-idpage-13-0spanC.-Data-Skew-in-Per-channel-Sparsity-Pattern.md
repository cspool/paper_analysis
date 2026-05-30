# <span id="page-13-0"></span>C. Data Skew in Per-channel Sparsity Pattern

Fig. [C.1](#page-13-1) provides the distribution of nonzero entries per output channel across different linear layers in the first LLaMA-7B block. This plot shows that the nonzero distribution is heavily skewed, with a few channels containing a much larger proportion of nonzero values. This skewed distribution makes it challenging to efficiently perform computations using the sparse matrix, as it is difficult to distribute the nonzero elements evenly across parallel processing units. This motivates our modified kernel for handling channels with a large number of outliers in order to reduce the runtime overhead of the sparse matrices. As outlined in Tab. [C.1,](#page-13-2) we observed over 100% added runtime overhead when employing a standard CSR-based

<span id="page-14-1"></span>Table D.2. Ablation study comparing sensitivity-agnostic and sensitivity-based non-uniform quantization on the LLaMA-7B model with 3-bit quantization, measured by perplexity on the C4 benchmark. The baseline model in FP16 achieves a perplexity of 7.08.

| Method             | Sensitivity-Agnostic (↓) | Sensitivity-Based (↓) |  |  |
|--------------------|--------------------------|-----------------------|--|--|
| SqueezeLLM         | 18.08                    | 7.75                  |  |  |
| SqueezeLLM (0.05%) | 8.10                     | 7.67                  |  |  |
| SqueezeLLM (0.45%) | 7.61                     | 7.56                  |  |  |

<span id="page-14-2"></span>![](_page_14_Figure_3.jpeg)

Figure D.2. (Left) Model size (normalized by the size of the FP16 model) and perplexity trade-off with different percentages of sensitive values included in the sparse matrix. Here, no outlier values are included in the sparse matrix. (Right) Comparison of the performance when the sensitive values are not removed as the sparse matrix (only outlier values are removed) to the case where 0.05% of the sensitive values are removed. In both cases, the trade-offs are obtained by controlling the percentage of outlier values included in the sparse matrix.

kernel. However, if we allocate each thread to process a fixed number of nonzeros (rather than having each thread process an entire row) we were able to drastically reduce the runtime overhead to 10-20% with both sensitive values and outliers.

