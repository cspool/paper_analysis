# <span id="page-17-0"></span>A.5 IMPLEMENTATION OF FASTGEN ON LONG-CONTEXT BENCHMARKS

Due to the lack of official implementation of the FastGen [\(Ge et al.](#page-13-2) [\(2024\)](#page-13-2)) algorithm, we reproduce it using a community codebase [\(Adams et al.](#page-11-12) [\(2024\)](#page-11-12)), which is referenced by FastGen's official repository. In the FastGen algorithm, the pruning ratio cannot be directly configurable; instead, the recovery ratio T is used to control sparsity as outlined in the FastGen paper. To quantify sparsity, we calculated the average KV cache usage across all test cases as the overall measure of sparsity. For the Llama-2-7B model, we set the recovery ratio to 0.7, ensuring the average KV cache budget was over 25% of the full KV cache. Similarly, for the Llama-3-8B model, we set the recovery ratio to 0.87, ensuring the average KV cache budget was more than 50% of the full KV cache. Additionally, since FastGen uses the full attention map of the user-provided prompt to profile the types of different heads, it results in an O(n 2 ) attention map complexity. Therefore, we are unable to test its performance in long contexts. For the long context benchmark, we used 8 A100-80G GPUs, achieving sequence lengths of up to 24k tokens for the Llama-2-7B model and up to 32k tokens for the Llama-3-8B model. In addition to the needle-in-the-haystack benchmark shown in Figure [6,](#page-5-0) we also evaluated

<span id="page-18-1"></span>Table 4: Full LongBench results with Llama-2-7B-Instruct-32K. DuoAttention achieves the best performance with a 25% KV cache budget on most datasets.

| Dataset             | Full  | H2O (25%) | SLLM (25%) | TOVA (25%) | Duo (25%) |
|---------------------|-------|-----------|------------|------------|-----------|
| Average             | 37.52 | 26.84     | 27.80      | 29.78      | 34.49     |
| 2WikiMQA            | 35.59 | 28.87     | 29.69      | 31.18      | 33.37     |
| DuReader (zh)       | 25.10 | 15.56     | 13.96      | 15.51      | 23.99     |
| GovReport           | 31.23 | 20.66     | 24.14      | 22.88      | 27.98     |
| HotpotQA            | 47.98 | 39.60     | 40.39      | 47.45      | 50.44     |
| LCC                 | 51.21 | 45.78     | 44.25      | 47.91      | 48.34     |
| LSHT (zh)           | 34.50 | 16.50     | 17.50      | 18.50      | 25.50     |
| MultiNews           | 27.11 | 19.21     | 20.54      | 21.41      | 25.03     |
| MultiFieldQA-en     | 33.95 | 21.01     | 16.69      | 18.19      | 25.49     |
| MultiFieldQA-zh     | 45.79 | 19.81     | 22.50      | 24.96      | 39.23     |
| Musique             | 22.97 | 20.63     | 20.09      | 21.00      | 19.27     |
| NarrativeQA         | 24.11 | 19.14     | 21.13      | 23.06      | 20.49     |
| Passage Count       | 0.00  | 0.53      | 0.58       | 0.00       | 0.33      |
| PassageRetrieval-en | 50.92 | 19.50     | 19.08      | 30.17      | 47.25     |
| PassageRetrieval-zh | 37.68 | 11.75     | 16.77      | 32.38      | 40.93     |
| Qasper              | 33.23 | 16.84     | 17.68      | 20.85      | 26.59     |
| QMSum               | 20.79 | 18.89     | 20.05      | 20.16      | 21.48     |
| RepoBench-P         | 51.58 | 45.16     | 45.25      | 49.03      | 48.58     |
| SAMSum              | 42.10 | 39.73     | 37.43      | 36.17      | 33.10     |
| TREC                | 71.50 | 48.50     | 56.50      | 47.00      | 68.50     |
| TriviaQA            | 86.21 | 85.16     | 85.24      | 85.65      | 86.15     |
| VCSUM (zh)          | 14.45 | 10.71     | 14.36      | 11.85      | 12.35     |

<span id="page-18-0"></span>Table 5: Comparison of FastGen and DuoAttention on a subset of LongBench using the Llama-3-8B-Instruct-1048K model.

|                     | FastGen (>50%) | DuoAttention (50%) |
|---------------------|----------------|--------------------|
| Average             | 32.82          | 40.01              |
| 2WikiMQA            | 18.61          | 29.08              |
| DuReader (zh)       | 20.22          | 29.31              |
| HotpotQA            | 33.08          | 41.63              |
| LCC                 | 46.50          | 44.16              |
| MultiNews           | 18.18          | 27.72              |
| MultiFieldQA-en     | 44.05          | 51.44              |
| MultiFieldQA-zh     | 42.15          | 52.40              |
| Musique             | 13.58          | 24.65              |
| Passage Count       | 0.09           | 0.00               |
| PassageRetrieval-en | 93.12          | 87.00              |
| PassageRetrieval-zh | 40.75          | 62.15              |
| Qasper              | 26.51          | 26.93              |
| QMSum               | 24.03          | 24.20              |
| SAMSum              | 34.12          | 41.83              |
| TriviaQA            | 69.92          | 87.14              |
| VCSUM (zh)          | 0.23           | 10.46              |

FastGen on LongBench for both models. However, due to the quadratic memory consumption of FastGen, we only report results for datasets that were feasible to run on 8x A100-80G GPUs using FastGen. As shown in Table [5](#page-18-0) and Table [6,](#page-19-0) DuoAttention can consistently outperform FastGen on LongBench datasets.

<span id="page-19-0"></span>Table 6: Comparison of FastGen and DuoAttention on a subset of LongBench using the Llama-2-7B-32K-Instruct model.

|                     | FastGen (>25%) | DuoAttention (25%) |
|---------------------|----------------|--------------------|
| Average             | 19.01          | 32.81              |
| 2WikiMQA            | 28.05          | 33.37              |
| MultiNews           | 12.60          | 25.03              |
| MultiFieldQA-en     | 28.58          | 25.49              |
| MultiFieldQA-zh     | 22.44          | 39.23              |
| PassageRetrieval-zh | 3.38           | 40.93              |