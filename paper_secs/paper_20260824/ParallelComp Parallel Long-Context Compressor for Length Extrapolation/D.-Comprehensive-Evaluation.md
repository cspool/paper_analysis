# **D.** Comprehensive Evaluation

This section presents additional experiments and implementation details to further contextualize and substantiate the claims in the main text. Specifically, we investigate comparisons with Retrieval-Augmented Generation baselines, provide extended benchmarks with APE (Yang et al., 2025b) and StarAttention (Acharya et al., 2024), report on hyperparameter sensitivity, discuss latency and memory efficiency, and elaborate on our design choices for attention bias sparsification.

## <span id="page-17-1"></span>D.1. Comparison with Retrieval-Augmented Generation Method

| Model           | QM    | QASP  | MSQ   | HQA   | MFQA  | AVG   |
|-----------------|-------|-------|-------|-------|-------|-------|
| ChatQA-2        | 11.64 | 28.85 | 27.81 | 53.81 | 51.02 | 34.63 |
| ChatQA-2 w/ RAG | 13.20 | 28.85 |       |       | 51.15 | 36.16 |
| Ours            | 24.18 | 39.05 | 33.25 | 49.58 | 42.66 | 37.74 |

Table 9: Performance comparison with RAG and non-RAG baselines on LongBench.

| Model           | KV Retrieval | Numbe String | Passkey | En.MC | AVG   |
|-----------------|--------------|--------------|---------|-------|-------|
| ChatQA-2        | 72.00        | 100.00       | 100.00  | 64.19 | 84.05 |
| ChatQA-2 w/ RAG | N/A          | N/A          | N/A     | N/A   | N/A   |
| Ours            | 92.80        | 99.83        | 100.00  | 54.59 | 86.81 |

<span id="page-17-2"></span>Table 10: Performance comparison on InfiniteBench. RAG methods completely fail on InfiniteBench, so we do not provide further results.

To clarify the role of length extrapolation versus retrieval-augmented generation (RAG), we compare the proposed method to leading RAG-enhanced models such as ChatQA-2 (Xu et al., 2024) on representative benchmarks. Table 9 and Table 10 summarize results on LongBench and InfiniteBench, respectively. The proposed method demonstrates strong robustness and competitive or superior performance in challenging long-context retrieval scenarios such as InfiniteBench, where RAG-based methods may encounter instability or diminished effectiveness.

### D.2. Comparison with APE and StarAttention

To comprehensively evaluate our approach in the context of existing chunked long-context processing methods, we conduct extensive experiments comparing PARALLELCOMP with both APE (Yang et al., 2025b) and StarAttention (Acharya et al., 2024). APE leverages a shared prefix to minimize distributional disparities, incorporates a low-temperature mechanism to

#### ParallelComp: Parallel Long-Context Compressor for Length Extrapolation

<span id="page-18-0"></span>

| Method       | NARR  | QAS   | MUL   | HOPT  | 2WKI  | MUS   | GOV   | QMS   | NEWS  | TREC  | TRIV  | SSM   | PCNNT | PREN  | LCC   | REP   AVG          |
|--------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------------------|
| APE          | 23.63 | 39.11 | 50.06 | 49.47 | 43.70 | 25.99 | 27.78 | 22.79 | 11.22 | 43.50 | 90.17 | 9.79  | 0.50  | 59.00 | 23.93 | 24.28   34.06      |
| StarAttn     | 3.74  | 11.90 | 24.81 | 14.17 | 14.37 | 8.19  | 34.90 | 22.54 | 27.11 | 65.33 | 87.84 | 43.71 | 3.80  | 65.17 | 50.54 | <b>45.40</b> 32.72 |
| ParallelComp | 29.45 | 45.98 | 50.67 | 48.36 | 46.56 | 23.32 | 32.60 | 24.29 | 27.34 | 38.50 | 86.72 | 25.93 | 0.05  | 95.00 | 14.15 | 21.42   38.15      |

Table 11: Comparison with APE and StarAttention denoted as **StarAttn** on LongBench. Temperature and scaling factors for APE are indicated as APE T+S. In our experiments, we set APE's temperature to 0.5 and scaling factor to 0.8; for StarAttention, the chunk size is set to 2K. In ParallelComp, we reuse 6K position encodings to facilitate length extrapolation. All evaluations are conducted on the Llama-3.1-8B-Instruct base model with a KV cache size of 24K.

sharpen attention, and utilizes a scaling factor to compensate for temperature changes. Its objective is to better align the attention patterns between parallel and sequential encoding. StarAttention is designed for chunk-based training of models with long contexts. At each generation step, it recalculates attention for every chunk, whereas PARALLELCOMP computes attention once during the prefill phase and then efficiently reuses the compressed KV cache for subsequent generation. This distinction leads to substantial improvements in computational efficiency.

We first compared the performance of different methods under the same KV cache budget. Hyperparameters are selected according to those reported in the original publications or official releases. Table 11 summarizes the representative results, where all models are assessed using a standardized evaluation infrastructure. Our analysis emphasizes the memory bottleneck encountered during length extrapolation. This experiment emphasizes the memory bottleneck encountered during length extrapolation. The other two methods are forced to truncate the input during extrapolation, resulting in significantly lower performance on certain tasks compared to our approach.

<span id="page-18-1"></span>

| Method       | NARR  | QAS   | MULT  | HOPT  | 2WKI  | MUS   | GOV   | QMS   | NEWS  | TREC  | TRIV  | SSM   | PCNNT | PREN  | LCC   | REP   | AVG   |
|--------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| APE0.8+0.8   | 25.92 | 41.99 | 53.79 | 53.64 | 50.54 | 26.46 | 30.15 | 25.42 | 20.68 | 50.50 | 88.70 | 9.72  | 6.50  | 89.00 | 16.71 | 25.78 | 38.47 |
| APE0.5+0.8   | 23.63 | 39.11 | 50.06 | 49.47 | 43.70 | 25.99 | 27.78 | 22.79 | 11.22 | 43.50 | 90.17 | 9.79  | 0.50  | 59.00 | 23.93 | 24.28 | 34.06 |
| APE0.2+0.8   | 18.83 | 26.53 | 41.70 | 44.63 | 35.91 | 17.71 | 24.31 | 20.14 | 7.96  | 35.75 | 88.54 | 9.72  | 1.50  | 34.50 | 23.86 | 23.22 | 28.43 |
| APE0.8+0.4   | 9.04  | 11.48 | 19.59 | 31.41 | 24.68 | 10.16 | 5.32  | 13.80 | 8.20  | 0.50  | 87.06 | 9.71  | 0.00  | 7.00  | 13.72 | 16.51 | 16.76 |
| APE0.5+0.4   | 7.59  | 9.74  | 16.13 | 31.89 | 25.72 | 9.62  | 5.20  | 9.56  | 8.19  | 0.00  | 87.60 | 9.69  | 0.00  | 5.00  | 14.26 | 15.58 | 15.99 |
| APE0.2+0.4   | 4.90  | 8.97  | 13.99 | 29.71 | 26.66 | 8.74  | 5.16  | 9.45  | 8.26  | 0.50  | 87.57 | 9.71  | 0.00  | 4.00  | 14.20 | 16.24 | 15.50 |
| StarAttn4K   | 3.74  | 11.90 | 24.81 | 14.17 | 14.37 | 8.19  | 34.90 | 22.54 | 27.11 | 65.33 | 87.84 | 43.71 | 3.80  | 65.17 | 50.54 | 45.40 | 32.72 |
| StarAttn6K   | 4.65  | 13.63 | 21.05 | 14.47 | 15.57 | 6.38  | 34.80 | 22.67 | 26.27 | 66.00 | 65.54 | 47.91 | 8.00  | 70.00 | 56.48 | 45.42 | 32.43 |
| ParallelComp | 29.45 | 45.98 | 50.67 | 48.36 | 46.56 | 23.32 | 32.60 | 24.29 | 27.34 | 38.50 | 86.72 | 25.93 | 0.05  | 95.00 | 14.15 | 21.42 | 38.15 |

Table 12: Performance comparison across different methods. APE X+Y indicates temperature = X and scaling factor = Y. StarAttn4K and StarAttn6K represent StarAttention using chunk sizes of 4K and 6K, respectively.

In order to further compare with the full-size context models StarAttention and APE, we conducted ablation experiments as shown in Table 12. We find that even after carefully tuning the hyperparameters of both models, their average performance only surpasses that of our PARALLELCOMP by 0.32%, which further demonstrates the effectiveness of our method.