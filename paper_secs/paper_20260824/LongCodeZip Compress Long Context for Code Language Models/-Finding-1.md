# **?** Finding 1

LongCodeZip is effective across various downstream tasks, with up to 5.6x compression ratio without sacrificing downstream performance.

#### B. RQ2: Ablation Study

To understand the contribution of each component in Long-CodeZip, we conduct an ablation study on the Long Code Completion task using Qwen2.5-Coder-7B. For all ablations, the total token budget and other hyper-parameters are set the same as the full method. We systematically remove or modify key components to analyze their individual impact. For coarsegrained ablations, we replace our conditional perplexity-based

TABLE VIII: Cross-model Results

<span id="page-7-2"></span>

| Compression Model   | DS-6.7B | Seed-8B | Qwen-7B | Avg. ES |
|---------------------|---------|---------|---------|---------|
| No Compression      | 57.14   | 64.04   | 56.36   | 59.18   |
| No Context          | 41.29   | 41.88   | 38.14   | 40.44   |
| DEEPSEEK-CODER-6.7B | 60.58   | 61.48   | 56.55   | 59.54   |
| SEED-CODER-8B       | 60.86   | 63.11   | 55.95   | 59.97   |
| QWEN2.5-CODER-0.5B  | 61.12   | 62.68   | 56.58   | 60.13   |
| QWEN2.5-CODER-1.5B  | 60.89   | 62.79   | 56.18   | 59.95   |
| QWEN2.5-CODER-3B    | 60.74   | 63.10   | 56.79   | 60.21   |
| QWEN2.5-CODER-7B    | 61.34   | 62.62   | 57.55   | 60.58   |

ranking with similarity-based ranking, and compare against random function ranking to establish a lower bound. For fine-grained ablations, we test four variants: removing fine-grained compression entirely (coarse-grained selection only), removing adaptive budget allocation (uniform budget allocation), replacing meta-chunking with simple line-based chunking, and using random line selection within selected functions.

Different components contribute varying degrees to performance. The coarse-grained ranking mechanism is most critical - conditional perplexity-based ranking outperforms similarity-based approaches by 7.89% and random selection by 17.79% in ES score. This demonstrates that semantic relevance through conditional perplexity is superior to lexical similarity. For fine-grained components, adaptive budget allocation improves ES by 2.34%, enabling important functions to retain more detail. Perplexity-based chunking outperforms simple line chunking by 1.57% in ES while being more computationally efficient, as line-by-line compression ranking would incur higher overhead compared to block-based analysis. Knapsack-based selection outperforms random line selection by 2.48% in ES, confirming relevance-guided selection helps compression quality.

#### Finding 2

Coarse-grained conditional perplexity ranking has the most impact on the performance of LongCodeZip, while fine-grained optimizations further improve the compression information density.

TABLE IX: Efficiency Analysis of Different Methods

<span id="page-8-0"></span>

| Method                  | Comp. Time (s) | Comp. GPU Mem (GB) | Gen. Time (s) | Gen. GPU Mem (GB) | Ratio | ES    | EM    |
|-------------------------|----------------|--------------------|---------------|-------------------|-------|-------|-------|
| No Compression          | 0.0            | 0.0                | 15.70         | Base + 3.48       | 1.0x  | 56.36 | 31.80 |
| No Context              | 0.0            | 0.0                | 0.68          | Base + 0.06       | -     | 38.14 | 9.60  |
| RAG (Function Chunking) | 0.53           | 1.07               | 7.57          | Base + 1.13       | 3.1x  | 52.79 | 26.00 |
| LLMLingua-2             | 0.65           | 4.71               | 6.53          | Base + 0.79       | 4.4x  | 41.29 | 12.20 |
| DietCode                | 15.23          | 0.0                | 7.26          | Base + 1.03       | 3.4x  | 43.91 | 13.20 |
| SlimCode                | 0.35           | 0.0                | 6.48          | Base + 0.78       | 4.5x  | 40.85 | 12.20 |
| LongCodeZip             | 2.58           | Base + 0.69        | 6.59          | Base + 0.81       | 4.3x  | 57.55 | 32.40 |

Note: Comp.: Compression, Gen.: Generation, Mem: Memory. Base model parameters memory: 28.37 GB.

