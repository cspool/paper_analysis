# <span id="page-18-0"></span>**D More Details about Query Identification**

In this section, we evaluate the capability of contemporary LLMs to distinguish between global and localized queries. We assess the alignment between LLM predictions and human annotations by computing classification accuracy across three benchmarks: MLVU [\[54\]](#page-14-0), LongVideoBench [\[55\]](#page-14-1), and VideoMME [\[56\]](#page-14-2). The ground truth labels for these query types are derived from human annotations, as detailed in Section [B.](#page-17-1)

**LLMs exhibit strong alignment with human annotation.** As presented in Table [4,](#page-18-2) nearly all evaluated LLMs achieve an overall classification accuracy exceeding 80%. This indicates that off-the-shelf LLMs possess sufficiently robust reasoning capabilities to effectively differentiate between localized and global queries without extensive fine-tuning when given a proper prompt.

**Localized queries are more readily identifiable.** Table [4](#page-18-2) further reveals that accuracy on localized queries consistently surpasses that of global queries. While GQ accuracy is comparatively lower, this has a negligible impact on final model performance; it primarily incurs a minor computational overhead. This is because, as established previously, performance differences between query-aware frame selection and uniform sampling are minimal for global queries. In addition, the critical metric is LQ accuracy that may influence the final performance. On this metric, almost all LLMs achieve an accuracy greater than 90%, ensuring the final performance is good. And to make a tradeoff between compute cost and final model performance, we choose to use Qwen3-Next-80B-A3B-Instruct [\[73\]](#page-15-4) in our main experiments.

<span id="page-18-2"></span>**Table 4:** *Accuracy (%) of different LLMs in identifying localized queries (LQ) and global queries (GQ) across multiple benchmarks.*

| LLM                               |       | MLVU [54] |         |       |     | LongVideoBench [55] | VideoMME [56] |       |         |  |
|-----------------------------------|-------|-----------|---------|-------|-----|---------------------|---------------|-------|---------|--|
|                                   | LQ    | GQ        | Overall | LQ    | GQ  | Overall             | LQ            | GQ    | Overall |  |
| Qwen3-Next-80B-A3B-Instruct [73]  | 87.02 | 38.26     | 78.52   | 97.53 | N/A | 97.53               | 89.13         | 65.76 | 83.90   |  |
| Llama-3.1-8B-Instruct [78]        | 93.65 | 24.01     | 81.50   | 98.20 | N/A | 98.20               | 96.99         | 34.24 | 82.95   |  |
| GPT-OSS-20B [79]                  | 82.00 | 74.93     | 80.77   | 93.04 | N/A | 93.04               | 89.20         | 69.97 | 84.90   |  |
| DeepSeek-R1-Distill-Qwen-32B [80] | 93.03 | 26.38     | 81.42   | 99.18 | N/A | 99.18               | 97.21         | 52.85 | 87.28   |  |

