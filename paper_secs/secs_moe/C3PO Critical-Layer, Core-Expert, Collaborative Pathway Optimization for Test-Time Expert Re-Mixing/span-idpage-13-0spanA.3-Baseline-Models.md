# <span id="page-13-0"></span>A.3 Baseline Models

We briefly introduce each model categorized by active parameter size as follows:

### LMs with ∼1B active parameters:

• Pythia-1B [\(Biderman et al.,](#page-9-7) [2023\)](#page-9-7): A 1-billion-parameter dense model, trained by EleutherAI using standard autoregressive training techniques.

- Llama3.2-1B [\(Grattafiori et al.,](#page-10-12) [2024\)](#page-10-12): A compact variant of the Llama family, featuring approximately 1 billion parameters designed by Meta.
- OLMo-1B [\(Groeneveld et al.,](#page-10-13) [2024\)](#page-10-13): An open-source dense transformer model with around 1 billion parameters, developed by Allen Institute for AI (AI2).
- TinyLyne-1B-7B [Tang et al.](#page-11-4) [\(2024\)](#page-11-4): A sparse mixture-of-experts (MoE) model with 1 billion active parameters from a total of 7 billion parameters.

### LMs with ∼2-3B active parameters:

- OpenMoE-3B-9B [\(Xue et al.,](#page-11-5) [2024\)](#page-11-5): An MoE architecture having 3 billion active parameters selected from a total of 9 billion parameters.
- StableLM-2B [\(Bellagente et al.,](#page-9-8) [2024\)](#page-9-8): A dense transformer-based language model by Stability AI, containing around 2 billion parameters.
- JetMoE-2B-9B [\(Shen et al.,](#page-10-14) [2024\)](#page-10-14): A sparse mixture-of-experts model from the Jet series with 2 billion active parameters chosen from a pool of 9 billion.
- Gemma2-3B [\(Team et al.,](#page-11-6) [2024\)](#page-11-6): A dense transformer model developed by Google Deep-Mind with approximately 3 billion parameters.
- Qwen1.5-3B-14B [\(Yang et al.,](#page-11-7) [2024\)](#page-11-7): A large-scale MoE model by Alibaba, featuring 3 billion active parameters selected from a total of 14 billion parameters.

### LMs with ∼7-9B active parameters:

- Llama2-7B [\(Touvron et al.,](#page-11-8) [2023\)](#page-11-8): Meta's open-source dense language model with approximately 7 billion parameters.
- Qwen-7B [\(Yang et al.,](#page-11-7) [2024\)](#page-11-7): A 7-billion-parameter dense transformer model developed by Alibaba.
- Mistral-7B [\(Jiang et al.,](#page-10-15) [2023\)](#page-10-15): A dense language model by Mistral AI, consisting of roughly 7 billion parameters.
- DeepSeek-7B [\(Bi et al.,](#page-9-9) [2024\)](#page-9-9): An open-source transformer-based dense language model with 7 billion parameters.
- Llama3.1-8B [\(Grattafiori et al.,](#page-10-12) [2024\)](#page-10-12): Meta's latest generation dense transformer model with about 8 billion parameters.
- OLMo2-7B [\(OLMo et al.,](#page-10-16) [2024\)](#page-10-16): An advanced 7-billion-parameter dense model by Allen Institute for AI, building upon the OLMo architecture.

## <span id="page-14-0"></span>A.4 Ablation Study

Layer optimization strategies determine which specific layers' routing weights should be modified in each token, directly influencing the model's performance after optimization. Table [8](#page-15-0) analyzes different layer optimization strategies for routing weights in OLMoE. We systematically explore various combinations within the OLMoE's 16 layers, revealing that the location of optimized layers significantly impacts performance. Single-layer optimization shows best results when targeting the last layer, while two-layer combinations including the last layer consistently outperform other configurations. Most importantly, optimizing only the final five layers (Last5) achieves the best performance across all benchmarks, surpassing even the full 16-layer optimization (All16). This suggests that focusing optimization on the deeper layers near the output is more effective than modifying the entire network, highlighting the importance of targeted layer selection in MoE architectures.

Token optimization strategies determine which specific token numbers and positions should be modified in the sequence, significantly affecting the inference results after optimization. Table [9](#page-16-0) examines the impact of optimizing routing weights at different token positions in OLMoE. We systematically analyze various positions (first, middle, last) and quantities (one, three tokens). Results clearly show that token position significantly affects performance, with last token optimization consistently outperforming other configurations across all benchmarks. Notably, optimizing only the last token yields the best results, achieving improvements of +7.7% on MMLU and +15.0%

| OLMoE                                 | MMLU                                    | HellaSwag | ARC-C | ARC-E | PIQA | WinoGrande |  |  |
|---------------------------------------|-----------------------------------------|-----------|-------|-------|------|------------|--|--|
| Base model                            | 57.8                                    | 77.9      | 51.3  | 79.8  | 80.7 | 72.2       |  |  |
| 1 Layer Optimization                  |                                         |           |       |       |      |            |  |  |
| First 1                               | 59.4                                    | 78.9      | 52.8  | 80.3  | 82.5 | 73.9       |  |  |
| Middle 1                              | 58.3                                    | 78.1      | 51.9  | 79.9  | 81.2 | 72.8       |  |  |
| Last 1                                | 60.2                                    | 79.7      | 53.5  | 81.6  | 82.9 | 74.5       |  |  |
| 2 Layers Routing Weights Optimization |                                         |           |       |       |      |            |  |  |
| First 1 + Middle 1                    | 60.5                                    | 80.2      | 54.6  | 82.3  | 83.1 | 75.2       |  |  |
| First 1 + Last 1                      | 61.8                                    | 81.3      | 55.8  | 83.7  | 84.5 | 76.8       |  |  |
| Middle 1 + Last 1                     | 60.9                                    | 80.7      | 54.9  | 82.8  | 83.4 | 75.7       |  |  |
| First 2                               | 60.7                                    | 80.6      | 55.3  | 83.1  | 84.0 | 76.1       |  |  |
| Middle 2                              | 59.9                                    | 79.5      | 53.9  | 81.9  | 82.3 | 74.1       |  |  |
| Last 2                                | 62.3                                    | 81.9      | 56.7  | 84.2  | 85.1 | 77.3       |  |  |
| 5 Layers Routing Weights Optimization |                                         |           |       |       |      |            |  |  |
| First 2 + Middle 3                    | 63.2                                    | 82.8      | 59.4  | 85.1  | 85.6 | 79.2       |  |  |
| First 2 + Last 3                      | 64.3                                    | 83.7      | 62.8  | 86.5  | 87.1 | 80.7       |  |  |
| Middle 2 + Last 3                     | 63.7                                    | 83.1      | 61.5  | 85.3  | 86.2 | 79.8       |  |  |
| First 5                               | 63.9                                    | 83.5      | 62.1  | 85.9  | 86.7 | 80.3       |  |  |
| Middle 5                              | 62.5                                    | 82.3      | 58.7  | 84.6  | 84.9 | 78.5       |  |  |
| Last 5                                | 65.5                                    | 85.3      | 66.3  | 87.4  | 88.0 | 82.7       |  |  |
|                                       | All Layers Routing Weights Optimization |           |       |       |      |            |  |  |
| All (16) Layers                       | 64.1                                    | 84.3      | 63.7  | 86.1  | 86.8 | 81.2       |  |  |

<span id="page-15-0"></span>Table 8: Comparison of C3PO applied to different layers in OLMoE. Performance comparison of different layer optimization strategies.

on ARC-C compared to the baseline. Expanding optimization to three tokens actually decreases performance, suggesting that focusing exclusively on the final token provides the most effective routing optimization strategy.

|                       | MMLU | HellaSwag | ARC-C | ARC-E | PIQA | WinoGrande |  |
|-----------------------|------|-----------|-------|-------|------|------------|--|
| Base model            | 57.8 | 77.9      | 51.3  | 79.8  | 80.7 | 72.2       |  |
| 1 Token Optimization  |      |           |       |       |      |            |  |
| First 1 Token         | 61.4 | 81.5      | 58.7  | 83.6  | 84.2 | 77.3       |  |
| Middle 1 Token        | 59.2 | 79.1      | 53.0  | 81.2  | 82.1 | 73.8       |  |
| Last 1 Token          | 65.5 | 85.3      | 66.3  | 87.4  | 88.0 | 82.7       |  |
| 3 Tokens Optimization |      |           |       |       |      |            |  |
| First 3 Token         | 60.8 | 80.7      | 57.5  | 82.9  | 83.5 | 76.4       |  |
| Middle 3 Token        | 58.6 | 78.5      | 52.4  | 80.5  | 81.3 | 73.1       |  |
| Last 3 Token          | 64.1 | 84.3      | 64.8  | 86.2  | 86.7 | 81.3       |  |

Table 9: Performance comparison of different token optimization strategies.

<span id="page-16-0"></span>**Neighborhood selection** Table 10 examines different neighborhood selection strategies for routing weight optimization in OLMoE. We evaluate two approaches: an  $\epsilon$ -neighborhood method with various thresholds and a k-nearest neighbors (kNN) approach with different k values. While both methods significantly improve performance over the baseline, the kNN approach with k=3 consistently delivers the best results across all benchmarks, achieving improvements of +7.7% on MMLU and +15.0% on ARC-C. The  $\epsilon$ -neighborhood method shows strong performance at  $\epsilon$ =0.5, but still falls short of kNN's effectiveness. These results indicate that selecting a moderate number of nearest neighbors provides the optimal strategy for neighborhood-based routing optimization.

|                                                                                         | MMLU        | HellaSwag   | ARC-C       | ARC-E       | PIQA        | WinoGrande  |
|-----------------------------------------------------------------------------------------|-------------|-------------|-------------|-------------|-------------|-------------|
| Base model                                                                              | 57.8        | 77.9        | 51.3        | 79.8        | 80.7        | 72.2        |
| $ \begin{aligned} \epsilon &= 0.3 \\ \epsilon &= 0.5 \\ \epsilon &= 0.7 \end{aligned} $ | 60.4        | 80.5        | 57.2        | 83.4        | 84.1        | 76.5        |
|                                                                                         | 63.2        | 83.7        | 63.5        | 85.8        | 86.3        | 80.2        |
|                                                                                         | 62.8        | 84.1        | 62.9        | 85.1        | 86.5        | 79.8        |
| k = 1 $k = 3  (Ours)$ $k = 5$                                                           | 61.7        | 82.3        | 59.8        | 84.2        | 85.3        | 78.4        |
|                                                                                         | <b>65.5</b> | <b>85.3</b> | <b>66.3</b> | <b>87.4</b> | <b>88.0</b> | <b>82.7</b> |
|                                                                                         | 63.9        | 84.5        | 63.7        | 86.1        | 86.7        | 81.3        |

Table 10: Performance comparison of different optimization strategies.

<span id="page-16-1"></span>**Step numbers** Table 11 examines how the number of optimization steps affects routing weight performance in OLMoE. Results show significant improvements as steps increase from 3 to 10, with substantial early gains (+2.5% on MMLU from 3 to 5 steps) that gradually diminish due to our cosine annealing learning rate schedule. Importantly, performance plateaus beyond 10 steps, with minimal fluctuations at 20 and 50 steps across all benchmarks. This indicates that 10 optimization steps provide a better balance between computational efficiency and performance improvement, as additional steps yield negligible benefits.

**Learning rate** Table 12 demonstrates the impact of learning rate schedules on model performance across six benchmarks. The cosine learning rate schedule ( $10e-2 \rightarrow 10e-5$ ) consistently outperforms other methods, achieving improvements of +7.7% on MMLU, +7.4% on HellaSwag, and +15.0% on ARC-C over the base model. Step decay ( $10e-2 \rightarrow 10e-5$ ) shows comparable but slightly lower gains, while fixed learning rates (1e-4 and 1e-3) yield more modest improvements. These results highlight that adaptive learning rate strategies, particularly cosine scheduling, significantly enhance model performance.

| #Steps     | MMLU | HellaSwag | ARC-C | ARC-E | PIQA | WinoGrande |
|------------|------|-----------|-------|-------|------|------------|
| Base model | 57.8 | 77.9      | 51.3  | 79.8  | 80.7 | 72.2       |
| 3          | 61.3 | 81.2      | 58.3  | 83.3  | 84.0 | 77.2       |
| 5          | 63.8 | 83.4      | 62.5  | 85.4  | 86.2 | 80.1       |
| 7          | 64.8 | 84.7      | 65.2  | 86.8  | 87.3 | 81.7       |
| 10 (Ours)  | 65.5 | 85.3      | 66.3  | 87.4  | 88.0 | 82.7       |
| 20         | 65.4 | 85.7      | 66.5  | 87.2  | 88.3 | 82.4       |
| 50         | 65.7 | 85.2      | 66.1  | 87.5  | 87.9 | 82.9       |

Table 11: Performance comparison with different numbers of optimization steps.

<span id="page-17-0"></span>

| Learning Rate | MMLU | HellaSwag | ARC-C | ARC-E | PIQA | WinoGrande |
|---------------|------|-----------|-------|-------|------|------------|
| Base model    | 57.8 | 77.9      | 51.3  | 79.8  | 80.7 | 72.2       |
| Fixed(1e-3)   | 59.1 | 79.4      | 53.0  | 81.2  | 82.1 | 73.9       |
| Fixed(1e-4)   | 61.5 | 81.6      | 57.1  | 83.5  | 84.3 | 76.8       |
| Step Decay    | 64.8 | 84.7      | 65.3  | 86.8  | 87.2 | 81.9       |
| Cosine(Ours)  | 65.5 | 85.3      | 66.3  | 87.4  | 88.0 | 82.7       |

Table 12: Performance comparison with different learning rate schedules.

<span id="page-17-1"></span>Embedding model Table [13](#page-17-2) demonstrates the significant impact of embedding model quality on performance across six benchmarks. NV-Embed-V2 consistently outperforms other embedding models, achieving improvements of up to +15.0% on ARC-C compared to the base model. The results show the clear improvement from All-Mini-V6 to our NV-Embed-V2. This trend confirms that higher-quality embeddings enable more effective identification of relevant neighbors in the reference set, which directly translates to better optimization of routing weights and enhanced performance on downstream tasks.

| Embedding<br>Model    | MMLU | HellaSwag | ARC-C | ARC-E | PIQA | WinoGrande |
|-----------------------|------|-----------|-------|-------|------|------------|
| Base model            | 57.8 | 77.9      | 51.3  | 79.8  | 80.7 | 72.2       |
| All-Mini-V6           | 58.9 | 78.6      | 53.5  | 80.3  | 82.3 | 73.9       |
| Sentence-Bert         | 61.2 | 80.8      | 56.1  | 83.7  | 83.1 | 77.4       |
| Stella-En-1.5B-V5     | 62.1 | 83.4      | 61.2  | 84.2  | 85.8 | 78.3       |
| Gte-Qwen2-7B-instruct | 64.5 | 83.9      | 62.8  | 86.5  | 85.2 | 81.4       |
| NV-Embed-V2 (Ours)    | 65.5 | 85.3      | 66.3  | 87.4  | 88.0 | 82.7       |

Table 13: Performance comparison with different embedding models.

<span id="page-17-2"></span>Kernel choice Table [14](#page-18-0) compares different kernel functions for NGD across six benchmarks. The Gaussian kernel consistently outperforms alternatives, achieving substantial improvements over the linear baseline (+7.7% on MMLU, +7.4% on HellaSwag, +15.0% on ARC-C). This result suggests the Gaussian kernel's effectiveness stems from its superior ability to model non-linear relationships in high-dimensional embedding spaces.

| Kernel          | MMLU | HellaSwag | ARC-C | ARC-E | PIQA | WinoGrande |
|-----------------|------|-----------|-------|-------|------|------------|
| Linear          | 57.8 | 77.9      | 51.3  | 79.8  | 80.7 | 72.2       |
| Polynomial      | 61.2 | 79.4      | 58.7  | 81.5  | 82.9 | 76.3       |
| Matern          | 62.9 | 83.1      | 61.8  | 85.2  | 84.5 | 80.2       |
| Gaussian (Ours) | 65.5 | 85.3      | 66.3  | 87.4  | 88.0 | 82.7       |

<span id="page-18-0"></span>Table 14: Performance comparison with different kernel functions.