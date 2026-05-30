# **B** Implementation Details

This section presents the implementation details of ZSMerge.

The KV cache compression framework is built upon the Transformers library. To minimize deviations from the original framework and reduce redevelopment complexity, only the forward propagation function was replaced globally. As a result, a single process cannot simultaneously hold two instances with different compression modes. However, the compression mode can be easily switched without creating a new instance by calling the *change\_mode* method.

Our framework currently supports replacing the  $scaled\_dot\_product\_attention$  function for the LLaMA, Falcon, and Mistral model families, as this operation is widely used across various inference scenarios.

The initialization of the attention score s based on the full history of attention scores during the prefilling stage imposes a substantial computational burden. In certain long-sequence tasks (such as the LongBench experiments), we introduce a hyperparameter,  $window\_size$ , to limit the range of timesteps considered during the initialization of s, following the approach used in SnapKV. This optimization has a minimal impact on generation quality but significantly accelerates the prefilling process.

#### <span id="page-13-1"></span>C Extended Experiments

## C.1 Latency and Throughput Evaluation Across Sequence Lengths and Batch Sizes

Table 4 presents additional experimental results on workload-scalable KV cache compression. The validation of the LESS and H2O frameworks was conducted using the code provided at https://github.com/hdong920/LESS. The results were obtained from a single experiment run, as the observed conclusions were clear and consistent. In short-sequence and low-batch-size settings, the FullKV method shows slight advantages, as compression introduces additional computational overhead. However, in other scenarios, ZSMerge demonstrates significant performance gains, even when compared to other compression methods.

<span id="page-14-1"></span>Table 4: Workload-Scalable KV Cache Compression: ZSMerge Outperforms Baselines in Throughput (tokens/sec) and Latency (seconds) Across Sequence Lengths and Batch Sizes.

|            |            |            | THROUGHPUT(TOKENS/S) ↑ / LATENCY(S) ↓ |              |               |               |  |  |  |  |  |  |
|------------|------------|------------|---------------------------------------|--------------|---------------|---------------|--|--|--|--|--|--|
| SEQ.LENGTH | BATCH SIZE | MODEL SIZE | FULLKV                                | H2O (5%)     | LESS (5%)     | ZSMERGE (5%)  |  |  |  |  |  |  |
| 1024+1024  | 4          | 7B         | 117.5 / 34.9                          | 83.5 / 49.0  | 22.7 / 180.6  | 81.5 / 50.3   |  |  |  |  |  |  |
| 1024+1024  | 8          | 7B         | 177.8 / 46.1                          | 104.9 / 78.1 | 48.8 / 167.9  | 161.6 / 50.7  |  |  |  |  |  |  |
| 2048+2048  | 8          | 7B         | 110.8 / 147.9                         | 72.2 / 227.0 | 25.1 / 654.2  | 163.2 / 100.4 |  |  |  |  |  |  |
| 2048+2048  | 16         | 7B         | 133.1 / 246.2                         | 86.1 / 380.1 | OOM           | 281.9 / 178.4 |  |  |  |  |  |  |
| 4096+4096  | 4          | 7B         | 62.4 / 262.1                          | 43.9 / 372.5 | 15.0 / 1086.2 | 77.0 / 212.7  |  |  |  |  |  |  |
| 4096+4096  | 8          | 7B         | 65.5 / 500.3                          | OOM          | OOM           | 146.7 / 223.2 |  |  |  |  |  |  |
| 4096+4096  | 16         | 7B         | OOM                                   | OOM          | OOM           | 271.6 / 241.3 |  |  |  |  |  |  |
| 8192+4096  | 4          | 7B         | 38.9 / 420.8                          | OOM          | OOM           | 74.3 / 220.4  |  |  |  |  |  |  |
| 8192+8192  | 4          | 7B         | 33.1 / 989.3                          | OOM          | OOM           | 78.3 / 418.5  |  |  |  |  |  |  |
| 8192+4096  | 8          | 7B         | OOM                                   | OOM          | OOM           | 132.0 / 248.2 |  |  |  |  |  |  |
| 8192+8192  | 8          | 7B         | OOM                                   | OOM          | OOM           | 142.6 / 459.6 |  |  |  |  |  |  |
| 256+256    | 2          | 13B        | 62.5 / 8.2                            | 49.2 / 10.4  | 31.1 / 16.5   | 30.7 / 16.7   |  |  |  |  |  |  |
| 512+512    | 2          | 13B        | 61.4 / 16.7                           | 43.5 / 23.6  | 27.1 / 37.8   | 31.1 / 32.9   |  |  |  |  |  |  |
| 1024+1024  | 2          | 13B        | 54.4 / 37.7                           | 33.2 / 61.6  | 16.8 / 121.8  | 31.1 / 65.8   |  |  |  |  |  |  |
| 2048+2048  | 2          | 13B        | 43.0 / 95.2                           | 25.5 / 160.4 | 12.7 / 322.7  | 31.3 / 131.0  |  |  |  |  |  |  |
| 2048+2048  | 4          | 13B        | 59.7 / 137.2                          | 40.7 / 201.2 | 16.5 / 497.5  | 61.5 / 133.3  |  |  |  |  |  |  |
| 4096+4096  | 4          | 13B        | 37.1 / 441.8                          | 24.9 / 657.9 | OOM           | 60.0 / 273.2  |  |  |  |  |  |  |
| 4096+4096  | 8          | 13B        | OOM                                   | OOM          | OOM           | 110.8 / 295.7 |  |  |  |  |  |  |
| 4096+4096  | 16         | 13B        | OOM                                   | OOM          | OOM           | 178.2 / 367.6 |  |  |  |  |  |  |
| 4096+8192  | 16         | 13B        | OOM                                   | OOM          | OOM           | 666.3 / 196.7 |  |  |  |  |  |  |
| 4096+4096  | 32         | 13B        | OOM                                   | OOM          | OOM           | 397.5 / 329.7 |  |  |  |  |  |  |
| 8192+8192  | 4          | 13B        | OOM                                   | OOM          | OOM           | 617.0 / 53.1  |  |  |  |  |  |  |
| 8192+8192  | 8          | 13B        | OOM                                   | OOM          | OOM           | 642.3 / 102.0 |  |  |  |  |  |  |
| 8192+4096  | 16         | 13B        | OOM                                   | OOM          | OOM           | 466.7 / 140.4 |  |  |  |  |  |  |
| 8192+8192  | 16         | 13B        | OOM                                   | OOM          | OOM           | 812.6 / 161.3 |  |  |  |  |  |  |
| 8192+8192  | 32         | 13B        | OOM                                   | OOM          | OOM           | OOM           |  |  |  |  |  |  |

#### <span id="page-14-0"></span>**C.2** Hyperparameter Sensitivity Validation

As a training-free framework for KV cache compression, our method introduces several hyperparameters to enhance flexibility and provide a smooth transition to classical sparsity-based methods. We conduct comprehensive sensitivity analysis to offer empirical recommendations for practical deployment scenarios.

**Budget Distribution Strategy**: Our framework follows a hierarchical budget allocation strategy. First, the proximity maintenance budget  $B_p$  is controlled by the cache tail ratio  $(B_p/B)$ . Then, a small portion of the remaining budget is allocated to the residual budget  $B_r$ , controlled by the cache dense parameter  $(B_r/(B-B_p))$ . Finally, the remaining budget is distributed to the context preservation budget  $B_c$ .

**Experimental Setup**: We conduct sensitivity analysis on LLaMA2-7B using the XSUM summarization task. We fix an anchor configuration and systematically vary each hyperparameter to isolate its individual impact on performance, measured by ROUGE-1, ROUGE-2, and ROUGE-L scores.

- Proximity Maintenance Ratio  $(B_p/B)$ : This parameter governs the allocation between proximity maintenance and context preservation. Our analysis reveals that extreme partitions significantly hinder performance. Values below 0.3 or above 0.7 show notable degradation, with ROUGE-1 scores dropping by 4-8% at the extremes (0.1 and 0.9). The optimal range lies between 0.3 and 0.7, with peak performance around 0.5, consistent with established methods like H2O. We recommend setting this ratio to 0.5 for balanced performance.
- Residual Budget Ratio  $(B_r/(B-B_p))$ : The residual budget demonstrates the effectiveness of our token merging operation. When  $B_r=0$ , the method degrades to a pure eviction strategy, showing 6-11% performance drops across all ROUGE metrics. Small positive values (0.01-0.03) provide stable benefits, with 0.02 showing optimal performance. Higher values (0.05-0.08) compress the context budget excessively, leading to performance degradation of 7-18%. We recommend setting this ratio to 0.02 to achieve stable benefits while preserving sufficient context budget.
- Scale Factor ( $\alpha$ ): This parameter controls the influence of merged cache tokens. Our analysis shows that increasing the scale factor from 0.0 to 1.0 progressively improves performance, with

ROUGE scores improving by 1-5%. Setting  $\alpha=0$  degenerates the method to standard sparse approaches, while  $\alpha=1.0$  provides optimal performance. This validates the effectiveness of our token merging operation. We restrict the value to 1.0 in our standard experiments.

<span id="page-15-0"></span>![](_page_15_Figure_1.jpeg)

Figure 5: Hyperparameter Sensitivity Analysis: (top) Proximity Maintenance Ratio  $(B_p/B)$ , (middle) Residual Budget Ratio  $(B_r/(B-B_p))$ , (bottom) Scale Factor  $(\alpha)$ 

**Key Findings**: Figure 5 illustrates the sensitivity patterns across all three hyperparameters. The analysis confirms that **ZSMerge** demonstrates robust performance across most hyperparameter configurations. The method shows particular sensitivity to extreme budget allocations but maintains stable performance within recommended ranges. The effectiveness of token merging is clearly demonstrated through the consistent improvements observed when enabling residual budgets and scale factors, distinguishing our approach from pure eviction-based methods.

**Practical Recommendations**: For deployment scenarios, we recommend the following configuration:  $B_p/B=0.5,\,B_r/(B-B_p)=0.02$ , and  $\alpha=1.0$ . This configuration provides robust performance while maintaining compatibility with existing sparse attention frameworks.

#### C.3 Details on LongBench benchmark

**Dataset**: We evaluate ZSMerge on the LongBench benchmark, a comprehensive suite for assessing long-context understanding in LLMs. The benchmark comprises 21 tasks spanning six categories:

- · Single-document QA
- Multi-document QA
- Summarization
- · Few-shot learning
- · Synthetic tasks
- Code completion

The datasets (English and Chinese) feature context lengths of 5,000-15,000 tokens and are standardized for automated evaluation [43].

**Evaluation Framework**: Our experiments utilize the benchmarking methods implemented in the KVCache-Factory repository (https://github.com/Zefan-Cai/KVCache-Factory).

This framework supports various KV cache compression methods, including PyramidKV, SnapKV, StreamingLLM, and H2O. It is compatible with attention mechanisms such as Flash Attention v2 and SDPA, allowing for efficient evaluation under different memory constraints.

**Models and Configuration**: We conduct experiments on two backbone models: LLaMA2-7B and Mistral-7B. To simulate memory-constrained scenarios, we set the cache size constraints to 512 and 1024 tokens. Notably, for the Mistral-7B model, the H2O baseline encountered out-of-memory (OOM) errors and was excluded from the comparison.

<span id="page-16-0"></span>Table 5: Performance Comparison of KV Cache Compression Methods on LongBench Tasks

| Method           | 2wikimga       | dureader          | gov_report     | hotpotqa       | lcc            | lsht           | multi_news     | multifieldqa_en | multifieldqa_zh | musique        | narrativeqa    | passage_count | passage_retrieval_en | passage_retrieval_zh | qasper         | unsub | repobench-p    | samsum         | trec           | triviaqa       | vcsum          |
|------------------|----------------|-------------------|----------------|----------------|----------------|----------------|----------------|-----------------|-----------------|----------------|----------------|---------------|----------------------|----------------------|----------------|-------|----------------|----------------|----------------|----------------|----------------|
|                  |                |                   |                |                |                |                |                |                 |                 | Ll             | laMA2-7.       | В             |                      |                      |                |       |                |                |                |                |                |
| FullKV           | 10.54          | 23.35             | 27.16          | 7.77           | 68.13          | 20.25          | 3.01           | 23.93           | 18.78           | 4.26           | 17.33          | 1.50          | 5.52                 | 8.00                 | 9.94           | 20.55 | 62.25          | 32.11          | 68.00          | 88.92          | 9.89           |
|                  | Cache          | size=512          |                |                |                |                |                |                 |                 |                |                |               |                      |                      |                |       |                |                |                |                |                |
| H2O              | 7.33           | 3.72              | 1.18           | 4.81           | 22.06          | 0.00           | 1.79           | 5.62            | 1.90            | 2.05           | 5.53           | 2.36          | 4.03                 | 0.08                 | 3.16           | 10.99 | 17.19          | 3.08           | 18.50          | 7.90           | 0.10           |
| Stream           | 9.43           | 13.97             | 0.98           | 6.60           | 64.73          | 16.67          | 0.29           | 17.21           | 11.82           | 3.41           | 11.45          | 1.88          | 5.12                 | 5.96                 | 5.93           | 18.74 | 57.79          | 30.34          | 56.00          | 87.51          | 8.28           |
| SnapKV           | 10.70          | 17.87             | 18.18          | 8.33           | 66.35          | 17.25          | 2.61           | 21.76           | 17.00           | 4.41           | 17.39          | 2.75          | 6.26                 | 7.50                 | 7.26           | 19.98 | 59.76          | 33.96          | 67.50          | 87.34          | 8.37           |
| ZSMerge          | 10.72          | 17.98             | 17.62          | 8.34           | 66.32          | 17.25          | 2.65           | 21.52           | 16.59           | 4.41           | 17.30          | 2.75          | 6.26                 | 7.00                 | 7.29           | 20.05 | 59.71          | 33.62          | 67.50          | 87.29          | 8.37           |
| H2O              | 8.49           | size=102<br>6.41  | 4.25           | 6.08           | 38.60          | 0.50           | 2.91           | 11.14           | 4.84            | 2.70           | 6.34           | 2.23          | 6.67                 | 0.00                 | 7.55           | 19.51 | 19.65          | 30.86          | 19.50          | 30.52          | 0.14           |
| Stream           | 9.29           | 13.33             | 0.99           | 6.49           | 66.61          | 17.00          | 0.91           | 16.99           | 12.10           | 3.51           | 11.47          | 1.62          | 3.92                 | 7.81                 | 6.66           | 19.05 | 59.27          | 31.87          | 62.50          | 88.54          | 7.77           |
| SnapKV           | 10.67          | 21.78             | 22.54          | 8.02           | 67.64          | 18.75          | 2.74           | 22.56           | 18.04           | 4.73           | 17.49          | 2.54          | 5.98                 | 6.75                 | 8.52           | 20.58 | 61.26          | 32.97          | 68.00          | 88.38          | 7.33           |
| ZSMerge          | 10.67          | 22.12             | 22.05          | 7.97           | 67.63          | 18.75          | 2.82           | 22.39           | 18.07           | 4.73           | 17.55          | 2.54          | 5.76                 | 6.75                 | 8.49           | 20.67 | 61.19          | 33.13          | 68.00          | 88.38          | 7.39           |
|                  |                |                   |                |                |                |                |                |                 |                 | Mintral        | 7B-Instri      | tot vO 3      |                      |                      |                |       |                |                |                |                |                |
| FullKV           | 39.01          | 32.38             | 34.89          | 49.37          | 61.56          | 40.25          | 27.83          | 52.88           | 32.26           | 28.58          | 29.07          | 5.50          | 98.00                | 96.50                | 41.58          | 25.77 | 62.63          | 47.51          | 76.00          | 88.59          | 16.08          |
|                  | Cache          | size=512          |                | .,             |                |                |                |                 |                 |                |                |               |                      | ,                    |                |       |                |                |                |                |                |
| Stream           | 31.86          | 17.64             | 22.10          | 41.05          | 59.37          | 18.50          | 23.20          | 29.91           | 15.60           | 17.60          | 24.21          | 6.00          | 81.00                | 9.50                 | 25.95          | 20.25 | 56.42          | 43.79          | 65.50          | 86.95          | 13.65          |
| SnapKV           | 38.72          | 24.02             | 25.85          | 49.53          | 60.32          | 37.75          | 24.96          | 54.05           | 28.27           | 26.72          | 28.79          | 5.00          | 96.00                | 93.50                | 36.34          | 24.08 | 60.60          | 46.75          | 75.00          | 89.44          | 13.82          |
| ZSMerge          | 38.56          | 23.68             | 25.47          | 49.67          | 60.21          | 37.75          | 24.85          | 53.98           | 28.38           | 26.58          | 29.01          | 5.00          | 95.00                | 93.50                | 35.94          | 24.22 | 60.76          | 46.67          | 75.00          | 89.28          | 13.81          |
| g.               |                | size=102<br>17.17 |                | 40.05          | 61.04          | 21.25          | 25.40          | 21.16           | 10.00           | 10.02          | 24.01          |               | 02.50                | 14.50                | 25.05          | 20.01 | 50.01          | 45.50          | 60.50          | 00.71          |                |
| Stream<br>SnapKV | 32.65<br>38.86 | 26.13             | 24.59<br>28.30 | 43.35<br>49.14 | 61.04<br>61.34 | 21.25<br>38.25 | 25.48<br>26.72 | 31.16<br>52.64  | 16.46<br>29.73  | 18.03<br>27.81 | 24.81<br>29.09 | 5.50<br>5.50  | 82.50<br>98.00       | 14.50<br>96.00       | 27.95<br>37.76 | 20.81 | 59.21<br>61.86 | 45.59<br>46.22 | 68.50<br>76.00 | 88.71<br>88.99 | 14.11<br>14.76 |
| ZSMerge          | 38.86          | 25.89             | 28.05          | 49.14          | 61.35          | 38.25          | 26.67          | 52.64           | 29.73           | 27.81          | 29.09          | 5.50          | 98.00                | 96.00                | 38.03          | 25.00 | 61.83          | 46.42          | 76.00          | 88.99          | 14.70          |
| Zowierge         | 50.00          | 25.07             | 20.03          | 77.17          | 01.55          | 30.23          | 20.07          | 32.04           | 27.50           |                |                |               | 70.00                | 70.00                | 30.03          | 25.00 | 01.05          | 40.42          | 70.00          | 00.77          | 14.02          |
| E-111737         | 16.20          | 21.45             | 24.12          | 15.02          | (5.00          | 10.50          | 26.70          | 27.02           | 20.02           |                | 3.1-8B-I       |               | 70.70                | 76.20                | 12.00          | 22.44 | 57.46          | 12.50          | 70.00          | 01.27          | 16.14          |
| FullKV           | 16.39          | 31.45<br>size=512 | 34.13          | 15.93          | 65.06          | 40.50          | 26.70          | 27.02           | 20.03           | 9.97           | 21.06          | 7.34          | 72.79                | 76.30                | 12.80          | 22.44 | 57.46          | 43.56          | 70.00          | 91.37          | 16.14          |
| SnapKV           | 15.26          | 23.65             | 25.02          | 16.17          | 63.93          | 40.00          | 24.19          | 25.97           | 20.43           | 8.71           | 20.00          | 7.72          | 72.28                | 79.51                | 11.10          | 22.94 | 54.27          | 42.34          | 67.50          | 91.67          | 13.79          |
| ZSMerge          | 14.83          |                   | 25.24          | 16.36          | 64.45          | 40.00          | 24.21          | 24.98           | 19.24           | 8.97           | 21.73          | 7.35          | 71.25                | 74.57                | 10.45          | 22.70 | 55.08          | 43.27          | 66.00          | 91.47          | 14.23          |
|                  | Cache          | size=102          | 24             |                |                |                |                |                 |                 |                |                |               |                      |                      |                |       |                |                |                |                |                |
| SnapKV           | 15.02          | 25.37             | 27.55          | 16.54          | 64.41          | 40.00          | 25.49          | 27.63           | 19.94           | 9.73           | 20.21          | 6.24          | 72.40                | 74.99                | 11.16          | 23.31 | 55.97          | 42.96          | 69.50          | 91.39          | 14.10          |
| ZSMerge          | 15.01          | 25.13             | 27.82          | 16.35          | 64.33          | 40.00          | 25.65          | 26.27           | 19.63           | 8.88           | 20.49          | 7.25          | 72.29                | 76.07                | 11.12          | 22.88 | 57.17          | 43.52          | 69.00          | 91.08          | 14.77          |
|                  |                |                   |                |                |                |                |                |                 |                 |                |                |               |                      |                      |                |       |                |                |                |                |                |

The experimental results (Table 5) reveal several key patterns. Across both LLaMA2-7B and Mistral-7B models, the uncompressed FullKV baseline achieves the highest performance but serves primarily as an upper-bound reference rather than a practical solution. When comparing compression methods under memory constraints, ZSMerge and SnapKV demonstrate superior capability in preserving model accuracy compared to StreamingLLM and H2O, particularly in challenging scenarios with cache sizes limited to 512 or 1024 tokens.

The Mistral-7B model consistently outperforms LLaMA2-7B, showing particularly strong results in question answering and retrieval tasks, where it achieves near-perfect scores in some cases. This performance gap highlights Mistral's architectural advantages for long-context processing. Meanwhile, LLaMA2 struggles more noticeably with certain synthetic tasks and Chinese language datasets, suggesting limitations in its multilingual and reasoning capabilities.

Cache size plays a measurable but not decisive role in performance. While increasing the cache from 512 to 1024 tokens provides modest improvements, ZSMerge maintains competitive accuracy even with the smaller cache, demonstrating its efficiency. In contrast, H2O shows severe degradation under constrained settings, failing completely on some tasks.

Task-specific analysis indicates that summarization and code-related tasks benefit most from methods that better preserve context, like ZSMerge and SnapKV. Retrieval-focused tasks, however, show less sensitivity to cache size, with Mistral achieving consistently high scores regardless of compression. Overall, ZSMerge emerges as a balanced solution, delivering near-FullKV performance while operating efficiently within strict memory limits.

To further evaluate the applicability of ZSMerge under realistic deployment settings, we benchmarked it using the LLaMA-3.1-8B-Instruct model on the LongBench suite. As shown in Table 5, we report results exclusively for our method, since other contextually adaptive baselines (e.g., StreamingLLM, H2O) currently do not provide support for LLaMA-3 architecture. Despite this, ZSMerge exhibits robust performance under constrained cache settings (512 and 1024 tokens), achieving scores close

to the uncompressed FullKV baseline across a wide range of tasks. In particular, ZSMerge preserves strong performance on summarization and retrieval tasks, indicating that our token merging strategy can mitigate MQA's context sensitivity without auxiliary modules or task-specific tuning. These results reinforce the generalization and plug-and-play capability of ZSMerge, even when deployed with newer model architectures featuring aggressive attention simplification.

#### <span id="page-17-0"></span>**C.4** Extended Architecture Validation

To validate our zero-shot compatibility claims, we extended ZSMerge evaluation to modern LLM architectures. This section presents comprehensive results on LLaMA3 and discusses ongoing experiments with additional contemporary models.

**InfiniteBench Long-Context Evaluation** We successfully implemented **ZSMerge** on LLaMA3-8B architecture, demonstrating that our framework maintains compatibility with stronger, more recent backbones. The implementation required no architectural modifications or hyperparameter retuning, confirming the architecture-agnostic design principles of our approach.

To address concerns about evaluation scope, we conducted comprehensive experiments on InfiniteBench [45] using LLaMA3-8B, a demanding benchmark featuring contexts exceeding 100K tokens. This evaluation provides critical validation under extreme long-context scenarios that stresstest compression methods beyond conventional limits.

Table 6: InfiniteBench Results: ZSMerge vs. State-of-the-Art Methods on LLaMA3-8B

<span id="page-17-1"></span>

| Method        | En.Dia | En.MC | En.Sum | Math.Find | Retrieve.KV | Retrieve.Number | Retrieve.PassKey | Zh.QA | Avg   |
|---------------|--------|-------|--------|-----------|-------------|-----------------|------------------|-------|-------|
| FullKV        | 37.37  | 23.70 | 21.72  | 55.55     | 73.73       | 98.00           | 100.00           | 25.47 | 54.69 |
| H2O           | 43.17  | 23.33 | 21.30  | 63.30     | 26.36       | 72.52           | 98.40            | 24.63 | 46.63 |
| InfLLM        | 24.67  | 15.85 | 16.91  | 50.85     | 0.00        | 98.00           | 100.00           | 34.87 | 42.64 |
| OmniKV        | 30.83  | 23.33 | 21.99  | 55.00     | 48.67       | 98.00           | 100.00           | 24.83 | 50.33 |
| Streaming LLM | 14.63  | 13.88 | 20.31  | 40.10     | 0.00        | 5.83            | 2.73             | 17.67 | 14.39 |
| Minference    | 24.95  | 21.55 | 20.91  | 62.11     | 17.03       | 75.84           | 57.27            | 22.65 | 37.79 |
| FlexPrefill   | 33.56  | 23.47 | 21.44  | 70.51     | 53.53       | 98.17           | 99.49            | 27.99 | 53.52 |
| ZSMerge       | 36.44  | 22.79 | 21.34  | 55.16     | 67.96       | 98.00           | 100.00           | 24.84 | 52.95 |

The InfiniteBench results (Table 6) demonstrate several critical findings:

**Performance Preservation:** ZSMerge achieves 96.8% of FullKV performance (52.95 vs. 54.69 average), maintaining quality despite compression. This validation occurs with context lengths of at least 100K tokens that test compression robustness.

**Competitive Positioning:** The method outperforms traditional eviction-based approaches (H2O: 46.63, InfLLM: 42.64) and matches the performance of recent specialized methods (FlexPrefill: 53.52, OmniKV: 50.33). **ZSMerge** performs well in retrieval tasks (Retrieve.KV: 67.96), demonstrating information preservation for memory-intensive operations.

**Task-Specific Analysis:** The method shows strength in:

- Retrieval tasks: Scores of 100.00 on PassKey and 98.00 on Number retrieval
- Long-form reasoning: Performance on English Dialogue (36.44) and Math Finding (55.16)
- Cross-lingual capability: Performance on Chinese QA (24.84), indicating multilingual robustness

**Extended Baseline Comparison with SnapKV** We conducted evaluations with SnapKV on models that were not originally supported, including Qwen2.5 [19], Yi-1.5 [20], and LLaMA-3.1.

**Implementation Adaptations:** For comparison, we implemented technical adaptations including GQA model support for Grouped Query Attention models (LLaMA-3 and Qwen2.5) by averaging grouped query scores to align them with KV heads, Qwen compatibility by resolving interface differences in the RoPE method to maintain compatibility with the original Qwen behavior, and architecture extension by enabling SnapKV support for Qwen2.5-7B, Yi-1.5-6B, and LLaMA-3.1-8B-Instruct models.

Experimental Setup: We evaluated on the GSM-Infinite-8k [\[46\]](#page-11-14) benchmark across three difficulty levels (symbolic, medium, hard). The methodology included length bucketing where test samples were bucketed by input length to account for tokenizer variations despite the nominal 8k limit, adaptive cache budgets using cache budgets of 9k, 4k, and 2k tokens for samples with input lengths >10k, 5k–10k, and <5k tokens respectively, and ensuring all methods used identical experimental conditions and evaluation metrics. The results are presented in Table [7.](#page-18-0)

<span id="page-18-0"></span>Table 7: Extended Baseline Comparison: ZSMerge vs. SnapKV on GSM-Infinite-8k

| Model                 | Method  | Symbolic | Medium | Hard   |
|-----------------------|---------|----------|--------|--------|
|                       | FullKV  | 9.35%    | 17.12% | 12.04% |
| Qwen2.5-7B-Instruct   | SnapKV  | 0.00%    | 3.60%  | 8.33%  |
|                       | ZSMerge | 4.66%    | 9.91%  | 9.26%  |
|                       | FullKV  | 0.00%    | 5.83%  | 7.46%  |
| Yi-1.5-6B             | SnapKV  | 0.00%    | 4.85%  | 6.47%  |
|                       | ZSMerge | 0.00%    | 5.34%  | 5.97%  |
|                       | FullKV  | 20.24%   | 11.65% | 13.93% |
| LLaMA-3.1-8B-Instruct | SnapKV  | 16.74%   | 10.19% | 12.93% |
|                       | ZSMerge | 15.38%   | 11.17% | 13.43% |

Key Findings: The results demonstrate that ZSMerge consistently matches or outperforms SnapKV across all tested models and difficulty levels:

- Qwen2.5-7B: ZSMerge shows superior performance on symbolic and medium difficulty tasks, with competitive results on hard tasks.
- Yi-1.5-6B: ZSMerge achieves slightly better performance across medium and hard difficulties while maintaining identical symbolic task performance.
- LLaMA-3.1-8B: ZSMerge demonstrates consistent advantages across all difficulty levels, particularly excelling in symbolic reasoning tasks.

These results validate that ZSMerge maintains its effectiveness when compared against state-of-theart baselines across diverse modern architectures, confirming the robustness and generalizability of our approach.