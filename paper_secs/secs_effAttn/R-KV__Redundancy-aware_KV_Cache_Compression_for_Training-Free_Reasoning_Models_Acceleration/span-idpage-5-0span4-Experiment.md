# <span id="page-5-0"></span>4 Experiment

### 4.1 Experimental Setup

**Models and Datasets** In our experiments, we use variants of the DeepSeek-R1 distilled model: DeepSeek-R1-Distill-Llama-8B, and DeepSeek-R1-Distill-Qwen-14B [1], which we refer to as R1-Llama-8B and R1-Qwen-14B, respectively, for brevity throughout the paper.

We evaluate the models' mathematical reasoning capabilities using three benchmarks: MATH-500 [8] and AIME 2024 [9].

**Hyperparameters** We set  $B_{\text{buffer}} = 128$ ,  $\alpha = 8$  and  $\lambda = 0.1$ , with an analysis of  $\lambda$  in §5.1.

**Baselines** We compare our method against SnapKV [3], originally designed for long prefilling. To adapt it for decoding, we apply the same compression interval as our method, i.e., compressing the KV cache every 128 decoding steps using identical  $B_{\text{budget}}$  and  $B_{\text{buffer}}$ . Our approach focuses on improving KV cache eviction through a hybrid strategy, and we therefore restrict comparison to state-of-the-art attention-based eviction methods. Budget allocation techniques (e.g., head-level [6] and layer-level [5]) are orthogonal to our work and not included. We also report results for FullKV, which retains the full KV cache and serves as the gold standard for decoding quality.

**Evaluation Setup** We set the maximum generation length to 16,384 tokens for MATH-500 and 32,768 tokens for AIME 2024 and AIME 2025, because further increasing the generation length has shown no improvement on model performance on these datasets from our attempts. We find that using greedy decoding to evaluate long-output reasoning models results in significant variability across different setups. Following existing works [1], we utilize pass@k evaluation [10] and report

pass@1 using a non-zero temperature. We use the recommended sampling temperature and top-p value for each model, i.e., sampling temperature of 0.6 and a top-p value of 0.95 for DeepSeek-R1 Distilled models. We generate 64 responses for each question. Pass@1 is then calculated as Pass@1 =  $\frac{1}{k}\sum_{i=1}^{k}p_i$ , where  $p_i$  denotes the correctness of the i-th response. This method provides more reliable performance estimates.

#### <span id="page-6-1"></span>4.2 Results

The accuracy performance of R-KV compared with all baselines is shown in Figure 4, with detailed accuracy numbers in Appendix B.2. The KV cache budget ratio is calculated based on the KV cache budget and the average generation length of tokens, i.e., R1-Llama-8B: 2,979.1 on MATH-500 and 15,535.8 on AIME24; R1-Qwen-14B: 2,833.04 on MATH-500 and 12,402 on AIME24. Our method significantly outperforms the baseline SnapKV, achieving up to 40% Acc. improvement. We provide two KV cache budget and performance analysis. Fixed budget analysis is more practical because when the model outputs longer (i.e., from 2,979.1 on MATH-500 to 15,535.8 on AIME24), the KV cache budget needed for lossless compression increases less (i.e., 512). In the KV cache budget ratio perspective, the changes of lossless compression ratio is dominated by generation length.

**Ratio Budget** For R1-Llama-8B, R-KV achieves lossless compression with 34% KV cache budget on the MATH-500 dataset and with 10% KV cache budget on the AIME-2024 dataset. Given 16% KV cache budget, our method even surpasses the FullKV baseline, reaching 105% of its accuracy. Similarly, for R1-Qwen-14B, R-KV achieves lossless compression with 54% KV cache budget on the MATH-500 dataset and with 25% KV cache budget on the AIME-2024 dataset. Given 33% KV cache budget, our method achieves 105% of FullKV accuracy.

**Fixed Budget** For R1-Llama-8B, R-KV achieves lossless compression with 1024 KV cache budget on the MATH-500 dataset and with 1536 KV cache budget on the AIME-2024 dataset. For R1-Llama-8B, R-KV achieves lossless compression with 1536 KV cache budget on the MATH-500 dataset and with 3072 KV cache budget on AIME-2024.

### 5 Discussion

### <span id="page-6-0"></span>5.1 How to Choose $\lambda$ ?

Figure 5 shows the distributions of the Importance Score ( $I^h$ ) and Redundancy Estimation ( $R^h$ ) for head h=0 at the top layer ( $N_{\text{layer}}=31$ ). The figure reveals that  $I^h$  is sparse and dominated by a few outlier values, while the similarity distributions (which inform  $R^h$ ) are relatively dense. When  $\lambda=0$ , the token retention strategy is overned entirely by Redundancy Estimation ( $R^h$ ). As shown in Figure 5, the initial four tokens are not guaranteed to be preserved. As highlighted by prior work [7], evicting these initial tokens can severely impair the generative capabilities of LLMs. Therefore, it is crucial to select a  $\lambda$  value that starts from at least 0.01. On the other hand, as  $\lambda$  increases beyond 0.1, the selection metric becomes increasingly dominated by attention scores. These observations suggest that an optimal  $\lambda$  lies within the range of  $0.01 \le \lambda \le 0.1$ , effectively balancing the contributions of Importance Score and Redundancy Estimation.

Figure 6 presents the accuracy (Acc.) performance of R-KV on the DeepSeek-Distill-R1-Llama-8B model using the MATH-500 dataset. The results further guide the choice of  $\lambda$  for optimal performance. The figure demonstrates that  $\lambda=0.1$  yields the highest accuracy. In contrast, strategies relying solely on redundancy ( $\lambda=0$ ) or solely on attention ( $\lambda=1$ ) exhibit the poorest performance, underscoring the complementary nature of these two metrics and the importance of a balanced approach. Thus, based on this finding, we select  $\alpha=0.1$  for all evaluations detailed in Figure 4.

## 5.2 Failure of Attention-Based Methods to Capture Redundancy

To thoroughly investigate the advantages of R-KV's hybrid selection metrics (combining attention and redundancy) over pure attention-based importance metrics, we compared the tokens selected by R-KV against those chosen by a pure attention-based method (SnapKV). We present a case where R-KV correctly completes the task while the comparison method fails. As illustrated in Figure 7,

<span id="page-7-0"></span>![](_page_7_Figure_0.jpeg)

![](_page_7_Figure_1.jpeg)

Figure 5: KV sekection score comparison of attention-only metric v.s. redundency-only metric v.s. R-KV with different  $\lambda$ . When  $\lambda \geq 0.1$ , the selection score starts to be dominated by attention score.

Figure 6: Performance Comparison of the same methods as Figure 5.

<span id="page-7-1"></span>![](_page_7_Figure_4.jpeg)

Figure 7: Comparison of selected key-value (KV) tokens for an example between SnapKV (left) and R-KV (right). Grey tokens are unselected, while the gradient from light to dark red indicates the number of attention heads selecting each token (darker = more heads). R-KV selects a more diverse and broadly distributed set of tokens, capturing richer contextual information.

grey tokens represent unselected tokens, while the gradient from light orange to red indicates the number of heads selecting each token, with darker red signifying selection by more heads.

When considering the tokens selected by all heads, we observe that R-KV selects a more diverse set of tokens that cover a broader range and contain more effective information. These selections are more evenly distributed throughout the decoded output, capturing a more comprehensive context representation. In contrast, SnapKV's selected tokens exhibit more limited coverage. It tends to favor tokens positioned close to the query token, which are often selected multiple times by various heads, indicating a concentration of attention in localized areas. Furthermore, SnapKV also selects tokens that are not in close proximity to the query but still constitute largely redundant and unimportant segments (i.e., "3 students are leaving early." and "But in the initial").

## 5.3 Efficiency Analysis

**Memory Saving** R-KV achieves improved memory efficiency by allocating fixed-size buffers for both the retained KV cache and newly generated tokens. Unlike FullKV, which scales memory linearly with sequence length, R-KV 's memory footprint remains constant, enabling substantial savings during long-form generation. Detailed memory accounting is provided in Appendix C.1.

<span id="page-8-0"></span>

| Gen. Length | Method | Budget                                                            | Mem. Saving (%)         | Batch                               | Throughput (tok/s)               | Tokens Gen.                       | Dec. Time (s)                    |
|-------------|--------|-------------------------------------------------------------------|-------------------------|-------------------------------------|----------------------------------|-----------------------------------|----------------------------------|
| 8K          | FullKV | -                                                                 | -                       | 1<br>62 (max)                       | 75.44<br>849.13                  | 8 094<br>501 828                  | 107.30<br>590.99                 |
|             | R-KV   | Fixed – 1024<br>Fixed – 1024<br>Fixed – 1536                      | 87.50<br>87.50<br>81.25 | 1<br>402 (max)<br>287 (max)         | 80.46<br>3 251.52<br>2 525.75    | 8 094<br>3 253 788<br>6 546 972   | 100.60<br>1 000.70<br>919.72     |
|             |        | Ratio – 10% – 819<br>Ratio – 34% – 2785<br>Ratio – 54% – 4423     | 90.00<br>66.00<br>46.00 | 479 (max)<br>167 (max)<br>105 (max) | 3 809.15<br>1 608.01<br>1 257.83 | 3 877 026<br>1 351 698<br>849 870 | 1 017.82<br>840.61<br>675.66     |
| 16K         | FullKV | -<br>-                                                            | -<br>-                  | 1<br>30 (max)                       | 69.41<br>347.03                  | 16 286<br>488 580                 | 234.65<br>1 407.89               |
|             | R-KV   | Fixed – 1024<br>Fixed – 1024<br>Fixed – 1536                      | 93.75<br>93.75<br>90.63 | 1<br>402 (max)<br>287 (max)         | 80.95<br>3 188.82<br>2 447.61    | 16 286<br>6 546 972<br>4 674 082  | 201.18<br>2 053.10<br>1 909.65   |
|             |        | Ratio – 10% – 1 638<br>Ratio – 34% – 5 570<br>Ratio – 54% – 8 847 | 90.00<br>66.00<br>46.00 | 271 (max)<br>82 (max)<br>46 (max)   | 2 300.28<br>797.43<br>584.77     | 4 413 506<br>1 335 452<br>749 156 | 1 918.68<br>1 674.70<br>1 281.12 |

Table 1: Memory saving, throughput, and decoding-time comparison for Llama3-8B under various generation length and KV cache compression budget settings.

**Computation Overhead** While R-KV introduces additional computation for importance and redundancy scoring, the total overhead is modest and often outweighed by the reduced attention cost over a compressed KV cache. This trade-off becomes increasingly favorable as sequence length grows. Complexity comparisons can be found in Appendix C.1

**Real-time analysis** We present the real-time analysis of memory saving and end-to-end throughput improvement in Table 1. When the batch size is 1, R-KV exhibits a slight throughput advantage over FullKV. This suggests that the acceleration achieved by R-KV through reduced attention computation outweighs computational overhead of R-KV. However, this direct speedup constitutes a minor portion of the overall benefit. The primary throughput improvement from R-KV stems from enabling significantly larger inference batch sizes due to KV cache compression.

We evaluate end-to-end throughput under both ratio-based and fixed KV cache budgets. R-KV consistently enables much larger batch sizes and higher throughput than FullKV, with benefits becoming more pronounced at longer sequence lengths. For example, at a sequence length of 16K, R-KV achieves up to  $9\times$  larger batch sizes and over  $6.6\times$  higher throughput under a 10% compression ratio, and  $13.4\times$  larger batch sizes with  $9.2\times$  throughput under a fixed budget of 1024. Detailed analysis are provided in Appendix C.2.

## 6 Related Work

**KV Cache Compression** The optimization of KV cache memory efficiency in LLMs has garnered increasing attention as model sizes and context windows expand. Existing approaches primarily fall into three categories: dynamic token eviction[3, 11, 12], quantization[13, 14, 15], merging[16, 17, 18], and low-rank decomposition[19, 20, 21]. Previous eviction methods like SnapKV[3], PyramidKV[5], Ada-KV[22], HeadKV[6] dynamically prune tokens based on attention scores, but mainly focus on evicting tokens for prefilling stage. StreamingLLM[7] and H2O[4] are proposed for decoding. However, these general-purpose techniques often struggle with reasoning-intensive tasks, where aggressive eviction risks disrupting critical intermediate steps in CoT, and suffers from reasoning models' inherent redundency.

**Efficient Reasoning** Recent works in efficient reasoning focus on training the model to generate less CoT without sacrificing performance. [23, 24, 25] use RL optimization with length penalty rewards to encourage models to produce more concise chains-of-thought (CoT). [26, 27] employs variable-length CoT datasets to supervised fine-tune (SFT) the LLM to reduce token usage while preserving reasoning correctness. Both RL and SFT methods require additional training. [27, 28, 29] use test-time prompting to reduce generation length, but these methods may hurt the performance. As a KV cache compression work for reasoning models, R-KV is able to achieve lossless compression without extensive training and prompting.

