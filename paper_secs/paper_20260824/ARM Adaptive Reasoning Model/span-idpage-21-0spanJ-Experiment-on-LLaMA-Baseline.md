# <span id="page-21-0"></span>J Experiment on LLaMA Baseline

Based on the superior performance of the Qwen model and its inherent adaptability to reinforcement learning [\[55;](#page-12-10) [48\]](#page-12-12), we adopt Qwen as the backbone model in our experiments. To examine the generality of our method, we further train models on LLaMA-3.2-3B [\[10\]](#page-9-13) and report the results in Table [10.](#page-21-1) Consistent with our main findings, ARM achieves comparable performance to the GRPO baseline while using fewer tokens across diverse task domains and complexity levels, demonstrating that our method can generalize to different backbone models.

✝ ✆

We also note that token reduction is less pronounced on LLaMA than on Qwen (e.g., 15.7% vs. 55.2% on GSM8K). Upon closer analysis, we find this is caused by repetitive outputs on occasion produced by the LLaMA-based model—a phenomenon also observed in prior work [\[48\]](#page-12-12)—which may lead to longer response lengths. This discrepancy may stem from differences in model architecture or pretraining data, and we leave further investigation to future work.

<span id="page-21-1"></span>Table 10: Comparison of the GRPO baseline and ARM on LLaMA-3.2-3B [\[10\]](#page-9-13) across CSQA, GSM8K, and AIME'25. ARM achieves similar accuracy while reducing token usage.

| LLaMA-3.2-3B  |             |                      | CSQA              |                      | GSM8K             | AIME'25           |                      |  |
|---------------|-------------|----------------------|-------------------|----------------------|-------------------|-------------------|----------------------|--|
|               | k           | Acc.                 | Tok.              | Acc.                 | Tok.              | Acc.              | Tok.                 |  |
| GRPO Baseline | 1<br>8<br>1 | 76.4<br>76.8<br>76.2 | 347<br>350<br>158 | 87.5<br>90.3<br>86.1 | 677<br>662<br>546 | 3.3<br>3.3<br>3.3 | 4616<br>4375<br>3534 |  |
| ARM           | 8           | 76.5                 | 162               | 89.8                 | 558               | 3.3               | 3713                 |  |
| ∆             |             | -0.3                 | -53.7%            | -0.5                 | -15.7%            | 0                 | -15.1%               |  |

<span id="page-22-3"></span>Table 11: Performance of L1-Exact under different specified token budgets across benchmarks. "Spec." indicates the user-specified reasoning budget in tokens.

|       | CSQA |      | AIME |      | MA   | ATH  | Al   | МC   | olympiad_bench |      |  |
|-------|------|------|------|------|------|------|------|------|----------------|------|--|
| Spec. | Acc. | Tok. | Acc. | Tok. | Acc. | Tok. | Acc. | Tok. | Acc.           | Tok. |  |
| 512   | 45.8 | 328  | 3.3  | 623  | 71.0 | 590  | 47.0 | 641  | 31.7           | 608  |  |
| 1024  | 46.6 | 589  | 6.7  | 1291 | 77.2 | 1182 | 45.8 | 1283 | 37.2           | 1184 |  |
| 2048  | 46.0 | 2004 | 13.3 | 1935 | 79.6 | 1751 | 55.4 | 1950 | 39.7           | 1813 |  |
| 3600  | 46.1 | 4747 | 26.7 | 3696 | 81.8 | 3478 | 72.3 | 3525 | 43.7           | 3460 |  |

Table 12: Comparison between L1 and ARM across multiple benchmarks.

<span id="page-22-5"></span>

|           |        |              | Ea         | asy          |            |              | Medium     |              |            |              |            |              |            |              | Hard         |              | vg.          |  |
|-----------|--------|--------------|------------|--------------|------------|--------------|------------|--------------|------------|--------------|------------|--------------|------------|--------------|--------------|--------------|--------------|--|
| 7B Models |        | CS           | QA         | OB           | QA         | GSN          | 18K        | MA           | TH         | SVA          | MP         | BE           | ВН         | AIM          | E'25         |              | 8-           |  |
|           | k      | Acc.         | Tok.       | Acc.         | Tok.       | Acc.         | Tok.       | Acc.         | Tok.       | Acc.         | Tok.       | Acc.         | Tok.       | Acc.         | Tok.         | Acc.         | Tok.         |  |
| L1        | 1 8    | 62.4<br>65.6 | 232<br>234 | 69.2<br>74.8 | 341<br>345 | 89.8<br>92.9 | 273<br>272 | 85.9<br>88.6 | 943<br>944 | 89.7<br>91.3 | 231<br>231 | 64.6<br>69.9 | 628<br>628 | 30.0<br>33.3 | 3949<br>3964 | 70.2<br>73.8 | 942<br>945   |  |
| Arm       | 1<br>8 | 66.3<br>67.2 | 237<br>234 | 68.6<br>69.6 | 316<br>322 | 90.1<br>93.9 | 311<br>306 | 85.6<br>93.1 | 945<br>933 | 90.7<br>93.3 | 251<br>242 | 65.6<br>71.8 | 617<br>623 | 40.0<br>40.0 | 5413<br>5858 | 72.4<br>75.6 | 1156<br>1217 |  |

### <span id="page-22-0"></span>**K** Further Discussion on Length-Penalty Strategies

### **K.1** Implementations

To ensure fair comparisons, we follow the official settings of L1 [1] and THINKPRUNE [16], adopting their specified minimum allowed lengths when evaluating on easy tasks. We set the temperature to 0.6 and top-p to 0.95, consistent with both papers. Specifically, we use L1-Qwen-1.5B-Exact<sup>5</sup> at 512 tokens for L1 and DeepSeek-R1-Distill-Qwen-1.5B-thinkprune-iter2k<sup>6</sup> for THINKPRUNE.

#### <span id="page-22-6"></span>**K.2** Inaccurate Estimation of Length-Penalty Strategies

We provide an in-depth analysis of L1 here, which applies a length penalty strategy during training. At inference time, users are required to explicitly specify token budgets in the instructions for the task. We follow the official setting and additionally extend the evaluation to the CSQA benchmark. Results are presented in Table 11. We clarify our claims as follows:

Users' inaccurate token estimation hurts performance, since the length penalty strategy assumes prior knowledge of the task to predefine the appropriate reasoning length. If the user's estimation is not accurate enough, the performance degradation is significant: 1) **Underestimated budgets hurt performance on complex tasks.** On harder benchmarks like AIME, performance is severely degraded under low budgets: only 3.3% at 512 tokens, improving gradually to 26.7% at 3600 tokens. Such under-estimation of required reasoning length severely limits performance on challenging benchmarks. 2) **Large budgets waste resources on simple tasks.** On easier benchmarks like CSQA, enforcing large reasoning budgets leads to token usage increases with minimal, or even detrimental, gains. For example, CSQA accuracy rises marginally from 45.8% at 512 tokens to 46.1% at 3600 tokens. In contrast, ARM learns to allocate longer reasoning only when necessary, avoiding both under- and overestimation.

#### K.3 Further Comparison with L1

We further conduct a comparison between ARM-7B and L1-Qwen-7B-Exact<sup>7</sup>. For fairness, we align the backbone model by using DS-R1-Distill for both models. During L1 inference, we set token budgets to match ARM's average token usage on each dataset for fair comparison.

<span id="page-22-1"></span><sup>&</sup>lt;sup>5</sup>https://huggingface.co/l3lab/L1-Qwen-1.5B-Exact

<span id="page-22-2"></span><sup>&</sup>lt;sup>6</sup>https://huggingface.co/Shiyu-Lab/DeepSeek-R1-Distill-Qwen-1.5B-thinkprune-iter2k

<span id="page-22-4"></span><sup>&</sup>lt;sup>7</sup>https://huggingface.co/l3lab/L1-Qwen-7B-Exact

As shown in Table [12,](#page-22-5) ARM consistently outperforms L1 with similar token usage, demonstrating the advantage of adaptive reasoning over length constraints. Notably, L1 requires human intervention to set the token budget manually, and an inaccurate assignment of the token budget would bring a performance drop (As detailed in Appendix [K.2\)](#page-22-6). In contrast, ARM autonomously adjusts its reasoning length based on task complexity through format selection, enabling better efficiencyperformance trade-offs without manual token tuning.

