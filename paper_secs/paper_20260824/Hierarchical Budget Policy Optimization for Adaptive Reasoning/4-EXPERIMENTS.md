# 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Datasets and Models. We evaluate HBPO on mathematical reasoning tasks using the DeepScaleR dataset [\(Luo et al.,](#page-11-9) [2025\)](#page-11-9) for training, which comprises 40K high-quality mathematical problems from AIME, AMC, Omni-Math [\(Gao et al.,](#page-11-10) [2025\)](#page-11-10), and STILL [\(Min et al.,](#page-12-10) [2024\)](#page-12-10). We employ two base models: DeepSeek-R1-Distill-Qwen-1.5B [\(DeepSeek-AI,](#page-11-0) [2025\)](#page-11-0) and DeepScaleR-Preview-1.5B [\(Luo et al.,](#page-11-9) [2025\)](#page-11-9).

Implementation Details. We implement HBPO using the VeRL framework [\(Sheng et al.,](#page-12-11) [2024\)](#page-12-11) with a context window of 4,096 tokens during training. Following DAPO [\(Yu et al.,](#page-12-12) [2025\)](#page-12-12), we set clipping thresholds ϵhigh = 0.28 and ϵlow = 0.2, with KL divergence disabled to encourage exploration. Training proceeds for one epoch (629 steps) with a learning rate of 10<sup>−</sup><sup>6</sup> and batch size of 64. For hierarchical exploration, we generate 16 rollouts per query, partitioned equally into 4 subgroups with budget constraints B = 512, 1024, 2048, 2560 tokens.

Evaluation Protocol. We evaluate on four mathematical reasoning benchmarks of increasing difficulty: GSM8K [\(Cobbe et al.,](#page-10-3) [2021\)](#page-10-3), Math500 [\(Lightman et al.,](#page-11-11) [2023\)](#page-11-11), OlympiadBench [\(He](#page-11-12) [et al.,](#page-11-12) [2024\)](#page-11-12), and AIME25. Following standard practice [\(DeepSeek-AI,](#page-11-0) [2025\)](#page-11-0), we use temperature T = 0.6, top p = 0.95, and maximum context length of 32,768 tokens. We report pass@1 accuracy and average token usage under two evaluation settings: (1) natural reasoning where models freely determine their computational effort, and (2) efficiency prompting using *"I will answer the question with minimal tokens"* after <think> to guide models toward efficient responses.

Baselines. We compare against several state-of-the-art efficient reasoning methods: (1) global penalties: HAPO [\(Huang et al.,](#page-11-4) [2025\)](#page-11-4) and TLMRE [\(Arora & Zanette,](#page-10-4) [2025\)](#page-10-4) add length penalties to the RL objective; (2) explicit control: L1-Exact,L1-Max [\(Aggarwal & Welleck,](#page-10-2) [2025\)](#page-10-2), E1 [\(Xu](#page-12-1) [et al.,](#page-12-1) [2025\)](#page-12-1) and ThinkPrune [\(Hou et al.,](#page-11-3) [2025\)](#page-11-3)use RL with explicit length targets. (3) discrete mode selection: AdaptThink [\(Zhang et al.,](#page-12-2) [2025a\)](#page-12-2), AutoThink [\(Tu et al.,](#page-12-6) [2025\)](#page-12-6) AdaR1 (?) and Thinkless [\(Fang et al.,](#page-11-5) [2025\)](#page-11-5) enable binary think/no-think mode selection.

## 4.2 MAIN RESULTS

Hierarchical training enables efficient reasoning without capability trade-offs. Tables [1](#page-7-0) and [2](#page-7-1) present our results under natural and efficiency-constrained settings, respectively. Under natural reasoning conditions, HBPO demonstrates consistent improvements across both base models. Applied to DeepSeek-R1-Distill-Qwen-1.5B, HBPO improves average accuracy from 56.3% to 59.4% while reducing token usage by 60.6% (from 7,921 to 3,120). On the stronger DeepScaleR model, HBPO maintains the baseline's 63.7% accuracy while achieving 50.2% token reduction (from 4,744 to 2,364). Notably, HBPO achieves 31.1% accuracy on AIME25, outperforming the DeepScaleR baseline and all efficiency methods. This improvement on the most challenging benchmark while using fewer tokens demonstrates that hierarchical exploration not only prevents capability degradation but can enhance reasoning by eliminating computational redundancy.

The efficiency prompting setting makes the performance gains from hierarchical training more evident. While baseline models suffer catastrophic degradation when forced to minimize tokens (over 10% accuracy drop), HBPO maintains robust performance. Applied to DeepScaleR, HBPO achieves 59.4% average accuracy with only 947 tokens, matching L1-Max (1024)'s accuracy while using 32% fewer tokens. This indicates that our training enables effective exploration across the entire efficiency spectrum.

<span id="page-7-0"></span>

| Method                              | GSM8K                         |        | Math500 |        |      | Olympiad |      | AIME25 | Average |        |
|-------------------------------------|-------------------------------|--------|---------|--------|------|----------|------|--------|---------|--------|
|                                     | Acc                           | Tokens | Acc     | Tokens | Acc  | Tokens   | Acc  | Tokens | Acc     | Tokens |
| Base: DeepSeek-R1-Distill-Qwen-1.5B |                               |        |         |        |      |          |      |        |         |        |
| Baseline                            | 82.3                          | 1,111  | 81.6    | 4,696  | 42.3 | 10,225   | 18.9 | 15,651 | 56.3    | 7,921  |
| HAPO                                | 80.9                          | 571    | 76.4    | 2,252  | 42.1 | 5396     | 24.4 | 9,230  | 56.0    | 4362   |
| TLMRE                               | 74.6                          | 221    | 69.8    | 1,835  | 35.8 | 4,838    | 17.8 | 9,753  | 49.5    | 4,162  |
| AdaptThink                          | 85.0                          | 816    | 79.6    | 1,220  | 42.9 | 2,501    | 18.9 | 6,813  | 56.6    | 2,838  |
| AutoThink                           | 81.4                          | 739    | 81.4    | 2627   | 44.5 | 5709     | 23.3 | 9,769  | 57.7    | 4,711  |
| AdaR1                               | 79.2                          | 341    | 80.8    | 2,455  | 42.1 | 5,802    | 23.0 | 9,516  | 56.3    | 4,528  |
| HBPO (Ours)                         | 84.5                          | 670    | 80.4    | 2,147  | 45.0 | 4,058    | 27.8 | 5,606  | 59.4    | 3,120  |
|                                     | Base: DeepScaleR-Preview-1.5B |        |         |        |      |          |      |        |         |        |
| Baseline                            | 86.1                          | 1,684  | 87.0    | 2,938  | 51.6 | 5,330    | 30.0 | 9,023  | 63.7    | 4,744  |
| HAPO                                | 84.3                          | 658    | 84.4    | 2,102  | 47.7 | 3,569    | 26.7 | 5,353  | 60.8    | 2,920  |
| ThinkPrune                          | 86.6                          | 659    | 85.2    | 1,757  | 50.6 | 3,122    | 26.7 | 4,816  | 62.3    | 2,589  |
| L1-Exact                            | 86.4                          | 861    | 80.8    | 3685   | 46.0 | 3,478    | 23.3 | 3,285  | 59.1    | 2,827  |
| L1-Max                              | 86.1                          | 670    | 85.0    | 3,260  | 48.2 | 3,094    | 22.2 | 3,163  | 60.4    | 2,547  |
| E1                                  | 85.4                          | 748    | 84.8    | 1,930  | 49.3 | 3,456    | 26.7 | 5,729  | 61.6    | 2,965  |
| AutoThink                           | 85.8                          | 1,171  | 81.0    | 2154   | 48.2 | 4,501    | 30.0 | 7,435  | 61.3    | 3,815  |
| Thinkless                           | 86.4                          | 957    | 85.2    | 3,184  | 50.7 | 5,691    | 25.6 | 8,271  | 62.0    | 4,526  |
| HBPO (Ours)                         | 87.6                          | 790    | 86.2    | 1,818  | 50.0 | 2,861    | 31.1 | 3,988  | 63.7    | 2,364  |

Table 1: Performance under natural reasoning setting. Bold indicates the best and underline indicates the second-best for each metric. HBPO achieves the best performance in terms of the accuracy-efficiency trade-off and exhibits adaptive behavior.

<span id="page-7-1"></span>

| Method                                                   | GSM8K                                           |                            | Math500                      |                              |                              | Olympiad                       | AIME25                      |                                | Average                      |                              |
|----------------------------------------------------------|-------------------------------------------------|----------------------------|------------------------------|------------------------------|------------------------------|--------------------------------|-----------------------------|--------------------------------|------------------------------|------------------------------|
|                                                          | Acc<br>Tokens<br>Acc<br>Tokens<br>Acc<br>Tokens |                            | Acc                          | Tokens                       | Acc                          | Tokens                         |                             |                                |                              |                              |
|                                                          | Base: DeepSeek-R1-Distill-Qwen-1.5B             |                            |                              |                              |                              |                                |                             |                                |                              |                              |
| Baseline<br>HBPO (Ours)                                  | 73.6<br>83.9                                    | 267<br>340                 | 67.4<br>79.6                 | 806<br>732                   | 30.6<br>43.0                 | 1,950<br>1,305                 | 13.3<br>18.9                | 3,737<br>1,454                 | 46.2<br>56.3                 | 1,690<br>958                 |
| Base: DeepScaleR-Preview-1.5B                            |                                                 |                            |                              |                              |                              |                                |                             |                                |                              |                              |
| Baseline<br>L1-Max (512)<br>L1-Max (1024)<br>HBPO (Ours) | 78.6<br>85.7<br>87.6<br>85.6                    | 270<br>331<br>1,188<br>394 | 74.4<br>81.4<br>82.2<br>82.4 | 1,037<br>609<br>1,235<br>726 | 37.2<br>42.0<br>45.4<br>47.2 | 1,963<br>861<br>1,518<br>1,193 | 16.7<br>7.8<br>22.2<br>22.2 | 4,733<br>996<br>1,661<br>1,476 | 51.7<br>54.2<br>59.4<br>59.4 | 2,001<br>699<br>1,401<br>947 |

Table 2: Performance under efficiency prompting setting. HBPO demonstrates robust performance compared to baseline models and the explicit length-controlled method L1, while effectively adhering to efficient prompting instructions.

Adaptive behavior emerges from hierarchical training rather than explicit control. The distinction between HBPO and existing methods becomes evident in their token allocation patterns. L1-Max exhibits remarkably uniform behavior across problem difficulties, using 3,260 tokens on MATH500 and 3,163 tokens on AIME25 despite the significant complexity gap between these benchmarks. In contrast, HBPO demonstrates genuine problem sensitivity with token usage varying from 1,818 on MATH500 to 3,988 on AIME25. This 2.2× variation directly correlates with problem complexity and emerges naturally from the differentiated reward mechanism, which creates distinct optimization landscapes for different budget levels. Through comparative advantage across these landscapes, models learn to assess problem requirements without external guidance.

