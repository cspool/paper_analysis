# 5 ANALYSIS

## 5.1 ANALYSIS OF HIERARCHICAL STRUCTURE

<span id="page-8-0"></span>

| Configuration          | GSM8K |            | Math500 |            | Olympiad |            | AIME25 |            | Average |        |
|------------------------|-------|------------|---------|------------|----------|------------|--------|------------|---------|--------|
|                        | Acc   | Tokens Acc |         | Tokens Acc |          | Tokens Acc |        | Tokens Acc |         | Tokens |
| Single (b=1536)        | 85.6  | 327        | 83.4    | 1,055      | 48.1     | 2,301      | 22.2   | 3,686      | 59.8    | 1,842  |
| Dual (b ∈ {512, 2560}) | 86.4  | 816        | 85.6    | 1,849      | 48.2     | 2,938      | 27.8   | 4,104      | 61.7    | 2,427  |
| 4-budget               | 87.6  | 790        | 86.2    | 1,818      | 50.0     | 2,861      | 31.1   | 3,988      | 63.7    | 2,364  |
| 6-budget               | 87.0  | 809        | 87.2    | 1,893      | 50.9     | 3,084      | 26.7   | 3,934      | 62.9    | 2,430  |
| 8-budget               | 87.4  | 864        | 85.6    | 1,836      | 49.9     | 2,899      | 28.9   | 4,019      | 62.9    | 2,405  |

Table 3: Impact of hierarchical granularity on performance. The 4-budget configuration achieves optimal balance between and within-group learning and exploration diversity.

## Optimal hierarchy emerges from balancing intra-group learning and inter-group exploration.

To understand the impact of hierarchical structure on performance, we systematically analyze different budget configurations while maintaining a constant average budget of 1,536 tokens. Table [3](#page-8-0) reveals a clear performance progression: single-budget training achieves only 59.8% average accuracy, demonstrating the limitations of uniform exploration. The performance improves to 61.7% with dual budgets and reaches an optimal of 63.7% with our 4-budget configuration.

Single-budget training reduces to traditional uniform sampling without inter-budget reward differentiation. Dual budgets introduce basic differentiation between short (512) and long (2,560) reasoning, improving accuracy by 1.9%. The 4-budget configuration achieves optimal performance by offering sufficient granularity for adaptive learning, while ensuring enough samples per subgroup to support effective intra-group optimization. Further increasing the number of budgets to 6 or 8 slightly degrades performance, with a 0.8% drop, as fewer samples per subgroup weaken intragroup learning signals. This reveals a fundamental trade-off: exploration diversity must be balanced with statistical reliability for effective policy learning.

HBPO achieves efficiency through adaptive resource allocation rather than uniform compression. As results shown in Table [4,](#page-8-1) traditional GRPO with cosine reward achieves some efficiency (average 1,150 tokens) but suffers significant accuracy degradation, particularly on complex tasks where it achieves only 23.3% on AIME25. The model learns to generate universally short responses regardless of problem requirements, a form of mode collapse that sacrifices capability for efficiency.

<span id="page-8-1"></span>Table 4: Comparison with traditional efficient reasoning methods under natural inference conditions.

| Method                    | GSM8K |        |      | MATH500 |      | Olympiad | AIME25 |        |
|---------------------------|-------|--------|------|---------|------|----------|--------|--------|
|                           | Acc   | Tokens | Acc  | Tokens  | Acc  | Tokens   | Acc    | Tokens |
| Classic Reward            | 86.2  | 661    | 86.2 | 1,605   | 49.1 | 3,174    | 24.4   | 4,309  |
| Cosine Reward             | 83.0  | 195    | 77.6 | 478     | 42.0 | 1,271    | 23.3   | 2,657  |
| HBPO(Budget-aware Reward) | 87.6  | 790    | 86.2 | 1,818   | 50.0 | 2,861    | 31.1   | 3,988  |

Figure [3](#page-9-0) presents the training dynamics of entropy, mean generating length, and validation on the Math500 dataset, highlighting the advantages of hierarchical structures and budget-aware reward mechanism. HBPO (4-budget) setting significantly increases entropy throughout training, outperforming both the dual-budget and single-budget baselines. This suggests that a more finegrained budget hierarchy encourages more diverse and effective exploration, thereby preventing exploration collapse. When comparing cosine reward to HBPO(budget-aware reward), the cosine reward leads to a sharp drop in generation length during the early training stages (steps 0–100), which results in excessive compression and poor generalization on the Math500 validation set.

<span id="page-9-0"></span>> **[图片提取文字 (无描述)]:**
> Entropy for Different Hierarchical Structures Training: Mean Generating Length Validation: Accuracy on Math500 Validation: Token Count Distribution 0.7 Cosine Reward Cosine Reward 3500 2500 Budget-aware Reward 0.84 Budget-aware Reward Single Budget - Dual Budget 3000 0.6 0.82 HBPO(4 Budget) 2000 # 2500 0.80 g Token 2000 å 1000 0.76 0.3 1500 500 0.74 Cosine Reward 1000 Budget-aware Reward 100 400 600 200 400 500 600 100 200 500 600 60 180 300 540 Training Steps Training Steps Training Steps Training Steps
![](_page_9_Figure_1.jpeg)

Figure 3: Training dynamics. (Left) Entropy Comparison of different hierarchical structures. (Right) Comparison of training dynamics and validation performance between cosine and budget-aware reward methods.

In contrast, HBPO maintains a stable average generation length of approximately 1,400 tokens. This stability stems from its hierarchical structure, which encourages effective exploration through budget-aware rewards rather than uniform compression. As a result, the model gradually discovers the most efficient reasoning length on the Math500 validation set during training and consistently improves its validation accuracy.

#### 5.2 REASONING PATTERN ANALYSIS

**HBPO develops different reasoning strategies based on problem complexity.** To understand how models improve efficiency, we analyze reasoning patterns through two lenses: the proportion of exploratory thinking versus direct solution generation, and the frequency of reflection keywords that indicate deliberative processes. Figure 4 reveals striking differences between methods.

<span id="page-9-1"></span>> **[图片提取文字 (无描述)]:**
> **Efficiency Prompting** Natural Reasoning L1 (Thinking) L1 (Solution) HBPO (Thinking) HBPO (Solution) Autothink (Thinking) Autothink (Solution) Baseline Token Distribution **Keyword Distribution Keyword Distribution** 54.1 40.0 7435 40 -50 -7000 -35.0 35 -31.7 32.1 6000 -29.8 40 -30 -Keyword Count - 02 - 05 Count 5000 -Token Count 4501 24.0 25 -3988 4000 -Keyword 3260 3163 3094 2861 20.3 3000 -100% 15 -2154 11.0 13.513.1 999 12.8 1818 9.1 2000 -10.6 10 -10.2 89% 5.5 6.2 92% 87% 1171 10 -6.6 670 <sup>790</sup> 1000 -83% 600 1.8 Math500 Olympiad Bench AIME25 GSM8K Math500 Olympiad Bench AIME25 GSM8K GSM8K Math500 Olympiad Bench AIME25
![](_page_9_Figure_6.jpeg)

Figure 4: Reasoning pattern analysis across methods and problem difficulties. Thinking proportions and reflection keyword frequencies show HBPO's adaptive adjustment, with keywords properly contained within thinking segments.

HBPO exhibits clear adaptation to problem difficulty. The proportion of thinking content increases monotonically from 81% on GSM8K to 89% on AIME25, while reflection keywords (wait, alternatively, but, remember, check, and verify) rise from 6 to 30 occurrences per problem. This pattern supports our differentiated reward design, showing that the model learns to identify when longer reasoning adds value.

L1-Max improves efficiency through uniform length control, maintaining nearly constant thinking proportions (90-92%) and keyword frequencies (29-32) across three datasets. This rigidity reveals mechanical optimization rather than intelligent adaptation. AutoThink attempts adaptive reasoning but exhibits problematic patterns: excessive thinking on simple problems (1171 tokens on GSM8K) and insufficient adjustment for complex ones. Moreover, AutoThink exhibits an average of 1.8 and 4.0 reasoning-related keywords per problem in the solution segments on the MATH500 and Olympiad benchmarks, indicating that reasoning processes leak into what should be direct answers.

The efficiency prompting setting provides further insight into adaptive capabilities. When instructed to minimize tokens, HBPO exhibits progressive keyword scaling (1.8 on GSM8K to 13.1 on AIME25), demonstrating that the model has internalized problem-complexity relationships. L1- Max, when explicitly prompted to "think for 1024 tokens", shows minimal variation (10.6 to 13.5), revealing its inability to differentiate between problem requirements even under explicit efficiency instructions. These patterns confirm that hierarchical training enables genuine adaptive reasoning rather than uniform optimization.

Generalization to scientific reasoning validates domain-agnostic efficiency learning. To assess whether hierarchical exploration enables general efficiency principles rather than task-specific optimization, we evaluate on GPQA-Diamond, a challenging scientific reasoning benchmark outside our training domain. Table [5](#page-10-5) shows that HBPO maintains the highest accuracy (34.72%) while reducing token usage by 55% compared to baseline. This performance on out-of-distribution tasks demonstrates that hierar-

<span id="page-10-5"></span>Table 5: Performance on GPQA-Diamond

| Model      | Acc   | Tokens |  |  |
|------------|-------|--------|--|--|
| DeepScaleR | 33.84 | 4,762  |  |  |
| L1-Max     | 33.33 | 1,227  |  |  |
| AutoThink  | 34.41 | 3,787  |  |  |
| HBPO       | 34.72 | 2,101  |  |  |

chical training teaches fundamental principles of computational resource allocation that transfer across reasoning domains.

These analyses collectively demonstrate that HBPO's hierarchical exploration framework addresses the fundamental challenges in efficient reasoning. By maintaining exploration diversity through budget hierarchies and enabling adaptive learning through differentiated rewards, HBPO teaches models to recognize the computational requirements of different problems and allocate resources accordingly. The result is a system that achieves efficiency not through constraint but through understanding.

