# HIERARCHICAL BUDGET POLICY OPTIMIZATION FOR ADAPTIVE REASONING

Shangke Lyu<sup>1,\*</sup>, Linjuan Wu<sup>1,\*</sup>, Yuchen Yan<sup>1</sup>, Xingyu Wu<sup>1</sup>, Hao Li<sup>2</sup>
Yongliang Shen<sup>1</sup>, Peisheng Jiang<sup>2</sup>, Weiming Lu<sup>1</sup>, Jun Xiao<sup>1</sup>, Yueting Zhuang<sup>1</sup>
<sup>1</sup>Zhejiang University <sup>2</sup>SF Technology
{lyusk, wulinjuan525, syl, luwm}@zju.edu.cn

GitHub: https://github.com/zju-real/hbpo
Project: https://zju-real.github.io/hbpo

## **ABSTRACT**

Large reasoning models achieve remarkable performance through extensive chainof-thought generation, yet they suffer from a critical inefficiency: applying uniformly extensive reasoning regardless of problem complexity. We present Hierarchical Budget Policy Optimization (HBPO), a reinforcement learning framework that enables models to learn problem-specific reasoning depths without sacrificing capability. Unlike existing approaches that impose rigid constraints or rely on discrete mode selection, HBPO partitions the exploration space into budget-constrained hierarchies (512-2560 tokens), each with differentiated reward structures that preserve both efficiency incentives and reasoning capabilities. This design addresses a fundamental challenge in efficient reasoning training: traditional length penalties systematically bias models away from necessary long reasoning paths, causing exploration space collapse. Through hierarchical sampling and budget-aware rewards, HBPO maintains exploration diversity while teaching models to recognize when extended deliberation is warranted. Extensive experiments demonstrate that HBPO reduces average token usage by up to 60.6% while improving accuracy by 3.14% across four reasoning benchmarks. Most notably, HBPO exhibits emergent adaptive behavior where models automatically adjust reasoning depth based on problem complexity. Our results suggest that reasoning efficiency and capability are not inherently conflicting, and can be simultaneously optimized through appropriately structured hierarchical training that preserves exploration diversity.

## 1 Introduction

Advances in large reasoning models have led to impressive performance on complex reasoning tasks through chain-of-thought methodologies (OpenAI, 2024; DeepSeek-AI, 2025). However, these models exhibit fundamental inefficiency: they generate unnecessarily long reasoning chains even for simple problems, sometimes consuming thousands of tokens for basic arithmetic (Chen et al., 2025; 2024). This phenomenon reveals a fundamental misalignment, as current reasoning models lack the ability to adapt their computational effort to the actual complexity of problems.

Recent empirical findings challenge the conventional belief that longer reasoning always leads to better outcomes. Research shows that models can maintain competitive accuracy even without intermediate steps (Ma et al., 2025), and in some cases, shorter reasoning paths perform comparably or even better on simpler tasks (Li et al., 2025). This is further supported by stark variations in optimal reasoning lengths across tasks. For instance, L1 (Aggarwal & Welleck, 2025) achieves peak performance with  $\sim$ 1,100 tokens on GSM8K, but requires over 3,000 tokens on OlympiadBench. Such heterogeneity highlights a key insight: the computational requirements for effective reasoning are inherently problem-dependent, yet current models apply uniform reasoning strategies regardless of task complexity.

<sup>\*</sup> The first two authors have equal contributions.

<span id="page-1-0"></span>> **[图片提取文字 (无描述)]:**
> Length-constrained Method Explicit Ours: Hierarchical Budget Policy Optimization Think for 1k tokens. Constrained  $A_n$ Budget 1 Reward Design 512 512 Length Penalty 00 Budget 2 Discrete mode method 1024 1024 1024 Hierarchical Budget-aware ... Budget A1 Budget 3 2048 2048 reward 2048 think think Budget 4 2560 2560 2560 no-think no-think
![](_page_1_Figure_1.jpeg)

Figure 1: HBPO provides budget-aware reward through hierarchical budget exploration, which enables fine-grained adaptive reasoning. While length-constrained methods use global constraint or length penalty, and discrete mode methods dichotomize problem difficulty, HBPO partitions the exploration space into budget-constrained hierarchies (512, 1024, 2048, 2560 tokens). This structure maintains reasoning diversity throughout training, enabling emergent adaptive behavior where models match computational resources to problem complexity.

To address these inefficiencies, an increasing number of studies aim to improve the inference efficiency of reasoning models. Current approaches fall into two primary categories. *Length-constrained methods* directly constrain generation through explicit mechanisms or incorporate length penalties into training objectives: prompts like "think for n tokens" and corresponding length-control rewards in L1 (Aggarwal & Welleck, 2025); progressively limits on the model's reasoning space during training in ThinkPrune (Hou et al., 2025); enforces budget constraints through forced termination in Scalable Chain of Thoughts (Xu et al., 2025); and HAPO (Huang et al., 2025) leverages history-aware optimization to track minimal sufficient reasoning lengths. *Discrete mode methods* dichotomize problem difficulty and omit the reasoning process for simple instances, which enables the model to operate in a think/no-think manner. Thinkless (Fang et al., 2025) first performs format training for mode switching via fine-tuning. AdaptThink (Zhang et al., 2025a) employs importance sampling to enable the model to switch between reasoning patterns. While effective at reducing token usage, these methods share a key limitation: they prioritize efficiency or mode selection at the cost of accuracy performance, lacking fine-grained mechanisms for models to autonomously decide appropriately efficient reasoning length.

We identify two key challenges that hinder existing methods from achieving genuine reasoning efficiency. **First, length penalties introduce systematic training biases that impair reasoning capabilities.** In standard reinforcement learning settings (DeepSeek-AI, 2025), correct solutions receive equal rewards regardless of length, allowing for unbiased exploration. However, length penalties disrupt this balance by consistently favoring shorter outputs, leading policies to gradually abandon long-reasoning strategies (Hou et al., 2025; Huang et al., 2025; Lou et al., 2025). **Second, static efficiency constraints fail to capture the continuous nature of reasoning complexity.** Even adaptive methods rely on coarse mechanisms, such as binary think/no-think decisions (Zhang et al., 2025a; Fang et al., 2025) or fixed confidence thresholds (Qiao et al., 2025), which overlook the nuanced relationship between problem characteristics and computational requirements.

These limitations raise a fundamental question: rather than enforcing uniform constraints, can models learn differentiated reasoning strategies through structured exploration? This question motivates our study of hierarchical budget exploration, where efficiency emerges not from rigid control but from structured exploration within budget-constrained subspaces.

We propose **Hierarchical Budget Policy Optimization** (**HBPO**) illustrated in Figure 1, a reinforcement learning framework that enables models to learn problem-specific reasoning strategies while retaining their ability to perform complex reasoning. The core idea is to partition the exploration space into multiple budget-constrained subgroups, allowing models to preserve reasoning diversity and uncover natural alignments between problem characteristics and required computational effort. Specifically, HBPO employs a hierarchical sampling strategy that partitions rollout samples into subgroups, each governed by a distinct token budgets. We implement this by inserting length prompts (e.g., "I will answer the question within n tokens") after the reasoning tag, thereby constructing multiple exploration spaces with budgets ranging from 512 to 2560 tokens. Unlike uniform sampling, this structure encourages the model to explore both concise and extended

reasoning paths throughout training, effectively mitigating the systematic degradation of reasoning capabilities caused by global length penalties.

To enable efficient reasoning within each budget hierarchy, we design a piecewise reward function with distinct behaviors inside and outside budget boundaries. Within the assigned budget, rewards are monotonically non-decreasing to preserve exploratory flexibility. Beyond the budget, cosine decay and length deviation penalties are applied to encourage the model to return to its designated exploration space. This creates a natural gradient of incentives: shorter budgets favor concise solutions with higher rewards, while longer budgets retain standard rewards for extended reasoning. The result is a reward landscape that teaches models not just to reason efficiently within constraints, but to recognize which constraint level matches the problem at hand.

HBPO achieves a superior accuracy-efficiency trade-off compared to existing methods on four reasoning benchmarks. Crucially, it exhibits adaptive behavior by dynamically allocating computational resources based on problem complexity. For example, on GSM8K, it uses only 670 tokens. On AIME25, it uses 5,606 tokens, representing a more than eightfold increase in token usage. In both cases, it improves accuracy by 2.2% and 8.9% compared to the base model DeepSeek-R1- Distill-Qwen-1.5B, demonstrating effective resource allocation.

Our contributions are threefold:

- We introduce Hierarchical Budget Policy Optimization, a reinforcement learning framework that partitions the exploration space into budget-constrained hierarchies with differentiated rewards, preserving reasoning diversity while enabling adaptive resource allocation.
- We demonstrate that uniform efficiency constraints systematically collapse the exploration space and degrade reasoning capabilities, validating the necessity of structured exploration for maintaining model performance.
- We provide evidence of emergent adaptive reasoning, where HBPO-trained models automatically adjust reasoning depth based on problem characteristics, achieving up to 60.6% reduction in token usage while improving accuracy by 3.14% across mathematical reasoning benchmarks.

# 2 RELATED WORKS

# 2.1 EFFICIENT REASONING

Recent advances in reasoning models have spurred various efforts to reduce computational overhead while preserving performance. Existing approaches can be broadly categorized into three types: Length-constrained methods explicitly restrict generation through predefined mechanisms. For example, L1 [\(Aggarwal & Welleck,](#page-10-2) [2025\)](#page-10-2) introduces token budget prompts with corresponding rewards; ThinkPrune [\(Hou et al.,](#page-11-3) [2025\)](#page-11-3) progressively tightens constraints via iterative training; and Scalable Chain of Thoughts [\(Xu et al.,](#page-12-1) [2025\)](#page-12-1) separates the thinking and solution phases, each with its budget. While effective in limiting token usage, these methods require manual budget specification and lack adaptability to varying problem complexity. Reward-based methods incorporate efficiency into training objectives more implicitly. HAPO [\(Huang et al.,](#page-11-4) [2025\)](#page-11-4) incentivizes concise reasoning by tracking minimal correct response lengths, while "Think When You Need" [\(Jiang et al.,](#page-11-7) [2025\)](#page-11-7) balances brevity and quality through pairwise comparisons and adaptive target lengths. These approaches offer finer control but still impose global objectives across diverse problem types, limiting flexibility. Training-free approaches [\(Muennighoff et al.,](#page-12-4) [2025;](#page-12-4) [Yang et al.,](#page-12-5) [2025\)](#page-12-5) intervene at inference time through symbolic control tokens or confidence-based early stopping. While costeffective, these methods are heuristic-driven and lack learning-based adaptation. Despite their differences, all these approaches share a fundamental limitation: they treat efficiency as a uniform constraint, overlooking the fact that optimal reasoning length varies significantly with problem complexity.

# 2.2 ADAPTIVE REASONING

Recognizing heterogeneous reasoning requirements, recent work explores adaptive strategies that adjust computational effort based on problem characteristics. Binary mode selection represents the

<span id="page-3-0"></span>> **[图片提取文字 (无描述)]:**
> Policy Hierarchical Budget Group Reward query Model Rollout Mechanism Computation Inter-subgroup Intra-subgroup Reward Reward In how many ways can 8 Token usage Budget limit 8 5 5 5 5 people sit around a round table if 3 of the people - $R(b|n_{gen})$ Pierre, Rosa, and Thomas - all (PO) 6 want to sit together? Two seatings are considered the same if one is a rotation of the other. (ground truth:720) ...The answer is 360. (20 tokens)  $A_{1.1}$ Answer with To determine the number of ways 8 people can sit around a round ... budget limit ... Thus, the total number of 512 tokens.  $A_{1.2}$ arrangements is the product of these two results:  $(6-1)! \times 3! = 5! \times 3! = 120 \times 6 = 720$ . (2000 tokens) ...Within the block, Pierre, Rosa, and Thomas can be arranged in 3! ways. ...The answer is 720. (800 tokens)  $A_{2,1}$ Answer with If we have 8 people and we're grouping 3 of them together... we have 5 individuals plus the block, totaling 6  $A_{2.2}$ 1024 tokens. units to arrange around the table. ... budget limit ... The answer is 720. (1200 tokens) ... Wait, let's take an example. Let's say n=8, k=3. Let's fix Pierre's position. Then Rosa and Thomas must be on  $A_{3,1}$ either side of Pierre. ... The answer is 720. (1100 tokens) Answer with 2048 tokens.  $A_{3,2}$ ...So 4! \* 3! = 144. (100 tokens) ...The number of distinct seating arrangements is (n - 1)! instead of n!...The answer is 720. (1000 tokens)  $A_{4,1}$ Answer with ...Alternatively, another approach: If we fix one person to account for the circular arrangement, maybe that's 2560 tokens.  $A_{4.2}$ another way to think. Let's say we fix Pierre's position. ...The answer is 720. (2200 tokens)
![](_page_3_Figure_1.jpeg)

Figure 2: Overview of Hierarchical Budget Policy Optimization. Given a query, HBPO generates responses across multiple budget-constrained subgroups (512, 1024, 2048, 2560 tokens), each guided by a piecewise reward function that preserves exploration within budgets while penalizing excess through deviation penalties. The advantage computation decomposes into intra-subgroup advantages (comparing responses against budget-specific baselines) and inter-subgroup advantages (enabling cross-budget learning through global comparison). This hierarchical structure enables models to learn efficient reasoning within constraints and adaptive budget selection based on problem complexity.

most common approach, with models choosing between thinking and non-thinking modes [\(Lou](#page-11-6) [et al.,](#page-11-6) [2025;](#page-11-6) [Zhang et al.,](#page-12-2) [2025a;](#page-12-2) [Fang et al.,](#page-11-5) [2025\)](#page-11-5). These methods employ various techniques including selective loss masking, simplified mode definitions, and decoupled optimization to prevent mode collapse. Multi-stage training strategies [\(Jiang et al.,](#page-11-7) [2025;](#page-11-7) [Tu et al.,](#page-12-6) [2025;](#page-12-6) [Zhang et al.,](#page-12-7) [2025b\)](#page-12-7) use sophisticated reward designs and batch-level balancing to achieve better mode distributions. Beyond binary selection, multi-modal approaches define richer reasoning taxonomies: ARM [\(Wu](#page-12-8) [et al.,](#page-12-8) [2025\)](#page-12-8) uses four modes with adaptive scaling, while PATS [\(Wang et al.,](#page-12-9) [2025\)](#page-12-9) enables steplevel switching between complexity levels. Some methods introduce auxiliary components like regression models for mode prediction [\(Liang et al.,](#page-11-8) [2025\)](#page-11-8) or self-budgeting mechanisms [\(Li et al.,](#page-11-2) [2025\)](#page-11-2). While these adaptive approaches demonstrate significant efficiency gains, they operate within discrete categories rather than enabling continuous adaptation. Complex multi-stage procedures and predefined mode taxonomies limit their flexibility and generalization. In contrast, our hierarchical budget exploration framework enables continuous adaptation through a unified policy optimization process. Without relying on manually defined modes or external modules, our approach allows the model to learn problem-specific reasoning depths, leading to emergent adaptive behavior that naturally aligns computational effort with problem complexity.

# 3 METHOD

We present Hierarchical Budget Policy Optimization, as shown in Figure [2,](#page-3-0) which extends the Group Relative Policy Optimization (GRPO) [\(DeepSeek-AI,](#page-11-0) [2025\)](#page-11-0) framework to enable adaptive reasoning through structured exploration. The core innovation lies in partitioning the exploration space into budget-constrained hierarchies and designing differentiated reward mechanisms that preserve reasoning diversity. We first introduce the hierarchical rollout strategy (Section [3.1\)](#page-4-0), then detail the budget-aware reward design (Section [3.2\)](#page-4-1)), and finally describe the training procedure (Section [3.3\)](#page-5-0)).

## <span id="page-4-0"></span>3.1 HIERARCHICAL BUDGET EXPLORATION

The fundamental challenge in efficient reasoning training is that uniform length penalties systematically bias models away from necessary long reasoning paths. To address this, we partition rollout samples into hierarchical subgroups, each operating within distinct token budget constraints. This structure ensures that models maintain exposure to diverse reasoning lengths throughout training.

Given a query q, we generate n rollout samples and partition them into k subgroups {G1, G2, ..., Gk}, where each subgroup G<sup>i</sup> is associated with a token budget b<sup>i</sup> . We implement this through budget-specific prompts embedded after the reasoning tag: "I will answer the question within b<sup>i</sup> tokens". The budget values form an ascending sequence (b<sup>1</sup> < b<sup>2</sup> < ... < bk), spanning from compact reasoning (e.g., 512 tokens) to extended deliberation (e.g., 2560 tokens).

This hierarchical structure serves two key purposes. First, it prevents exploration space collapse, a common issue in efficiency training where models abandon long reasoning. By preserving separate exploration spaces, HBPO ensures sampling across diverse reasoning lengths. Second, it enables structured comparative learning: the model discovers the suitable computation for each problem by contrasting performance across budget levels, rather than relying on global optimization.

## <span id="page-4-1"></span>3.2 BUDGET-AWARE REWARD DESIGN

The effectiveness of hierarchical exploration hinges on careful reward design. Existing methods either use uniform rewards—supporting fair exploration but lacking efficiency incentives—or apply global length penalties, which improve efficiency at the cost of reasoning ability. HBPO addresses this trade-off with a piecewise reward function that integrates the strengths of both approaches.

## 3.2.1 INTRA-BUDGET REWARD FUNCTION

Within each budget-constrained subgroup, we design a reward function that balances reason exploration and efficiency. For a given budget b, the reward integrates length-based penalties f<sup>1</sup> that promote token efficiency with classical rewards f<sup>2</sup> that encourage diverse reasoning. The reward is formally defined as:

<span id="page-4-2"></span>
$$R(n_{\rm gen} \mid b) = \begin{cases} f_1(n_{\rm gen}, b), & \text{if correct, } n_{\rm gen} > b, \text{ and } n_{\rm gen} \leq L_{\rm max} \\ f_2(b), & \text{if correct, } n_{\rm gen} \leq b, \text{ and } n_{\rm gen} \leq L_{\rm max} \\ 0, & \text{otherwise} \end{cases} \tag{1}$$

where:

$$f_1(n_{\text{gen}}, b) = \beta \cdot \cos\left(\frac{\pi n_{\text{gen}}}{2L_{\text{max}}}\right) - \alpha |n_{\text{gen}} - b|$$
 (2)

<span id="page-4-3"></span>
$$f_2(b) = \beta \cdot \cos\left(\frac{\pi b}{2L_{\text{max}}}\right)$$
 (3)

Here, ngen denotes the number of generated tokens, Lmax is the maximum context length, β is a scaling factor, and α controls deviation sensitivity. The piecewise structure serves distinct purposes across different generation lengths. When ngen > b, the reward follows f1, incorporating both cosine decay and deviation penalty to guide the model back to its designated exploration space. When ngen ≤ b, the reward is bounded by f2, ensuring monotonic non-decreasing behavior that preserves exploration within the budget.

## 3.2.2 INTER-BUDGET REWARD DIFFERENTIATION

The hierarchical structure naturally induces reward differentiation across budgets. For a fixed generation length ngen, different budget assignments yield different rewards according to Equation [1,](#page-4-2) signaled as R(b | ngen). This creates systematic preferences that align with problem complexity.

When ngen < min(bi), all budgets yield rewards determined by f2, and smaller budgets receive higher rewards due to the monotonic decrease of the cosine function over the interval. This preference for smaller budgets on short responses encourages efficiency for simple problems. Conversely, when ngen > max(bi), larger budgets provide higher rewards through smaller deviation

## <span id="page-5-1"></span>Algorithm 1 Hierarchical Budget Policy Optimization (HBPO)

```
Require: Initial policy \pi_{\theta_0}, budget levels \mathcal{B} = \{b_1, ..., b_k\}, learning rate \eta
 1: for iteration t = 1, 2, ..., T do
 2:
           Sample batch of queries Q from training data
 3:
           for each query q \in \Omega do
 4:
                 for each budget b_i \in \mathcal{B} do
 5:
                       Generate n/k responses with prompt "I will answer within b_i tokens"
 6:
                       Store responses in subgroup G_i
 7:
                 end for
 8:
                 for each subgroup G_i do
                       Compute rewards \{R_{i,j}\} using Equation 1
 9:
                      Compute intra-subgroup mean reward: \mu_i = \frac{1}{|G_i|} \sum_{j=1}^{|G_i|} R_{i,j} Compute budget rewards R_{b_i} using Equation 3
10:
11:
                       Compute intra-subgroup advantage: A_i^{intra} = \mu_i - R_{b_i}
12:
13:
                 Compute inter-subgroup advantage: A_{i,j}^{\text{inter}} = \frac{R_{i,j} - \frac{1}{n} \sum_{i,j} R_{i,j}}{\text{std}(R)}
Normalize final advantage: A_{i,j} = A_i^{\text{intra}} + A_{i,j}^{\text{inter}}
14:
15:
16:
           Update policy: \theta_{t+1} \leftarrow \theta_t - \eta \nabla_{\theta} \mathcal{L}(\theta_t)
17:
18: end for
```

penalties  $|n_{gen} - b_i|$  in  $f_1$ , preserving the model's ability to engage in extended reasoning when necessary.

As  $n_{\rm gen}$  increases from below  $\min(b_i)$  to above  $\max(b_i)$ , the reward functions corresponding to different budgets transition in relative preference. The intersection points between reward curves represent complexity thresholds where the optimal budget choice transitions. Through comparative advantage across these differentiated rewards, the model learns to match computational resources to problem requirements without explicit complexity labels or external guidance.

#### <span id="page-5-0"></span>3.3 Training Procedure

HBPO extends the standard GRPO framework by incorporating hierarchical sampling and budget-aware advantage computation into the policy optimization process, the algorithm is shown in Algorithm 1. During each training iteration t, the model generates n responses for a given query, which are automatically partitioned into k subgroups based on their associated budget constraints. Each response is generated with an embedded budget prompt "I will answer the question within  $b_i$  tokens", where  $b_i \in \{b_1, b_2, ..., b_k\}$  represents the predetermined budget levels.

The advantage computation leverages the hierarchical structure to enable both efficient reasoning within budgets and adaptive budget selection across problems. For the j-th response in the i-th subgroup, we compute the reward  $R_{i,j}$  using the budget-aware reward function described in Section 3.2. To capture the hierarchical nature of our exploration, we decompose the advantage into two complementary components that guide different aspects of learning.

The intra-subgroup advantage measures how well responses perform relative to their budget expectation:  $A_i^{\text{intra}} = \mu_i - R_{b_i}$ , where  $\mu_i = \frac{1}{|G_i|} \sum_{j=1}^{|G_i|} R_{i,j}$  is the mean reward within subgroup i, and  $R_{b_i}$  represents the budget-specific baseline computed using Equation 3. This term encourages optimization within each budget constraint, teaching the model to reason efficiently given a specific token allocation.

The inter-subgroup advantage enables comparative learning across different budgets:

$$A_{i,j}^{\text{inter}} = \frac{R_{i,j} - \frac{1}{n} \sum_{i,j} R_{i,j}}{\text{std}(R)}$$

$$\tag{4}$$

This term compares each response against the global mean, creating natural preferences for budget selection. Responses from shorter budgets that achieve high rewards receive positive advantages,

while unnecessarily long responses receive negative advantages, teaching the model to match computational effort to problem requirements.

The final advantage combines both components with normalization for stable training:

$$A_{i,j} = A_i^{\text{intra}} + A_{i,j}^{\text{inter}} \tag{5}$$

The policy optimization adopts GRPO's clipped objective to prevent destructive updates:

$$\mathcal{L}(\theta) = -\mathbb{E}_{(s,a) \sim \pi_{\theta_{\text{old}}}} \left[ \min \left( \rho_{\theta}(s,a) A(s,a), \text{clip}(\rho_{\theta}(s,a), 1 - \epsilon_{\text{low}}, 1 + \epsilon_{\text{high}}) A(s,a) \right) \right]$$
 (6)

where ρθ(s, a) = πθ(a|s)/πθold (a|s) represents the probability ratio. The hierarchical advantages Ai,j naturally flow through this objective, enabling the model to improve both within-budget efficiency and cross-budget selection without requiring separate optimization objectives or complex multi-stage training procedures.

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

# 6 CONCLUSION

We introduced Hierarchical Budget Policy Optimization, a framework that enables reasoning models to achieve efficient computation without sacrificing capability. By maintaining diverse exploration through budget-constrained hierarchies and budget-aware rewards, HBPO prevents the exploration collapse and an optimized allocation of the length budget. Our experiments demonstrate that models trained with HBPO significantly reduce inference costs while improving performance, exhibiting adaptive behavior that naturally matches computational effort to problem complexity.

# REFERENCES

<span id="page-10-2"></span>Pranjal Aggarwal and Sean Welleck. L1: controlling how long A reasoning model thinks with reinforcement learning. *CoRR*, abs/2503.04697, 2025. doi: 10.48550/ARXIV.2503.04697. URL <https://doi.org/10.48550/arXiv.2503.04697>.

<span id="page-10-4"></span>Daman Arora and Andrea Zanette. Training language models to reason efficiently. *CoRR*, abs/2502.04463, 2025. doi: 10.48550/ARXIV.2502.04463. URL [https://doi.org/10.](https://doi.org/10.48550/arXiv.2502.04463) [48550/arXiv.2502.04463](https://doi.org/10.48550/arXiv.2502.04463).

<span id="page-10-0"></span>Qiguang Chen, Libo Qin, Jinhao Liu, Dengyun Peng, Jiannan Guan, Peng Wang, Mengkang Hu, Yuhang Zhou, Te Gao, and Wanxiang Che. Towards reasoning era: A survey of long chainof-thought for reasoning large language models. *CoRR*, abs/2503.09567, 2025. doi: 10.48550/ ARXIV.2503.09567. URL <https://doi.org/10.48550/arXiv.2503.09567>.

<span id="page-10-1"></span>Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. Do NOT think that much for 2+3=? on the overthinking of o1-like llms. *CoRR*, abs/2412.21187, 2024. doi: 10.48550/ARXIV.2412.21187. URL [https://doi.org/10.48550/arXiv.2412.](https://doi.org/10.48550/arXiv.2412.21187) [21187](https://doi.org/10.48550/arXiv.2412.21187).

<span id="page-10-3"></span>Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John

- Schulman. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*, 2021.
- <span id="page-11-0"></span>DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *CoRR*, abs/2501.12948, 2025. doi: 10.48550/ARXIV.2501.12948. URL [https://doi.org/](https://doi.org/10.48550/arXiv.2501.12948) [10.48550/arXiv.2501.12948](https://doi.org/10.48550/arXiv.2501.12948).
- <span id="page-11-5"></span>Gongfan Fang, Xinyin Ma, and Xinchao Wang. Thinkless: LLM learns when to think. *CoRR*, abs/2505.13379, 2025. doi: 10.48550/ARXIV.2505.13379. URL [https://doi.org/10.](https://doi.org/10.48550/arXiv.2505.13379) [48550/arXiv.2505.13379](https://doi.org/10.48550/arXiv.2505.13379).
- <span id="page-11-10"></span>Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen, Runxin Xu, Zhengyang Tang, Benyou Wang, Daoguang Zan, Shanghaoran Quan, Ge Zhang, Lei Sha, Yichang Zhang, Xuancheng Ren, Tianyu Liu, and Baobao Chang. Omni-math: A universal olympiad level mathematic benchmark for large language models. In *The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025*. OpenReview.net, 2025. URL [https://openreview.net/forum?id=](https://openreview.net/forum?id=yaqPf0KAlN) [yaqPf0KAlN](https://openreview.net/forum?id=yaqPf0KAlN).
- <span id="page-11-12"></span>Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems, 2024.
- <span id="page-11-3"></span>Bairu Hou, Yang Zhang, Jiabao Ji, Yujian Liu, Kaizhi Qian, Jacob Andreas, and Shiyu Chang. Thinkprune: Pruning long chain-of-thought of llms via reinforcement learning. *CoRR*, abs/2504.01296, 2025. doi: 10.48550/ARXIV.2504.01296. URL [https://doi.org/10.](https://doi.org/10.48550/arXiv.2504.01296) [48550/arXiv.2504.01296](https://doi.org/10.48550/arXiv.2504.01296).
- <span id="page-11-4"></span>Chengyu Huang, Zhengxin Zhang, and Claire Cardie. HAPO: training language models to reason concisely via history-aware policy optimization. *CoRR*, abs/2505.11225, 2025. doi: 10.48550/ ARXIV.2505.11225. URL <https://doi.org/10.48550/arXiv.2505.11225>.
- <span id="page-11-7"></span>Lingjie Jiang, Xun Wu, Shaohan Huang, Qingxiu Dong, Zewen Chi, Li Dong, Xingxing Zhang, Tengchao Lv, Lei Cui, and Furu Wei. Think only when you need with large hybrid-reasoning models. *CoRR*, abs/2505.14631, 2025. doi: 10.48550/ARXIV.2505.14631. URL [https://](https://doi.org/10.48550/arXiv.2505.14631) [doi.org/10.48550/arXiv.2505.14631](https://doi.org/10.48550/arXiv.2505.14631).
- <span id="page-11-2"></span>Zheng Li, Qingxiu Dong, Jingyuan Ma, Di Zhang, and Zhifang Sui. Selfbudgeter: Adaptive token allocation for efficient LLM reasoning. *CoRR*, abs/2505.11274, 2025. doi: 10.48550/ARXIV. 2505.11274. URL <https://doi.org/10.48550/arXiv.2505.11274>.
- <span id="page-11-8"></span>Guosheng Liang, Longguang Zhong, Ziyi Yang, and Xiaojun Quan. Thinkswitcher: When to think hard, when to think fast. *CoRR*, abs/2505.14183, 2025. doi: 10.48550/ARXIV.2505.14183. URL <https://doi.org/10.48550/arXiv.2505.14183>.
- <span id="page-11-11"></span>Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step, 2023.
- <span id="page-11-6"></span>Chenwei Lou, Zewei Sun, Xinnian Liang, Meng Qu, Wei Shen, Wenqi Wang, Yuntao Li, Qingping Yang, and Shuangzhi Wu. Adacot: Pareto-optimal adaptive chain-of-thought triggering via reinforcement learning. *CoRR*, abs/2505.11896, 2025. doi: 10.48550/ARXIV.2505.11896. URL <https://doi.org/10.48550/arXiv.2505.11896>.
- <span id="page-11-9"></span>Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Tianjun Zhang, Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1 preview with a 1.5b model by scaling rl. [https://pretty-radio-b75.notion.site/](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL \ -19681902c1468005bed8ca303013a4e2) [DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL \ -19681902c1468005bed8ca303013a4e2)\ [-19681902c1468005bed8ca303013a4e2](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL \ -19681902c1468005bed8ca303013a4e2), 2025. Notion Blog.
- <span id="page-11-1"></span>Wenjie Ma, Jingxuan He, Charlie Snell, Tyler Griggs, Sewon Min, and Matei Zaharia. Reasoning models can be effective without thinking. *CoRR*, abs/2504.09858, 2025. doi: 10.48550/ARXIV. 2504.09858. URL <https://doi.org/10.48550/arXiv.2504.09858>.

- <span id="page-12-10"></span>Yingqian Min, Zhipeng Chen, Jinhao Jiang, Jie Chen, Jia Deng, Yiwen Hu, Yiru Tang, Jiapeng Wang, Xiaoxue Cheng, Huatong Song, Wayne Xin Zhao, Zheng Liu, Zhongyuan Wang, and Ji-Rong Wen. Imitate, explore, and self-improve: A reproduction report on slow-thinking reasoning systems. *CoRR*, abs/2412.09413, 2024. doi: 10.48550/ARXIV.2412.09413. URL [https:](https://doi.org/10.48550/arXiv.2412.09413) [//doi.org/10.48550/arXiv.2412.09413](https://doi.org/10.48550/arXiv.2412.09413).
- <span id="page-12-4"></span>Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel J. Candes, and Tatsunori Hashimoto. s1: Simple test- ` time scaling. *CoRR*, abs/2501.19393, 2025. doi: 10.48550/ARXIV.2501.19393. URL [https:](https://doi.org/10.48550/arXiv.2501.19393) [//doi.org/10.48550/arXiv.2501.19393](https://doi.org/10.48550/arXiv.2501.19393).
- <span id="page-12-0"></span>OpenAI. Learning to reason with llms. *OpenAI Blog*, 2024. URL [https://openai.com/](https://openai.com/index/learning-to-reason-with-llms/) [index/learning-to-reason-with-llms/](https://openai.com/index/learning-to-reason-with-llms/). Accessed: 2025-07-22.
- <span id="page-12-3"></span>Ziqing Qiao, Yongheng Deng, Jiali Zeng, Dong Wang, Lai Wei, Fandong Meng, Jie Zhou, Ju Ren, and Yaoxue Zhang. Concise: Confidence-guided compression in step-by-step efficient reasoning. *CoRR*, abs/2505.04881, 2025. doi: 10.48550/ARXIV.2505.04881. URL [https://doi.org/](https://doi.org/10.48550/arXiv.2505.04881) [10.48550/arXiv.2505.04881](https://doi.org/10.48550/arXiv.2505.04881).
- <span id="page-12-11"></span>Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. *arXiv preprint arXiv: 2409.19256*, 2024.
- <span id="page-12-6"></span>Songjun Tu, Jiahao Lin, Qichao Zhang, Xiangyu Tian, Linjing Li, Xiangyuan Lan, and Dongbin Zhao. Learning when to think: Shaping adaptive reasoning in r1-style models via multi-stage RL. *CoRR*, abs/2505.10832, 2025. doi: 10.48550/ARXIV.2505.10832. URL [https://doi.org/](https://doi.org/10.48550/arXiv.2505.10832) [10.48550/arXiv.2505.10832](https://doi.org/10.48550/arXiv.2505.10832).
- <span id="page-12-9"></span>Yi Wang, Junxiao Liu, Shimao Zhang, Jiajun Chen, and Shujian Huang. PATS: process-level adaptive thinking mode switching. *CoRR*, abs/2505.19250, 2025. doi: 10.48550/ARXIV.2505. 19250. URL <https://doi.org/10.48550/arXiv.2505.19250>.
- <span id="page-12-8"></span>Siye Wu, Jian Xie, Yikai Zhang, Aili Chen, Kai Zhang, Yu Su, and Yanghua Xiao. ARM: adaptive reasoning model. *CoRR*, abs/2505.20258, 2025. doi: 10.48550/ARXIV.2505.20258. URL <https://doi.org/10.48550/arXiv.2505.20258>.
- <span id="page-12-1"></span>Yuhui Xu, Hanze Dong, Lei Wang, Doyen Sahoo, Junnan Li, and Caiming Xiong. Scalable chain of thoughts via elastic reasoning. *CoRR*, abs/2505.05315, 2025. doi: 10.48550/ARXIV.2505.05315. URL <https://doi.org/10.48550/arXiv.2505.05315>.
- <span id="page-12-5"></span>Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Zheng Lin, Li Cao, and Weiping Wang. Dynamic early exit in reasoning models. *CoRR*, abs/2504.15895, 2025. doi: 10. 48550/ARXIV.2504.15895. URL <https://doi.org/10.48550/arXiv.2504.15895>.
- <span id="page-12-12"></span>Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Weinan Dai, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. DAPO: an opensource LLM reinforcement learning system at scale. *CoRR*, abs/2503.14476, 2025. doi: 10. 48550/ARXIV.2503.14476. URL <https://doi.org/10.48550/arXiv.2503.14476>.
- <span id="page-12-2"></span>Jiajie Zhang, Nianyi Lin, Lei Hou, Ling Feng, and Juanzi Li. Adaptthink: Reasoning models can learn when to think. *CoRR*, abs/2505.13417, 2025a. doi: 10.48550/ARXIV.2505.13417. URL <https://doi.org/10.48550/arXiv.2505.13417>.
- <span id="page-12-7"></span>Xiaoyun Zhang, Jingqing Ruan, Xing Ma, Yawen Zhu, Haodong Zhao, Hao Li, Jiansong Chen, Ke Zeng, and Xunliang Cai. When to continue thinking: Adaptive thinking mode switching for efficient reasoning. *CoRR*, abs/2505.15400, 2025b. doi: 10.48550/ARXIV.2505.15400. URL <https://doi.org/10.48550/arXiv.2505.15400>.