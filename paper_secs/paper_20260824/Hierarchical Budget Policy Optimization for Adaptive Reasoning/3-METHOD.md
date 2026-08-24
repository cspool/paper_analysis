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

