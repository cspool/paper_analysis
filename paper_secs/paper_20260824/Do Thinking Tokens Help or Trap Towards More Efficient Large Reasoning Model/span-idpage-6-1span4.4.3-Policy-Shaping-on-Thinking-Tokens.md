# <span id="page-6-1"></span>4.4.3 Policy Shaping on Thinking Tokens

Our third innovation ensures consistent gradient flow for thinking token suppression by addressing GRPO's gradient clipping limitation.

## GRPO Gradient Clipping Limitation.

As established in Section 4.3, GRPO's policy gradient for token  $\tau_{i,t}$  is zeroed when the importance ratio  $r_i^t$  falls outside the clipping bounds  $[1 - \epsilon, 1 + \epsilon]$ . For thinking tokens in unpreferred trajectories (where  $A_i^t < 0$ ), gradients are preserved only when  $r_i^t > 1 - \epsilon$ , which requires:

$$\pi_{\theta}(\boldsymbol{\tau}_{i,t}|\boldsymbol{q},\boldsymbol{\tau}_{i,< t}) > (1 - \epsilon) \cdot \pi_{\theta_{\text{old}}}(\boldsymbol{\tau}_{i,t}|\boldsymbol{q},\boldsymbol{\tau}_{i,< t})$$

However, this condition is difficult to satisfy due to thinking tokens' inherently high prediction probabilities. Section 3.2 shows that the thinking token wait being around 0.88 in average probability, implying a high suppression threshold. With the standard  $\epsilon = 0.2$ , thinking tokens are suppressed only when  $\pi_{\theta}(\tau_{i,t}|q,\tau_{i,< t}) > (1-0.2) \times 0.88 = 0.704$ . Such high threshold causes critical suppression signals to be clipped away when we need them most.

Old Policy Calibration. To ensure consistent gradient flow for thinking token suppression, we calibrate the old policy probability for thinking tokens. Inspired by LUFFY (Yan et al., 2025), we reshape the old policy probability as:

$$\hat{\pi}_{\theta_{\text{old}}}(\boldsymbol{\tau}_{i,t}|\boldsymbol{q},\boldsymbol{\tau}_{i,< t}) = \frac{\gamma}{1-\epsilon}, \quad \forall \boldsymbol{\tau}_{i,t} \in \mathcal{S}_{\text{think}}$$
 (7)

where  $\gamma$  is a small constant (we use  $\gamma=0.1$ ). Then, the importance ratio is modified as:

$$\hat{r}_{i}^{t} = \frac{\pi_{\theta}(\boldsymbol{\tau}_{i,t}|q,\boldsymbol{\tau}_{i,< t})}{\hat{\pi}_{\theta_{\text{old}}}(\boldsymbol{\tau}_{i,t}|q,\boldsymbol{\tau}_{i,< t})} = \frac{\pi_{\theta}(\boldsymbol{\tau}_{i,t}|q,\boldsymbol{\tau}_{i,< t})}{\gamma} \cdot (1 - \epsilon)$$
(8)

Guaranteed Gradient Flow. With this calibration, the clipping condition  $\hat{r}_i^t > 1 - \epsilon$  is satisfied whenever  $\pi_{\theta}(\tau_{i,t}|q,\tau_{i,< t}) > \gamma$ . Since thinking tokens typically maintain probabilities well above  $\gamma = 0.1$  during training, this ensures that suppression gradients for thinking tokens are rarely clipped. Combined with our token-level advantage scaling, this strategy provides reliable and consistent learning signals for thinking token regulation throughout the training process.

#### <span id="page-7-2"></span>5 Experiments

In this section, we discuss our experimental setup, detailed implementation and baselines.

#### <span id="page-7-3"></span>5.1 Setup

**Model.** The training process is initialized with DeepSeek-R1-Distill-Qwen-1.5B, which is a popular LRM developed by DeepSeek-AI (2025).

Training Data. Inspired by recent works (Yang et al., 2025a; Bae et al., 2025), we prioritise the quality over quantity for RL training data, curating 1,000 problems of medium difficulty from the DAPO-MATH-17K dataset (Yu et al., 2025). Specifically, we processed 28,000 random DAPO-MATH-17K samples with the LRM, using 0.6 temperature to generate 4 responses per question with 16K maximum length. The final dataset comprised entries exhibiting an answer correctness rate between 0.25 and 0.5, coupled with an average response length exceeding 8192 tokens. These selected data are of moderate difficulty and feature longer responses, making them prone to the thinking trap. We name the curated dataset as DuPPO-1K.

Validation Data. We utilise AIME24 as the validation set, following the setting of DAPO (Yu et al., 2025) and VAPO (Yue et al., 2025). During inference on this validation set, the temperature is set to 0. Adhering to the principle of balancing performance and token usage optimally, we report the evaluation results for the following checkpoints: DeepSeek-R1-Distill-Qwen-1.5 with GRPO at 90-step, DeepSeek-R1-Distill-Qwen-1.5 with DuP-PO at 80-step.

Benchmarks and Metrics. To evaluate performance, we selected six popular math reasoning benchmarks: AIME

2024, AIME 2025, AMC (Jia LI et al., 2024), MINERVA (Lewkowycz et al., 2022), OLYMPIADBENCH (He et al., 2024), and MATH500 (Lightman et al., 2024). Given the comparatively smaller test sets of AIME 2024, AIME 2025, and AMC, we present results using Avg@32. For MINERVA, OLYMPIADBENCH and MATH500, Pass@1 is used. The temperature is set to 0.6, and the max response length is set to 8,192 for testing. To mitigate systematic errors caused by sampling randomness, all reported results represent the average of three inference runs.

#### <span id="page-7-1"></span>5.2 Baselines

We compare DuP-PO against the base model (DeepSeek-R1-Distill-Qwen-1.5B) and two test-time efficiency baselines:

- NoThink (Ma et al., 2025a) bypasses internal reasoning by appending "Okay, I think I have finished thinking.</think>" after "<think>" in prompts, forcing models to generate concise answers directly.
- ThinkTokenPenalty (Wang et al., 2025b) applies the logit penalty to thinking tokens (e.g., however, alternatively) during inference. We adopt an aggressive variant that penalizes all thinking tokens to prevent their sampling entirely. This technique aligns with sampling from the rectified policy  $\pi_r$  described in Section 4.4.

## <span id="page-7-4"></span>5.3 RL Practice

#### <span id="page-7-0"></span>5.3.1 Implementation Details

Inference. We Training and VeRL (Sheng et al., 2024) to implement DuP-PO, where the training context size, batch size, and the learning rate are 8,192, 128 and 2e-6. The actor policy update batch size is 128. The total rollout number is set as 8, including N=4 responses sampling from the normal policy  $\pi_n$ , and M=4 responses sampling from the rectified policy  $\pi_r$ . We remove the KL loss term by setting  $\beta = 0$ and set the entropy loss coefficient to 0.01. In terms of the newly added hyperparameters of DuP-PO, we set the enhancement factor  $\alpha$  and the suppression factor  $\beta$  as 2 for advantage scaling. For the 1.5B model, we train them within 100 steps. All training experiments are conducted on a single 8xH800

<span id="page-8-0"></span>Table 1: Main results on six mathematical reasoning benchmarks. Score represents the average performance, using Pass@1 for MINERVA, MATH500, and OLYMPIADBENCH (for short OLYMPIAD), and Avg@32 for AMC, AIME24, and AIME25. Len denotes the average response length. Higher Score values and lower Len values indicate better performance. Bold entries highlight the best result in each column.

| Model                         | MATH500   |         | OLYMPIAD  |         | MINERVA   |         | AIME24    |         | $\mathbf{AMC}$ |         | AIME25    |         | Average   |                    |                    |                                  |
|-------------------------------|-----------|---------|-----------|---------|-----------|---------|-----------|---------|----------------|---------|-----------|---------|-----------|--------------------|--------------------|----------------------------------|
|                               | Score (†) | Len (↓) | Score (†) | Len (↓) | Score (†) | Len (↓) | Score (†) | Len (↓) | Score (†)      | Len (↓) | Score (†) | Len (↓) | Score (†) | Len $(\downarrow)$ | $\Delta$ Score (†) | $\Delta \text{Len} (\downarrow)$ |
| DeepSeek-R1-Distill-Qwen-1.5B |           |         |           |         |           |         |           |         |                |         |           |         |           |                    |                    |                                  |
| Base model                    | 79.2      | 3760    | 43.0      | 5992    | 29.3      | 4821    | 21.8      | 7410    | 55.4           | 5812    | 20.4      | 7277    | 43.9      | 6105               | -                  | -                                |
| + NoThink                     | 72.3      | 2022    | 40.5      | 4052    | 24.4      | 1272    | 21.9      | 6157    | 53.8           | 4153    | 18.5      | 6335    | 41.8      | 4502               | -2.1               | -26.3%                           |
| + ThinkTokenPenalty           | 77.7      | 2547    | 43.8      | 4081    | 27.7      | 2392    | 20.1      | 5694    | 57.2           | 3956    | 18.8      | 5053    | 44.0      | 4234               | +0.1               | -30.6%                           |
| + GRPO                        | 81.4      | 3345    | 45.0      | 5614    | 30.1      | 4482    | 24.0      | 7169    | 58.7           | 5359    | 23.2      | 6956    | 46.6      | 5724               | +2.7               | -6.2%                            |
| + DuP-PO                      | 82.7      | 2830    | 47.4      | 5033    | 30.6      | 3585    | 25.0      | 6795    | 60.9           | 4722    | 21.7      | 6499    | 47.9      | 5162               | +4.0               | -15.4%                           |

node, and all inference experiments use four NVIDIA RTX 4090 GPUs.

Thinking Tokens. For thinking token identification, we categorize common thinking tokens in LRM responses into reflection tokens (wait, hmm, hold on, okay) and thought transition tokens (alternatively, maybe, but, however). Our analysis in Section 3.2 focuses on the wait token as a representative case, while both DuP-PO and ThinkTokenPenalty penalize all identified thinking tokens to maximize token efficiency.

#### <span id="page-8-1"></span>5.3.2 Reward Design

We design a rule-based reward function that incorporates both correctness and formatting rewards, similar to the general RL training approach used in DeepSeek-R1 (DeepSeek-AI, 2025). The reward function evaluates two components: answer correctness through Math-Verify parsing and response formatting quality. Specifically, correct answers receive a base reward of 1.0, with an additional 0.1 bonus for well-formatted responses. The reward function is formally defined as:

$$R(\boldsymbol{\tau}, \boldsymbol{a}) = \begin{cases} 1.1, & \text{if is\_equivalent}(\boldsymbol{a}, \boldsymbol{\tau}) \\ & \tau \text{ is well-formatted} \\ 1.0, & \text{if is\_equivalent}(\boldsymbol{a}, \boldsymbol{\tau}), \\ & \tau \text{ is not well-formatted} \\ 0.1, & \text{if not\_equivalent}(\boldsymbol{a}, \boldsymbol{\tau}), \\ & \tau \text{ is well-formatted} \\ 0, & \text{if not\_equivalent}(\boldsymbol{a}, \boldsymbol{\tau}), \\ & \tau \text{ is not well-formatted} \end{cases}$$
(9)

where is\_equivalent( $a, \tau$ ) determines whether the ground truth answer a can be successfully parsed from the response trajectory  $\tau$ .

#### <span id="page-8-2"></span>6 Results and Discussion

We conduct comprehensive experiments across six widely-adopted mathematical reasoning benchmarks to evaluate DuP-PO's effectiveness. Our empirical analysis demonstrates consistent improvements in both performance and token efficiency, as summarized in Table 1.

