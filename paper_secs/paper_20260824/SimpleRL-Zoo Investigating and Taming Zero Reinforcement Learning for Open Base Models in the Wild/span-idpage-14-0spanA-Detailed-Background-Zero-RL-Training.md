# <span id="page-14-0"></span>A Detailed Background: "Zero RL Training"

In our study, we follow the zero RL training recipe in DeepSeek-AI et al. (2025a) using various open base models, employing the GRPO algorithm (Shao et al., 2024). Here, zero RL training refers to reinforcement learning directly from the base model without any prior supervised fine-tuning (SFT). GRPO optimizes computational efficiency by eliminating the need for a separate value model; instead, it directly utilizes group-normalized rewards to estimate advantages. For a query q and a set of responses  $O = \{o_1, o_2, \ldots, o_G\}$  sampled from the old policy model  $\pi_{\text{old}}$ , we adopt a token-level, length-rectified GRPO objective to optimize the policy model  $\pi$ :

$$\mathcal{J}_{\text{GRPO}}(\theta) = \underbrace{\frac{1}{\sum_{i=1}^{G} |o_{i}|} \sum_{i=1}^{G} \sum_{t=1}^{|o_{i}|} \min \left[ r_{i,t}(\theta) \hat{A}_{i}, \operatorname{clip}\left(r_{i,t}(\theta); 1 - \epsilon, 1 + \epsilon\right) \hat{A}_{i} \right]}_{\text{Clipped policy update}} - \underbrace{\frac{\beta \mathbb{D}_{\text{KL}}[\pi_{\theta} \parallel \pi_{\text{ref}}]}{\text{KL penalty}}}_{\text{KL penalty}}$$
where  $r_{i,t}(\theta) = \frac{\pi_{\theta}(o_{i,t} \mid q, o_{i, < t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i, < t})}$ 
(1)

where  $\pi_{\text{ref}}$  represents the reference model, and the term  $\mathbb{D}_{KL}$  introduces a KL divergence constraint to limit how much the model can deviate from this reference. The advantage estimate  $\hat{A}_i$  measures how much better the response  $o_i$  is compared to the average response, which is computed using a group of rewards  $\{r_1, r_2, \ldots, r_G\}$  for the responses in set O:

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, r_2, \dots, r_G\})}{\text{std}(\{r_1, r_2, \dots, r_G\})}$$
(2)

### <span id="page-14-1"></span>B Detailed Experimental Setup

#### **B.1** Dataset

To keep the training recipe simple, we select training data exclusively from the GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021) datasets. For the MATH dataset, following prior studies (Lightman et al., 2023; Wang et al., 2023; Sun et al., 2024), we reserve the MATH500 subset as the test set, uniformly sample an additional 500 problems for validation, and combine the remaining 4,000 test problems with the original 7,500 training problems to form our training set. Each example in the MATH dataset is originally labeled with a difficulty level ranging from 1 to 5. In our experiments, we find that data difficulty is critical for successful zero RL (§3.2) and it is necessary to use data that aligns with the model's capability. To investigate this phenomenon, we categorize the data into three difficulty levels: Easy (GSM8K and MATH lv.1), Medium (MATH lv.1–4), and Hard (MATH lv.3–5), with each category containing roughly 8,000 problems. For our main training runs, we use Easy for LLama-3.1-8B, Mistral- v0.1-7B, and DeepSeek-Math-7B; Medium for Qwen-2.5-0.5B; Hard for Mistral-Small-24B, Qwen-2.5-Math-7B, and Qwen-2.5-1.5B/7B/14B/32B, and we will report ablation study on data difficulty in §3.2.

#### **B.2** Reward

We use a rule-based reward function that assigns rewards solely based on the correctness of the generated response: a correct final answer receives a reward of +1, while an incorrect one receives a reward of 0. Recent studies (Luo et al., 2025; Chen et al., 2025) often incorporate format-based rules into reward calculation, encouraging the model to follow specific output

<span id="page-14-2"></span><sup>&</sup>lt;sup>2</sup>The original GRPO objective has a length normalization term that introduces length biases. We remove the length normalization term similar to concurrent works (Yu et al., 2025; Liu et al., 2025) – this length-rectified objective was the default implementation of GRPO in our adapted codebase, verl (Sheng et al., 2024).

formats. However, we find that this approach may hinder the model's exploration and ultimately harm its performance particularly for the base models which struggle with following the format in the initial stage, as detailed in [§3.1.](#page-7-0)

#### **B.3 Models**

We conduct zero RL training experiments on Llama-3.1-8B [\(Dubey et al.,](#page-11-3) [2024\)](#page-11-3), DeepSeek-Math-7B [\(Shao et al.,](#page-12-5) [2024\)](#page-12-5), Mistral-v0.1-7B [\(Jiang et al.,](#page-11-2) [2023\)](#page-11-2), Mistral-Small-24b-Base-2501 [\(Mistral AI,](#page-12-4) [2025\)](#page-12-4), and Qwen-2.5 (0.5B, 1.5B, 7B, 14B, 32B) [\(Yang et al.,](#page-12-6) [2024a\)](#page-12-6). As we perform experiments for a variety of models, under extremely simple settings with small, simple datasets and only correctness reward, we refer to our obtained models as *SimpleRL-Zoo* to represent a simple training recipe for a zoo of open base models. For models with weaker instruction-following capabilities (Llama-3.1-8B, Mistral-v0.1-7B, and Qwen-2.5-0.5B/1.5B), we employ simpler prompts [\(Chern et al.,](#page-9-4) [2023\)](#page-9-4) requiring only stepby-step reasoning. For models with stronger instruction-following abilities, we use more complex prompts [\(Yang et al.,](#page-12-6) [2024a\)](#page-12-6) that require the final answers to be placed in boxes. In our preliminary experiments, we observe that using complex prompts with models that have weak instruction-following capabilities often results in large amounts of irrelevant or nonsensical content being generated early in training, leading to instability. The content of simpler prompts and more complex prompts is shown in Figure [10](#page-16-1) in Appendix.

#### **B.4 Benchmark**

We evaluate performance on standard mathematical reasoning benchmarks, including GSM8K [\(Cobbe et al.,](#page-10-2) [2021\)](#page-10-2), MATH 500 [\(Hendrycks et al.,](#page-11-4) [2021\)](#page-11-4), Minerva Math [\(Lewkowycz](#page-11-6) [et al.,](#page-11-6) [2022\)](#page-11-6), and OlympiadBench [\(He et al.,](#page-11-7) [2024\)](#page-11-7), as well as on competition-level benchmarks such as AIME 2024 and AMC 2023.

#### **B.5 Training and Evaluation Details**

We train our models using the verl [\(Sheng et al.,](#page-12-15) [2024\)](#page-12-15) framework. And we typically use the same set of hyperparameters to train and evaluate all models in the SimpleRL-Zoo series in default main experiment setting. We use a prompt batch size of 1,024 and generate 8 rollouts per prompt, with a maximum rollout length of 8,192 tokens. Training is performed using a mini-batch size of 256. The default sampling temperature is set to 1.0, and the clip ratio is 0.2. For models ranging from 0.5B to 14B parameters, we use a KL loss coefficient of 1e-4. For models larger than 14B, the KL loss coefficient is set to 1e-3. We build our evaluation script based on [Yang et al.](#page-12-7) [\(2024b\)](#page-12-7), using a temperature of 1.0 and a maximum generation length of 16K tokens. To ensure consistency, we adopt the same prompt template used during training. For most benchmarks, we report pass@1 results. However, for AIME 2024, which contains fewer problems, we report both pass@1 and average accuracy (avg@32), computed over 32 generated samples per problem.

