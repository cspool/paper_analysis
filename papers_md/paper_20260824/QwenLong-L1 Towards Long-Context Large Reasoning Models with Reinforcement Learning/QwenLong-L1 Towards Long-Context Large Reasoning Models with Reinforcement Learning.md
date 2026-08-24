# **QWENLONG-L1: Towards Long-Context Large Reasoning Models with Reinforcement Learning**

Fanqi Wan, Weizhou Shen, Shengyi Liao, Yingcheng Shi, Chenliang Li, Ziyi Yang, Ji Zhang, Fei Huang, Jingren Zhou, Ming Yan\*
Tongyi Lab, Alibaba Group

![](_page_0_Picture_3.jpeg)

https://github.com/Tongyi-Zhiwen/QwenLong-L1 https://huggingface.co/Tongyi-Zhiwen/QwenLong-L1-32B https://modelscope.cn/models/iic/QwenLong-L1-32B

## **Abstract**

Recent large reasoning models (LRMs) have demonstrated strong reasoning capabilities through reinforcement learning (RL). These improvements have primarily been observed within the short-context reasoning tasks. In contrast, extending LRMs to effectively process and reason on long-context inputs via RL remains a critical unsolved challenge. To bridge this gap, we first formalize the paradigm of long-context reasoning RL, and identify key challenges in suboptimal training efficiency and unstable optimization process. To address these issues, we propose QWENLONG-L1, a framework that adapts short-context LRMs to long-context scenarios via progressive context scaling. Specifically, we utilize a warm-up supervised fine-tuning (SFT) stage to establish a robust initial policy, followed by a curriculum-guided phased RL technique to stabilize the policy evolution, and enhanced with a difficulty-aware retrospective sampling strategy to incentivize the policy exploration. Experiments on seven long-context document question-answering benchmarks demonstrate that QWENLONG-L1-32B outperforms flagship LRMs like OpenAI-o3-mini and Qwen3-235B-A22B, achieving performance on par with Claude-3.7-Sonnet-Thinking, demonstrating leading performance among stateof-the-art LRMs. This work advances the development of practical long-context LRMs capable of robust reasoning across information-intensive environments.

> **[图片提取文字 (无描述)]:**
> QwenLong-L1-32B R1-Distill-Qwen-32B o3-mini Qwen3-235B-A22B Claude-3.7-Sonnet-Thinking DeepSeek-R1 o1-preview 100 .5 89.9 86.5 87.5 83.0 83.5<sup>84.484.4</sup>82.583.5 79.680.8 80 75.574.6 70.470.670.7 72.172.9 70.9 66.567.567.5 Accuracy (%) 59.060.061.5 56.0 55.0 56.0 5 DocMath Frames 2WikimQA HotpotQA Musique NarrativeQA Qasper Average
![](_page_0_Figure_7.jpeg)

Figure 1: Overall results of QWENLONG-L1 across seven long-context reasoning benchmarks. Starting from R1-Distill-Qwen-32B, QWENLONG-L1-32B achieves an average gain of 5.1 points, surpassing OpenAI-o3-mini, Qwen3-235B-A22B, and comparable to Claude-3.7-Sonnet-Thinking.

<sup>\*</sup> Corresponding author.

<span id="page-1-0"></span>> **[图片提取文字 (无描述)]:**
> Long-Context Reasoning RL Short-Context Reasoning RL More KL Greater Length 0.250 0.05 -Fluctuations Value/Variance 0.85 1400 edence 0.04 0.225 Reward Score 0.70 6 0.200 Response .ž ∩ 0.02 0.175 0.01 1000 0.150 Slower Reward More Entropy 0.65 Improvements Reduction 0.00 40 40 Training Steps Training Steps Training Steps Training Steps (c) KL Divergence (a) Reward Score (b) Entropy (d) Response Length
![](_page_1_Figure_0.jpeg)

Figure 2: Comparison of training dynamics between short-context and long-context reasoning RL. The long-context reasoning RL demonstrates two key challenges: *suboptimal training efficiency*, with slower improvements in reward score caused by more reduction in entropy, and *unstable optimization process*, with more fluctuations in KL divergence introduced from greater variance in longer output.

# 1 Introduction

Recent breakthroughs in large reasoning models (LRMs) have showcased significant improvements in reasoning capabilities, achieving performance comparable to human experts in complex problemsolving scenarios [\[49\]](#page-22-0). These advancements, exemplified by OpenAI-o1 [\[26,](#page-21-0) [15\]](#page-20-0), DeepSeek-R1 [\[6,](#page-19-0) [11\]](#page-20-1), and Qwen-QwQ [\[41,](#page-21-1) [43\]](#page-21-2), have sparked extensive research efforts to explore and enhance a broad spectrum of reasoning tasks through reinforcement learning (RL), ranging from foundational logical reasoning [\[29,](#page-21-3) [47\]](#page-22-1) to advanced challenges in programming [\[8,](#page-19-1) [7\]](#page-19-2) and mathematics [\[24,](#page-21-4) [39\]](#page-21-5), with innovations in RL algorithms driving progress in reasoning quality enhancements [\[3,](#page-19-3) [55,](#page-22-2) [23,](#page-20-2) [54\]](#page-22-3).

Following RL fine-tuning, LRMs exhibit a phenomenon analogous to human "slow thinking" [\[4\]](#page-19-4), characterized by the emergence of sophisticated problem-solving strategies such as divide-and-conquer and backtracking mechanisms in their extended chain-of-thought (CoT) reasoning outputs [\[46\]](#page-22-4). While this process enhances reasoning performance on short context tasks (e.g., 4K tokens) [\[45,](#page-21-6) [22\]](#page-20-3), its scalability to long-context scenarios (e.g., 120K tokens), which requires robust contextual grounding and multi-step reasoning, remains unexplored. This limitation poses a significant barrier to practical applications requiring interaction with external knowledge, such as deep research [\[38,](#page-21-7) [27,](#page-21-8) [40\]](#page-21-9), where LRMs must collect and process information from knowledge-intensive environments.

To shed light on this topic, we first introduce the concept of *long-context reasoning RL*. Different from *short-context reasoning RL*, which primarily relies on internal knowledge stored within model parameters, long-context reasoning RL instead necessitates that LRMs perform retrieval and grounding of relevant information from long-context inputs, followed by generation of reasoning chains based on the incorporated information [\[12,](#page-20-4) [30,](#page-21-10) [52\]](#page-22-5). To illustrate the differences between short-context and long-context reasoning RL, we conduct a preliminary experiment to compare the training dynamics in Figure [2.](#page-1-0) Our results reveal that long-context reasoning RL exhibits *suboptimal training efficiency* compared to the short-context counterpart with *(a) delayed reward convergence*. This discrepancy stems from *(b) marked reduction in output entropy* when processing long-context inputs, which restricts exploratory behavior during policy optimization. Furthermore, we identify *unstable optimization process*, evidenced by *(c) intermittent spikes in KL divergence*. These instabilities are introduced by the inherent variance amplification due to *(d) longer output length with heterogeneous input length distributions*, leading to greater variability during policy updating.

To address these challenges, we propose QWENLONG-L1, a novel RL framework designed to facilitate the transition of LRMs from short-context proficiency to robust long-context generalization, as shown in Figure [3.](#page-2-0) Inspired by recent studies on context extension during pretraining [\[9,](#page-20-5) [48,](#page-22-6) [10\]](#page-20-6), our framework enhances short-context LRMs through progressive context scaling during RL training. The framework comprises three core components: a warm-up supervised fine-tuning (SFT) phase to initialize a robust policy, a curriculum-guided RL phase that facilitates stable adaptation from short to long contexts, and a difficulty-aware retrospective sampling mechanism that adjusts training complexity across stages to incentivize policy exploration. Leveraging recent RL algorithms, including GRPO [\[34\]](#page-21-11) and DAPO [\[54\]](#page-22-3), our framework integrates hybrid reward functions combining rule-based and model-based binary outcome rewards to balance precision and recall. Through strategic utilization of group relative advantages during policy optimization, it guides LRMs to

<span id="page-2-0"></span>> **[图片提取文字 (无描述)]:**
> Policy Updating Phase I Cshort  $A_1$ Rule-Based Verification Policy Progressive Group  $A_2$ Model Context Scaling Computation LLM-as-a-... ... ... Judge Phase K  $c_{long}$  $y_G$  $r_G$  $A_G$
![](_page_2_Figure_0.jpeg)

Figure 3: Overview of QWENLONG-L1, which is a novel *long-context reasoning RL* training framework. The proposed framework integrates group-relative RL algorithms, hybrid reward mechanisms, and progressive context scaling strategies to enable stable adaptation from short-context to longcontext LRMs with robust contextual grounding and multi-step reasoning capabilities.

learn effective reasoning patterns essential for long-context reasoning scenarios, resulting in robust long-context grounding and superior reasoning capabilities.

In our experiments, we focus on document question answering (DocQA) [\[51,](#page-22-7) [17,](#page-20-7) [13\]](#page-20-8) as a representative real-world long-context reasoning task. Specifically, we introduce DOCQA-RL-1.6K, a specialized RL training dataset comprising 1.6K DocQA problems spanning mathematical, logical, and multi-hop reasoning domains. Experimental results across seven long-context DocQA benchmarks demonstrate the superiority of QWENLONG-L1 compared to various proprietary and open-source LRMs. Notably, QWENLONG-L1-14B achieves superior performance over Gemini-2.0-Flash-Thinking and Qwen3-32B, while QWENLONG-L1-32B outperforms OpenAI-o3-mini, Qwen3-235B-A22B, and even matches Claude-3.7-Sonnet-Thinking. Our analysis further identifies several critical insights in long-context reasoning RL optimization: (1) progressive context scaling promotes higher entropy and stabilizes KL divergence, enhancing training efficiency; (2) SFT proves to be an economical way for performance enhancement, whereas RL unlocks the potential to achieve optimal performance; (3) RL naturally fosters specialized long-context reasoning behaviors that boost final performance, but imitating these behaviors do not translate into gains when applied to SFT. Our key contributions are summarized as follows:

- We conceptualize the paradigm of *long-context reasoning RL* and identify its unique challenges, making a further step towards developing practical long-context LRMs capable of grounding and integrating information for complex, real-world reasoning scenarios.
- We present QWENLONG-L1, the first RL framework designed for long-context LRMs. Through progressive context scaling, QWENLONG-L1 enables stable short-to-long context adaptation via group-relative RL optimization and hybrid reward mechanisms.
- We showcase the effectiveness of QWENLONG-L1 through comprehensive experiments across seven long-context document question answering benchmarks. Our results reveal that QWENLONG-L1 achieves substantial performance gains compared to cutting-edge LRMs, offering a fundamental recipe and practice for long-context reasoning optimization.

# 2 Long-Context Reasoning Reinforcement Learning

While existing works have explored reinforcement learning (RL) for enhancing short-context reasoning tasks [\[14,](#page-20-9) [56,](#page-22-8) [16\]](#page-20-10), the extension of long-context reasoning RL remains an unresolved challenge. To this end, we introduce QWENLONG-L1, the first RL framework specifically designed for long-context reasoning. In this section, we first describe the preliminaries, and then detail the implementation, including the RL algorithms for long-context reasoning, the progressive context scaling strategies for stable optimization, and the hybrid reward mechanisms to balance evaluation precision and recall.

# 2.1 Preliminaries

The standard RL objective for language models optimizes a KL-regularized expected reward [\[32\]](#page-21-12):

$$\max_{\pi_{\theta}} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(\cdot \mid x)} \left[ r_{\phi}(x, y) \right] - \beta \mathbb{D}_{\text{KL}} \left[ \pi_{\theta}(y \mid x) \mid \mid \pi_{\text{ref}}(y \mid x) \right]$$
 (1)

where rϕ(x, y) denotes the reward for output y given input x from the policy model πθ, and πref represents the reference model for DKL regularization.

Unlike prior works wherein the input x typically is a short question, requiring the policy model  $\pi_{\theta}$  to generate output y based on its parametric knowledge, we extend this formulation to incorporate an additional long-context c. Therefore, the policy model  $\pi_{\theta}$  needs to first ground relevant information from c, and then provide reasoning chains to solve the question x:

$$\max_{\pi_{0}} \mathbb{E}_{x,c \sim \mathcal{D}, y \sim \pi_{\theta}(\cdot \mid x, c)} \left[ r_{\phi}(x, \boldsymbol{c}, y) \right] - \beta \mathbb{D}_{\text{KL}} \left[ \pi_{\theta}(y \mid x, \boldsymbol{c}) \mid\mid \pi_{\text{ref}}(y \mid x, \boldsymbol{c}) \right]$$
(2)

<span id="page-3-0"></span>The context c and the question x are concatenated by the following prompt template:

Table 1: Prompt template to concatenate the input context and question.

Please read the following text and answer the question below.

<text> context </text> question

Format your response as follows: "Therefore, the answer is (insert answer here)".

## 2.2 RL Algorithms

**Proxy Policy Optimization (PPO)** To optimize the policy model  $\pi_{\theta}$  using the above objective, the PPO [33] algorithm proposes to maximize the clipped surrogate objective:

$$\mathcal{J}_{PPO}(\theta) = \mathbb{E}_{x,c \sim \mathcal{D}, y \sim \pi_{\theta_{\text{old}}}(\cdot | x, c)}$$

$$\left[\frac{1}{|y|}\sum_{t=1}^{|y|}\min\left(\frac{\pi_{\theta}(y_t\mid x, c, y_{< t})}{\pi_{\theta_{\text{old}}}(y_t\mid x, c, y_{< t})}A_t, \operatorname{clip}\left(\frac{\pi_{\theta}(y_t\mid x, c, y_{< t})}{\pi_{\theta_{\text{old}}}(y_t\mid x, c, y_{< t})}, 1 - \varepsilon, 1 + \varepsilon\right)A_t\right)\right]$$
(3)

where  $\pi_{\theta}$  is the current policy model,  $\pi_{\theta_{\text{old}}}$  is the old policy model before updating,  $\varepsilon$  is the clipping hyperparameter, and  $A_t$  is the advantage estimator of the t-th token. For a given input context c and question x,  $\pi_{\theta_{\text{old}}}$  first generate sequential output y, and then  $A_t$  is calculated to update  $\pi_{\theta}$  using the generalized advantage estimation (GAE) [31] with a learned value function  $V_{\phi}$ .

For long-context inputs, the quadratic complexity of attention mechanisms renders value network training computationally prohibitive. Therefore, we opt for the group-relative RL algorithms, including GRPO [34] and DAPO [54], to estimate the advantage through group-normalized rewards instead of an additional value network.

Group Relative Policy Optimization (GRPO) In GRPO, given the context c and the question x, the old policy model  $\pi_{\theta_{\text{old}}}$  generates a group of G outputs  $\{y_i\}_{i=1}^G$ , with rewards  $\{r_i\}_{i=1}^G$  calculated by the reward function. The optimization objective for the policy model  $\pi_{\theta}$  becomes:

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{x,c \sim \mathcal{D},\{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|x,c)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \left( \min \left( \frac{\pi_{\theta}(y_{i,t}|x,c,y_{i,< t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x,c,y_{i,< t})} A_{i,t}, \right) \right) \right]$$

$$\operatorname{clip}\left( \frac{\pi_{\theta}(y_{i,t}|x,c,y_{i,< t})}{\pi_{\theta_{\text{old}}}(y_{i,t}|x,c,y_{i,< t})}, 1 - \varepsilon, 1 + \varepsilon \right) A_{i,t} \right) - \beta \mathbb{D}_{KL}(\pi_{\theta}||\pi_{\text{ref}}) \right) \right]$$
(4)

where the advantage for token  $y_{i,t}$  is estimated by normalizing the group-level rewards:

$$A_{i,t} = \frac{r_i - \text{mean}(\{r_i\}_{i=1}^G)}{\text{std}(\{r_i\}_{i=1}^G)}$$
(5)

In our implementation, we remove the KL term in the GRPO objective to encourage the exploration capacity of the policy model, following the common suggestions in recent works [14, 23, 54].

**Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO)** DAPO integrates several techniques to ensure a more stable and efficient RL process: (1) a higher clip threshold to avoid entropy collapse, (2) a dynamic sampling strategy to remove examples with zero reward variance, (3) a token-level loss to mitigate the length bias, and (4) an overlong reward shaping to avoid excessively long outputs. The optimization objective for the policy model  $\pi_{\theta}$  is:

$$\mathcal{J}_{\text{DAPO}}(\theta) = \mathbb{E}_{(x,c)\sim\mathcal{D},\{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|x,c)} \left[ \frac{1}{\sum_{i=1}^G |y_i|} \sum_{i=1}^G \sum_{t=1}^{|y_i|} \min\left( \frac{\pi_{\theta}(y_t \mid x, c, y_{< t})}{\pi_{\theta_{\text{old}}}(y_t \mid x, c, y_{< t})} A_{i,t}, \right) \right] \\
\text{clip}\left( \frac{\pi_{\theta}(y_t \mid x, c, y_{< t})}{\pi_{\theta_{\text{old}}}(y_t \mid x, c, y_{< t})}, 1 - \varepsilon_{\text{low}}, 1 + \varepsilon_{\text{high}} \right) A_{i,t} \right) \right]$$
(6)

where  $\varepsilon_{\text{low}}$  and  $\varepsilon_{\text{high}}$  are the low and high clip thresholds, respectively. The dynamic sampling strategy ensures that the examples have a non-zero reward variance, and the token-level loss sets an equal weight for loss calculation for each token in the outputs. The overlong reward shaping set a soft length-award punishment as follows:

$$r_{\text{final}_{i}} = \begin{cases} r_{i}, & |y_{i}| \leq L_{\text{max}} - L_{\text{cache}} \\ r_{i} + \frac{(L_{\text{max}} - L_{\text{cache}}) - |y_{i}|}{L_{\text{cache}}}, & L_{\text{max}} - L_{\text{cache}} < |y_{i}| \leq L_{\text{max}} \\ r_{i} - 1, & L_{\text{max}} < |y_{i}| \end{cases}$$
(7)

where  $L_{\text{max}}$  is the maximum sequence length and  $L_{\text{cache}}$  is a buffer zone for gradual length penalization. Different from GRPO, the final advantage for  $y_{i,t}$  in DAPO becomes:

$$A_{i,t} = \frac{r_{\text{final}_i} - \text{mean}(\{r_{\text{final}_i}\}_{i=1}^G)}{\text{std}(\{r_{\text{final}_i}\}_{i=1}^G)}$$
(8)

## <span id="page-4-0"></span>2.3 Progressive Context Scaling

Training LRMs for long-context reasoning presents unstable optimization dynamics. To address these issues, we propose a progressive context scaling framework, including a curriculum-guided RL strategy to stabilize short-to-long context optimization, a difficulty-aware retrospective sampling mechanism to prioritize exploration of complex instances, and a warm-up supervised fine-tuning (SFT) phase to provide robust initialization before RL training.

Curriculum-Guided Phased Reinforcement Learning The RL process is divided into K discrete phases, with the target context lengths of  $L_k$  for each phase. Starting from an initial input length  $L_1$ , each subsequent phase increases the input length until reaching the maximum target length  $L_K$ . During phase k, the policy model  $\pi_{\theta}$  is trained exclusively on examples that satisfies:

$$L_{k-1} < |x| + |c| \le L_k, \quad L_0 = 0$$
 (9)

where |x| and |c| denote the question length and supporting context length, respectively.

**Difficulty-Aware Retrospective Sampling** Building on the crucial efficacy of instance difficulty in previous data selection studies [59, 19], we adopt a difficulty-aware retrospective sampling method to strategically incorporate instances from preceding phases. Specifically, we implement importance sampling weighted by difficulty scores to curate retrospective instances:

$$diff(x,c) = \frac{1}{\text{mean}(\{r_i\}_{i=1}^G)}$$
 (10)

where diff(·) denotes the difficulty function, quantified as the inverse mean reward  $\{r_i\}_{i=1}^G$  from a group of outputs generated by the base model. Lower mean rewards correspond to higher difficulty scores, prioritizing challenging instances during retrospective sampling.

**Warm-Up Supervised Fine-Tuning** Prior to initiating RL training, we employ a warm-up supervised fine-tuning (SFT) stage to establish a robust initial policy model capable of grounding information from the long-context inputs. This critical preparatory stage ensures the policy model develops fundamental capabilities in context comprehension, reasoning chain generation, and answer extraction before exposure to the instability of RL optimization.

The SFT process utilizes high-quality demonstrations  $\mathcal{D}_{SFT}$  distilled from a teacher LRM, where each example contains a question x, supporting context c, and a gold-standard reasoning path  $y^*$  with verified correctness. To align with the progressive scaling curriculum, we construct  $\mathcal{D}_{SFT}$  within the initial input length  $L_1$  in curriculum-guided RL. The model is trained to minimize the standard negative log-likelihood objective:

$$\mathcal{L}_{SFT}(\theta) = -\mathbb{E}_{(x,c,y^*) \sim \mathcal{D}_{SFT}} \frac{1}{|y^*|} \sum_{t=1}^{|y^*|} \log \pi_{\theta}(y_t^* \mid x, c, y_{< t}^*)$$
(11)

The resulting SFT model serves as the initial policy  $\pi_{\theta}$  for RL training, providing stable starting parameters. In Section 4.2, we demonstrate the effectiveness of the proposed three strategies for stable short-to-long context scaling in reasoning RL.

# <span id="page-5-0"></span>2.4 Hybrid Reward Mechanisms

Prior works on short-context reasoning tasks in mathematics, coding, and logical reasoning [\[24,](#page-21-4) [7,](#page-19-2) [47\]](#page-22-1) typically utilize rule-based reward functions that prioritize precision through strict answer matching and format verification to mitigate reward hacking risks [\[35\]](#page-21-15). However, long-context reasoning tasks such as open-domain question answering present unique challenges due to their inherent answer diversity. Overly restrictive rule-based rewards in such contexts risk constraining valid answer variations, potentially compromising overall performance. To address these limitations, we propose a hybrid reward mechanism that combines rule-based verification [\[11\]](#page-20-1) with LLM-as-a-judge [\[58\]](#page-22-10), thereby balancing precision and recall through complementary evaluation.

Rule-Based Verification The rule-based component rrule ensures precision by verifying strict adherence to task-specific correctness criteria. For question answering tasks, we first extract the final answer yans from model generations y using regular expressions aligned with the structured prompt template in Table [1,](#page-3-0) and then perform exact string matching against the gold answer ygold:

$$\mathbf{r}_{\text{rule}}(y) = \mathbb{I}(y_{\text{ans}} = y_{\text{gold}}) \tag{12}$$

where I represents the indicator function. Notably, we intentionally omit the format reward for answer extraction, as the base model demonstrates sufficient inherent format compliance capabilities; excessive format rewards could oversimplify the learning objective, potentially hindering the model's ability for reasoning chain exploration [\[56,](#page-22-8) [16\]](#page-20-10).

LLM-as-a-Judge To complement the precision-oriented rule-based component and address potential false negatives in string matching, we introduce an LLM-based evaluator rLLM that assesses semantic equivalence between generated and gold answers:

$$\mathbf{r}_{\mathsf{LLM}}(x,y) = \mathsf{LLM}(x,y_{\mathsf{ans}},y_{\mathsf{gold}}) \tag{13}$$

where the LLM judge produces a binary correctness score based on the evaluation template as follows:

Table 2: Prompt template for LLM-as-a-judge to compare the predicted answer and the gold answer given the question quesiton.

You are an expert in verifying if two answers are the same. Your input is a problem and two answers, Answer 1 and Answer 2. You need to check if they are equivalent. Your task is to determine if two answers are equivalent, without attempting to solve the original problem. Compare the answers to verify they represent identical values or meaning, even when written in different forms or notations. Your output must follow the following format:

- 1) Provide an explanation for why the answers are equivalent or not.
- 2) Then provide your final answer in the form of: [[YES]] or [[NO]]

Problem: question Answer 1: predicted answer Answer 2: gold answer

Combined Reward Formulation The integrated reward function combines both rule-based verification and LLM-as-a-judge through maximum selection:

$$r_{\phi}(x,y) = \max(\mathbf{r}_{\text{rule}}(y), \mathbf{r}_{\text{LLM}}(x,y)) \tag{14}$$

Given the relative simplicity of answer comparison tasks, we employ a small model, e.g., Qwen2.5- 1.5B-Instruct [\[50\]](#page-22-11), with a temperature of zero for deterministic scoring. This configuration enables efficient reward computation during online RL training while maintaining evaluation reliability.

# 3 Experimental Setup

In our experiments, we employ document question answering (DocQA) as our primary evaluation task for long-context reasoning capabilities, as it inherently requires both contextual grounding and multi-step reasoning. This section details our experimental setup for training and evaluation.

## 3.1 Training Datasets

RL Dataset To construct a challenging RL dataset for verifiable long-context reasoning, we develop DOCQA-RL-1.6K, which comprises 1.6K DocQA problems across three reasoning domains: (1)

<span id="page-6-1"></span>Table 3: Detailed statistics of our train and test datasets. Length is calculated by the Qwen tokenizer.

| Statistics  | Train Dataset |        | Test Dataset |         |        |        |        |        |        |  |
|-------------|---------------|--------|--------------|---------|--------|--------|--------|--------|--------|--|
|             | SFT           | RL     | DocMath      | Frames  | 2Wiki  | HQA    | Musi   | NarQA  | Qasp   |  |
| # Examples  | 5,305         | 1,591  | 200          | 824     | 200    | 200    | 200    | 200    | 200    |  |
| Avg. Length | 13,064        | 11,437 | 17,645       | 15,756  | 7,530  | 13,431 | 16,327 | 29,887 | 5,074  |  |
| Max. Length | 20,003        | 59,559 | 176,285      | 117,131 | 17,035 | 17,640 | 17,883 | 65,357 | 21,927 |  |

**Mathematical Reasoning**: We use 600 problems from the DocMath [57] dataset, requiring numerical reasoning across long and specialized documents such as financial reports<sup>2</sup>; (2) **Logical Reasoning**: We employ DeepSeek-R1 [11] to synthesize 600 multi-choice questions requiring logic analysis of real-world documents spanning legal, financial, insurance, and production domains from our curated collection; (3) **Multi-Hop Reasoning**: We sample 200 examples from MultiHopRAG [36] and 200 examples from Musique [44], emphasizing cross-document reasoning.

**SFT Dataset** To establish a robust starting point for RL optimization, we distill 5.3K high-quality question-document-answer triplets through DeepSeek-R1 [11]. Aligned with recent data curation methods for LRMs [25, 53], we clean and filter questions based on quality, complexity, and diversity. Additionally, we control the quality and length of the documents to ensure precise contextual information. In Table 3, we provide the statistics of our RL and SFT datasets.

# 3.2 Training Details

**Base Model** In our experiments, we initialize our base model with R1-Distill-Qwen-14B and R1-Distill-Qwen-32B [11], subsequently implementing SFT and RL optimization phases<sup>3</sup>.

**RL Training** As depicted in Section 2.3, we propose a progressive context scaling mechanism for long-context reasoning RL optimization. Specifically, the training process follows a two-phase curriculum context scaling, with 20K input length  $L_1$  in phase I, and 60K input length  $L_2$  in phase II. We incorporate difficulty-aware retrospective sampling to maintain the most difficult samples with an average accuracy of zero from phase I to II. The training is conducted on 32xA100-80G GPUs, with a train batch size of 128, a mini batch size of 32, a rollout number of 8, and a learning rate of 2e-6. We set a temperature to 0.7 and a top-p to 0.95 with a maximum output length of 10K for sampling.

**SFT Training** The input length in the SFT stage is set to 20K. The training is conducted on 32xA100-80G GPUs for 3 epochs, with a train batch size of 128, and a learning rate of 5e-6.

# 3.3 Evaluation Details

**Benchmarks** We conduct evaluation on seven long-context DocQA benchmarks, including multi-hop reasoning benchmarks<sup>4</sup> such as 2WikiMultihopQA [13], HotpotQA [51], Musique [44], NarrativeQA [17], Qasper [5], and Frames [18] as well as mathematical reasoning benchmarks like DocMath [57]. We report the maximum of exact match and LLM-judged accuracy as the final score, aligned with the reward function in Section 2.4. We use DeepSeek-V3 [21] as the judge model with a temperature of 0.0 to provide a reliable evaluation. The benchmark statistics are shown in Table 3.

**Configurations** We evaluate our long-context LRMs with a maximum input length of 120K and output length of 10K. For the proprietary LRMs with a limited context length, we set the maximum input length to 50K. We conduct a zero-shot evaluation with a temperature of 0.7 and a top-p of 0.95.

#### 3.4 Baselines

We compare QWENLONG-L1 against the following state-of-the-art LRMs.

**Proprietary LRMs** OpenAI-o1-preview [15], Claude-3.7-Sonnet-Thinking [1], OpenAI-o3-mini [28], Qwen3-Plus [42], QwQ-Plus [43], and Gemini-2.0-Flash-Thinking [37].

**Open-Source LRMs** DeepSeek-R1 [11], Qwen3-235B-A22B [42], Qwen3-32B [42], QwQ-32B [43], R1-Distill-Qwen-32B [11], and R1-Distill-Qwen-14B [11].

<span id="page-6-0"></span><sup>&</sup>lt;sup>2</sup>For DocMath, we sample 75% items from each subset from its valid split for training and 25% for evaluation.

<span id="page-6-2"></span><sup>&</sup>lt;sup>3</sup>We exclude 7B/1.5B variants due to their mathematical reasoning feature inherent from Qwen2.5-Math [50].

<span id="page-6-3"></span><sup>&</sup>lt;sup>4</sup>We use the data from LongBench [2] for 2WikimultihopQA, HotpotQA, Musique, NarrativeQA, and Qasper.

<span id="page-7-0"></span>Table 4: Main results across seven long-context DocQA benchmarks. We highlight the **top-1** and **top-3** performance.  $\Delta$  indicates the performance gains and declines compared to the base models.

| Models                          | DocMath | Frames | 2Wiki  | HQA                 | Musi   | NarQA  | Qasp        | Avg.   |  |  |  |
|---------------------------------|---------|--------|--------|---------------------|--------|--------|-------------|--------|--|--|--|
| Proprietary LRMs                |         |        |        |                     |        |        |             |        |  |  |  |
| OpenAI-o1-preview               | 64.5    | 80.8   | 87.5   | 83.5                | 69.0   | 68.0   | 57.0        | 72.9   |  |  |  |
| Claude-3.7-Sonnet-Thinking      | 67.5    | 70.9   | 86.5   | 84.4                | 68.3   | 61.5   | 56.0        | 70.7   |  |  |  |
| OpenAI-o3-mini                  | 66.5    | 75.5   | 86.5   | 83.5                | 66.5   | 59.0   | 55.0        | 70.4   |  |  |  |
| Qwen3-Plus                      | 66.0    | 73.6   | 90.5   | 82.4                | 69.8   | 57.5   | 52.5        | 70.3   |  |  |  |
| QwQ-Plus                        | 64.5    | 73.5   | 89.0   | 81.0                | 66.5   | 62.0   | 53.5        | 70.0   |  |  |  |
| Gemini-2.0-Flash-Thinking       | 63.0    | 69.8   | 82.9   | 79.5                | 62.5   | 57.0   | 45.5        | 65.7   |  |  |  |
| Open-Source LRMs                |         |        |        |                     |        |        |             |        |  |  |  |
| DeepSeek-R1                     | 66.0    | 79.6   | 89.9   | 82.5                | 74.5   | 59.5   | 53.0        | 72.1   |  |  |  |
| Qwen3-235B-A22B                 | 67.5    | 74.6   | 91.5   | 84.4                | 63.3   | 60.0   | 53.0        | 70.6   |  |  |  |
| QwQ-32B                         | 59.5    | 72.9   | 90.5   | 78.5                | 66.0   | 58.0   | <i>57.5</i> | 69.0   |  |  |  |
| Qwen3-32B                       | 58.0    | 70.0   | 87.0   | 83.4                | 62.8   | 57.5   | 56.0        | 67.8   |  |  |  |
| R1-Distill-Qwen-32B             | 62.5    | 67.0   | 84.0   | 80.5                | 61.0   | 54.0   | 50.0        | 65.6   |  |  |  |
| R1-Distill-Qwen-14B             | 61.0    | 64.2   | 87.0   | 77.5                | 58.0   | 51.0   | 51.0        | 64.2   |  |  |  |
| Ours Methods                    |         |        |        |                     |        |        |             |        |  |  |  |
| R1-Distill-Qwen-14B-SFT         | 60.0    | 65.7   | 88.5   | 80.5                | 60.0   | 52.0   | 48.5        | 65.0   |  |  |  |
| $\Delta$ to R1-Distill-Qwen-14B | (-1.0)  | (+1.5) | (+1.5) | (+3.0)              | (+2.0) | (+1.0) | (-2.5)      | (+0.8) |  |  |  |
| QWENLONG-L1-14B-GRPO            | 65.0    | 68.7   | 88.5   | 86.5                | 63.5   | 53.5   | 51.5        | 68.2   |  |  |  |
| $\Delta$ to R1-Distill-Qwen-14B | (+4.0)  | (+4.5) | (+1.5) | ( <del>+9.0</del> ) | (+5.5) | (+2.5) | (+0.5)      | (+4.0) |  |  |  |
| QWENLONG-L1-14B-DAPO            | 65.5    | 67.4   | 89.0   | 84.0                | 63.0   | 57.0   | 52.5        | 68.3   |  |  |  |
| $\Delta$ to R1-Distill-Qwen-14B | (+4.5)  | (+3.2) | (+2.0) | (+6.5)              | (+5.0) | (+6.0) | (+1.5)      | (+4.1) |  |  |  |
| R1-Distill-Qwen-32B-SFT         | 65.0    | 71.6   | 87.0   | 80.5                | 65.5   | 57.5   | 54.0        | 68.7   |  |  |  |
| $\Delta$ to R1-Distill-Qwen-32B | (+2.5)  | (+4.6) | (+3.0) | (+0.0)              | (+4.5) | (+3.5) | (+4.0)      | (+3.2) |  |  |  |
| QWENLONG-L1-32B-GRPO            | 68.0    | 72.2   | 87.0   | 82.0                | 66.0   | 61.0   | 56.0        | 70.3   |  |  |  |
| $\Delta$ to R1-Distill-Qwen-32B | (+5.5)  | (+5.2) | (+3.0) | (+1.5)              | (+5.0) | (+7.0) | (+6.0)      | (+4.7) |  |  |  |
| QWENLONG-L1-32B-DAPO            | 67.5    | 70.1   | 90.5   | 83.0                | 69.0   | 56.0   | 58.5        | 70.7   |  |  |  |
| $\Delta$ to R1-Distill-Qwen-32B | (+5.0)  | (+3.1) | (+6.5) | (+2.5)              | (+8.0) | (+2.0) | (+8.5)      | (+5.1) |  |  |  |

<span id="page-7-1"></span>> **[图片提取文字 (无描述)]:**
> DocMath 2WikimQA Average Frames 99. 85 85 82-80.0 o1-preview = 80.8 96. 80-76.5 80 DeepSeek-R1 = 79.6 76-93. 75 75o1-preview = 72.9 DeepSeek-R1 = 89.9 70-DeepSeek-R1 = 72.1 90-70-Accuracy (%) DeepSeek-R1 = 66.0 o1-preview = 87.5 o1-preview = 64.5 64 16 16 16 16 HotpotQA Musique NarrativeQA Qasper 864 79-74 93 80 71-66. 89-74 DeepSeek-R1 = 74.5 o1-preview = 68.0 o1-preview = 69.2 o1-preview = 57.0 63 -58 85-DeepSeek-R1 = 59.5 68-DeepSeek-R1 = 53.0 o1-preview = 83.5 50-DeepSeek-R1 = 82.5 81 55 16 16 16 Number of Samples K (Log Scale)
![](_page_7_Figure_2.jpeg)

Figure 4: Pass@K rates of QWENLONG-L1-14B with different sample numbers across all benchmarks. We show that QWENLONG-L1-14B surpasses DeepSeek-R1 with a small sampling number.

# 4 Experimental Results

# 4.1 Main Results

Table 4 presents the overall performance of QWENLONG-L1 across seven long-context document question answering (DocQA) benchmarks. The key findings are as follows:

<span id="page-8-1"></span>> **[图片提取文字 (无描述)]:**
> Phased RL 0.60 Baseline Phased RL 68.3 Phased RL + RS 0.8 Phased RL II Accuracy (%) Accuracy (%) Reward Score 0.75 0.70 65.0 65.0 64.5 64.5 Higher Reward 0.50 0.10 0.65 Lower Gradient Norm GRPO SFT+GRPO 30 30 DAPO SFT+DAPO 10 20 Training Steps Training Steps (a) Overall Performance Comparison (b) Training Dynamics with/without SFT - Phased RL I Phased RL II w. RS 0.24 Phased RL II Phased RL II w/o. RS 0.19 w/o, Phased RL KL Divergence 0.22 Higher Entropy Reward Score More Stable KL Higher Entropy Entropy 81.0 Eutropy 0.18 0.17 0.16 0.01 0.14 0.16 Lower Reward 0.45 0.00 30 50 50 20 40 20 40 Training Steps Training Steps Training Steps Training Steps (d) Training Dynamics with/without RS (c) Training Dynamics with/without Phased RL
![](_page_8_Figure_0.jpeg)

Figure 5: Ablation studies of progressive context scaling strategy, where "Baseline" refers to the base or SFT model before RL training, "RL" refers to the naive single-stage RL, and "Phased RL" refers to the curriculum-guided phased RL. "RS" refers to the difficulty-aware retrospective sampling.

Limited Efficacy of SFT for Long-Context Reasoning. Since the base model, R1-Distill-Qwen, is primarily optimized for short-context reasoning tasks in mathematics, coding, and scientific domains, we conduct SFT to adapt it for long-context reasoning before RL training, as outlined in Section [2.3.](#page-4-0) Despite this intervention, the SFT model only shows an average gain of 0.8 points on 14B and 3.2 points on 32B. Furthermore, the improvements exhibit significant variability across benchmarks, suggesting limited generalizability of the SFT approach to long-context reasoning scenarios.

Significant Improvements via RL Integration. Through the integration of RL, QWENLONG-L1 exhibits remarkable advancements in long-context reasoning performance. Notably, QWENLONG-L1-14B achieves an average improvement of 4.1 and 4.0 points over the base model with DAPO and GRPO, surpassing the 0.4 points improvement of the SFT baseline by a significant margin. Furthermore, when scaling to 32B base model, QWENLONG-L1-32B even demonstrates a 5.1 and 4.7 points performance increase with DAPO and GRPO. These results highlight the advanced capacity of RL approaches in refining the output distribution to address intricate, context-dependent reasoning problems through group-relative advantage estimation and incentives for on-policy sampled outputs.

Leading Performance among Flagship LRMs. Our evaluation demonstrates that QWENLONG-L1 achieves superior performance compared to leading proprietary and open-source LRMs. Specifically, QWENLONG-L1-14B achieves an average score of 68.3, surpassing Gemini-2.0-Flash-Thinking, R1-Distill-Qwen-32B, and Qwen3-32B, while mathcing the performance of QwQ-32B. Moreover, QWENLONG-L1-32B achieves an average score of 70.7, exceeding the performance of QwQ-Plus, Qwen3-Plus, Qwen3-235B-A22B, and OpenAI-o3-mini, even comparable to Claude-3.7-Sonnet-Thinking, demonstrating leading performance among state-of-the-art flagship LRMs.

Additional Enhancements by Test-Time Scaling. We further conduct experiments to analyze the test-time scaling performance of QWENLONG-L1. Following established works [\[11,](#page-20-1) [45\]](#page-21-6), we generate 16 candidate outputs per input question and evaluate Pass@K to quantify exploratory capability across all benchmarks. As illustrated in Figure [4,](#page-7-1) QWENLONG-L1-14B exhibits consistent performance enhancements with increased sampling scales. Notably, QWENLONG-L1-14B demonstrates remarkable gains, even surpassing DeepSeek-R1 and OpenAI-o1-preview with a small sample size. Specifically, it achieves an average Pass@2 rate of 73.7 across all benchmarks, outperforming both DeepSeek-R1's 72.1 and OpenAI-o1-previews's 72.9, highlighting the efficacy of test-time scaling. Moreover, the significant gap between Pass@K and Pass@1 metrics indicates further potential for RL training to better bridge the transition from diverse exploration to optimal exploitation.

# <span id="page-8-0"></span>4.2 Ablation Studies

In this section, we conduct ablation studies to investigate the key components in QWENLONG-L1 that enable successful progressive context scaling for long-context reasoning RL, including warmup supervised fine-tuning, curriculum-guided phased reinforcement learning, and difficulty-aware retrospective sampling, with the experimental results shown in Figure 5.

Warm-up Supervised Fine-tuning. To illustrate the influence of warm-up SFT, we first evaluate the overall performance of models trained with and without this preparatory phase across seven benchmarks, using various RL algorithms and training strategies. As illustrated in Figure 5 (a), integration warm-up SFT yields significant performance improvements in all experimental setups. To further explore the mechanism of warm-up SFT in RL dynamics, Figure 5 (b) tracks the reward scores and gradient norm during training. The results reveal that warm-up SFT not only accelerates reward improvements but also sustains lower gradient norm during different RL phases, validating its capacity to prioritize performance gains over format alignment when transitioning models from short-context to long-context reasoning tasks. These findings highlight the necessity of integrating SFT as a precursor to providing a robust and efficient initialization for RL training.

Curriculum-Guided Phased Reinforcement Learning. As shown in Figure 5 (a), we conduct a comparative analysis between naive single-stage RL and the proposed curriculum-guided phased RL, with different training configurations: GRPO, DAPO, SFT + GRPO, and SFT + DAPO. The results demonstrate that our phased RL methodology achieves substantial performance improvements. We also note that this improvement is less pronounced when models are initialized with SFT, suggesting that warm-up training partially compensates for curriculum design. Further analysis in Figure 5 (c) reveals that single-stage RL exhibits heightened instability, as demonstrated by fluctuating KL divergence and entropy collapse. These results confirm the pivotal role of curriculum-guided phased training in the stable policy evolution from short-context to long-context reasoning RL.

**Difficulty-Aware Retrospective Sampling**. To maintain a wild exploration of hard examples, we introduce a difficulty-aware retrospective sampling strategy to integrate a subset of hard samples from prior training phases into the current training data. As illustrated in 5 (a), this strategy yields further performance enhancements with phased RL. Notably, in Figure 5 (d), despite undergoing phase I RL training, these retained hard examples also lead to significantly lower reward and higher policy entropy, which incentivize the policy model to augment the exploration process.

### 4.3 Additional Analysis

In this section, we investigate the questions pertaining to the development of long-context LRMs, focusing on the trade-off between SFT and RL in optimizing long-context reasoning capabilities, and the emergence and dynamics of long-context reasoning behaviors during training.

Trade-off between SFT and RL in Optimization. As discussed in Section 4.2, SFT offers a robust initialization for RL training. However, given that the initial SFT phase in our experiments relied on short-context training data, a critical question arises regarding the role of long-context SFT and its impact on RL. To this end, we train a long-context SFT model using 10K context-question-answer triplets distilled from DeepSeek-R1, maintaining the same data distribution as the short-context SFT phase. This long-context SFT model serves as the starting point for single-stage RL training, without progressive context scaling due to its inherent long-context capability.

<span id="page-9-0"></span>> **[图片提取文字 (无描述)]:**
> 68.2 Before RL 68 After RL 67.4 Accuracy (%) 67.166.7 65.0 64.5 64 Baseline Short SFT Long SFT
![](_page_9_Figure_7.jpeg)

Figure 6: Comparison between different models before and after RL, where "Baseline" denotes the base model, "Short SFT" denotes the short-context SFT model, and "Long SFT" denotes the long-context SFT model.

As shown in Figure 6, the long-context SFT model surpasses both the base model by 2.6 points and the short-context SFT model by 2.1 points. Despite the requirement for more training data, SFT offers distinct practical advantages, including reduced computational complexity, minimal infrastructure demands, and diminished reliance on specialized technical expertise, thereby positioning it as an economical strategy for performance enhancement [11]. However, further RL applied to the long-context SFT model yields marginal improvements, with only 0.3 points gains and a 67.4 final score, significantly underperforming the 3.2 points improvements and a 68.2 final score achieved when RL is applied to the short-context SFT model. These results highlight two insights for long-context LRM development: (1) SFT and RL exhibit distinct yet complementary purposes—SFT achieves acceptable performance with less effort, whereas RL is indispensable for attaining optimal results;

<span id="page-10-0"></span>> **[图片提取文字 (无描述)]:**
> Backtracking Verification Subgoal Setting Grounding SFT = 4.79 $\Gamma = 9.17$ SFT = 3.33SFT = 2.908 SFT = 65= 65= 65SFT = 65Beha 2.2 40 40 40 Training Steps Training Steps Training Steps Training Steps
![](_page_10_Figure_0.jpeg)

Figure 7: The change in reasoning behavior over training steps. We focus on four core reasoning behaviors, including long-context specific *grounding* and three general reasoning strategies: *subgoal setting*, *backtracking*, and *verification*. During RL training, we show that all the behaviors increase progressively with the corresponding performance gains. However, despite SFT leading to significantly increased reasoning behaviors, these efforts fail to improve the final performance.

(2) Maximizing performance necessitates prioritizing RL over SFT, as excessive focus on SFT risks trapping models in local optima, thereby constraining opportunities for RL improvements.

Emergence and Dynamics of Long-Context Reasoning Behaviors. The reasoning behaviors critically shape LRMs' reasoning trajectories and rewards [11, 14]. To investigate these dynamics, we follow recent studies [56, 23] to analyze the evolution of reasoning behaviors during SFT and RL training. Specifically, we use DeepSeek-V3 [21] to extract and track the shifts of the average count of four core reasoning behaviors over training steps, including long-context specific *grounding* and three general reasoning behaviors: *subgoal setting*, *backtracking*, and *verification*:

- **Grounding**: The model recalls related information in the long context to support subsequent reasoning, e.g., "Let me look through the provided text to find...".
- **Subgoal Setting**: The model decomposes complex questions into multiple manageable subgoals to solve them step-by-step, *e.g.*, "*To solve this, we first need to...*".
- **Backtracking**: The model identifies errors in generations and go back to revise its approach iteratively, *e.g.*, "*This approach won't work because...*".
- **Verification**: The model validates the predicted answers systematically to ensure solution correctness with self-reflection, *e.g.*, "Let's verify this result by...".

The results in Figure 7 reveal three insights: (1) All LRMs exhibit marked reasoning behaviors, with long-context grounding occurring most frequently, underscoring its effectiveness in managing contextual dependencies during reasoning. (2) RL training amplifies these behaviors progressively, correlating with significant performance gains, suggesting RL's efficacy in refining the output space to prioritize reasoning patterns conducive to accurate solutions. (3) In contrast, while SFT models demonstrate increased reasoning behaviors, these adjustments fail to transform into performance improvements, likely due to SFT's inherent reliance on imitation learning, which prioritizes superficial pattern alignment over substantive reasoning skill development [59, 20].

### 5 Conclusion and Future Work

In this study, we explore the development of long-context LRMs with robust contextual grounding and reasoning capabilities through reinforcement learning (RL). We first propose the paradigm of *long-context reasoning RL* and identify *suboptimal training efficiency* and *unstable optimization process*. To address these challenges, we present QWENLONG-L1, a progressive context scaling RL framework designed to bridge the gap between short-context proficiency and long-context generalization. Specifically, the training process begins with a warm-up SFT, followed by a curriculum-guided phased RL, with a difficulty-aware retrospective sampling strategy. Experiments across seven long-context document question-answering benchmarks demonstrate that QWENLONG-L1 achieves leading performance among state-of-the-art proprietary LRMs. Specifically, QWENLONG-L1-14B outperforms Gemini-2.0-Flash-Thinking and Qwen3-32B, while QWENLONG-L1-32B further surpasses OpenAI-o3-mini, Qwen3-235B-A22B, and even matches Claude-3.7-Sonnet-Thinking. Our analysis yields three key insights for long-context reasoning RL, including the pivotal role of progressive context scaling in enabling stable adaptation, the necessity of prioritizing RL for optimal performance, and the increase of long-context reasoning behaviors during RL training for performance improvements.

Future work should prioritize three key avenues to advance long-context LRMs. First, scaling real-world tasks, like automated scientific research and long video analysis, will provide appropriate environments to enhance long-context comprehension and decision-making capabilities. Second, developing advanced architectures is essential, including optimized attention mechanisms, e.g., linear and sparse attention, and efficient infrastructures, e.g., asynchronous actor rollout and parameter updating. Third, rethinking long-context RL paradigms, such as transitioning from token-level to turnlevel markov decision process (MDP), might enable the breakdown of long-context into sequential interactions and optimizing them iteratively, paving the way for infinite-context RL systems.

# 6 Case Study

To demonstrate QWENLONG-L1's capabilities, we show two illustrative examples from our analysis.

Case complong-testmini-183: When evaluating QWENLONG-L1-14B against R1-Distill-Qwen-14B and R1-Distill-Qwen-14B-SFT, distinct behavioral differences emerge. In this instance, R1- Distill-Qwen-14B is misled by the detailed *"Interest is payable semi–annually beginning October 15, 2011"*. It grounds *irrelevant temporal* and *financial information*, and consequently *miscalculates the first-year interest payment*. Although R1-Distill-Qwen-14B-SFT is fine-tuned on high-quality SFT triplets distilled from DeepSeek-R1, it still fails to resolve this issue. Instead, it demonstrates persistent limitations by *entering a loop of over-analysis on unrelated documents*. The *uncertainty* and *self-doubting* behavior eventually causes it to *exhaust the maximum generation limit* (10,000 tokens) without providing the final answer. In contrast, while QWENLONG-L1-14B initially exhibits similar distractions, it quickly *engages in effective self-reflection*. Through *timely verification and backtracking*, it successfully *filters out the irrelevant details and arrives at the correct answer*.

Case complong-testmini-265: This example illustrates how long-context reasoning behaviors (e.g. *grounding*, *backtracking*, *verification*, *subgoal setting*) emerge and interplay in practice, complementing our behavior evolution analysis in Figure [7.](#page-10-0) This task requires the model to *infer the total interest accrued over two years based on dense financial filings*. Here, the model *correctly computes that the total interest is \$980,000* based on a principal of *\$4,900,000 at 10% annual interest over two years*. This success is attributed to the emergence of multiple reasoning behaviors, showcasing how long-context reasoning RL enhances not only improves answer accuracy but also fosters the development of internal cognitive behaviors during reasoning.

# DocMath complong-testmini-183

Ground Truth: 32.4

Prompt: Please read the following text and answer the question below.

<text>

...*127Docs*...

128: On March 22, 2011, we issued \$300.0 million in aggregate principal amount of 8.0% senior unsecured notes due 2019 (the "Notes") at an offering price equal to 100% of par. The Notes were sold in a private placement to eligible purchasers in a transaction exempt from the registration requirements of the Securities Act of 1933, as amended.

129: We received net proceeds of \$291.6 million, after deducting the discount of \$7.5 million and offering expenses of \$0.9 million. We used the net proceeds to repay indebtedness under our existing credit facility. The discount and the offering expenses are being amortized over the life of the Notes. The amortization is included in "Interest expense" on our unaudited condensed consolidated statements of operations.

130: The Notes were issued under an indenture dated March 22, 2011, (the "Indenture"), mature April 15, 2019, and bear interest at 8.0%. Interest is payable semi–annually beginning October 15, 2011. The Notes are general unsecured obligations and are effectively junior in right of payment to any of our secured indebtedness to the extent of the value of the collateral securing such indebtedness.

...*271Docs*...

</text>

What is the total capital cost, combining both the offering cost of the senior notes and interest payment in the first year, in million dollars?

Format your response as follows: "Therefore, the answer is (insert answer here)".

#### R1-Distill-Qwen-14B:

#### Response (2166 tokens):

## <think>

Alright, so I need to figure out the total capital cost for EV Energy Partners, L.P., combining both the offering cost of the senior notes and the interest payment in the first year. Let me go through the provided financial statements step by step to find all the relevant information.

First, I'll look for the senior notes offering costs. [...] The offering expenses were \$0.9 million, and the discount was \$7.5 million. [...] So the total offering costs (discount and expenses) are \$7.5 + \$0.9 = \$8.4 million.

Next, I need to find the interest payment for the first year. The senior notes have an 8% annual interest rate and are due in 2019. They were issued in March 2011, so the first interest period would be from March to September 2011, but the interest is actually payable semi-annually starting in October 2011. [...]

Since the notes were outstanding for six months, the semi-annual interest would be 8% / 2 = 4% of the principal. Therefore, the interest for the first year would be 4% of \$300 million, which is \$12 million. [...]

The interest is semi-annual, so the first payment would be in October 2011, and the second in April 2012. So for the first year (2011), they would have paid \$12 million in interest on the senior notes. [...]

So combining the offering cost (\$8.4 million) and the interest payment for the first year (\$12 million), the total capital cost is \$8.4 + \$12 = \$20.4 million.

[...]

### </think>

The total capital cost includes the offering costs and the interest payment for the first year. The offering costs for the senior notes were \$8.4 million, and the interest expense for the first year (2011) was \$12 million. Therefore, the total capital cost is \$8.4 million + \$12 million = \$20.4 million.

Therefore, the answer is \$20.4 million.

#### Grounding Behavior:

# Instances of Grounding Behavior:

- 1. Explicit Grounding: "Let me go through the provided financial statements step by step to find all the relevant information."
- 2. Explicit Grounding: "In the Notes section, under NOTE 5: FAIR VALUE MEASUREMENTS, I see that on March 22, 2011, they issued \$300 million in senior notes."
- 3. Explicit Grounding: "Wait, but the offering cost is mentioned in NOTE 9: OWNER'S EQUITY."
- 4. Explicit Grounding: "Looking at the balance sheet, under June 30, 2011, the long-term debt is \$480,183 thousand, which is \$480.183 million."
- 5. Explicit Grounding: "Wait, in NOTE 7: LONG-TERM DEBT, under the credit facility, it says that borrowings bear interest at a floating rate based on LIBOR."
- 6. Grounding Verification: "Wait, the financial statements show interest expense for the three months ended June 30, 2011 as \$8,124 thousand, which is \$8.124 million."
- 7. Grounding Verification: "Wait, but in the condensed statements, the interest expense for the three months is \$8.124 million, and for six months, it's \$13,283 thousand."
- 8. Grounding Verification: "Wait, the notes were issued in March 2011, so from March 2011 to March 2012 is the first year."
- 9. Grounding Verification: "Wait, in the financial statements, the interest expense for the six months ended June 30, 2011 is \$13.283 million, which includes the senior notes' interest for three months and the credit facility's interest for the entire six months."
- 10. Grounding Verification: "Wait, the senior notes were issued in March 2011, so the first interest period is from March 22, 2011 to September 22, 2011, which is about six months, and the interest would be 4% of \$300 million, which is \$12 million."
- 11. Grounding Verification: "Wait, but the offering cost is a one-time expense, and the interest is an ongoing expense."

12. Grounding Verification: "Wait, but the question is about the total capital cost, which usually includes both the initial costs and the interest expense over the period."

Each of these instances involves either explicit references to the document or verification steps that lead back to the document, thus counting as grounding behavior.

#### R1-Distill-Qwen-14B-SFT:

#### Response (10000 tokens):

#### <think>

Okay, let's tackle this question. [...]

First, I need to find where the offering costs for the senior notes are mentioned. [...] So the total offering costs here are \$7.5 million (discount) plus \$0.9 million (expenses), which adds up to \$8.4 million. Next, the interest payment for the first year. The senior notes were issued in March 2011 with a 8.0% annual interest rate. The total principal amount is \$300 million. [...] However, the notes mention that interest is payable semi-annually starting October 15, 2011. [...] Wait, the notes were issued on March 22, 2011, and the first interest payment is on October 15, 2011. [...]

The interest expense for the three months ended June 30, 2011, under "Interest expense" is \$8,124 thousand.

[...]

In the Condensed Consolidated Statements of Operations for the six months ended June 30, 2011, the interest expense is \$13,283 thousand.

[...]

In Note 7, under the senior notes, it says: "The discount and the offering expenses are being amortized over the life of the Notes. The amortization is included in 'Interest expense' on our unaudited condensed consolidated statements of operations." So the total initial costs for the senior notes are \$7.5 million + \$0.9 million = \$8.4 million. These are amortized over the life of the notes, so each year, \$8.4 million / 8 = \$1.05 million per year. For the first year (2011), that would be \$1.05 million. The question is about the interest payment in the first year.

[...]

The senior notes [...] bear 8% interest, payable semi-annually. The first payment is on October 15, 2011 [...]. So in the first year (2011), there's only one payment in October.

[...]

The coupon rate is 8%, so each payment is 4% of the principal. The principal is \$300 million, so each payment is \$12 million. The first payment is in October 2011, so in 2011, only one payment of \$12 million would be made. [...] So total capital cost would be \$8.4 million + \$12 million = \$20.4 million. [...]

Alternatively, [...] the amortization of the offering costs is \$8.4 million over 8 years, so for the first year, that's \$1.05 million. Therefore, [...] the interest paid would be \$12 million (coupon) + \$1.05 million (amortization) = \$13.05 million. The offering costs are \$8.4 million. So total capital cost would be \$8.4 million + \$13.05 million = \$21.45 million.

Alternatively, [...] the offering costs are \$8.4 million. The interest expense for the first year (2011) would be the interest paid on the senior notes

[...] (Reach maximum token limits)

## Grounding Behavior:

## Explanation of Grounding Behavior:

- 1. "Looking through the financial statements, in Note 7 under Long-Term Debt, there's a section about the 8.0% Senior Notes due 2019."
  - *Explicit Grounding:* The model directly references the financial statements and Note 7 to locate information about the senior notes.
- 2. "Under 'Proceeds from debt offering,' it says they received net proceeds of \$291.6 million after deducting a discount of \$7.5 million and offering expenses of \$0.9 million."
  - *Explicit Grounding:* The model extracts specific details from the document about the net proceeds, discount, and offering expenses.

- 3. "The interest expense for the six months ended June 30, 2011, is listed as \$8,124 thousand, but that's semi-annual."
  - *Explicit Grounding:* The model references the financial statements to find the interest expense for the six months ended June 30, 2011.
- 4. "Wait, the notes mention that interest is payable semi-annually starting October 15, 2011."
  - *Explicit Grounding:* The model refers back to the notes to verify the timing of interest payments.
- 5. "The interest expense for the three months ended June 30, 2011, under 'Interest expense' is \$8,124 thousand."
  - *Explicit Grounding:* The model again references the financial statements to confirm the interest expense for the three months ended June 30, 2011.
- 6. "Wait, the notes were issued on March 22, 2011, and the first interest payment is on October 15, 2011."
  - *Explicit Grounding:* The model returns to the document to verify the issuance date and the first interest payment date.
- 7. "Let's check the Notes again. In Note 7, under the senior notes due 2019, it says the net proceeds were \$291.6 million, after deducting a discount of \$7.5 million and offering expenses of \$0.9 million."
  - *Explicit Grounding:* The model revisits Note 7 to confirm the details about the net proceeds, discount, and offering expenses.
- 8. "Let's look at the interest expense line in the Statements of Operations."
  - *Explicit Grounding:* The model refers to the Statements of Operations to verify the interest expense.
- 9. "In the Condensed Consolidated Statements of Operations for the six months ended June 30, 2011, the interest expense is \$13,283 thousand."
  - *Explicit Grounding:* The model extracts the interest expense figure from the financial statements.
- 10. "In Note 7, under 'Interest expense,' it mentions that the amortization of the discount and deferred loan costs is included in interest expense."
  - *Explicit Grounding:* The model refers back to Note 7 to confirm how the amortization of discount and expenses is treated.
- 11. "Let's check the Notes again. In Note 7, the senior notes' discount and expenses are amortized over the life of the notes."
  - *Explicit Grounding:* The model revisits Note 7 to verify the amortization period for the discount and expenses.
- 12. "Let's check the interest expense in the Statements of Operations. For the six months ended June 30, 2011, the interest expense is \$13,283 thousand."
  - *Explicit Grounding:* The model refers to the Statements of Operations to confirm the interest expense figure.

Each of these steps involves explicit grounding behavior, where the model directly references or returns to the original document to verify or extract information.

# QWENLONG-L1-14B:

#### Response (5304 tokens):

# <think>

Alright, so I need to figure out the total capital cost for EV Energy Partners, L.P. in the first year, combining both the offering cost of the senior notes and the interest payment. Let me go through the provided financial statements step by step to find the necessary information.

First, I'll look at the Notes to the Condensed Consolidated Financial Statements. In Note 7, under "8.0% Senior Notes due 2019," [...] The discount was \$7.5 million, and the offering expenses were \$0.9 million. So, the total offering costs here would be \$7.5 million (discount) + \$0.9 million (expenses) = \$8.4 million.

Next, I need to find the interest payment for the first year.

[...]

The notes were issued in March 2011, so the full year's interest would be 8% of \$300 million, which is \$24 million. But since the interest is semi-annual and the first payment was in October 2011, the interest for the first year (2011) might only be half of that, so \$12 million. Wait, no, that's not correct because the interest is calculated on the outstanding principal for each period. Let me think again.

[...]

The senior notes have a fixed rate of 8.0%. The interest expense for the senior notes would be 8% of \$300 million, which is \$24 million annually [...]. I think we should consider the full year's interest, which would be \$24 million.

[...]

I'm going to go with the full year's interest, which is \$24 million, plus the offering costs of \$8.4 million, totaling \$32.4 million.

## </think>

The total capital cost includes both the offering costs and the interest payment for the first year. The offering costs for the senior notes were \$8.4 million (discount of \$7.5 million and expenses of \$0.9 million). The interest payment for the first year (2011) is the full annual interest of 8% on the \$300 million senior notes, which is \$24 million. Adding these together gives a total capital cost of \$32.4 million.

Therefore, the answer is \$32.4 million.

#### Grounding Behavior:

#### Explanation of Grounding Behavior:

- 1. Explicit Grounding: The model repeatedly references specific sections of the document, such as "Note 7," "Condensed Consolidated Financial Statements," and "Condensed Consolidated Statements of Operations," to extract information about the senior notes, offering costs, and interest expenses. For example:
  - "First, I'll look at the Notes to the Condensed Consolidated Financial Statements. In Note 7, under '8.0% Senior Notes due 2019,' it mentions that on March 22, 2011, they issued \$300 million in aggregate principal amount of senior notes."
  - "Looking at the financial statements, in the Condensed Consolidated Statements of Operations, under 'Interest expense,' for the six months ended June 30, 2011, it's \$13,283 thousand."
- 2. Grounding Verification: The model frequently double-checks its understanding by returning to the document to verify details, such as the timing of interest payments, the breakdown of interest expenses, and the calculation of offering costs. For example:
  - "Wait, in the financial statements, for the six months ended June 30, 2011, the interest expense is \$13,283 thousand. Let me check if that includes the senior notes."
  - "Wait, in Note 7, it says that the net proceeds from the senior notes were \$291.6 million, after deducting the discount of \$7.5 million and offering expenses of \$0.9 million."
- 3. Repeated Verifications: The model revisits the same sections multiple times to confirm calculations or clarify ambiguities, such as the interest payment for the first year and the breakdown of offering costs. For example:
  - "Wait, let me go back to the question: 'What is the total capital cost, combining both the offering cost of the senior notes and interest payment in the first year, in million dollars?"'
  - "Wait, in the financial statements, the interest expense for the six months ended June 30, 2011, is \$13,283 thousand. Let me check if that includes the senior notes."

## DocMath complong-testmini-265

Ground Truth: 980000.0

Prompt: Please read the following text and answer the question below.

<text>

...*329Docs*...

330: In connection with our July 2021 acquisition of 100% of the equity of TopPop, on July 26, 2021, we issued to the sellers promissory notes in the aggregate principal amount of \$4,900,000 (the "TopPop Notes"). The TopPop Notes bear interest at the rate of 10% per annum, matured on July 26, 2022 and are secured by all of the outstanding membership interest in TopPop. Under the terms of the TopPop Notes, we have a five-day grace period to July 31, 2022 before an event of default under the TopPop

Notes occurs. Upon an event of default under the TopPop Notes, the holders of such TopPop Notes may exercise all rights and remedies available under the terms of the TopPop Notes or applicable laws, including to foreclose on certain collateral consisting of the membership interests of TopPop. On July 26, 2022, the total principal amount outstanding under the TopPop Notes was \$4,900,000, exclusive of accrued and unpaid interest.

331: We are currently in discussions with holders of the TopPop Notes regarding possible solutions for the payment of the TopPop Notes, including the possible extension of the maturity date of the TopPop Notes for an additional year. There can be no assurance that our discussions will be successful and if we are not successful in finding an acceptable resolution to the existing default or the impending event of default, the noteholders will be able to seek judgement for the full amount due and may seek to foreclose on our assets. If this occurs, any such remedy will have a material adverse effect on our business, results of operations and financial condition and is likely to negatively impact the price of our common stock. Holders of approximately \$3.55 million of these notes have agreed to extend the term for 30 days and have indicated that they will not seek cash settlement prior to August 2023.

...*30Docs*...

## </text>

How much total interest will the company have to pay under the new agreement until August 2023 in dollars if the company managed to extend the "TopPop Notes" maturity date for one more year? Format your response as follows: "Therefore, the answer is (insert answer here)".

#### QWENLONG-L1-14B:

#### Response (2192 tokens):

# <think>

Okay, so I need to figure out how much total interest the company will have to pay under the new agreement until August 2023 if they extend the "TopPop Notes" maturity date for one more year. Let me start by understanding the problem.

From the text, I see that the TopPop Notes were issued on July 26, 2021, with a principal amount of \$4,900,000. They bear an interest rate of 10% per annum. The original maturity was on July 26, 2022, but there's a five-day grace period until July 31, 2022, before a default occurs. However, the company is in discussions to extend the maturity date by an additional year, so the new maturity would be around August 2023.

I need to calculate the interest for this extended period. The original term was from July 26, 2021, to July 26, 2022, which is one year. If they extend it by one more year, the new maturity would be July 26, 2023, but the user mentioned until August 2023, so I'll consider the period up to August 2023.

First, I should determine the exact period for which the interest is to be calculated. The original notes were from July 26, 2021, to July 26, 2022. If extended for another year, the new maturity is July 26, 2023, but the user specifies until August 2023, which is a bit beyond that. However, I think the interest is calculated up to the maturity date, so maybe just until July 26, 2023, unless the extension is until August, in which case I need to adjust.

But the user says the company managed to extend the maturity date for one more year, so the total period would be two years from the original issue date. Let me check the details again.

Wait, the original notes matured on July 26, 2022, and the company is discussing an extension for an additional year. So the new maturity would be July 26, 2023, making the total term two years. But the user mentions until August 2023, so perhaps the extension is until August 31, 2023, or just an extra year beyond the original 2022 date.

I think the key is to calculate the interest for the two-year period from July 26, 2021, to July 26, 2023, and then maybe add a bit more if the extension is beyond that. But let's focus on the period until August 2023.

The interest rate is 10% per annum, and the notes are simple interest, I believe, unless stated otherwise. The interest is calculated on the principal amount, which is \$4,900,000.

So, the interest for one year would be \$4,900,000 \* 10% = \$490,000. If they extend it for another year, that's another \$490,000, making the total interest \$980,000.

But wait, the original notes had a maturity in 2022, and the extension is for one more year, so from 2022 to 2023. Let me check if the interest is only for the extended year or for the entire two years.

Looking at the text, it says the TopPop Notes bear interest at 10% per annum and mature on July 26, 2022. The extension is for an additional year, so the interest would be for the period from July 26, 2022, to July 26, 2023, which is one year. Therefore, the interest for the extended period would be \$490,000. But the user is asking for the total interest until August 2023. So if the extension is until August 2023, that's a bit more than a year. However, typically, interest is calculated on an annual basis, and if the extension is for a full year, the interest would be \$490,000. If the extension is only until August 2023, that's about 13 months, so the interest would be \$4,900,000 \* 10% \* (13/12) ≈ \$508,333.33, but I'm not sure if the extension is for a full year or just until August.

Wait, in the text under Note 7, it says: "The Company has agreed to keep this agreement in place and for the three months ended March 31, 2022, the Company has recorded consulting expense of \$45,000." That's about the leasehold improvements, not the TopPop Notes.

Looking again at Note 14, which is under the leasehold improvements section, I don't see any specific info on the TopPop Notes beyond their principal and interest rate. So, I'll assume that the extension is for an additional year beyond the original 2022 maturity, making the total term two years.

Therefore, the total interest would be \$490,000 for the first year and another \$490,000 for the second year, totaling \$980,000.

Wait, but in Note 7, under the leasehold improvements, it says: "The Company has agreed to keep this agreement in place and for the three months ended March 31, 2022, the Company has recorded consulting expense of \$45,000." No, that's a different note. I'm getting confused.

Wait, let me go back to the relevant sections. In Note 7 (which is actually under Item 7, line 15), it says: "The Promissory Notes bear interest at the rate of 10% per annum and mature on July 26, 2022." Then, in the Recent Developments section (Item 2, line 25), it mentions that the TopPop Notes have a maturity on July 26, 2022, and that the company is in discussions to possibly extend the maturity date for an additional year.

So, if the maturity is extended to August 2023, that's an additional year beyond 2022. Therefore, the interest for the original year (2021-2022) was \$490,000. Now, with the extension, they would have to pay interest for another year, so another \$490,000, totaling \$980,000.

But wait, in the text under Item 2, line 25, it says: "the Company is currently in discussions with holders regarding possible solutions for the payment of the Promissory Notes, including the possible extension for an additional year." So, the extension is for an additional year beyond the original 2022 maturity, making the new maturity July 26, 2023.

So, the interest would be for two years: from July 26, 2021, to July 26, 2023. Therefore, total interest would be \$4,900,000 \* 10% \* 2 = \$980,000.

But let me double-check. The original maturity was July 26, 2022, and the extension is for an additional year, so the notes would now mature on July 26, 2023. Therefore, the interest would be for the period from July 26, 2021, to July 26, 2023, which is exactly two years.

So, the total interest would be \$4,900,000 \* 10% \* 2 = \$980,000.

Wait, but in the text under Item 15, line 15, it says: "the Company was late in filing its Annual Report on Form 10-K for the year ended December 31, 2021, which was filed on June 15, 2022. During the first quarter ended March 31, 2022, there was no business activity or transactions." That doesn't affect the interest calculation.

Another point: the notes are secured by the membership interests in TopPop, but the interest calculation is straightforward. It's simple interest unless compounded, which isn't mentioned here. The text doesn't specify compounding, so I'll assume it's simple interest.

So, for each year, the interest is 10% of the principal. Since the principal remains \$4,900,000, and the period is two years, the total interest is \$490,000 per year \* 2 = \$980,000.

Wait, but in the recent developments (Item 2, line 25), it mentions that the TopPop Notes have a maturity on July 26, 2022, and the company is discussing an extension for an additional year. So, the extension would be until July 26, 2023, making it a two-year period.

Therefore, the total interest the company would have to pay until August 2023 under the new agreement is \$980,000.

# </think>

The TopPop Notes have a principal amount of \$4,900,000 and an annual interest rate of 10%. If the maturity date is extended for one additional year until August 2023, the total interest for the two-year period would be \$490,000 per year, totaling \$980,000.

Therefore, the answer is \$980, 000 .

## Grounding Behavior:

# Explanation of Grounding Behavior:

- 1. "From the text, I see that the TopPop Notes were issued on July 26, 2021, with a principal amount of \$4,900,000."
- 2. "Looking at the text, it says the TopPop Notes bear interest at 10% per annum and mature on July 26, 2022."

- 3. "Wait, in the text under Note 7, it says: 'The Company has agreed to keep this agreement in place and for the three months ended March 31, 2022, the Company has recorded consulting expense of \$45,000.'"
- 4. "Looking again at Note 14, which is under the leasehold improvements section, I don't see any specific info on the TopPop Notes beyond their principal and interest rate."
- 5. "Wait, let me go back to the relevant sections. In Note 7 (which is actually under Item 7, line 15), it says: 'The Promissory Notes bear interest at the rate of 10% per annum and mature on July 26, 2022.'"

### Backtracking Behavior:

- 1. First Backtracking: The model initially considers the period up to August 2023 but then questions whether the interest is calculated up to the maturity date (July 26, 2023) or beyond. It revises its approach to focus on the two-year period from July 26, 2021, to July 26, 2023.
- 2. Second Backtracking: The model initially calculates the interest for two years but then questions whether the interest is only for the extended year (2022-2023) or for the entire two years. It revises its approach to consider the interest for the extended year only.
- 3. Third Backtracking: The model considers the possibility of the extension being until August 2023 (13 months) and calculates the interest accordingly. However, it then questions whether the extension is for a full year or just until August and revises its approach to assume a full year.
- 4. Fourth Backtracking: The model mistakenly references Note 7 (leasehold improvements) instead of the relevant section for the TopPop Notes. It corrects itself by going back to the relevant sections (Note 14 and Item 2).
- 5. Fifth Backtracking: The model initially calculates the total interest as \$980,000 but then doublechecks the period and confirms that the interest is for two years (July 26, 2021, to July 26, 2023).
- 6. Sixth Backtracking: The model considers the possibility of compounded interest but revises its approach to assume simple interest since compounding is not mentioned in the text.

#### Verification Behavior:

The chain-of-reasoning provided contains several instances where the model checks and verifies intermediate results or reasoning steps. Here are the key instances of verification behavior:

#### 1. Verification of the period for interest calculation:

- The model checks the exact period for interest calculation, considering both original and extended maturity dates.
- Example statements:

"Let me check the details again"

"Wait, the original notes matured on July 26, 2022, and the company is discussing an extension for an additional year."

#### 2. Verification of the interest calculation:

- The model repeatedly verifies the interest calculation for the extended period.
- Example statements:

"So, the interest for one year would be \$4,900,000 \* 10% = \$490,000. If they extend it for another year, that's another \$490,000, making the total interest \$980,000."

"Wait, but in the text under Item 2, line 25, it says: 'the Company is currently in discussions with holders regarding possible solutions for the payment of the Promissory Notes, including the possible extension for an additional year.'"

#### 3. Verification of the extension period:

- The model examines whether the extension is for a full year or partial period.
- Example statement:

"But the user is asking for the total interest until August 2023. So if the extension is until August 2023, that's a bit more than a year."

#### 4. Verification of the interest type:

- The model confirms whether simple or compound interest applies.
- Example statement:

"The text doesn't specify compounding, so I'll assume it's simple interest."

#### 5. Final verification of the total interest:

- The model double-checks the complete interest calculation.
- Example statements:

"So, the total interest would be \$4,900,000 \* 10% \* 2 = \$980,000."

"Wait, but in the recent developments (Item 2, line 25), it mentions that the TopPop Notes have a maturity on July 26, 2022, and the company is discussing an extension for an additional year."

#### Subgoal Setting Behavior:

- 1. Understand the problem: The model starts by identifying the need to calculate the total interest under the new agreement.
- 2. Determine the period for interest calculation: The model breaks down the task by figuring out the exact period for which the interest needs to be calculated, considering the original and extended maturity dates.
- 3. Clarify the extension details: The model attempts to clarify whether the extension is for a full year or until August 2023, which affects the interest calculation.
- 4. Calculate interest for the extended period: The model calculates the interest for the extended period, considering whether it is a full year or a partial year.
- 5. Verify assumptions: The model checks the text to confirm assumptions about the interest rate, compounding, and the exact period of the extension.
- 6. Finalize the total interest: The model concludes by calculating the total interest for the two-year period based on the verified assumptions.

# References

- <span id="page-19-6"></span>[1] Anthropic. Claude 3.7 sonnet system card, Feburary 2025.
- <span id="page-19-7"></span>[2] Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. LongBench: A bilingual, multitask benchmark for long context understanding. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 3119–3137, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.172.
- <span id="page-19-3"></span>[3] Ganqu Cui, Lifan Yuan, Zefan Wang, Hanbin Wang, Wendi Li, Bingxiang He, Yuchen Fan, Tianyu Yu, Qixin Xu, Weize Chen, et al. Process reinforcement through implicit rewards. *arXiv preprint arXiv:2502.01456*, 2025.
- <span id="page-19-4"></span>[4] Kahneman Daniel. *Thinking, fast and slow*. 2017.
- <span id="page-19-5"></span>[5] Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A Smith, and Matt Gardner. A dataset of information-seeking questions and answers anchored in research papers. In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pages 4599–4610, 2021.
- <span id="page-19-0"></span>[6] DeepSeek-AI. Deepseek-r1-lite-preview is now live: unleashing supercharged reasoning power!, November 2024.
- <span id="page-19-2"></span>[7] Ahmed El-Kishky, Alexander Wei, Andre Saraiva, Borys Minaev, Daniel Selsam, David Dohan, Francis Song, Hunter Lightman, Ignasi Clavera, Jakub Pachocki, et al. Competitive programming with large reasoning models. *arXiv preprint arXiv:2502.06807*, 2025.
- <span id="page-19-1"></span>[8] Hugging Face. Open r1: A fully open reproduction of deepseek-r1, January 2025.

- <span id="page-20-5"></span>[9] Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Hannaneh Hajishirzi, Yoon Kim, and Hao Peng. Data engineering for scaling language models to 128k context. In *International Conference on Machine Learning*, 2024.
- <span id="page-20-6"></span>[10] Tianyu Gao, Alexander Wettig, Howard Yen, and Danqi Chen. How to train long-context language models (effectively). *arXiv preprint arXiv:2410.02660*, 2024.
- <span id="page-20-1"></span>[11] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.
- <span id="page-20-4"></span>[12] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented language model pre-training. In *International conference on machine learning*, pages 3929–3938. PMLR, 2020.
- <span id="page-20-8"></span>[13] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. In *Proceedings of the 28th International Conference on Computational Linguistics*, pages 6609–6625, 2020.
- <span id="page-20-9"></span>[14] Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, and Heung-Yeung Shum Xiangyu Zhang. Open-reasoner-zero: An open source approach to scaling reinforcement learning on the base model, 2025.
- <span id="page-20-0"></span>[15] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. *arXiv preprint arXiv:2412.16720*, 2024.
- <span id="page-20-10"></span>[16] Bowen Jin, Hansi Zeng, Zhenrui Yue, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. *arXiv preprint arXiv:2503.09516*, 2025.
- <span id="page-20-7"></span>[17] Tomáš Kocisk ˇ y, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, Gábor ` Melis, and Edward Grefenstette. The narrativeqa reading comprehension challenge. *Transactions of the Association for Computational Linguistics*, 6:317–328, 2018.
- <span id="page-20-12"></span>[18] Satyapriya Krishna, Kalpesh Krishna, Anhad Mohananey, Steven Schwarcz, Adam Stambler, Shyam Upadhyay, and Manaal Faruqui. Fact, fetch, and reason: A unified evaluation of retrieval-augmented generation. *arXiv preprint arXiv:2409.12941*, 2024.
- <span id="page-20-11"></span>[19] Ming Li, Yong Zhang, Zhitao Li, Jiuhai Chen, Lichang Chen, Ning Cheng, Jianzong Wang, Tianyi Zhou, and Jing Xiao. From quantity to quality: Boosting llm performance with selfguided data selection for instruction tuning. In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, pages 7595–7628, 2024.
- <span id="page-20-14"></span>[20] Bill Yuchen Lin, Abhilasha Ravichander, Ximing Lu, Nouha Dziri, Melanie Sclar, Khyathi Chandu, Chandra Bhagavatula, and Yejin Choi. The unlocking spell on base llms: Rethinking alignment via in-context learning. In *The Twelfth International Conference on Learning Representations*, 2023.
- <span id="page-20-13"></span>[21] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. *arXiv preprint arXiv:2412.19437*, 2024.
- <span id="page-20-3"></span>[22] Jiaheng Liu, Dawei Zhu, Zhiqi Bai, Yancheng He, Huanxuan Liao, Haoran Que, Zekun Wang, Chenchen Zhang, Ge Zhang, Jiebin Zhang, et al. A comprehensive survey on long context language modeling. *arXiv preprint arXiv:2503.17407*, 2025.
- <span id="page-20-2"></span>[23] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. *arXiv preprint arXiv:2503.20783*, 2025.

- <span id="page-21-4"></span>[24] Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Tianjun Zhang, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl, 2025.
- <span id="page-21-18"></span>[25] Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling. *arXiv preprint arXiv:2501.19393*, 2025.
- <span id="page-21-0"></span>[26] OpenAI. Learning to reason with llms, September 2024.
- <span id="page-21-8"></span>[27] OpenAI. Introducing deep research, February 2025.
- <span id="page-21-19"></span>[28] OpenAI. Openai o3-mini system card, January 2025.
- <span id="page-21-3"></span>[29] Jiayi Pan, Junjie Zhang, Xingyao Wang, Lifan Yuan, Hao Peng, and Alane Suhr. Tinyzero: Clean, minimal, accessible reproduction of deepseek r1-zero, Janurary 2025.
- <span id="page-21-10"></span>[30] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. In-context retrieval-augmented language models. *Transactions of the Association for Computational Linguistics*, 11:1316–1331, 2023.
- <span id="page-21-14"></span>[31] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. Highdimensional continuous control using generalized advantage estimation. *arXiv preprint arXiv:1506.02438*, 2015.
- <span id="page-21-12"></span>[32] John Schulman, Xi Chen, and Pieter Abbeel. Equivalence between policy gradients and soft q-learning. *arXiv preprint arXiv:1704.06440*, 2017.
- <span id="page-21-13"></span>[33] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*, 2017.
- <span id="page-21-11"></span>[34] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*, 2024.
- <span id="page-21-15"></span>[35] Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov, and David Krueger. Defining and characterizing reward gaming. *Advances in Neural Information Processing Systems*, 35:9460– 9471, 2022.
- <span id="page-21-16"></span>[36] Yixuan Tang and Yi Yang. Multihop-rag: Benchmarking retrieval-augmented generation for multi-hop queries. In *First Conference on Language Modeling*, 2024.
- <span id="page-21-21"></span>[37] Gemini Team. Gemini 2.0 flash thinking, December 2024.
- <span id="page-21-7"></span>[38] Gemini Team. Try deep research and our new experimental model in gemini, your ai assistant, December 2024.
- <span id="page-21-5"></span>[39] NovaSky Team. Unlocking the potential of reinforcement learning in improving reasoning models, Feburary 2025.
- <span id="page-21-9"></span>[40] Perplexity Team. Introducing perplexity deep research, February 2025.
- <span id="page-21-1"></span>[41] Qwen Team. Qwq: Reflect deeply on the boundaries of the unknown, November 2024.
- <span id="page-21-20"></span>[42] Qwen Team. Qwen3: Think deeper, act faster, April 2025.
- <span id="page-21-2"></span>[43] Qwen Team. Qwq-32b: Embracing the power of reinforcement learning, March 2025.
- <span id="page-21-17"></span>[44] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop questions via single-hop question composition. *Transactions of the Association for Computational Linguistics*, 10:539–554, 2022.
- <span id="page-21-6"></span>[45] Liang Wen, Yunke Cai, Fenrui Xiao, Xin He, Qi An, Zhenyu Duan, Yimin Du, Junchen Liu, Lifu Tang, Xiaowei Lv, Haosheng Zou, Yongchao Deng, Shousheng Jia, and Xiangzheng Zhang. Light-r1: Curriculum sft, dpo and rl for long cot from scratch and beyond. *arXiv preprint arXiv:2503.10460*, 2025.

- <span id="page-22-4"></span>[46] Siwei Wu, Zhongyuan Peng, Xinrun Du, Tuney Zheng, Minghao Liu, Jialong Wu, Jiachen Ma, Yizhi Li, Jian Yang, Wangchunshu Zhou, et al. A comparative study on reasoning patterns of openai's o1 model. *arXiv preprint arXiv:2410.13639*, 2024.
- <span id="page-22-1"></span>[47] Tian Xie, Zitian Gao, Qingnan Ren, Haoming Luo, Yuqian Hong, Bryan Dai, Joey Zhou, Kai Qiu, Zhirong Wu, and Chong Luo. Logic-rl: Unleashing llm reasoning with rule-based reinforcement learning. *arXiv preprint arXiv:2502.14768*, 2025.
- <span id="page-22-6"></span>[48] Wenhan Xiong, Jingyu Liu, Igor Molybog, Hejia Zhang, Prajjwal Bhargava, Rui Hou, Louis Martin, Rashi Rungta, Karthik Abinav Sankararaman, Barlas Oguz, et al. Effective long-context scaling of foundation models. In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, pages 4643–4663, 2024.
- <span id="page-22-0"></span>[49] Fengli Xu, Qianyue Hao, Zefang Zong, Jingwei Wang, Yunke Zhang, Jingyi Wang, Xiaochong Lan, Jiahui Gong, Tianjian Ouyang, Fanjin Meng, et al. Towards large reasoning models: A survey of reinforced reasoning with large language models. *arXiv preprint arXiv:2501.09686*, 2025.
- <span id="page-22-11"></span>[50] An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, et al. Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement. *arXiv preprint arXiv:2409.12122*, 2024.
- <span id="page-22-7"></span>[51] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 2369–2380, 2018.
- <span id="page-22-5"></span>[52] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In *International Conference on Learning Representations*, 2023.
- <span id="page-22-13"></span>[53] Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. Limo: Less is more for reasoning. *arXiv preprint arXiv:2502.03387*, 2025.
- <span id="page-22-3"></span>[54] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. *arXiv preprint arXiv:2503.14476*, 2025.
- <span id="page-22-2"></span>[55] Yufeng Yuan, Yu Yue, Ruofei Zhu, Tiantian Fan, and Lin Yan. What's behind ppo's collapse in long-cot? value optimization holds the secret. *arXiv preprint arXiv:2503.01491*, 2025.
- <span id="page-22-8"></span>[56] Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild. *arXiv preprint arXiv:2503.18892*, 2025.
- <span id="page-22-12"></span>[57] Yilun Zhao, Yitao Long, Hongjun Liu, Ryo Kamoi, Linyong Nan, Lyuhao Chen, Yixin Liu, Xiangru Tang, Rui Zhang, and Arman Cohan. Docmath-eval: Evaluating math reasoning capabilities of llms in understanding long and specialized documents. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 16103–16120, 2024.
- <span id="page-22-10"></span>[58] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. *Advances in Neural Information Processing Systems*, 36:46595–46623, 2023.
- <span id="page-22-9"></span>[59] Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. *Advances in Neural Information Processing Systems*, 36:55006–55021, 2023.