# Pairwise Comparison Reward Mechanism.

tion for the score using text elements."

}} "'

We use the Qwen-Plus [\(Yang et al.,](#page-11-11) [2024\)](#page-11-11) model to judge the quality of the generated responses. The pairwise comparison prompts used in our experiment are adapted from [\(Zheng et al.,](#page-11-6) [2023\)](#page-11-6) and [\(Wu et al.,](#page-11-2) [2025c\)](#page-11-2).

For the training samples in LongWriter [\(Bai](#page-9-1) [et al.,](#page-9-1) [2024b\)](#page-9-1) dataset, we use the original evaluation dimensions and the prompt is as follows.

Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie. NOTE: If the response contains

severe repetition or redundancy, it should be viewed as low quality score, losing the comparison.

User Question {question}

The Start of Assistant A's Answer {answer\_a}

The End of Assistant A's Answer

The Start of Assistant B's Answer {answer\_b}

The End of Assistant B's Answer

For the training samples in WritingBench (Wu et al., 2025c) training dataset, we use the generated criteria as the original paper recommends and the prompt is as follows.

### Criteria Pairwise Comparison Prompt

Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider the following dimensions. criteria

Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie. NOTE: If the response contains severe repetition or redundancy, it should be viewed as low quality score, losing the comparison.

User Question {question}

The Start of Assistant A's Answer

{answer\_a}
The End of Assistant A's Answer

The Start of Assistant B's Answer {answer\_b}
The End of Assistant B's Answer

### A.2 Training Parameters

We display the key training parameters used in our training experiments. We adopt the effective reinforcement training framework VeRL (Sheng et al., 2024) to train our models. In our experiment, we use the proximal policy optimization (PPO) (Schulman et al., 2017) algorithm with generalized advantage estimation (GAE) as the advantage estimator. In the parameters about GAE, we set  $\gamma = 1.0$  and  $\lambda = 1.0$ . The critic model shares the same backbone as the policy model and is initialized with identical parameters. As for the learning rate, we set a linear warm-up phase to stablize training: a low learning rate  $(1 \times 10^{-6})$  with a warm-up ratio of 0.4 for the actor model, while the critic adopts a higher learning rate  $(1 \times 10^{-5})$  with a warm-up ratio of 0.05.

The training process is conducted using a batch size of 32 for training, with a maximum prompt length of 4,096 tokens and response length capped at 10,000 tokens to accommodate long-form generation tasks. We enable the parameter/optimizer offloading via Fully Sharded Data Parallel (FSDP) to support efficient multi-GPU training and the training is conducted on 8 GPUs. We utilize a rollout strategy based on the vLLM engine with a tensor model parallel size of 2 and the sampling temperature during rollout is set to 1.0 throughout training to promote exploration. The KL divergence penalty is set to a modest coefficient of 0.001. We train each model for about 400 steps and evaluate the checkpoints on the validation set each 50 steps.

In terms of the parameters about the SFT stage, the details are documented in (Wu et al., 2025c) and we record the key parameters here. The models are trained using the AdamW optimizer with  $\beta_1 = 0.9$ ,  $\beta_2 = 0.999$ , and  $\epsilon = 1 \times 10^{-8}$ . The base learning rate was  $7 \times 10^{-6}$ , and a cosine decay scheduler with a warmup ratio of 0.1 (i.e., 10% of total training steps) was applied. Training runs for 5 epochs with a per-device batch size of 1 and gradient accumulation over 4 steps, yielding an effective batch size of 128 across 32 GPUs.

## <span id="page-14-0"></span>A.3 Reward model choice

To select an appropriate model to serve as the pairwise judge during training, we analyze the human agreement, cost and latency of several cuttingedge LLMs. As shown in Table [9,](#page-15-0) Qwen-plus has already achieved a high agreement with human judges, demonstrating its reward-giving capablities and making it a reliable choice for the training writer models. As shown in the following human evaluation results, qwen-plus has reached a remarkable agreement of 0.75, on par with R1 and surpassing gpt-4o-2024-11-20. Furthermore, GPT-4o and Claude models are widely adopted as judges in LLM benchmarks. If we use GPT series as training-time judges, the evaluation will be biased and unreliable. Therefore, we use a different training-time judge rather than the test-time judges.

RL requires a large amount of pairwise rewarding, therefore leading to huge API costs and high efficiency demands. As shown in the following results, qwen-plus has a remarkably lower price than gpt-4o and claude-3.7-sonnet and possesses the lowest first token latency.

## <span id="page-14-2"></span>A.4 Cost Analysis

To facilitate community reproduction and assess the feasibility of our approach, we provide a comprehensive breakdown of the monetary and computational costs involved. Our analysis distinguishes between the one-time offline costs of data curation (Stage 1) and the recurring costs of Reinforcement Learning (RL) training (Stage 2).

