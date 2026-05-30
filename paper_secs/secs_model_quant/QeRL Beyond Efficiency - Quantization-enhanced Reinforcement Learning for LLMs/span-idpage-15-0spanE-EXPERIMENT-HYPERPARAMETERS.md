# <span id="page-15-0"></span>E EXPERIMENT HYPERPARAMETERS

Training Data and Reward Function We trained the Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct, and Qwen2.5-32B-Instruct models, which are widely used for evaluating reasoning capabilities. Unlike other studies that rely on math-specialized models, we aim to evaluate training performance starting from general-purpose base models. Additionally, QeRL can be smoothly transferred to other model families, such as the Qwen3 series. For the GSM8K dataset, we primarily trained the Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct models using GRPO, while for the BigMath dataset, we focused on training the Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct, and Qwen2.5-32B-Instruct models using DAPO. Specifically, for the 7B and 14B models, we selected data with medium to high difficulty levels (grades 3–5), and for the 32B model, we used high-difficulty data (grades 4–5). For problem prompts, we append the suffix Solve the following math problem step by step. The reasoning process and direct answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>: <think> ... </think> <answer> ... </answer>.

RL Training Configuration For both GRPO and DAPO, we use the hyperparameters in Tab[.4,](#page-15-2) without using entropy or KL losses. For 4-bit training, the learning rate is set to 1e −5 . However, due to the fragile of the BF16 model with LoRA, the learning rate can not be larger than 5e −6 , or it will collapse in the late training stage.

<span id="page-15-2"></span>

| Hyperparameter             | Value                                          |
|----------------------------|------------------------------------------------|
| Optimizer                  | AdamW-8bit                                     |
| Policy learning rate       | −5<br>−6<br>1e<br>(QeRL, QLoRA) / 5e<br>(LoRA) |
| Training batch size        | 128                                            |
| Samples per prompt         | 8 (GSM8K) / 16 (BigMath)                       |
| Policy updates per rollout | 4 (GSM8K, off-policy) / 1 (BigMath, on-policy) |
| Max response length        | 4096 (GSM8K) / 8192 (BigMath)                  |
| Rollout temperature        | 1.0                                            |
| Clip range ϵlow, ϵhigh     | 0.2, 0.28                                      |
| Noise range Zstart, Zend   | 1e-2, 5e-4                                     |

Table 4: Hyperparameters of GRPO and DAPO training

