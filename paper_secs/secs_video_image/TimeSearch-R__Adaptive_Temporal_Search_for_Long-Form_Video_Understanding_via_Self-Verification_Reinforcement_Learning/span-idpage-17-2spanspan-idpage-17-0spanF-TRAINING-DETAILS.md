# <span id="page-17-2"></span><span id="page-17-0"></span>F TRAINING DETAILS

Table 5: Training hyperparameters of TimeSearch-R.

| Category             | Parameter                      | Value          |
|----------------------|--------------------------------|----------------|
|                      | Max FPS                        | 2              |
|                      | Max Frames per Video           | 768            |
| Video Processing     | Total Video Tokens             | 10,240         |
|                      | Min Tokens per Frame           | 12             |
|                      | Max Tokens per Frame           | 256            |
|                      | Max Search Turns               | 8              |
| Interaction Settings | Max Completion Length per Turn | 256            |
|                      | Number of Generations          | 8              |
|                      | KL Penalty Coefficient (β)     | 0.005          |
|                      | Scale Rewards                  | false          |
| GRPO Training        | Batch Size per GPU             | 1              |
|                      | Gradient Accumulation Steps    | 2              |
|                      | DeepSpeed Configuration        | ZeRO-3 Offload |
| Infrastructure       | VLLM Mode                      | colocate       |
|                      | Replay Buffer                  | true           |

We summarize the key hyperparameters in Table [5](#page-17-2) for reproducibility.

Training Configuration. TimeSearch-R employs a distributed training setup using PyTorch's native distributed data parallel framework with ZeRO-3 memory optimization through DeepSpeed. The training process leverages gradient accumulation to simulate larger batch sizes while maintaining memory efficiency on GPU clusters. We utilize mixed precision training with bfloat16 to accelerate computation while preserving numerical stability, coupled with Flash Attention 2.0 for efficient attention computation.

GRPO Training Setup. The reinforcement learning phase uses Group Relative Policy Optimization with 8 generations per prompt to provide sufficient policy gradient estimates. The KL divergence penalty coefficient β is set to 0.005 to balance between reward optimization and policy regularization. We employ VLLM in colocate mode for efficient inference during rollout generation, enabling faster

policy updates. This RL training stage is implemented on top of the TRL library [\(von Werra et al.,](#page-11-14) [2020\)](#page-11-14), following standard practice for outcome-driven policy optimization in large language models.

Video Processing Configuration. The model processes videos with a maximum of 768 frames and allocates up to 10,240 tokens for video content representation. Each interaction turn is limited to 8 search operations, with a maximum of 8 interaction turns per question to ensure comprehensive temporal exploration while maintaining computational efficiency. Frame tokens are dynamically allocated between 12 and 256 tokens per frame based on content complexity and relevance.

