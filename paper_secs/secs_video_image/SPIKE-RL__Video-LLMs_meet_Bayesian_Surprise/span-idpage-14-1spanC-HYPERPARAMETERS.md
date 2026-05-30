# <span id="page-14-1"></span>C HYPERPARAMETERS

### C.1 TRAINING

We train using 4 H100s on a single node with DeepSpeed ZeRO-3 offload. All runs use Qwen2.5- VL-7B-Instruct as the backbone, with FlashAttention-2, bfloat16 precision, and PEFT enabled.

| Hyperparameter                  | Value       |  |  |  |
|---------------------------------|-------------|--|--|--|
| Learning rate                   | 1 × 10−6    |  |  |  |
| GRPO β                          | 0.1         |  |  |  |
| Number of GRPO Rollouts         | 3           |  |  |  |
| Number of Hypotheses per window | 3           |  |  |  |
| Max prompt length               | 8192 tokens |  |  |  |
| Training samples                | 2000        |  |  |  |
| Epochs                          | 1           |  |  |  |
| Per-device batch size           | 1           |  |  |  |
| Effective global batch size     | 4           |  |  |  |
| Random seed                     | 42          |  |  |  |

Table 3: Key hyperparameters for GRPO training.

### C.2 INFERENCE

For both SPIKE and SPIKE-RL, we maintain a hypothesis set N = 3 per time step. We use a prior window of W = 4 frames, and the frames for surprise scoring are allocated in proportion to the video duration, F = f(duration). Videos up to a minute are assigned a base budget of 8 frames. For longer videos, the budget continues to double with each additional minute.

