# <span id="page-16-0"></span>D Implementation Details

Our training is performed using 8 NVIDIA A800 GPUs. The following settings are also applied to other baselines for fair comparisons.

#### D.1 Stage 1: SFT

We utilize the open-source training framework LLAMAFACTORY [\[58\]](#page-12-11) to perform SFT. The training is conducted with a batch size of 128 and a learning rate of 2e-4. We adopt a cosine learning rate scheduler with a 10% warm-up period over 6 epochs. To enhance training efficiency, we employ parameter-efficient training via Low-rank adaptation (LoRA) [\[17\]](#page-10-13) and DeepSpeed training with the ZeRO-3 optimization stage [\[36\]](#page-11-12). As a validation set, we sample 10% of the training data and keep the checkpoint with the lowest perplexity on the validation set for testing and the second stage.

Table 4: Effect of training with AIME-only data on reasoning format distribution.

<span id="page-17-3"></span>

| 7B Models               |          | CSQA          | OBQA     |               |  |  |
|-------------------------|----------|---------------|----------|---------------|--|--|
|                         | Long CoT | Other Formats | Long CoT | Other Formats |  |  |
| Training with AIME only | 79.4%    | 20.6%         | 83.0%    | 17.0%         |  |  |

<span id="page-17-4"></span>Table 5: Comparison of accuracy and token usage between different training recipes.

| 7B Models                             |              | CSQA       | OBQA         |            |
|---------------------------------------|--------------|------------|--------------|------------|
|                                       | Acc.         | Tok.       | Acc.         | Tok.       |
| Training with AIME only<br>ARM Recipe | 78.6<br>86.1 | 401<br>136 | 82.2<br>84.4 | 426<br>159 |

#### D.2 Stage 2: RL

We utilize the open-source training framework VeRL [\[40\]](#page-11-13) to perform RL. During training, we use a batch size of 1024 and generate 8 rollouts per prompt (G = 8), with a maximum rollout length of 4096 tokens. The model is trained with a mini-batch size of 180, a KL loss coefficient of 1e-3, and a total of 9 training epochs. The default sampling temperature is set to 1.0.

