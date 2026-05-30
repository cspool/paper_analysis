# B Experiments

#### B.1 Implementation Details

Fine-tuning. We adopt Qwen2.5-VL-7B-Instruct [Bai et al.](#page-10-18) [\[2025\]](#page-10-18) as our base VLM. The detailed training configurations are provided in Table [9.](#page-18-2) During fine-tuning, all components of the model are trainable except for the vision encoder. The training objective combines a cross-entropy loss for next-token prediction with a cosine similarity loss for aligning latent visual tokens, as described in Sec. [4.1.](#page-5-1) The loss weight γ for the visual alignment loss is set to the default value of 0.1. Both the training stage 1 and the training stage 2 employ the same configurations.

Table 9: Implementation details of Supervised Fine-tuning.

<span id="page-18-2"></span>

| Config                 | Value | Config                      | Value |
|------------------------|-------|-----------------------------|-------|
| optimizer              | Adam  | batch size                  | 8     |
| optimizer momentum β1  | 0.9   | gradient accumulation steps | 2     |
| optimizer momentum β2  | 0.95  | warmup steps                | 10    |
| optimizer weight decay | 0.01  | training epochs             | 10    |
| learning rate          | 1e-5  | loss weight γ               | 10    |

Table 10: Implementation details of Reinforcement Learning.

<span id="page-18-3"></span>

| Config                      | Value | Config                | Value |
|-----------------------------|-------|-----------------------|-------|
| prompt Length limit         | 1024  | response length limit | 1024  |
| learning rate               | 1e-6  | batch size            | 32    |
| gradient accumulation steps | 4     | rollout num           | 5     |
| training epochs             | 15    | mini batch size       | 8     |
| σf                          | 0.1   | σc                    | 0.9   |
| λkl                         | 0.01  | λen                   | 0.0   |

Reinforcement Learning. We adopt VERL [Sheng et al.](#page-11-18) [\[2024\]](#page-11-18) as the RL framework, and provide the detailed training settings in Tab. [10.](#page-18-3) Specifically, we utilize Group Relative Policy Optimization (GRPO) [Shao et al.](#page-11-13) [\[2024b\]](#page-11-13) for reinforcement learning. The reward function consists of a *format reward* and a *correctness reward*, weighted by σ<sup>f</sup> and σc, respectively. KL regularization is applied with a coefficient of λkl, while entropy regularization is disabled in the policy loss by setting λen = 0. For our Mirage, the KL divergence on latent visual tokens is omitted during RL training.

#### B.2 Efficiency Analysis

Both training stages of Mirage are conducted on a single NVIDIA H100 GPU. Taking the VSP spatial reasoning task as an example, Stage 1 completes in approximately 3.5 hours, while Stage 2 takes around 7.2 hours. For reference, text-only CoT SFT on the same hardware requires about 5.5 hours.