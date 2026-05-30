# **C.2** Training Configuration

This section elaborates on the experimental setup and hyperparameter configurations employed throughout our study. Table 18 documents the shared training parameters applied consistently across all backbone models and datasets. To address the specific requirements of different tasks and model architectures, we additionally provide dataset-specific and model-specific configurations in Table 19, including learning rate schedules, batch size adjustments, and task-specific optimization strategies.

#### <span id="page-32-0"></span>C.3 Split-LoRA

We implement Split-LoRA as a representative MoE-based LoRA method, following the general framework described in Section 2.2. In our experiments, we incorporate a sigmoid activation function in the router to normalize expert selection scores. Thus, the gating function operates as  $G(x) = \text{sigmoid}(\text{top-}k(W_gx))$ , which ensures differentiable routing while maintaining the sparsity of expert activation. This configuration allows Split-LoRA to serve as a representative baseline for evaluating the effectiveness of MoE structures in LoRA.

<span id="page-33-2"></span>Table 18: General Training Hyperparameters for FlyLoRA. Shared configuration across all experiments, including rank settings, optimizer details, and architectural choices.

| Parameter                  | Value                       |  |  |
|----------------------------|-----------------------------|--|--|
| Total rank (r)             | 32                          |  |  |
| Scaling factor (α)         | 64                          |  |  |
| Activated rank             | 8                           |  |  |
| Target modules             | {q,k,v,o,gate,down,up}_proj |  |  |
| Optimizer                  | AdamW                       |  |  |
| Warmup ratio               | 0.01                        |  |  |
| Gradient accumulated batch | 128                         |  |  |
| Dropout rate               | 0.00                        |  |  |

<span id="page-33-3"></span>Table 19: Dataset-Specific and Model-Specific Training Configurations for FlyLoRA. Taskoptimized settings for Llama-3.1-8B and Qwen-2.5-7B across four benchmarks, showing variations in epoch counts, learning rates, and sequence lengths based on dataset characteristics and model requirements.

| Model        | Parameter           | MMLU     | ScienceQA | GSM8K    | CodeAlpaca |
|--------------|---------------------|----------|-----------|----------|------------|
| Llama-3.1-8B | Epochs              | 1        | 20        | 1        | 2          |
|              | Learning rate       | 3 × 10−4 | 3 × 10−4  | 3 × 10−4 | 3 × 10−4   |
|              | Max sequence length | 128      | 256       | 512      | 512        |
|              | micro batch size    | 8        | 8         | 8        | 8          |
| Qwen-2.5-7B  | Epochs              | 1        | 20        | 1        | 2          |
|              | Learning rate       | 3 × 10−4 | 3 × 10−4  | 3 × 10−4 | 6 × 10−4   |
|              | Max sequence length | 128      | 256       | 512      | 512        |
|              | micro batch size    | 8        | 8         | 8        | 8          |

### C.4 Environments

Most experiments were conducted on a Linux server running Ubuntu 20.04.4 LTS, equipped with an Intel(R) Xeon(R) Platinum 8358P CPU at 2.60GHz and 8 NVIDIA GeForce RTX 3090 GPUs, using CUDA version 11.7. Experiments with Qwen-2.5-14B were conducted on a machine with 8 NVIDIA A100 GPUs.

