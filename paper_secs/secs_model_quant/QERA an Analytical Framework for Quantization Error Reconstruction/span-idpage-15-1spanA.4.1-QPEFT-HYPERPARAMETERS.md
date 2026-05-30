# <span id="page-15-1"></span>A.4.1 QPEFT HYPERPARAMETERS

We perform fine-tuning experiments on four NVIDIA A100 80GB GPUs with AMD EPYC 64-Core Processor with 1024GB RAM. The total fine-tuning time is around 2100 GPU hours.

RoBERTa-base on GLUE We sweep learning rates for each (method, task), and collect the best accuracy. Thus each (method, task) pair has its own tailored learning rate, ensuring the best performance of baselines and QERA under the same trainable parameter budget. The reported accuracy is the average value across random seeds 42, 1, and 2. The total batch size is 64 for all GLUE experiments and we train the models for 5 epochs. For 4-bit experiments, we use 4-bit floating point from the QLoRA implementation in PEFT. For 3-bit experiments, we use emulated MXINT [\(Darvish Rouhani et al., 2023\)](#page-10-13) with block size = 32 and for 2-bit experiments we use MXINT with block size = 16. Table [6](#page-16-2) lists the learning rates for each experiment.

LLaMA-2-7B/-3.1-8B on SlimPajama and GSM8K We adopt the learning rates in [Meng et al.](#page-11-15) [\(2024\)](#page-11-15). The reported perplexity/accuracy is the average value across random seeds 42, 1, and 2. For SlimPajama, we fine-tune the model on a subset for 1000 steps with rank = 8, total batch size = 64, sequence length = 1024, learning rate = 3e-5. For GSM8K, we fine-tune the model for 10 epochs with rank = 64, total batch size = 128, sequence length = 384, and learning rate = 3e-5.

<span id="page-16-2"></span>

| Rank | W-bits | Method                  | Learning rates                           |
|------|--------|-------------------------|------------------------------------------|
| -    | 16     | Full FT                 | 7e-5, 5e-5, 2e-5                         |
| 8    | 16     | LoRA                    | 1e-4, 2e-4, 3e-4                         |
| 8    | 4.25   | QLoRA/LoftQ/QERA-approx | 1e-4, 2e-4, 3e-4                         |
| 8    | 3.25   | QLoRA/LoftQ/QERA-approx | 1e-4, 2e-4, 3e-4                         |
| 64   | 2.50   | QLoRA/LoftQ/QERA-exact  | 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 9e-5, 1e-4 |

Table 6: Learning rates of RoBERTa-base experiments on GLUE.

### <span id="page-16-0"></span>A.4.2 PTQ HYPERPARAMETERS

We perform PTQ experiments on eight NVIDIA A6000 48GB GPUs with AMD EPYC 256-Core Processor with 1024GB RAM. The total quantization and evaluation time is around 4500 GPU hours. We report 0-shot accuracy or normalized accuracy (if available) for all tasks except Wiki-Text2, in which we report word perplexity. The sequence length for reporting word perplexity is the model's context length by default, except for Phi-3.5 and LLaMA-3.1. For these two models, we set the sequence length = 2048. We use the HuggingFace Transformers's implementation of HQQ, and reimplement ZeroQuant-V2 and LQER as baselines. We use MXINT with block size = 32 as the quantization format for all quantization methods except HQQ, which uses its built-in INT format with group size = 64. Thus, both formats have an average W-bits of 4.25. We evaluate quantized Vicuna-v1.5-7B, which is an instruction-tuned LLaMA-2-7B, with AlpacaEval 2.0. and use GPT4-Turbo as the evaluator. The reported win rate is the length-controlled win rate, which is a debiased version of the win rate that controls for the length of the generated outputs.

