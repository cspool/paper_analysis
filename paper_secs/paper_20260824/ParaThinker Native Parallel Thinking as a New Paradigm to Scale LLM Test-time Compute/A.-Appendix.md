# **A. Appendix**

#### <span id="page-19-3"></span>**A.1. Supervised Fine-Tuning (SFT) Details**

This section details the configuration used for supervised fine-tuning (SFT) of the large language model.

| Parameter                   | Value          |
|-----------------------------|----------------|
| Batch Size                  | 1              |
| Gradient Accumulation Steps | 8              |
| Learning Rate               | 10−5<br>1<br>× |
| Training Epochs             | 3              |
| Context Length              | 28,672         |
| Hardware                    | 4 GPUs         |
| Learning Rate Scheduler     | Constant       |
| Warmup Ratio                | 0.1            |
| Weight Decay                | 0.05           |
| Max Gradient Norm           | 0.5            |

Table 7 | SFT Training Configuration for DeepSeek-R1-Distill-Qwen-1.5B

| Parameter                   | Value                  |
|-----------------------------|------------------------|
| Batch Size                  | 1                      |
| Gradient Accumulation Steps | 4                      |
| Learning Rate               | 10−5<br>2<br>×         |
| Training Epochs             | 2                      |
| Context Length              | 28,672                 |
| Hardware                    | 8 GPUs                 |
| Learning Rate Scheduler     | Cosine with Minimum LR |
| Warmup Ratio                | 0.1                    |
| Weight Decay                | 0.05                   |
| Max Gradient Norm           | 0.5                    |

Table 8 | SFT Training Configuration for DeepSeek-R1-Distill-Qwen-7B

#### <span id="page-20-0"></span>**A.2. Examples of training data**

After sampling multiple answers from the teacher model, we construct the SFT training data as follows: (1) concatenate all sampled answers as separate reasoning paths; (2) insert special tokens to delineate the parallel reasoning paths; (3) append the ground-truth final answer. An example is shown below:

