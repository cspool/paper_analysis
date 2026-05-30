# A Appendix

## <span id="page-14-1"></span>A.1 Hyperparameters and Implementation Details

Table 3: Hyperparameter configurations of LoRA/DoRA and MixLoRA/MixDoRA for fine-tuning Gemma-2B, LLaMA2-7B/13B, and LLaMA3-8B on the commonsense reasoning tasks.

| Hyperparameters    | LoRA/DoRA | MixLoRA/MixDoRA            |
|--------------------|-----------|----------------------------|
| Cutoff Length      |           | 512                        |
| Learning Rate      |           | 2e-4                       |
| Optimizer          |           | AdamW                      |
| Batch size         |           | 16                         |
| Accumulation Steps |           | 8                          |
| Dropout            |           | 0.05                       |
| # Epochs           |           | 2                          |
| Where              |           | Q, K, V, O, Up, Down, Gate |
| LoRA Rank r        | 80        | 16                         |
| LoRA Alpha α       | 160       | 32                         |
| # Experts          | -         | 8                          |
| Top-K              | -         | 2                          |

All experiments are conducted with GPUs having 24GB memory (RTX 3090, RTX A5000, RTX 4090) for 7B models, GPUs having 48GB memory (RTX A6000) for 8B and 13B models, and setup with Python 3.10 and Ubuntu 22.04 on x86-64 CPUs.

## <span id="page-14-0"></span>A.2 Datasets

Table 4 presents detailed information about the datasets used in our experiments, including their task names, respective domains, the number of training and test sets, task types.

Table 4: Description of Datasets used in experiments.

| Task Name  | Domain               | # Train | # Test | Task Type           |
|------------|----------------------|---------|--------|---------------------|
| BoolQ      | Wikipedia            | 9,427   | 3,270  | Text Classification |
| ARC-E      | Natural Science      | 2,250   | 2,380  | Question Answering  |
| ARC-C      | Natural Science      | 1,120   | 1,170  | Question Answering  |
| OpenBookQA | Science Facts        | 4,957   | 500    | Question Answering  |
| PIQA       | Physical Interaction | 16,100  | 1,840  | Question Answering  |
| SIQA       | Social Interaction   | 33,410  | 1,954  | Question Answering  |
| HellaSwag  | Video Caption        | 39,905  | 10,042 | Sentence Completion |
| WinoGrande | Winograd Schemas     | 9,248   | 1,267  | Fill in the Blank   |

All datasets are downloaded from [HuggingFace](https://huggingface.co) using the DATASETS library in Python.

## <span id="page-15-0"></span>A.3 Multi-task Learning Evaluation Result using Gemma-2B

Table 5: Comparision of different peft methods for multi-task learning on various tasks, using Gemma 2B as the base model. Single-Task (ST) setup refers to training and evaluating PEFT modules for each task, while Multi-Task (MT) setup refers to training on mixed tasks, followed by separate evaluation. Reported results are accuracy scores.

| PEFT Method | # Params (%) | ST/MT | ARC-e | ARC-c | BoolQ | OBQA | PIQA | AVG. |
|-------------|--------------|-------|-------|-------|-------|------|------|------|
| LoRA        | 3.2%         | ST    | 71.9  | 43.2  | 62.1  | 71.4 | 80.9 | 65.9 |
|             | 3.2%         | MT    | 64.9  | 50.2  | 66.4  | 64.8 | 75.7 | 64.4 |
|             |              |       | -7.0  | 7.0   | 4.3   | -6.6 | -5.2 | -1.5 |
|             | 3.2%         | ST    | 71.5  | 46.2  | 62.2  | 70.4 | 81.6 | 66.4 |
| DoRA        | 3.2%         | MT    | 63.7  | 50.8  | 61.6  | 61.0 | 81.1 | 63.6 |
|             |              |       | -7.8  | 4.6   | -0.6  | -9.4 | -0.5 | -2.8 |
|             | 4.3%         | ST    | 76.3  | 47.4  | 65.8  | 75.8 | 81.1 | 69.3 |
| MixLoRA     | 4.3%         | MT    | 70.3  | 55.5  | 66.6  | 70.0 | 78.7 | 68.2 |
|             |              |       | -6.0  | 8.1   | 0.8   | -5.8 | -2.4 | -1.1 |
|             | 4.3%         | ST    | 77.0  | 54.3  | 67.2  | 75.4 | 81.8 | 71.1 |
| MixDoRA     | 4.3%         | MT    | 71.1  | 56.3  | 65.9  | 70.6 | 79.1 | 68.6 |
|             |              |       | -5.9  | 2.0   | -1.3  | -4.8 | -2.7 | -2.5 |

## <span id="page-15-1"></span>A.4 Experimental Results of Performance Metrics.

<span id="page-15-2"></span>Table 6: Experimental results of LLaMA-2 7B for performance metrics. The latency shown in the table represents the token computation latency, and the memory indicates the peak GPU memory collected by the profiler. To accurately measure the performance of LoRA and DoRA during inference, we conducted the experiments with weights unmerged. † represents methods with MIXLORA optimization.

|             |       | Training |        |          |      | Inference |       |         |      |        |
|-------------|-------|----------|--------|----------|------|-----------|-------|---------|------|--------|
| PEFT Method |       | Forward  |        | Backward |      | Memory    |       | Forward |      | Memory |
|             | µs    | %        | µs     | %        | GB   | %         | µs    | %       | GB   | %      |
| LoRA        | 245.3 | 100.0%   | 552.3  | 100.0%   | 15.2 | 100.0%    | 241.4 | 100.0%  | 13.7 | 100.0% |
| DoRA        | 659.4 | 268.8%   | 1193.8 | 216.1%   | 15.6 | 102.4%    | 645.3 | 267.3%  | 13.7 | 100.0% |
| MixLoRA     | 535.2 | 218.2%   | 1187.5 | 215.0%   | 15.1 | 99.5%     | 532.8 | 220.7%  | 13.7 | 100.0% |
| †MixLoRA    | 462.5 | 188.5%   | 1097.6 | 198.7%   | 15.1 | 99.5%     | 442.2 | 183.2%  | 13.7 | 100.0% |
| MixLoRA ×2  | 533.9 | 217.7%   | 1185.5 | 214.6%   | 8.8  | 57.7%     | 523.8 | 217.0%  | 7.2  | 52.5%  |
| †MixLoRA ×2 | 441.0 | 179.8%   | 1072.3 | 194.1%   | 8.8  | 57.7%     | 441.4 | 182.8%  | 7.2  | 52.5%  |

Table [6](#page-15-2) shows the results on LLaMA2 7B, demonstrating that MIXLORA exhibits lower token computation latency (DoRA requires 659.4µs for forward propagation, while MIXLORA only needs 535.2µs) and comparable peak GPU memory usage to DoRA (approximately 15GB). However, MIXLORA shows nearly twice the token computation latency of LoRA (245.3µs). This increased latency is due to MIXLORA sending each token to two experts for computation (when K = 2). Nonetheless, with our optimized algorithm, we reduced the token computation latency by nearly 30% for a single model (from 535.2µs to 462.5µs) and decreased the peak GPU memory per model by almost 45% when training or inferring with two models simultaneously (from 15.1GB to 8.8GB during training, and from 13.7GB to 7.2GB during inference). Appendix [7](#page-16-0) shows the results on Gemma 2B, corroborating these findings and proving that our algorithm maintains robustness across different model sizes. In conclusion, experiments show that MIXLORA offers a more balanced trade-off, providing higher performance with reduced latency compared to the current state-of-the-art method, DoRA.

<span id="page-16-0"></span>Table 7: Experimental results of Gemma 2B for performance metrics. The latency shown in the table represents the token computation latency, and the memory indicates the peak GPU memory collected by the profiler. To accurately measure the performance of LoRA and DoRA during inference, we conducted the experiments with weights unmerged. † represents methods with MIXLORA optimization.

|             |       |         |       | Training |      |        | Inference |         |      |        |  |
|-------------|-------|---------|-------|----------|------|--------|-----------|---------|------|--------|--|
| PEFT Method |       | Forward |       | Backward |      | Memory |           | Forward |      | Memory |  |
|             | µs    | %       | µs    | %        | GB   | %      | µs        | %       | GB   | %      |  |
| LoRA        | 151.1 | 100.0%  | 308.2 | 100.0%   | 11.4 | 100.0% | 152.0     | 100.0%  | 10.6 | 100.0% |  |
| DoRA        | 539.4 | 356.9%  | 919.9 | 298.5%   | 11.4 | 100.0% | 533.4     | 350.9%  | 10.6 | 100.0% |  |
| MixLoRA     | 250.6 | 165.8%  | 527.2 | 171.1%   | 11.2 | 97.7%  | 245.1     | 161.2%  | 10.5 | 99.8%  |  |
| †MixLoRA    | 226.5 | 149.9%  | 525.2 | 170.4%   | 11.2 | 97.7%  | 224.0     | 147.4%  | 10.5 | 99.8%  |  |
| MixLoRA ×2  | 249.6 | 165.1%  | 524.0 | 170.1%   | 7.6  | 66.9%  | 243.1     | 160.6%  | 6.5  | 61.5%  |  |
| †MixLoRA ×2 | 223.8 | 148.1%  | 523.7 | 169.9%   | 7.6  | 66.9%  | 221.2     | 145.6%  | 6.5  | 61.5%  |  |

## A.5 Robustness of MIXLORA Towards Different Rank

Table 8: Accuracy comparison of MIXLORA and MIXDORA with varying ranks for LLaMA2-7B on the commonsense reasoning tasks.

| PEFT Method | Rank r | # Params (%) | ARC-e | ARC-c | BoolQ | OBQA | Avg. |
|-------------|--------|--------------|-------|-------|-------|------|------|
|             | 2      | 0.38%        | 76.1  | 56.4  | 73.3  | 79.2 | 71.3 |
|             | 4      | 0.74%        | 76.2  | 56.5  | 73.8  | 80.8 | 71.8 |
| MixLoRA     | 8      | 1.46%        | 76.9  | 56.8  | 74.2  | 81.2 | 72.3 |
|             | 16     | 2.91%        | 77.7  | 58.1  | 72.7  | 84.4 | 73.2 |
|             | 32     | 5.80%        | 79.1  | 54.1  | 70.0  | 76.4 | 69.9 |
|             | 2      | 0.38%        | 75.3  | 52.7  | 73.3  | 80.4 | 70.4 |
|             | 4      | 0.74%        | 76.2  | 55.0  | 73.2  | 80.4 | 71.2 |
| MixDoRA     | 8      | 1.46%        | 76.7  | 55.5  | 73.5  | 78.6 | 71.1 |
|             | 16     | 2.91%        | 77.5  | 58.2  | 72.6  | 80.9 | 72.3 |
|             | 32     | 5.80%        | 75.5  | 53.6  | 72.0  | 77.6 | 69.7 |

