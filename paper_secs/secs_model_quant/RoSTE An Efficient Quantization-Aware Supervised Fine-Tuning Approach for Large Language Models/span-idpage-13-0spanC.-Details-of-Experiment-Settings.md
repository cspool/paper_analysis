# <span id="page-13-0"></span>C. Details of Experiment Settings

RoSTE algorithm. We set the lower level objective function in [\(12\)](#page-4-6) by drawing n = 128 samples from the fine tuning dataset for calibration.

<span id="page-13-1"></span>Hyper-parameters. We list the training configurations for SFT in (Exp.1) TL;DR summarization and (Exp.2) Tulu 3 experiments as suggested in [\(Huang et al.,](#page-9-19) [2024;](#page-9-19) [Lambert et al.,](#page-10-19) [2024\)](#page-10-19) in Table [4.](#page-13-1) For QA-SFT, we sweep through a number of hyper-parameters for STE and RoSTE to obtain the best performance, as listed in Table [5.](#page-13-2)

| Method                |           |             | SFT          |            |              |
|-----------------------|-----------|-------------|--------------|------------|--------------|
| Model                 | Pythia 1B | Pythia 6.9B | Qwen2.5 0.5B | Qwen2.5 7B | Llama 3.1 8B |
| Epoch                 | 1         | 1           | 1            | 1          | 2            |
| Batch Size (Per GPU)  | 16        | 1           | 16           | 1          | 1            |
| Gradient Accumulation | 1         | 16          | 1            | 16         | 16           |
| Optimizer             | AdamW     | AdamW       | AdamW        | AdamW      | AdamW        |
| Learning Rate         | 3e-5      | 3e-5        | 5e-5         | 1e-5       | 5e-6         |
| LR Schedule           | cosine    | cosine      | cosine       | cosine     | linear       |
| Warmup Ratio          | 0         | 0           | 0            | 0          | 0.03         |
| Max. Seq. Length      | 2048      | 2048        | 2048         | 2048       | 1024         |
| # Training Samples    | 117k      | 117k        | 117k         | 117k       | 100k         |

Table 4. Detailed training settings for SFT in the TL;DR summarization and Tulu 3 experiments.

<span id="page-13-2"></span>Table 5. Detailed training settings and hyper-parameters for QA-SFT in the TL;DR summarization and Tulu 3 experiments.

| Method                |                    |                    | QA-SFT (i.e., STE or RoSTE) |                    |                    |
|-----------------------|--------------------|--------------------|-----------------------------|--------------------|--------------------|
| Model                 | Pythia 1B          | Pythia 6.9B        | Qwen2.5 0.5B                | Qwen2.5 7B         | Llama 3.1 8B       |
| Epoch                 | 1                  | 1                  | 1                           | 1                  | 2                  |
| Batch Size (Per GPU)  | 16                 | 1                  | 16                          | 1                  | 1                  |
| Gradient Accumulation | 1                  | 16                 | 1                           | 16                 | 16                 |
| Optimizer             | AdamW              | AdamW              | AdamW                       | AdamW              | AdamW              |
| Learning Rate         | {3e-5, 6e-6, 3e-6} | {3e-5, 6e-6, 3e-6} | {5e-5, 1e-5, 5e-6}          | {5e-5, 1e-5, 5e-6} | {5e-6, 1e-6, 5e-7} |
| LR Schedule           | cosine             | cosine             | cosine                      | cosine             | linear             |
| Warmup Ratio          | 0                  | 0                  | 0                           | 0                  | 0.03               |
| Max. Seq. Length      | 2048               | 2048               | 2048                        | 2048               | 1024               |
| # Training Samples    | 117k               | 117k               | 117k                        | 117k               | 100k               |
| clipping factor       | {1, 0.95, 0.9}     | {1, 0.95, 0.9}     | {1, 0.95, 0.9}              | {1, 0.95, 0.9}     | {1, 0.95, 0.9}     |

Evalution. For the TL;DR summarization experiments, all final models are evaluated on the TL;DR test dataset using the ROUGE metric [\(Lin,](#page-10-18) [2004\)](#page-10-18), including ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-LSum. For the Tulu 3 experiments, all final models are evaluated on downstream tasks using EleutherAI LM Evaluation Harness [\(Gao et al.,](#page-9-21) [2021\)](#page-9-21). These tasks include TruthfulQA [\(Lin et al.,](#page-10-20) [2021\)](#page-10-20), MMLU-Pro [\(Wang et al.,](#page-11-14) [2024b\)](#page-11-14), BigBenchHard [\(Suzgun et al.,](#page-10-21) [2022\)](#page-10-21), AGIEval [\(Zhong et al.,](#page-11-15) [2023\)](#page-11-15), GSM8K [\(Cobbe et al.,](#page-9-3) [2021\)](#page-9-3), and MATH [\(Hendrycks et al.,](#page-9-22) [2020\)](#page-9-22). In Table [6,](#page-14-1) we list the detailed evaluation settings for these downstream tasks as suggested in the Tulu 3 paper [\(Lambert et al.,](#page-10-19) [2024\)](#page-10-19).

Table 6. Details of evaluation settings for the Tulu 3 experiments.

<span id="page-14-1"></span>

| Benchmark | TruthfulQA | MMLU-Pro | BigBenchHard | AGIEval | GSM8K         | Math |
|-----------|------------|----------|--------------|---------|---------------|------|
| # shot    | 6          | 0        | 3            | 0       | 8             | 4    |
| Metric    | Acc (mc1)  | EM       | EM           | Acc     | $\mathbf{EM}$ | EM   |
| CoT       | ✓          | ×        | ×            | X       | ✓             | X    |

