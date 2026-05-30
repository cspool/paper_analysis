# 8 Conclusion

In this work, we propose STARC, a clustering-based data mapping strategy that enables efficient sparse attention execution on PIM architectures. By co-locating semantically similar KV pairs and remapping them to contiguous memory regions, STARC bridges the gap between dynamic tokenwise sparsity and the rigid row-level access granularity of PIM. This co-design improves both throughput and energy efficiency without compromising model accuracy. Experiments show that STARC achieves up to 78% latency reduction and 65% energy savings on the attention layer compared to token-wise sparsity baselines. We hope that our work inspires further integration of PIM architectures with emerging LLM optimization techniques, ultimately enabling scalable and efficient LLM inference in real-world deployments.

### Acknowledgments

This work was supported in part by RPI-IBM Future of Computing Research Collaboration and the National Science Foundation under Award Number 2442271. We thank the anonymous reviewers for their constructive feedback and our shepherd Jongse Park for guidance throughout the revision process. We also thank Yinan Wang for insightful discussions.

#### **Appendix**

#### A Additional Results

To complement the main evaluation, we present additional LongBench results that examine STARC's effectiveness across a range of KV cache budgets (256, 512, and 2048), beyond the budget of 1024 used in the main results (Table 4, 5). These experiments illustrate how varying the KV cache budget affects model quality when serving long-context LLMs, and how STARC adapts its clustering-based mapping to improve efficiency while preserving model quality.

#### **B** Artifact Appendix

#### B.1 Abstract

This artifact provides a complete workflow to reproduce the key results of STARC, including (1) the implementation of STARC's selective token access with KV remapping and online clustering, (2) evaluation scripts to reproduce accuracy results on LongBench and RULER, and perplexity results on

PG-19, and (3) the simulator setup to reproduce the system-level performance/energy results on GPU-PIM platforms based on the AttAcc simulator (Ramulator-based).

#### B.2 Artifact check-list (meta-information)

- Algorithm: The STARC algorithm, which enables efficient long-context LLM inference by selectively accessing and remapping KV cache entries via online clustering under a fixed KV-cache budget.
- Program: The STARC artifact running public long-context benchmarks: LongBench (16 datasets) and RULER (13 datasets).
- Model: LongChat-7B-v1.5-32K; LLaMA-3.1-8B-Instruct; Mistral-7B-Instruct-v0.3, all of which are publicly available and can be downloaded from Huggingface.
- Dataset: LongBench (16 datasets; e.g., HotpotQA, QASPER, GovReport, etc.); PG-19; RULER (13 datasets; e.g., NIAH Single, Multi-key NIAH, Multi-value NIAH, etc.), all of which are publicly available and can be downloaded from Huggingface.
- Run-time environment: Linux; Python 3.10; CUDA 12.8.
- Hardware: See B.3.2.
- Metrics: LongBench task scores; PG-19 perplexity; RULER task scores; System metrics such as latency and energy.
- Output: Key results of our paper, including LongBench/RULER scores, PG-19 perplexity, and system-level performance and energy metrics with breakdowns.
- Experiments: See B.5
- How much disk space required (approximately)?: Approximately 80 GB in total.
- How much time is needed to prepare workflow (approximately)?: 20 minutes.
- How much time is needed to complete experiments (approximately)?: Excluding the additional results reported in

Table 4. LongBench results for STARC and baseline sparsity methods (KV cache budget: 256 tokens).

<span id="page-12-0"></span>

|           | Single-Document QA |          |       | Multi-Document QA |             |         | Sui     | Summarization  |           |       | Few-Shot Learning |       |          | Synthetic      |       | Code  |        |
|-----------|--------------------|----------|-------|-------------------|-------------|---------|---------|----------------|-----------|-------|-------------------|-------|----------|----------------|-------|-------|--------|
|           | Arr. Pot           | to store |       | Hohodoy           | Formation ? | Masique | Gorpeon | ONSUM<br>ONSUM | MultiNews |       | Privide A         | SAME  | to Count | æ <sup>e</sup> | 207   | 48.p  | A. 80. |
|           |                    |          |       |                   |             |         | KV      | Budget:        | 256       |       |                   |       |          |                |       |       |        |
| LongCha   | at                 |          |       |                   |             |         |         |                |           |       |                   |       |          |                |       |       |        |
| Full KV   | 19.51              | 25.98    | 43.80 | 31.94             | 23.20       | 11.38   | 31.77   | 21.66          | 26.06     | 66.00 | 82.00             | 20.79 | 2.00     | 30.00          | 53.86 | 48.68 | 33.66  |
| STARC     | 18.82              | 28.35    | 34.79 | 34.41             | 18.64       | 8.10    | 30.50   | 21.74          | 24.64     | 62.00 | 81.01             | 24.17 | 2.00     | 32.00          | 55.56 | 45.00 | 32.61  |
| SparQ     | 19.87              | 30.77    | 40.71 | 31.70             | 20.93       | 12.89   | 30.93   | 22.80          | 26.38     | 64.00 | 85.17             | 31.37 | 0.50     | 31.50          | 55.63 | 55.58 | 35.05  |
| InfiniGen | 13.68              | 27.47    | 36.05 | 27.86             | 20.41       | 7.75    | 26.27   | 20.49          | 24.97     | 62.00 | 77.22             | 32.47 | 2.00     | 18.00          | 52.70 | 50.28 | 31.23  |
| Quest     | 10.49              | 26.47    | 34.90 | 20.04             | 24.23       | 12.53   | 21.59   | 20.48          | 25.29     | 56.00 | 63.80             | 22.62 | 2.00     | 28.00          | 47.86 | 38.58 | 28.43  |
| Mistral   |                    |          |       |                   |             |         |         |                |           |       |                   |       |          |                |       |       |        |
| Full KV   | 23.94              | 40.07    | 57.58 | 49.10             | 36.71       | 22.27   | 35.66   | 25.77          | 26.80     | 80.00 | 87.67             | 47.35 | 4.00     | 98.00          | 58.98 | 56.36 | 46.89  |
| STARC     | 20.19              | 35.71    | 56.50 | 44.43             | 45.85       | 20.32   | 34.06   | 24.06          | 26.54     | 68.00 | 87.11             | 48.97 | 6.00     | 88.00          | 61.94 | 57.60 | 45.33  |
| SparQ     | 27.12              | 40.87    | 53.94 | 49.32             | 39.51       | 23.97   | 35.31   | 25.14          | 27.48     | 73.00 | 88.78             | 47.28 | 4.50     | 99.50          | 61.56 | 63.15 | 47.53  |
| InfiniGen | 19.52              | 37.95    | 54.54 | 42.28             | 38.98       | 10.34   | 31.14   | 22.82          | 27.21     | 74.00 | 83.45             | 47.70 | 4.00     | 90.00          | 61.70 | 52.34 | 43.62  |
| Quest     | 16.81              | 30.88    | 36.99 | 35.62             | 27.66       | 10.12   | 29.18   | 21.11          | 26.04     | 66.00 | 78.39             | 37.84 | 4.89     | 83.50          | 57.22 | 43.98 | 37.89  |
| Llama-3.  | .1                 |          |       |                   |             |         |         |                |           |       |                   |       |          |                |       |       |        |
| Full KV   | 27.02              | 13.98    | 28.04 | 18.30             | 17.45       | 13.01   | 35.83   | 23.66          | 25.91     | 74.00 | 89.77             | 44.56 | 3.92     | 97.50          | 63.30 | 55.06 | 39.46  |
| STARC     | 30.84              | 12.91    | 26.42 | 21.88             | 18.34       | 13.48   | 34.96   | 22.18          | 25.46     | 66.00 | 86.71             | 44.94 | 12.00    | 94.33          | 65.52 | 57.32 | 39.58  |
| SparQ     | 29.70              | 12.35    | 26.97 | 17.69             | 15.31       | 11.40   | 33.89   | 23.38          | 27.00     | 70.00 | 92.19             | 44.58 | 6.90     | 97.50          | 64.55 | 60.79 | 39.64  |
| InfiniGen | 21.86              | 16.53    | 29.63 | 21.47             | 17.76       | 5.36    | 32.38   | 22.70          | 25.50     | 68.00 | 86.40             | 44.58 | 7.25     | 96.00          | 67.36 | 55.38 | 38.64  |
| Quest     | 8.68               | 9.90     | 18.18 | 12.19             | 9.48        | 3.02    | 25.33   | 18.36          | 23.50     | 44.00 | 73.23             | 31.53 | 3.55     | 83.00          | 51.90 | 46.52 | 28.90  |

<span id="page-13-0"></span>Table 5. LongBench results for STARC and baseline sparsity methods (KV cache budget: 512 and 2048 tokens).

|                      | Single-Document QA |                |                | Multi-Document QA |                |                |                | Summarization   |                |                | Few-Shot Learning |                | Synthetic    |                | Code           |                |                |
|----------------------|--------------------|----------------|----------------|-------------------|----------------|----------------|----------------|-----------------|----------------|----------------|-------------------|----------------|--------------|----------------|----------------|----------------|----------------|
|                      | NrtvQA             | Qasper         | MF-en          | HotpotQA          | 2WikiMQA       | Musique        | GovReport      | QMSum           | MultiNews      | TREC           | TriviaQA          | SAMSum         | PCount       | PRe            | Lcc            | RB-P           | Avg.           |
|                      |                    |                |                |                   |                |                |                | KV Budget: 512  |                |                |                   |                |              |                |                |                |                |
| LongChat             |                    |                |                |                   |                |                |                |                 |                |                |                   |                |              |                |                |                |                |
| Full KV<br>STARC     | 19.51<br>17.59     | 25.98          | 43.80<br>39.44 | 31.94             | 23.20<br>18.70 | 11.38<br>10.21 | 31.77<br>30.46 | 21.66<br>20.49  | 26.06<br>25.11 | 66.00<br>64.00 | 82.00<br>79.81    | 20.79<br>22.48 | 2.00         | 30.00<br>30.00 | 53.86          | 48.68<br>48.38 | 33.66<br>33.13 |
| SparQ                | 19.20              | 29.86<br>29.18 | 40.81          | 33.92<br>32.27    | 22.30          | 13.43          | 30.81          |                 |                | 64.50          |                   | 29.05          | 2.00<br>0.00 | 30.00          | 57.60<br>55.11 |                | 34.76          |
| InfiniGen            | 16.37              | 25.37          | 38.10          | 28.48             | 18.15          |                | 28.71          | 22.81<br>21.10  | 26.29<br>25.06 | 64.00          | 84.70<br>79.03    |                | 0.00         | 28.00          | 52.60          | 55.70<br>53.42 | 32.74          |
| Quest                | 13.39              | 28.08          | 40.90          | 25.34             |                | 13.52<br>7.59  | 27.82          | 21.48           | 25.39          |                | 76.67             | 31.93<br>21.94 | 0.00         |                | 53.36          | 45.08          | 32.10          |
|                      |                    |                |                |                   | 24.59          |                |                |                 |                | 66.00          |                   |                |              | 36.00          |                |                |                |
| Mistral<br>Full KV   | 23.94              | 40.07          | 57.58          | 49.10             | 36.71          | 22.27          |                |                 | 26.80          |                | 87.67             | 47.35          | 4.00         | 98.00          | 58.98          | 56.36          | 46.89          |
| STARC                | 21.49              | 37.26          |                | 47.18             |                | 23.68          | 35.66<br>34.32 | 25.77<br>23.66  | 26.64          | 80.00<br>74.00 | 87.67             | 48.38          | 4.00         | 94.00          | 62.18          | 58.90          | 46.39          |
| SparQ                |                    |                | 58.73<br>53.70 |                   | 40.15<br>37.75 |                | 34.23          | 25.68           |                | 74.00          |                   | 47.35          | 5.00         |                | 60.72          |                | 47.66          |
| InfiniGen            | 29.00<br>22.76     | 40.09<br>36.82 | 58.67          | 50.43<br>49.17    | 31.64          | 26.49<br>15.34 | 33.80          | 23.87           | 27.50<br>26.51 | 78.00          | 89.07<br>83.67    |                | 2.00         | 99.50<br>96.00 |                | 62.07<br>58.04 | 45.55          |
| Quest                | 18.39              | 33.14          | 45.93          | 41.79             | 33.64          | 18.21          | 32.57          | 22.77           | 26.45          | 64.00          | 84.50             | 49.67<br>41.63 |              | 92.67          | 62.88<br>59.92 | 49.84          | 42.00          |
|                      |                    |                |                |                   |                |                |                |                 |                |                |                   |                | 6.50         |                |                |                |                |
| Llama-3.1<br>Full KV | 27.02              | 13.98          | 28.04          | 18.30             | 17.45          | 13.01          | 35.83          | 23.66           | 25.91          | 74.00          | 89.77             | 44.56          | 3.92         | 97.50          | 63.30          | 55.06          | 39.46          |
| STARC                | 31.78              | 13.06          | 28.77          | 18.49             | 18.58          | 14.24          | 34.33          | 22.65           | 25.80          | 70.00          | 88.57             | 44.26          | 4.67         | 95.83          | 63.56          | 60.08          | 39.67          |
| SparQ                | 30.30              | 13.30          | 26.19          | 17.90             | 16.12          | 10.43          | 34.11          | 23.83           | 27.17          | 70.50          | 91.97             | 43.80          | 8.29         | 98.08          | 64.43          | 61.34          | 39.86          |
| InfiniGen            | 23.36              | 16.90          | 27.18          | 22.17             | 18.30          | 8.76           | 33.69          | 22.79           | 25.85          | 66.00          | 89.90             | 45.10          | 6.67         | 98.07          | 66.46          | 50.72          | 38.87          |
| Quest                | 15.57              | 10.77          | 21.82          | 12.42             | 13.25          | 5.93           | 29.48          | 22.05           | 26.65          | 60.00          | 78.87             | 37.44          | 2.54         | 90.52          | 61.18          | 53.94          | 33.90          |
|                      |                    |                |                |                   |                |                |                |                 |                |                |                   |                |              |                |                |                |                |
| LongChat             |                    |                |                |                   |                |                |                | KV Budget: 2048 |                |                |                   |                |              |                |                |                |                |
| Full KV              | 19.51              | 25.98          | 43.80          | 31.94             | 23.20          | 11.38          | 31.77          | 21.66           | 26.06          | 66.00          | 82.00             | 20.79          | 2.00         | 30.00          | 53.86          | 48.68          | 33.66          |
| STARC                | 17.70              | 28.27          | 41.15          | 33.75             | 23.28          | 11.21          | 30.97          | 23.13           | 26.50          | 63.00          | 81.53             | 20.80          | 1.00         | 31.00          | 54.15          | 50.92          | 33.65          |
| SparQ                | 20.01              | 28.48          | 42.21          | 31.02             | 23.53          | 12.68          | 31.06          | 23.07           | 26.69          | 65.00          | 84.41             | 24.61          | 0.00         | 30.00          | 52.86          | 55.69          | 34.46          |
| InfiniGen            | 15.76              | 30.35          | 40.52          | 31.81             | 20.05          | 9.00           | 29.82          | 22.13           | 26.00          | 62.00          | 81.78             | 25.94          | 0.00         | 36.00          | 57.62          | 50.38          | 33.70          |
| Quest                | 14.93              | 31.48          | 45.33          | 31.60             | 19.70          | 12.93          | 30.83          | 22.07           | 25.61          | 62.00          | 81.33             | 20.35          | 2.00         | 30.00          | 55.68          | 49.92          | 33.49          |
| Mistral              |                    |                |                |                   |                |                |                |                 |                |                |                   |                |              |                |                |                |                |
| Full KV              | 23.94              | 40.07          | 57.58          | 49.10             | 36.71          | 22.27          | 35.66          | 25.77           | 26.80          | 80.00          | 87.67             | 47.35          | 4.00         | 98.00          | 58.98          | 56.36          | 46.89          |
| STARC                | 28.71              | 43.73          | 54.06          | 48.62             | 37.87          | 23.36          | 34.82          | 25.75           | 27.87          | 72.00          | 85.76             | 47.87          | 9.00         | 100.00         | 59.64          | 57.91          | 47.31          |
| SparQ                | 29.58              | 40.25          | 53.37          | 51.01             | 37.94          | 27.22          | 34.45          | 25.68           | 27.76          | 74.50          | 89.06             | 47.01          | 5.00         | 99.00          | 59.76          | 62.04          | 47.73          |
| InfiniGen            | 25.34              | 39.30          | 59.51          | 50.20             | 41.79          | 18.54          | 34.83          | 24.68           | 26.85          | 78.00          | 87.67             | 47.38          | 2.00         | 96.00          | 58.96          | 59.46          | 46.91          |
| Quest                | 23.48              | 40.55          | 58.73          | 48.94             | 37.63          | 25.41          | 32.79          | 24.07           | 27.28          | 70.00          | 88.33             | 47.07          | 6.00         | 98.00          | 57.86          | 60.54          | 46.67          |
| Llama-3.1            |                    |                |                |                   |                |                |                |                 |                |                |                   |                |              |                |                |                |                |
| Full KV              | 27.02              | 13.98          | 28.04          | 18.30             | 17.45          | 13.01          | 35.83          | 23.66           | 25.91          | 74.00          | 89.77             | 44.56          | 3.92         | 97.50          | 63.30          | 55.06          | 39.46          |
| STARC                | 30.61              | 13.88          | 27.94          | 20.85             | 19.62          | 11.53          | 34.56          | 22.75           | 26.30          | 72.00          | 88.57             | 45.54          | 2.92         | 99.00          | 62.66          | 55.54          | 39.64          |
| SparQ                | 29.76              | 13.06          | 26.61          | 17.30             | 16.85          | 11.26          | 34.02          | 23.50           | 26.69          | 71.00          | 91.48             | 43.82          | 6.37         | 98.01          | 63.38          | 59.78          | 39.56          |
| InfiniGen            | 27.98              | 13.33          | 32.01          | 19.49             | 18.79          | 12.86          | 35.45          | 23.10           | 26.66          | 72.00          | 89.81             | 44.63          | 7.00         | 96.67          | 61.12          | 55.74          | 39.79          |
| Quest                | 24.41              | 13.34          | 23.39          | 15.97             | 15.59          | 10.59          | 35.03          | 23.33           | 25.58          | 74.00          | 92.60             | 45.23          | 5.18         | 97.50          | 59.44          | 56.52          | 38.61          |

the appendix, the model accuracy experiments take approximately 12 hours. In addition, the system-level performance experiments take approximately 24 hours.

- Publicly available?: https://doi.org/10.5281/zenodo.18050293
- Code licenses (if publicly available)?: MIT license.

#### B.3 Description

B.3.1 How to access. The STARC algorithm, benchmarks, and scripts are available at GitHub: [EPIC-RPI/STARC](https://github.com/EPIC-RPI/STARC)

#### <span id="page-13-1"></span>B.3.2 Hardware dependencies.

• LLM accuracy evaluation (LongBench / PG-19 / RULER): Compatible with commonly used NVIDIA GPUs. We recommend NVIDIA H100 or L40 with sufficient GPU memory (e.g., at least 48 GB per GPU).

• System-level simulation: CPU-only execution is sufficient. Experiments in the paper were conducted on a dual-socket AMD EPYC 9334 system with 64 CPU cores in total (2×32 cores).

B.3.3 Software dependencies. The software is performed using Python 3.10, and CUDA version 12.8. The dependent Python packages can be found in the pyproject.toml file.

#### B.4 Installation

• Code access. First, please access the code by:

git clone -- recurse - submodules https :// github . com / EPIC - RPI / STARC cd STARC

• Environment setup. To better reproduce the results and avoid potential conflicts, we recommend using Python 3.10 and CUDA 12.8. We provide scripts for the recommended environment setup. Please follow the instructions to create the conda environment and install the STARC packages:

```
conda create -yn STARC python=3.10
conda activate STARC
pip install ninja==1.11.1.1 packaging
pip install -e .
pip install flash-attn==2.3.0
```

• **PIM system simulator setup.** Next is the setup for the PIM system simulator. In this artifact, we mainly build on the AttAcc simulator:

```
cd simulator_starc
git submodule update --init --recursive
```

• Build Ramulator2.

```
bash set_pim_ramulator.sh
cd ramulator2
mkdir build
cd build
cmake ..
```

#### <span id="page-14-0"></span>**B.5** Experiment workflow

This section describes how to reproduce the key results reported in the paper.

**E1:** LongBench accuracy. To reproduce the LongBench accuracy results, please run:

```
cd <Your Path>/STARC/scripts/
sh longbench.sh
```

If you want to evaluate more models, first you can find the corresponding model paths in:

```
STARC/evaluation/LongBench/config/model2path.json
```

By replacing the model name in longbench. sh, you can evaluate STARC under different models reported in the paper.

**E2: PG-19 perplexity.** To reproduce the perplexity results on PG-19, please run:

```
cd <Your Path>/STARC/scripts/
sh ppl_eval.sh
```

