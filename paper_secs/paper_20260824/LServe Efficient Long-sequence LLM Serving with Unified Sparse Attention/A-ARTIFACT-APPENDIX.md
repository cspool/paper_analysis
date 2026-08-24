# A ARTIFACT APPENDIX

#### A.1 Abstract

This artifact contains necessary scripts and dependencies to faithfully reproduce the crucial experiments presented in the paper. To successfully run the experiments, a host system with x86 64 CPUs is required, along with at least one A100 or L40S NVIDIA GPU. We also provide a pre-built docker image to simplify the environment setup process.

#### A.2 Artifact check-list (meta-information)

- Program: Accuracy evaluation and efficiency benchmarking code for LServe; efficiency benchmarking code for baseline systems such as vLLM.
- Compilation: Completed in the docker.
- Transformations: N/A.
- Binary: N/A.
- Model: We provide a quantized version of Llama-3-8B to simplify the evaluation process.
- Data set: Included in the docker image.
- Run-time environment: NVIDIA Container Toolkit (nvidia-docker).
- Hardware: A host with x86 64 CPUs and at least one NVIDIA A100 GPU (recommended) or L40S GPU.
- Run-time state: N/A.
- Execution: All benchmarks are executed on NVIDIA GPUs, while some data pre-processing code is executed on the host CPU;
- Metrics: Long-context benchmarks accuracy; LLM generation throughput.
- Output: Accuracy numbers and generation throughput (tokens/second).
- Experiments: Inference speed measurement for LServe and baseline systems such as vLLM; accuracy evaluation of LServe.
- How much disk space required (approximately)?: 128G.
- How much time is needed to prepare workflow (approximately)?: Around 1 hour to pull docker images depending on the Internet connection and CPU performance.
- How much time is needed to complete experiments (approximately)?: Around 2 hours to finish the efficiency benchmarks; and 1 hour to finish the accuracy benchmarks depending on the GPU performance and number of benchmark subsets to evaluate.
- Publicly available?: Yes.
- Code licenses (if publicly available)?: Apache License 2.0.
- Data licenses (if publicly available)?: MIT.
- Workflow framework used?: Docker.
- Archived (provide DOI)?: [https://doi.org/10.5281/](https://doi.org/10.5281/zenodo.14989916) [zenodo.14989916](https://doi.org/10.5281/zenodo.14989916)

#### A.3 Description

#### *A.3.1 How delivered*

We will provide AE reviewers with a pre-built docker image containing LServe, vLLM, and all necessary dependencies.

#### *A.3.2 Hardware dependencies*

A host machine with x86 64 CPUs and at least one NVIDIA A100 GPU.

#### *A.3.3 Software dependencies*

A GPU-compatible Docker runtime environment is required.

#### *A.3.4 Data sets*

We require datasets such as 2wikimqa, dureader, and hotpotqa for LongBench evaluation. All datasets are publicly available and included in the docker image.

#### A.4 Installation

We recommend that users utilize our pre-built Docker images to set up the environment and run all experiments within the GPUsupported Docker container.

```
1 docker run --gpus all -it -- workdir /
      workspace / projects shang12138 /lserve -
      mlsys25 -ae
```

#### A.5 Experiment workflow

#### *A.5.1 LServe Accuracy Evaluation*

We provide push-button solution for evaluating LServe on Long-Bench.

```
1 cd / workspace / projects / omniserve
2 bash eval / scripts / LongBench /
      submit_longbench_dense .sh
3 # Evaluate the baseline accuracy
4 bash eval / scripts / LongBench /
      submit_longbench_sparse .sh
5 # Evaluate LServe accuracy
6
7 # The evaluation results can be found at
      ./ eval / LongBench / pred / <model -name >/
      result . json
```

#### *A.5.2 Throughput Benchmark*

The generation throughputs of LServe and baseline system (i.e., vLLM) can be measured with the following commands.

```
1 # LServe benchmark
2 cd / workspace / projects / omniserve
3 bash scripts / lserve_benchmark / launch .sh
4 # Results in ./ results .csv
5
6 # vLLM benchmark
7 cd / workspace / projects / vllm
8 bash launch_server .sh # Start vLLM server
```

```
9 # When the vLLM server has been launched ,
10 # start a new terminal in the same docker
11 cd / workspace / projects / vllm
12 bash run_vllm .sh # Launch evaluation
13 # Results in ./ results . csv
```

#### A.6 Evaluation and expected result

We provide reference numbers for evaluation results in this section. Please note that absolute speed measurements may vary slightly, even on identical GPU platforms, due to differences in machine conditions. However, the relative acceleration ratios should remain consistent. Additionally, due to randomness inherent in LLM generation, accuracy results might show minor deviations from the reported reference numbers.

| Sequence Length | vLLM (ms) | LServe (ms) | Speedup |
|-----------------|-----------|-------------|---------|
| 64k             | 12.51     | 11.49       | 1.09x   |
| 96k             | 14.49     | 12.05       | 1.20x   |
| 128k            | 16.34     | 12.74       | 1.28x   |
| 160k            | 18.20     | 12.88       | 1.41x   |
| 192k            | 21.73     | 13.30       | 1.63x   |
| 224k            | 21.96     | 13.73       | 1.60x   |
| 256k            | 23.72     | 14.20       | 1.67x   |
| 320k            | 27.45     | 15.10       | 1.82x   |

Table 7: Generation latency of LServe and baseline (vLLM).

| Model     | Llama-3-8B |        |  |
|-----------|------------|--------|--|
| Benchmark | Dense      | LServe |  |
| 2WikiMQA  | 26.2       | 27.0   |  |
| DuReader  | 22.3       | 25.6   |  |
| HotpotQA  | 41.1       | 40.8   |  |
| MultiNews | 27.6       | 27.1   |  |
| Qasper    | 29.1       | 28.5   |  |

Table 8: LongBench evaluation results.

#### A.7 Methodology

Submission, reviewing and badging methodology:

- <http://cTuning.org/ae/submission-20190109.html>
- <http://cTuning.org/ae/reviewing-20190109.html>
- [https://www.acm.org/publications/policies/arti](https://www.acm.org/publications/policies/artifact-review-badging) [fact-review-badging](https://www.acm.org/publications/policies/artifact-review-badging)