# A.1 Abstract

We provide artifacts to reproduce all experimental results discussed in the evaluation section. These artifacts include the compiler implementation, kernels in our domain-specific languages, and scripts for running experiments and plotting figures. To simplify reproduction, all necessary dependencies are packaged within a Docker image. A Dockerfile is also included to demonstrate the image building process. We do not provide the models directly; instead, they and their metadata will be automatically downloaded from Hugging Face Hub when the experiment scripts are launched.

#### A.2 Artifact check-list (meta-information)

- Model: Gemma, Llama, Qwen
- Run-time environment: Linux, CUDA
- Hardware: NVIDIA L40s
- Metrics: Latency (ms), Speedup
- Output: Efficient kernels, Numerical results
- Experiments: Automated scripts in docker
- How much disk space required (approximately)?: 25 GiB
- How much time is needed to prepare workflow (approximately)?: 10 minutes
- How much time is needed to complete experiments (approximately)?: 3 hours
- Publicly available?: Yes
- Code licenses (if publicly available)?: Apache 2.0
- Workflow automation framework used?: Docker
- Archived (provide DOI)?:

<https://doi.org/10.5281/zenodo.16756859>

#### A.3 Description

A.3.1 How to access. We have open-sourced our artifacts at <https://github.com/yaoyaoding/tilus-artifacts>. To pull the Docker image and perform experiments, follow the guide in the README.md file. The code itself is several megabytes, while the Docker image is approximately 21 GiB. We only fetch model meta-information (e.g., number of layers, layer size) from Hugging Face Hub, and dummy weights are used; therefore, the models do not consume significant disk space.

A.3.2 Hardware dependencies. Our experiments were primarily conducted on NVIDIA L40s. To perform a hardware ablation study, we also ran some experiments on NVIDIA A100 and NVIDIA H100. Any NVIDIA GPU with compute capability ≥ 8.0 should be able to run our artifacts and observe speedup, though the specific numbers might vary slightly.

A.3.3 Software dependencies. We provide a Docker image with all software dependencies pre-installed. Therefore, only the following software is required:

- NVIDIA GPU driver ≥ 565.57.01
- NVIDIA container toolkit
- Docker

We have the following packages pre-installed:

- PyTorch v2.5.1
- Triton v3.1.0
- BitBLAS v0.0.1.dev15
- Marlin v0.1.1
- vLLM 0.7.3

A.3.4 Data sets. We use dummy inputs and weights because we are solely focused on system performance, which is independent of the input and weight content.

A.3.5 Models. The artifact uses three models for end-toend evaluation: Gemma-2-9B, QWen-2.5-32B, and Llama-3.3- 70B. The meta-information of these models will be automatically fetched from Hugging Face Hub (some may require a Hugging Face token).

#### A.4 Installation

First, clone the artifact Git repository:

git clone https://github.com/yaoyaoding/tilus-artifacts.git tilus Then, install Docker and the NVIDIA Container Toolkit by following the README.md in the artifact.

### A.5 Experiment workflow

All experiments can be executed with:

bash run.sh

This command will create a Docker container and sequentially run all experiments within it.

### A.6 Evaluation and expected results

Upon completion of the experiments, a folder named results or precompiled-results will be created under the artifact directory. This folder will contain four figures of the evaluation results, corresponding to those presented in the evaluation section.

### A.7 Methodology

Submission, reviewing and badging methodology:

- [https://www.acm.org/publications/policies/artifact](https://www.acm.org/publications/policies/artifact-review-and-badging-current) [-review-and-badging-current](https://www.acm.org/publications/policies/artifact-review-and-badging-current)
- <https://cTuning.org/ae>

