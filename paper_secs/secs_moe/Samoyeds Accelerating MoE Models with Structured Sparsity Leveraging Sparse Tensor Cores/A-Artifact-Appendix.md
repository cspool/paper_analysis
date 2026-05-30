# A Artifact Appendix

## A.1 Abstract

This artifact includes the source codes and experiments for replicating the evaluations in this paper.

## A.2 Description & Requirements

- A.2.1 How to access. All the source code and instructions can be accessed through the following platforms:
  - git: <https://github.com/guqiqi/Samoyeds.git>
  - zenodo: <https://doi.org/10.5281/zenodo.14880516>
  - docker image: kevinwu2017/samoyeds:1.0.0
- A.2.2 Hardware dependencies. GPUs with Sparse Tensor Core (such as NVIDIA GPUs with Ampere architecture or newer).
- A.2.3 Software dependencies. We recommend running the Samoyeds artifact on a Linux platform with Docker and an NVIDIA GPU driver supporting CUDA 11.4+. The artifact is pre-packaged in a Docker image.
- A.2.4 Benchmarks. The Samoyeds artifact requires several models and datasets, such as Bert, SQuAD 1.1, etc. All of these requirements will be automatically downloaded during runtime.

## A.3 Set-up

- 1 docker pull kevinwu2017/samoyeds:1.0.0
- 2 docker run −it −−gpus all −−name samoyeds−ae kevinwu2017/samoyeds:1.0.0

