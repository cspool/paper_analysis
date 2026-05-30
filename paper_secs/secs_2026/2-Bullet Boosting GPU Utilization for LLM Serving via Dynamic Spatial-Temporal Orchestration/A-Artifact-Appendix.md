# A Artifact Appendix

### A.1 Abstract

Bullet is an LLM serving system that enables intra-GPU prefill-decode disaggregation. Bullet is built on top of SGLang [78] and leverages libsmctrl [5] for SM partitioning. The artifact provides scripts to evaluate Llama3.1-8B [32] on an NVIDIA A100 80GB GPU, serving ShareGPT [61] workload with various request rates.

## A.2 Artifact check-list (meta-information)

- Compilation: CMake 3.17, GCC 10.2.1, CUDA 12.4.
- Model: Llama3.1-8B [32].
- Data set: ShareGPT [61].
- Run-time environment: Debian 5.10.0, CUDA 12.4, Python 3.12.9, PyTorch 2.6.0.
- Hardware: 1 NVIDIA A100 80GB GPU, x86 machine.
- How much disk space required (approximately)?: 20GB.
- How much time is needed to prepare workflow (approximately)?: 10 minutes.
- How much time is needed to complete experiments (approximately)?: 30 minutes.
- Publicly available?: Yes
- Archived (provide DOI)?: 10.5281/zenodo.17937105

## A.3 Description

**A.3.1 How to access.** The source code is available at Github (https://github.com/zejia-lin/BulletServe) and Zenodo (https://doi.org/10.5281/zenodo.17937105).

**A.3.2 Hardware dependencies.** The artifact requires 1 NVIDIA A100 80GB GPU and x86 machine for functional evaluation.

**A.3.3 Software dependencies.** Bullet uses libsmctrl [5] for SM masking, which is included in the csrc/ folder and requires CMake ≥ 3.17 for compilation. Bullet is tested with Python 3.12.9, and lower version may incur unexpected crash. Anaconda and uv is required to create environment and resolve Python dependencies.

**A.3.4 Data sets.** The experiment scripts automatically download the ShareGPT [61] dataset, which is available at https://huggingface.co/datasets/anon8231489123/ShareGPT\_Vicuna unfiltered.

A.3.5 Models. The artifact evaluates Llama3.1-8B [32], which is available at https://huggingface.co/meta-llama/Llama-3.1-8B. Since Hugging Face may require download permissions, users can obtain the model from alternative sources. To reduce setup time, the artifact uses dummy weights, requiring only JSON configuration files to be downloaded.

