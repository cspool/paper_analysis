# *B. Artifact check-list (meta-information)*

- Compilation: NVCC 12.4, GCC 11.4.0.
- Model: OPT-6.7B, Llama-2-7B, Llama-2-70B, Llama-3-8B, Qwen3-8B, Qwen3-14B.
- Data set: WikiText-2, ARC-e, HellaSwag, PiQA, Winogrande.
- Run-time environment: Ubuntu 22.04.5 LTS, CUDA 12.4, and PyTorch 2.6.0.
- Hardware: A server with an x86 processor and four NVIDIA RTX 6000 Ada GPUs.
- Output: Model perplexity and accuracy, simulator energy and performance.
- How much disk space is required?: About 240 GB.
- How much time is needed to prepare workflow?: It takes about 30 minutes to prepare the environment.
- How much time is needed to complete experiments (approximately)?: It takes approximately 100 hours to execute all experiments using the server equipped with GPUs. The most time-consuming experiment requires about 20 hours to finish.
- Publicly available?: Yes.
- Data licenses (if publicly available)?: The datasets are publicly available through their original licensing terms.
- Workflow automation framework used?: Conda, shell scripts.
- Archived DOI: <https://doi.org/10.5281/zenodo.19449314>

