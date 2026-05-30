# A.2 Artifact check-list (meta-information)

- Compilation: NVCC 12.4, GCC 11.4.0.
- Model: OPT-2.7B, OPT-6.7B, OPT-13B, OPT-30B, LLaMA2-7B, LLaMA2- 70B.
- Data set: WikiText-2, ARC-e, HellaSwag, PiQA, Winogrande, Pile.
- Run-time environment: Ubuntu 22.04.5 LTS, CUDA 12.4, and PyTorh 2.5.1.
- Hardware: A server with an x86 processor and four NVIDIA RTX 6000 Ada GPUs.
- Output: Model perplexity and accuracy, simulator energy and performance.
- How much disk space required (approximately)?: About 270GB.
- How much time is needed to prepare workflow (approximately)?: It takes about 30 minutes to prepare the environment.
- How much time is needed to complete experiments (approximately)?: It takes approximately 230 hours to execute all experiments using the server equipped with GPUs. The most timeconsuming experiment requires about 90 hours and about 550GB cpu memory to finish.
- Publicly available?: Yes.
- Code licenses (if publicly available)?: Not specified.
- Data licenses (if publicly available)?: The datasets are publicly available through their original licensing terms.
- Workflow automation framework used?: Conda, shell scripts.
- Archived (provide DOI)?: [https://doi.org/10.5281/zenodo.16895417.](https://doi.org/10.5281/zenodo.16895417)

