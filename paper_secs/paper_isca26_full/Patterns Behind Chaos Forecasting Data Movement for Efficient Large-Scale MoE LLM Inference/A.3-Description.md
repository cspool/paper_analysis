# *A.3 Description*

*A.3.1 How to Access:* Case Study 1 (wafer-scale GPU simulator): GitHub: [waferscale\\_gpu\\_moe\\_sim](https://github.com/zhongkaiyu/waferscale_gpu_moe_sim); DOI: [10.5281/zenodo.19617713.](https://doi.org/10.5281/zenodo.19617713)

Case Study 2 (real-GPU expert placement): GitHub: [moe\\_exp\\_placement](https://github.com/zhongkaiyu/moe_exp_placement); DOI: [10.5281/zenodo.19617695.](https://doi.org/10.5281/zenodo.19617695)

Each repository contains a README.md with setup, execution, and troubleshooting instructions. The Zenodo archives provide persistent snapshots of the evaluated artifact versions.

- *A.3.2 Hardware Dependencies:* Case Study 1 runs on a CPU server with at least 64 GB RAM and does not require a GPU. Case Study 2 requires an 8×NVIDIA H100 80 GB GPU server, CUDA 12.0 or newer, and about 300 GB of disk space. Reviewers without GPU access can still evaluate the primary simulator artifact.
- *A.3.3 Software Dependencies:* Case Study 1 requires Python ≥ 3.10 plus numpy, pandas, and matplotlib; the scripts install them automatically. Case Study 2 additionally requires PyTorch, a modified SGLang fork, DeepEP, and DeepGEMM. The repository documents exact installation commands and environment settings.
- *A.3.4 Datasets:* Both artifacts use pre-recorded MoE expert-selection traces from MMLU. The traces are hosted on HuggingFace and downloaded automatically by the AE scripts.

