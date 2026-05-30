# *A. Abstract*

This artifact appendix provides a guideline for using LILO for compressed LLM inference acceleration and how to reproduce the three key results of this paper: 1) LILO's inference throughput comparison to the uncompressed baseline 2) inference latency breakdown of LILO and the uncompressed baseline, and 3) throughput improvement with LILO over the uncompressed baseline across varying memory capacity. The following subsections outline the steps to access, setup the software environment, and to run experiments with LILO on a CPU system with Advanced Vector Extension (AVX), Advanced Matrix Extension (AMX), and In-memory Analytics (IAA) accelerators.

#### *B. Artifact check-list (meta-information)*

- Model: Llama3-405B and DeepSeek-R1
- Run-time environment: Ubuntu 22.04.4 LTS, Linux kernel 6.8.
- Hardware: Intel Xeon 6980P Processor, Micron 7450 NVMe M.2 SSD.
- Metrics: Inference latency (seconds) and throughput (tokens/s).
- Output: .log files containing inference latency measurements
- Experiments: LLM inference with LILO and the uncompressed
- baseline to reproduce Figures 11 and 12. • How much disk space required (approximately)?: 720 GB for the model weights and the Docker images.
- How much time is needed to prepare workflow (approximately)?: 1 hour.
- How much time is needed to complete experiments (approximately)?: 20 hours.
- Publicly available?: Yes.
- Code licenses (if publicly available)?: Apache-2.0 license.
- Archived: https://doi.org/10.5281/zenodo.17862931

