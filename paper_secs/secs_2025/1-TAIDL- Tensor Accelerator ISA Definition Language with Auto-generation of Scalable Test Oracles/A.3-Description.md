# A.3 Description

A.3.1 How to access.

Zenodo:<https://doi.org/10.5281/zenodo.16734309>

GitHub:<https://github.com/act-compiler/taidl-artifact-micro25>

A.3.2 Hardware dependencies.

Minimum (only TAIDL-TO (CPU)):

Any CPU (8GB+ RAM), No GPU required

Preferred (only TAIDL-TO (CPU) and TAIDL-TO (GPU)): Any CPU (8GB+ RAM), NVIDIA GPU (4GB+ VRAM)

Recommended (TAIDL-TO (CPU), TAIDL-TO (GPU), Baselines): Intel CPU (6th gen+, 8GB+ RAM), NVIDIA GPU (4GB+ VRAM)

Our artifact is built as Docker images that contain benchmarking environments for TAIDL-TO and the baselines. TAIDL-TOs can be benchmarked on personal laptops with TAIDL-TO (GPU) requiring an NVIDIA GPU. The baselines (Gemmini Spike and Intel SDE) only support amd64/x86\_64 CPU processor architecture. Therefore, Intel CPU + NVIDIA GPU is recommended for full evaluation.

- A.3.3 Software dependencies.
- (i) [Docker Engine](https://docs.docker.com/engine/install/) and (ii) [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

