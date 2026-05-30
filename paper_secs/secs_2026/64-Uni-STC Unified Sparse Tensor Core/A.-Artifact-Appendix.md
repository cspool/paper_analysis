# A. Artifact Appendix

#### A.1 Abstract

This artifact appendix describes the experimental workflow to reproduce the results presented in the paper "Uni-STC: Unified Sparse Tensro Core" (Paper #313). We provide a containerized environment (Docker) pre-installed with the simulators, scripts, and small-scale datasets. The experiments are categorized into two levels: Fast Verification (approx. 5 hours) for functional validation and Complete Verification (approx. 75 hours) for full reproduction.

#### A.2 Artifact check-list (meta-information)

- Program: Python 3.9, Bash Scripts, C++ Simulators.
- Compilation: GCC 9+, OpenMP 4.5+, OpenCV 4.x.
- Data set: SuiteSparse Matrix Collection (2,800+ matrices) and DLMC.
- Run-time environment: Ubuntu 22.04 LTS (via Docker).
- Hardware: X86-64 CPU, ≥64 GB DRAM.
- Storage: ≥150 GB (Fast Mode) / ≥500 GB (Complete Mode).
- Experiments: Format overhead analysis, Performance comparison, AMG solver, and Energy Efficiency Density.
- Prepare workflow time: 3 hours to download a 40GB Image.
- Execution time: Fast mode: 5 hours; complete mode: 75 hours.
- Publicly available: Yes.
- Workflow automation framework used: Yes.

#### A.3 Description

#### A.3.1 How to access

We provide a persistent artifact package hosted on Google Drive, which includes:

- 1. Docker Image (HPCA-Pap313-AE.tar<sup>1</sup> ): Contains the OS, dependencies, small data set, simulators, and plotting scripts.
- 2. Full Dataset (matrix.7z<sup>2</sup> ): The complete SuiteSparse collection required for complete verification.

#### A.3.2 Hardware dependencies

To fully reproduce the results reported in the paper, we recommend the following hardware configuration:

- Processor: X86-64 CPU with at least 16 cores.
- Memory: Minimum 64 GB DRAM is required to load large matrices in the complete dataset.
- Disk: 100 GB for the docker image and fast verification. 600 GB for the full dataset decompression.ss

#### A.3.3 Software dependencies

The artifact is encapsulated in a Docker container to ensure environment consistency. The host machine requires:

- OS: Linux (Ubuntu 20.04/22.04 recommended).
- Docker Engine: Version ≥ 20.10.

Inside the container, the environment is pre-configured with:

- Compilers: GCC 11.4, CMake 3.22.
- Python Env: Python 3.10 with necessary libraries.
- OpenCV: Version 4.x for image processing.

#### A.4 Installation

#### A.4.1 Deployment

- 1. Download and Decompress. Download HPCA-Pap313-AE.tar from the link<sup>3</sup> .
- 2. Load and Start Container. Load the image into your local Docker registry and launch the container in the background. Note that if you encounter permission errors, please prepend sudo.

```
$ docker load < HPCA-Pap313-AE.tar
# Optional: remove the tar file to save space
$ rm HPCA-Pap313-AE.tar
$ docker run -itd --name HPCA-Pap313 hpca-pap313-ae:v2
```

#### A.4.2 Initialization

Access the container, upgrade python package and execute the initialization script. This script compiles the simulator binaries and checks library integrity.

```
$ docker exec -it HPCA-Pap313 /bin/bash
(container)$ cd /root
# upgrade package and compile
(container)$ pip3 install pip setuptools wheel -U
(container)$ pip3 install quickstart-rhy -U
(container)$ ./init.sh
```

Expected Output: The initialization is successful if the following logs appear:

```
[INFO] Compile ResNet50 (sparse) Succeeded!
[INFO] Compile ResNet50 (dense) Succeeded!
[INFO] Compile Simulator (Scheduler = 8) Succeeded!
```

