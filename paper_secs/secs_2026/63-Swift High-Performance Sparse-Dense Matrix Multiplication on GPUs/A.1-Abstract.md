# A.1 Abstract

This section summarizes the artifact evaluation for this work. First, we provide the checklist for this artifact. Next, we describe the directory structure for the code. Finally, the installation, experiment workflow, and evaluation illustrate how to use the artifact to reproduce results and extend the implementation.

#### A.2 Artifact check-list (meta-information)

• Algorithm: Swift • Program: CUDA code • Compilation: NVCC

- Binary: After compilation, an executable file named test will be generated.
- Data set: SuiteSparse Matrix Collection (https://sparse.tamu.edu/).
- Run-time environment: Require isolated server as experiments sensitive to resource contention.
- Hardware: Platform 1: CPU: Intel(R) Core(TM) i9-14900K; GPU: RTX 4080 SUPER Platform 2: CPU: 12th Gen Intel(R) Core(TM) i9-12900K; GPU: RTX 3090Ti; Platform 3: CPU: Intel(R) Xeon(R) Gold 6151 CPU @ 3.00GHz; GPU: Tesla V100 Platform 4: Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz; GPU: NVIDIA A100
- Execution: Sole user
- Output: Experiments produce text files. When testing a single matrix, the results can be found in the \$<Swift dir>/src/data folder. For large-scale testing, the results can be found in the various subfolders under \$<Swift dir>/test/.
- How much disk space is required (approximately)?: 250 GB
- How much time is needed to prepare workflow (approximately)?: 1 hour
- How much time is needed to complete experiments (approximately)?: When testing a single matrix: about 1 minute. When testing all matrices in SuiteSparse Matrix Collection: At least 8 hours
- Publicly available?: Yes

