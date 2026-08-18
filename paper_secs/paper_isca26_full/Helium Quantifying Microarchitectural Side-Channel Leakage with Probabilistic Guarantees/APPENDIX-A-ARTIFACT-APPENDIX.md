# APPENDIX A ARTIFACT APPENDIX

#### *A. Abstract*

This artifact demonstrates leakage quantification with Helium, using both TracerSym and TracerSim to compute pointwise maximal leakage (PML) tail-bound privacy guarantees for multiple programs. The artifact provides a Dockerfile that builds a container image containing all required dependencies (including Python 3, the necessary Python packages, and Intel Pin), along with scripts to reproduce all four case studies presented in the paper. The artifact can be executed on any x86\_64 machine with Docker, Git, and Bash support, and requires at least 32 GB of RAM and 8 GB of available disk space.

#### *B. Artifact check-list (meta-information)*

- Data set: All workloads are precompiled and included in the artifact repository.
- Run-time environment: Docker, Git, Python3, and Bash must be installed on the local machine.
- Hardware: An x86-64 CPU with at least 32 GB of RAM and 8 GB of disk space is required.
- Metrics: PML tail-bound leakage guarantees for different programs under various leakage functions and runtime statistics.
- Output: Outputs include log files, tables, and plots (PDF figures).
- Experiments: There are four experiments, each corresponding to the case studies in Section [VII.](#page-8-0)
- How much disk space required (approximately)?: 8 GB
- How much time is needed to prepare workflow (approximately)?: Less than 5 minutes
- How much time is needed to complete experiments (approximately)?: Experiments were conducted on a dual-socket server equipped with 2 × Intel Xeon Gold 6226R CPUs at 2.90 GHz. Each socket contains 16 physical cores (32 physical cores total) with 2-way simultaneous multithreading (64 logical CPUs). The system has 512 GB RAM; however the experiments used a maximum of 32 GB of RAM. The total time to complete all four case studies on this system is approximately 10 hours.
- Publicly available?: Yes: [Helium-Artifact](https://github.com/samanthaarcher0/Helium-Artifact)
- Code licenses (if publicly available)?: MIT License
- Workflow automation framework used?: Docker
- Archived (provide DOI)?: [DOI](https://doi.org/10.5281/zenodo.19446913)

#### *C. Description*

- *1) How to access:* The code for this submission can be downloaded from the [Helium-Artifact](https://github.com/samanthaarcher0/Helium-Artifact) repository. The Helium-Artifact repository includes a Dockerfile that can be used to build the Docker image for the full evaluation of the artifact. The Docker image is also available at [DOI.](https://doi.org/10.5281/zenodo.19446913)
- *2) Hardware dependencies:* This artifact requires an x86\_64 machine with at least 32 GB of RAM and 8 GB of available disk space.
- *3) Software dependencies:* The host machine must support Docker, Git, and a POSIX-compatible shell (e.g., Bash). All other software dependencies (including Python 3, required Python packages, Intel Pin, and solver binaries) are installed inside the Docker container and do not need to be pre-installed on the host system.

*4) Data sets:* All evaluation workloads are precompiled and included in the artifact repository. The binaries are selfcontained with respect to third-party libraries (e.g., Libsodium and Firefox) and do not require these libraries to be installed at runtime.

#### *D. Installation*

To install the artifact, first clone the [Helium-Artifact](https://github.com/samanthaarcher0/Helium-Artifact) repository to your local machine. Next, build the Docker image (this step takes approximately less than 2 minutes). The commands are as follows:

```
$ git clone https://github.com/samanthaarcher0/
    Helium-Artifact.git
```

- \$ cd Helium-Artifact
- \$ docker build -f Dockerfile -t helium\_artifact .

After the image has been successfully built, launch the container:

\$ docker run -it helium\_artifact

#### *E. Experiment workflow*

The experimental workflow consists of running one bash script per case study in the paper (four in total) in the Docker container. Each bash script launches multiple processes to generate the figures and tables in the evaluation.

#### *F. Evaluation and expected results*

All four case studies, along with their corresponding outputs and figures, can be reproduced as described in this section. In total, the full evaluation requires approximately 10 hours on a dual-socket server equipped with two Intel Xeon Gold 6226R CPUs (2.90 GHz). The system has 32 physical cores with 2 way simultaneous multithreading (64 logical CPUs) and 512 GB of RAM. However, the experiments use at most 32 GB of memory. All workloads are provided as precompiled binaries; no additional build step is required.

1) *Case study I:* This case study evaluates cryptographic MAC Poly1305 under two multiplication µobs functions, zero-skip and digit-serial multiplication. It takes less than 6 minutes to run. The outputs can be found in results\_ case\_study\_I directory. To run:

```
### In the Docker container ###
$ ./run_case_study_I.sh
```

#### Generated outputs:

- Poly1305\_tail\_bound\_guarantees.log: The log contains the two tail-bound guarantees that are discussed in the text of [VII-A,](#page-8-1) one for Poly1305 under zero-skip multiplication and the other for Poly1305 under digitserial multiplication.
- Figure\_7\_poly1305\_ep\_delt\_under\_two\_lfs.pdf: Figure [7](#page-8-2) shows all possible tail-bound guarantees of Poly1305 under the two multiply optimizations.
- 2) *Case study II:* This case study evaluates the Firefox convolution SVG filter under the same two multiplication µobs functions from Case Study I. It takes less than a

minute to run. The outputs can be found in the results \_case\_study\_II directory. To run:

```
### In the Docker container ###
$ ./run_case_study_II.sh
```

#### Generated outputs:

- Table\_IV\_convolve\_tail\_bound\_guarantees. log: Table [IV](#page-9-2) with tail-bound guarantees of Firefox's convolution under zero-skip and digitserial multiplication µobs functions.
- 3) *Case study III:* This case study evaluates the scalability of TracerSym, measuring the increase in runtime and number of SMT queries as the number of instrumented instructions and the number of µobs per µobs function increases. It takes 1.5 hours to run. The outputs can be found in results\_case\_study\_III directory. To run:

```
### In the Docker container ###
$ ./run_case_study_III.sh
```

#### Generated outputs:

- Figure\_8\_scalability\_eval.pdf: Figure [8](#page-10-1) showing TracerSym's runtime and number of SMT queries with increasing numbers of instrumented instructions and increasing the number of µobs per µobs function, as described in [§VII-C.](#page-9-1)
- 4) *Case study IV:* This last case study computes PML tailbound guarantees using our simulation-based methodology, TracerSim, for four cryptographic programs studied in a recent work [\[37\]](#page-13-15). This case study takes less than 6 hours to run. Note, as TracerSim runs Monte Carlo simulations, the exact tail-bound guarantees will differ slightly between runs. However, all values should be reasonably close to those reported in [§VII-D.](#page-10-0) Further, due to slight updates in our tracing Pin tool, the table's instruction counts differ slightly, but do not meaningfully change the leakage guarantees. The outputs can be found in results\_case\_study\_IV directory. To run:

```
### In the Docker container ###
$ ./run_case_study_IV.sh
```

