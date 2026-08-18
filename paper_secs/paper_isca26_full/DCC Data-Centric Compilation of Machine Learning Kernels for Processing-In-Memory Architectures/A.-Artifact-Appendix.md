# A. Artifact Appendix

## A.1. Abstract

The Artifact Appendix describes how to reproduce the main results of this paper. It includes the source code of DCC, benchmark scripts, and step-by-step instructions for the key evaluation results. The experiments require a server equipped with a CPU with at least 64 hardware threads, 128GB of main memory, disk space of at least 128GB, and an NVIDIA GPU with an up-to-date NVIDIA driver installed. Note that our artifact needs Anaconda 25.6.1+ and CMake 3.16.3 to be installed in the server. We provide a README.md file that describes the required hardware and software dependencies and provides step-by-step instructions. This artifact is used to support our major claims (See Appendix [§A.6\)](#page-18-0), demonstrating DCC's performance benefits in Figures [8,](#page-10-0) [9](#page-10-1) and [11.](#page-11-0) We expect the full evaluation pipeline, including the setup and simulations, to take approximately 7-10 days. We also expect that the trace files generated for trace-driven simulations take approximately 90GB disk space.

## A.2. Artifact Check-list (meta-information)

- Program: DCC\_Artifact: In this artifact, we evaluate DCC compiler upon HBM-PIM and AttAcc PIM backends and compare HBM-PIM, HBM-PIM + DCC, AttAcc, AttAcc + DCC comparison points against GPU baseline.
- Compilation: This artifact strictly requires CMake 3.16.3 and GCC/G++ 11.4.
- Models: The workloads for kernel performance and inference performance have been configured to evaluate the GPT3-13B and LLAMA2-33B models.
- Run-time environment: Linux Ubuntu 20.04 (or newer) with Python 3.8, requiring CUDA 12.3 (or newer).
- Hardware: A server equipped with a CPU of at least 64 hardware threads and 128GB of main memory, and an NVIDIA GPU device with a minimum compute capability of 8.0 and at least 8GB GPU memory should be used to validate the results.
- Execution: Trace-driven simulations.
- Metrics: Execution time normalized as relative performance speedup.
- Output: Experimental results are stored as CSV files, and our scripts generate corresponding figures in PDF format. The generated figures are similar to the Figures [8,](#page-10-0) [9,](#page-10-1) and [11](#page-11-0) of the main paper.
- Experiments: Kernel performance and end-to-end inference performance. Detailed steps for the reproduction of evaluation experiments are provided in the README.md file.
- How much disk space required (approximately)?: 128 GB.
- How much time is needed to prepare workflow (approximately)?: 10 minutes (build the code and download Python packages).
- How much time is needed to complete experiments (approximately)?: 7-10 days (run simulations and generate figures).
- Publicly available?: Yes.
- Code licenses: MIT.

• Archived: [https://doi.org/10.5281/zenodo.](https://doi.org/10.5281/zenodo.19442321) [19442321](https://doi.org/10.5281/zenodo.19442321)

## A.3. Description

<span id="page-17-0"></span>A.3.1. How to Access. Download the compressed file DCC\_Artifact.zip from the Zenodo archive [https://doi.](https://doi.org/10.5281/zenodo.19442321) [org/10.5281/zenodo.19442321](https://doi.org/10.5281/zenodo.19442321) or our GitHub repository at <https://github.com/SPIN-Research-Group/DCC>.

A.3.2. Hardware Dependencies. The artifact should be tested on a sever with:

- x86-64 CPU with at least 64 hardware threads, 128GB of main memory and 128GB of disk storage.
- NVIDIA GPU device with a minimum compute capability (SM) of 8.0 and at least 8GB GPU memory.

A.3.3. Software Dependencies. The artifact requires the following software for installation:

- Ubuntu 20.04 (or newer)
- Python 3.8
- GNU compilers (gcc/g++) 11.4.0 (strict requirement)
- CUDA 12.3 (or newer)
- CMake 3.16.3 (strict requirement)
- Anaconda 25.6.1 (or newer)
- Package scikit-learn 1.3.2 (or newer)
- Package XGBoost 2.1.4 (or newer)

## A.4. Installation

Download the zip file containing the artifact source code in [§A.3.1.](#page-17-0) We provide detailed instructions in the README.md file under the root of source code directory to build the simulator and evaluate DCC on two PIM backends.

We next summarize the key steps:

1. Install Anaconda 25.6.1 (or newer) following the instructions provided in [https://www.anaconda.com/docs/](https://www.anaconda.com/docs/getting-started/main) [getting-started/main](https://www.anaconda.com/docs/getting-started/main). You may need to run the following command after installation to enable conda to create a new virtual environment. Please refer to [https://www.](https://www.anaconda.com/docs/getting-started/tos-plugin) [anaconda.com/docs/getting-started/tos-plugin](https://www.anaconda.com/docs/getting-started/tos-plugin).

#### \$ conda tos accept

- 2. Download and install Cmake 3.16.3 (strict requirement) following the instructions provided in [https://cmake.org/](https://cmake.org/download/) [download/](https://cmake.org/download/). Please note that we have not extensively tested all various CMake versions. We recommend using and building our artifact with Cmake 3.16.3.
- 3. Download the source code of DCC\_Artifact (See [§A.3.1\)](#page-17-0).
- 4. Setup the runtime environment for all experiments by running the following bash setup script under the root of source code directory:

\$ bash setup.sh

## A.5. Experiment Workflow

The artifact contains three experiments to conduct the evaluation of the Figures [8,](#page-10-0) [9,](#page-10-1) and [11](#page-11-0) of the main paper. We have a bash script to launch all the simulations of the three evaluation experiments, collect the raw results and save them under the results/ directory. We also include plotting scripts to parse

the raw results and generate the figures under the figures/ directory. Next, we provide more details.

Launch Experiments & Visualize the Results: We strongly recommend using a server with at least 64 hardware threads and at least 128GB of main memory. The following script (i) launches all the experiments required to reproduce the key results of our paper, (ii) stores the raw results under the results/ directory, and (iii) generates the Figures [8](#page-10-0) and [9,](#page-10-1) [11](#page-11-0) under the figures/ directory:

```
$ bash run_experiments.sh
```

Relaunch Failed Experiments (if any): If there are any failed simulations (the plotting scripts may also fail to visualize the figures), re-run the main bash script:

```
$ bash run_experiments.sh
```

## <span id="page-18-0"></span>A.6. Evaluation and Expected Result

Major Claims. For each of the three experiments and for the same workload configuration, we expect the reproduced results to be similar to those reported in the paper within ±4% due to randomness in the DCC's predictor offline training process. Specifically, DCC's predictor is trained offline by selecting a small set of random workload configurations (input token sizes, output token sizes, batch sizes etc.). We will focus on the range of speedup values in each evaluation experiment to verify the reproduction of the key results. We next clarify our major claims:

- 1. Figure [8:](#page-10-0) AttAcc + DCC achieves kernel performance speedups for the workload configurations of the ATTN, GEMV and RED kernels in the range of 9.22×–13.70×, 6.55×–10.18×, and 0.95×–4.99×, respectively, over GPU, and in the range of 1.01×–1.64×, 1.00×–1.80×, and 1.02×– 1.73×, respectively, over AttAcc.
- 2. Figure [9:](#page-10-1) HBM-PIM + DCC achieves kernel performance speedups for the workload configurations of the GEMV, RED, VA, and RELU kernels in the range of 4.63×–7.99×, 0.80×–3.48×, 1.29×–5.58×, and 1.60×–6.96×, respectively, over GPU, and in the range of 1.18×–2.45×, 1.35×–2.44×, 1.35×–2.43×, and 1.34×–2.42×, respectively, over HBM-PIM.
- 3. Figure [11:](#page-11-0) Attacc*Base* + DCC achieves end-to-end performance speedups for the workload configurations of the two LLMs in the range of 1.15×–1.73× over GPU, and in the range of 1.06×–1.41× over Attacc*Base*. Attacc*F ull* + DCC achieves end-to-end performance speedups in the range of 2.77×–8.02× over GPU, and in the range of 1.20×–2.85× over Attacc*F ull*.

# A. Artifact Appendix

## A.1. Abstract

The Artifact Appendix describes how to reproduce the main results of this paper. It includes the source code of DCC, benchmark scripts, and step-by-step instructions for the key evaluation results. The experiments require a server equipped with a CPU with at least 64 hardware threads, 128GB of main memory, disk space of at least 128GB, and an NVIDIA GPU with an up-to-date NVIDIA driver installed. Note that our artifact needs Anaconda 25.6.1+ and CMake 3.16.3 to be installed in the server. We provide a README.md file that describes the required hardware and software dependencies and provides step-by-step instructions. This artifact is used to support our major claims (See Appendix [§A.6\)](#page-18-0), demonstrating DCC's performance benefits in Figures [8,](#page-10-0) [9](#page-10-1) and [11.](#page-11-0) We expect the full evaluation pipeline, including the setup and simulations, to take approximately 7-10 days. We also expect that the trace files generated for trace-driven simulations take approximately 90GB disk space.

## A.2. Artifact Check-list (meta-information)

- Program: DCC\_Artifact: In this artifact, we evaluate DCC compiler upon HBM-PIM and AttAcc PIM backends and compare HBM-PIM, HBM-PIM + DCC, AttAcc, AttAcc + DCC comparison points against GPU baseline.
- Compilation: This artifact strictly requires CMake 3.16.3 and GCC/G++ 11.4.
- Models: The workloads for kernel performance and inference performance have been configured to evaluate the GPT3-13B and LLAMA2-33B models.
- Run-time environment: Linux Ubuntu 20.04 (or newer) with Python 3.8, requiring CUDA 12.3 (or newer).
- Hardware: A server equipped with a CPU of at least 64 hardware threads and 128GB of main memory, and an NVIDIA GPU device with a minimum compute capability of 8.0 and at least 8GB GPU memory should be used to validate the results.
- Execution: Trace-driven simulations.
- Metrics: Execution time normalized as relative performance speedup.
- Output: Experimental results are stored as CSV files, and our scripts generate corresponding figures in PDF format. The generated figures are similar to the Figures [8,](#page-10-0) [9,](#page-10-1) and [11](#page-11-0) of the main paper.
- Experiments: Kernel performance and end-to-end inference performance. Detailed steps for the reproduction of evaluation experiments are provided in the README.md file.
- How much disk space required (approximately)?: 128 GB.
- How much time is needed to prepare workflow (approximately)?: 10 minutes (build the code and download Python packages).
- How much time is needed to complete experiments (approximately)?: 7-10 days (run simulations and generate figures).
- Publicly available?: Yes.
- Code licenses: MIT.

• Archived: [https://doi.org/10.5281/zenodo.](https://doi.org/10.5281/zenodo.19442321) [19442321](https://doi.org/10.5281/zenodo.19442321)

## A.3. Description

<span id="page-17-0"></span>A.3.1. How to Access. Download the compressed file DCC\_Artifact.zip from the Zenodo archive [https://doi.](https://doi.org/10.5281/zenodo.19442321) [org/10.5281/zenodo.19442321](https://doi.org/10.5281/zenodo.19442321) or our GitHub repository at <https://github.com/SPIN-Research-Group/DCC>.

A.3.2. Hardware Dependencies. The artifact should be tested on a sever with:

- x86-64 CPU with at least 64 hardware threads, 128GB of main memory and 128GB of disk storage.
- NVIDIA GPU device with a minimum compute capability (SM) of 8.0 and at least 8GB GPU memory.

A.3.3. Software Dependencies. The artifact requires the following software for installation:

- Ubuntu 20.04 (or newer)
- Python 3.8
- GNU compilers (gcc/g++) 11.4.0 (strict requirement)
- CUDA 12.3 (or newer)
- CMake 3.16.3 (strict requirement)
- Anaconda 25.6.1 (or newer)
- Package scikit-learn 1.3.2 (or newer)
- Package XGBoost 2.1.4 (or newer)

## A.4. Installation

Download the zip file containing the artifact source code in [§A.3.1.](#page-17-0) We provide detailed instructions in the README.md file under the root of source code directory to build the simulator and evaluate DCC on two PIM backends.

We next summarize the key steps:

1. Install Anaconda 25.6.1 (or newer) following the instructions provided in [https://www.anaconda.com/docs/](https://www.anaconda.com/docs/getting-started/main) [getting-started/main](https://www.anaconda.com/docs/getting-started/main). You may need to run the following command after installation to enable conda to create a new virtual environment. Please refer to [https://www.](https://www.anaconda.com/docs/getting-started/tos-plugin) [anaconda.com/docs/getting-started/tos-plugin](https://www.anaconda.com/docs/getting-started/tos-plugin).

#### \$ conda tos accept

- 2. Download and install Cmake 3.16.3 (strict requirement) following the instructions provided in [https://cmake.org/](https://cmake.org/download/) [download/](https://cmake.org/download/). Please note that we have not extensively tested all various CMake versions. We recommend using and building our artifact with Cmake 3.16.3.
- 3. Download the source code of DCC\_Artifact (See [§A.3.1\)](#page-17-0).
- 4. Setup the runtime environment for all experiments by running the following bash setup script under the root of source code directory:

\$ bash setup.sh

## A.5. Experiment Workflow

The artifact contains three experiments to conduct the evaluation of the Figures [8,](#page-10-0) [9,](#page-10-1) and [11](#page-11-0) of the main paper. We have a bash script to launch all the simulations of the three evaluation experiments, collect the raw results and save them under the results/ directory. We also include plotting scripts to parse

the raw results and generate the figures under the figures/ directory. Next, we provide more details.

Launch Experiments & Visualize the Results: We strongly recommend using a server with at least 64 hardware threads and at least 128GB of main memory. The following script (i) launches all the experiments required to reproduce the key results of our paper, (ii) stores the raw results under the results/ directory, and (iii) generates the Figures [8](#page-10-0) and [9,](#page-10-1) [11](#page-11-0) under the figures/ directory:

```
$ bash run_experiments.sh
```

Relaunch Failed Experiments (if any): If there are any failed simulations (the plotting scripts may also fail to visualize the figures), re-run the main bash script:

```
$ bash run_experiments.sh
```

## <span id="page-18-0"></span>A.6. Evaluation and Expected Result

Major Claims. For each of the three experiments and for the same workload configuration, we expect the reproduced results to be similar to those reported in the paper within ±4% due to randomness in the DCC's predictor offline training process. Specifically, DCC's predictor is trained offline by selecting a small set of random workload configurations (input token sizes, output token sizes, batch sizes etc.). We will focus on the range of speedup values in each evaluation experiment to verify the reproduction of the key results. We next clarify our major claims:

- 1. Figure [8:](#page-10-0) AttAcc + DCC achieves kernel performance speedups for the workload configurations of the ATTN, GEMV and RED kernels in the range of 9.22×–13.70×, 6.55×–10.18×, and 0.95×–4.99×, respectively, over GPU, and in the range of 1.01×–1.64×, 1.00×–1.80×, and 1.02×– 1.73×, respectively, over AttAcc.
- 2. Figure [9:](#page-10-1) HBM-PIM + DCC achieves kernel performance speedups for the workload configurations of the GEMV, RED, VA, and RELU kernels in the range of 4.63×–7.99×, 0.80×–3.48×, 1.29×–5.58×, and 1.60×–6.96×, respectively, over GPU, and in the range of 1.18×–2.45×, 1.35×–2.44×, 1.35×–2.43×, and 1.34×–2.42×, respectively, over HBM-PIM.
- 3. Figure [11:](#page-11-0) Attacc*Base* + DCC achieves end-to-end performance speedups for the workload configurations of the two LLMs in the range of 1.15×–1.73× over GPU, and in the range of 1.06×–1.41× over Attacc*Base*. Attacc*F ull* + DCC achieves end-to-end performance speedups in the range of 2.77×–8.02× over GPU, and in the range of 1.20×–2.85× over Attacc*F ull*.

