# A Artifact Appendix

## A.1 Abstract

This artifact contains the source code of PIM-DL, including the implementation of model calibration, auto-tuner, and the inference engine. In addition, this artifact provides config files and scripts to reproduce the key experimental results reported in the paper.

## A.2 Artifact check-list (meta-information)

- Algorithm: LUT-based neural network (LUT-NN).
- Program: Python3, C, C++.
- Compilation: The compiler provided in UPMEM SDK (Version 2021.3.0), which is based on clang 10.0.0.
- Run-time environment: The system is developed and tested in Ubuntu 18.04.6 LTS (GNU/Linux 4.15.0-184-generic x86\_64).
- Hardware: The experiments were run on a machine with Intel Xeon 4210 CPU (dual-socket), 128 GB memory, and 8 UPMEM PIM-DIMMs (8GB/DIMM, DPU running at 350 MHz).
- Execution: Make sure no other workloads are running on the system during the experiment.
- Metrics: Normalized speedup and energy efficiency.
- Output: The resulting figures shown in paper for key experiments.
- Experiments: Scripts are included in the asplos24-ae folder. Detailed instructions are provided in asplos24-ae/README.md.
- How much disk space required (approximately)?:About 1GB.
- How much time is needed to prepare workflow (approximately)?: About 10 minutes.
- How much time is needed to complete experiments (approximately)?: About 3 hours.
- Publicly available?: Yes. Github link: [https://github.com/](https://github.com/leesou/PIM-DL-ASPLOS) [leesou/PIM-DL-ASPLOS](https://github.com/leesou/PIM-DL-ASPLOS).
- Code licenses (if publicly available)?: MIT License.
- Archived (provide DOI)?: Yes. DOI link: [https://doi.org/10.](https://doi.org/10.5281/zenodo.10531532) [5281/zenodo.10531532](https://doi.org/10.5281/zenodo.10531532)

## A.3 Description

A.3.1 How to access. For AE reviewers, considering it might be difficult to prepare a server equipped with UPMEM PIM-DIMMs, we provide ssh access to our server. For others who want to reproduce these experiments, we provide the open-sourced project on Github (Link: [https://github.com/](https://github.com/leesou/PIM-DL-ASPLOS) [leesou/PIM-DL-ASPLOS](https://github.com/leesou/PIM-DL-ASPLOS)), but the following dependencies should be satisfied.

A.3.2 Hardware dependencies. All experiments are run on a machine with Intel Xeon 4210 CPU (dual-socket), 128 GB memory, and 8 UPMEM PIM-DIMMs (8GB/DIMM, DPU running at 350 MHz).

A.3.3 Software dependencies. To use UPMEM PIM-DIM-Ms, UPMEM's SDK toolchain needs to be installed on the server. The SDK version on our server is 2021.3.0. Besides,

our server runs on Ubuntu 18.04.6 LTS (GNU/Linux 4.15.0- 184-generic x86\_64).

## A.4 Installation

Installation instructions are provided in the asplos24-ae folder. Please check asplos24-ae/README.md for more details.

### A.5 Experiment workflow

Experiment scripts are provided in the asplos24-ae folder. Please check asplos24-ae/README.md for more details.

## A.6 Evaluation and expected results

After finishing execution, all plotted results are saved in the asplos24-ae/results folder. These results should be in correspondence with Figure 10, 11, 12, and 13. Note that the results might be slightly different from that in the paper due to the runtime perturbation, but the trends should be similar. Please check asplos24-ae/README.md for more information on result validation.

