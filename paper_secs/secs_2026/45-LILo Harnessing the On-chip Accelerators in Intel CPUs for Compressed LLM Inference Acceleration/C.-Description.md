# *C. Description*

- *1) How to Access.:* The scripts and guidelines for the deployment of LILO are publicly available on GitHub (https://github.com/ece-fast-lab/HPCA-2026-LILo) and Zenodo (https://doi.org/10.5281/zenodo.17862931. The repository contains step-by-step scripts under scripts/ for environment preparation, baseline evaluation, decompression-based inference, and storage-offloading characterization, as well as a separate figure generation module.
- *2) Hardware Dependencies.:* A server with an Intel Xeon Scalable Processor (≥ 4th generation) equipped with at least one IAA is required. For best performance reproduction, four IAAs are recommended. The following BIOS settings must be enabled:
  - Hardware prefetch
  - LLC prefetch
  - Adjacent cache prefetch
- *3) Software Dependencies.:* We provide pre-built docker images that contain all required runtime dependencies and libraries, including:
  - Ubuntu 22.04 LTS
  - Linux kernel 6.8.0-49-generic
  - GCC 13.1.0
  - Intel Query Processing Library (QPL)
  - Intel idxd-config for IAA configuration

#### • Intel Extension for PyTorch (IPEX)

Note that the usage of IAA requires the installation of the Intel Query Processing Library (QPL) (https://github.com/ intel/qpl) and idxd-config (https://github.com/intel/idxdconfig), both of which are already incorporated into the provided docker images. To enable IAA, IOMMU also should be enabled via the following GRUB setting:

```
GRUB_CMDLINE_LINUX="quiet iommu=pt
   intel_iommu=on sm_on no5lvl splash
   intel_pstate=disable efi=nosoftreserve
   nokaslr"
```

#### *D. Installation*

First, download the Github repository as follows:

```
$ git clone https://github.com/ece-fast-lab/\
   HPCA-2026-LILo.git
$ cd HPCA-2026-LILo
```

The full reproduction pipeline is organized into four steps under scripts/. Users should follow the steps sequentially.

#### *E. Experiment Flow*

*1) Step 0: Environment Setup:* To prepare the software environment and system configuration, run the following commands:

```
$ cd scripts/step_0_env_setup
$ bash ./get_docker.sh
$ bash ./env_setup.sh
```

These commands download all required Docker images, fix the CPU frequency, and configure the IAA devices using idxd-config. The repository further provides instructions creating a cropped version of Llama3-405B and DeepSeek-R1 and randomizing MoE routing for DeepSeek-R1 with a patch.

*2) Step 1: Uncompressed Baseline Inference:* To collect inference latency for the uncompressed baseline, run the following command:

```
$ cd ../step_1_baseline
$ bash ./baseline.sh <docker_name>
```

This script launches a Docker container, executes the uncompressed baseline inference for both Llama and DeepSeek models, and saves the latency logs under the results directory. Note that you must update the mounted directory (-v) in baseline.sh to point to the storage location of your model weights on the local machine; the same update is also required for Steps 2 and 3.

*3) Step 2: Inference with Decompression (*LILO*):* To evaluate inference with on-the-fly decompression using LILO, run the following command:

```
$ cd ../step_2_decomp
$ bash ./decomp.sh <docker_name>
```

This script runs decompression-enabled inference for both Llama and DeepSeek models and writes the resulting latency logs to the results directory.

*4) Step 3: Storage-Offloading Characterization:* To characterize the overhead of storage-offloaded inference, run the following command:

```
$ cd ../step_3_storage_offload
$ bash ./storage_offload.sh <docker_name>
```

This step evaluates storage-offloaded inference using HuggingFace Accelerate by sweeping different amounts of offloaded data and recording the corresponding latency results. These measurements are later used to characterize the storageoffloading overhead by fitting a performance model, which is then combined with the latency results of both LILO and the uncompressed baseline.

#### *F. Evaluation and Expected Results*

To reproduce the key experimental results from the paper, the collected data is used to generate (1) Figure 11(a), which compares the inference throughput between LILO and the baseline, (2) Figure 11(b), which presents the latency breakdown, and (3) Figure 12, which shows throughput improvement under varying memory capacities. To generate all figures automatically, run the following command:

```
$ cd ../../fig_gen
$ python3 generate_figures.py
```