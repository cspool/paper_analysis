# Acknowledgement

We thank the anonymous ASPLOS reviewers and our shepherd, Todd Mytkowicz, for their comments and helpful feedback. This material is based upon work supported by NSF awards CNS-2147909, CNS-2211882, and CNS-2239351, and research awards from Amazon, Cisco, Google, Meta, Oracle, Qualcomm, and Samsung.

## A Artifact

#### A.1 Abstract

This artifact includes codes and scripts for reproducing all experiments in the paper. We also release the collected available trace using the g4dn spot instances in our artifact. To reproduce our experiments, we require twelve network-accessible AWS g4dn.12xlarge instances, each with 4 NVIDIA Tesla T4 GPUs, all of which require CUDA, NCCL, MPI, Python dependencies to be installed. This artifact consists of three components: Global Server (i.e. Inference Server), Params Client (i.e. Context Daemon), and modified FasterTransformer (i.e. Inference Engine). The first component is written in Python, while the other two are in C++. Our provided scripts will automatically launch all of them to perform experiment.

#### A.2 Artifact check-list (meta-information)

- Program: GlobalServer, ParamsClient, FasterTransformer(modified).
- Compilation: CMake.
- Run-time environment: CUDA, NCCL, MPI, Python3.
- Hardware: 12 AWS g4dn.12xlarge instances
- Execution: Inference are performing in GPUs, and the Global Server are managing requests and instances on CPUs.
- Metrics: End-to-end average/tail latency.
- Output: End-to-end latency for each requests, and figures.
- Experiments: End-to-End, Price Comparison, Fluctuating Workload, Ablation Study.
- How much disk space required (approximately)?:
   600GB per instance.
- How much time is needed to prepare workflow (approximately)?: 2 hours to install dependencies, build components and complete configuration.
- How much time is needed to complete experiments (approximately)?: 16 hours.
- Publicly available?: Yes.

- Code licenses (if publicly available)?: The SpotServe artifact is released under the Apache-2.0 License.
- Archived (provide DOI)?: https://doi.org/10.5281/zenodo. 10558752

#### A.3 Description

**A.3.1 How to access.** The artifact is available on Github: https://github.com/Hsword/SpotServe/blob/artifact/README.md, including installation guide and benchmark scripts to reproduce results.

**A.3.2 Hardware dependencies.** We conduct the experiments with twelve AWS g4dn.12xlarge instances, and each is equipped with four NVIDIA Tesla T4 GPUs and x86\_64 CPU. The network bandwidth across instances is 50Gbps.

**A.3.3 Software dependencies.** Following toolkits are required: CUDA ( $\geq 10.2$ ), NCCL ( $\geq 2.10$ ), MPI, and CMake ( $\geq 3.8$ ) is highly recommended for building the components.

#### A.4 Installation

To install the artifact, users need to build ParamsClient and our modified FasterTransformer individually. It is recommended that compile the components on single instance and send them to other nodes by rsync command later (See Experiment workflow).

*Install FasterTransformer.* If dependencies are not satisfied, CMake will report the missing dependencies:

```
cd ./FasterTransformer
mkdir build && cd build
cmake -DSM=75 -DCMAKE_BUILD_TYPE=Release
    -DBUILD_MULTI_GPU=ON ..
make multi_gpu_gpt_example_iter -j 8
```

#### **Install ParamsClient.** Installation command is similar:

```
cd ./ParamsClient
mkdir build && cd build
cmake ..
make -j 8
```

**Preparing Checkpoints.** Since we focus on the end-to-end latency, using randomized checkpoints is acceptable, we provide a python script to randomly generate model checkpoints. To save disk space, the first layer weights are the only generated files, all weights in succeeding layers are linked to the corresponding files of the first layer. Following command will generate checkpoint files for specified model that can be directly used by out system. Available candidates of model\_name are 6.7B, 20B, 30B.

```
cd ./ckpt
python generate_random_gpt_ckpt.py \
    -o <model_name>
```

Configure Environment. Configuration is required:

- ./elastic\_switch/trace/hostfile\_aws\_T4: The IP address of your instances, one entry each line, and at least 12 entries.
- ./elastic\_switch/scripts\_ae/env.sh: Set NIC, path to MPI, and your base directory. See its contents for details.

**Sync Codes and Data.** Make sure that all nodes are accessible to each other, and the Hostfile has been configured. We provide a Python script to automatically send built components and checkpoints (optional) to all the instances. Please set base directory and the IP address where components are built in sync\_code.py, and run following command:

```
python sync_code.py --n 12 --sync-dataset \
    --hostfile elastic-switch/trace/hostnameT4
```

## A.5 Experiment workflow

**Performing Experiments.** We provide shell scripts to generate per-request end-to-end latency, which will be used for plotting figures later. All scripts are located in ./elastic\_switch/scripts\_ae/, please set working directory to ./elastic\_switch/ before running the following scripts. It is not necessary to run all of them before go to next step.

• aws\_e2e.sh will start the end-to-end latency evaluation in §6.2. In the following command, the approach can be one of reparallelization, rerouting, spotserve, and the model\_name should be one of 6.7B, 20B, 30B, while the trace\_name should be one of As, Bs, As+o, Bs+o. Each execution will be corresponding to a single curve in Figure 6:

```
./scripts_ae/aws_e2e.sh <approach> \
          <model_name> <trace_name>
```

• aws\_ondemand.sh will start the monetary cost evaluation in §6.2, generating the dashed blue line in Figure 7. The num\_node can be one of 3, 4, 6, 8, but the dashed line will be plotted only when all of the four experiments have been conducted:

```
./scripts_ae/aws_ondemand.sh <num_node>
```

• aws\_workload.sh will start the fluctuating workload evaluation in §6.3 on specified trace (can be either A or B), where the approach can also be one of reparallelization, rerouting, spotserve:

```
./scripts_ae/aws_workload.sh \
    <approach> <trace_name>
```

• aws\_ablation.sh will start the ablation study evaluation in §6.2 on specified trace (can be either A or B), where the ablation\_level is from 0 to 4, corresponding to the five bars in Figure 9:

```
./scripts_ae/aws_ablation.sh \
     <ablation_level> <trace_name>
```

**Plotting Figures.** All the scripts above only generate the end-to-end latency for each request. To analysis these data and plot figures presented in the paper, we also provide a plot.py together with the scripts:

```
pip install matplotlib seaborn
python ./scripts_ae/plot.py <mode> \
    [-m MODEL] [-t TRACE]
```

This script works even when only part of the experiment has been completed (just ignoring missing results), allowing users to check partial experimental data. Here is the available options for mode:

- e2e Plot the corresponding figure as in Figure 6, both
   -m, -t flags are required to be specified.
- **price** Plot the monetary cost comparison figure as Figure 7, in which the scatters come from aws\_e2e.sh.
- workload-e2e Plot the end-to-end latency figure as Figure 8(e)(f) on the trace specified by -t flag.
- workload-case Plot the per-request latency figure as Figure 8(g)(h) on the trace specified by -t flag.
- **ablation** Plot the ablation study figure as Figure 9.

#### A.6 Evaluation and expected results

The specific results differ on the hardware, bandwidth, and sometimes sensitive to unpredictable GPU/network/batching fluctuations. However, we expect the results users reproduced roughly match the trends as the figures presented in the paper within the same environment. (i.e. Figures 6,7,8,9)

