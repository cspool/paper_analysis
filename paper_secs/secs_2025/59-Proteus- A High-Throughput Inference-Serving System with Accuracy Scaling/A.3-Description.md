# A.3 Description

**A.3.1 How to access.** The simulator code and workload traces can be accessed at https://github.com/UMass-LIDS/Proteus. It is also accessible at the following DOI: https://doi.org/10.5281/zenodo.10428550.

#### **A.3.2 Hardware dependencies.** None for the simulator.

- **A.3.3 Software dependencies.** We provide two installation methods. The first method only requires Docker, while the second method requires the following:
  - Gurobi optimization software
  - A Gurobi license
  - conda
  - A local installation of Python

### **A.3.4 Data sets.** The artifact has two types of datasets:

- 1. A *real-world* Twitter workload dataset to test end-to-end system performance.
- 2. Several *synthetic* workload datasets to test performance of isolated components as described in the paper
- **A.3.5 Models.** A mix of computer vision models for image classification and object detection is used, as well as natural

language models for text translation, sentiment analysis, and question answering. The complete list of all the DNN models and their variants used is given in Table 3 in the paper.

#### A.4 Installation

We provide two methods of installation:

- Docker: In order to quickly evaluate Proteus, we provide a Docker container with usage instructions at https://github.com/UMass-LIDS/Proteus/blob/main/DOCKER.md. We recommend this method for artifact evaluation and quick testing as it does not require obtaining a Gurobi license. For any other use cases, please use the second method described below.
- 2. **Local installation:** For extensive evaluation and general simulator usage, we recommend locally installing Proteus and obtaining a Gurobi license. The instructions are provided at https://github.com/UMass-LIDS/Proteus/blob/main/README.md.

## A.5 Experiment workflow

The simulator requires a JSON configuration file as input to set up the experiment. We have provided several example configuration files in the config folder of our GitHub repo. A configuration file describes the workload trace to use, the resource allocation algorithm, the adaptive batching algorithm, as well as any hyper-parameters (e.g., a  $\beta$  value of 1.05 for Proteus is used by default).

Depending on the installation method used, the experiments can be run using either of the following set of instructions:

- Docker: Follow instructions at https://github.com/ UMass-LIDS/Proteus/blob/main/DOCKER.md.
- Local installation: Follow instructions at https:// github.com/UMass-LIDS/Proteus/blob/main/EXAMPLES. md.

#### A.6 Evaluation and expected results

The simulator produces log files in the logs folder. These log files contain snapshots of the system at regular intervals containing not only aggregated information about user demand, system capacity, requests served/dropped/late, and accuracy seen by the requests, but they also contain detailed logs for all system events in logs/per\_predictor.

These log files are then ingested by the plotting scripts provided in the plotting folder to generate two graphs: an end-to-end evaluation of Proteus against the baselines on the Twitter trace, similar to the one in Section 6.3, as well as an evaluation of the responsiveness of Proteus vs. the baselines on a bursty trace, as in Section 6.4.

