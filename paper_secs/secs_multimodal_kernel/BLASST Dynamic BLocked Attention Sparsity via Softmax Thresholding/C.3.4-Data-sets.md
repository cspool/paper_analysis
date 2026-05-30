# *C.3.4 Data sets*

The core kernel benchmarks sweep across various threshold scale factors, evaluating throughput, memory bandwidth, and execution time on randomly initialized tensors. The artifact also includes closed sm100 binaries used to measure and collect exact sparsity percentages dynamically during execution.

#### C.4 Installation

To install and prepare the artifact, first clone the repository along with its required submodules:

```
git clone git@github.com:cameronshinn/
   blasst-ae-mlsys26.git --recursive
```

Next, initialize the containerized environment. The repository provides a convenience script to automatically launch the required Docker container (falling back to Singularity if Docker is unavailable) and mount the repository to the /workspace directory:

```
./start_docker.sh
cd /workspace
```

## C.5 Experiment workflow

The evaluation workflow is organized by target hardware architecture. For the NVIDIA Hopper architecture, the workflow is further decoupled by attention phase into separate directories. For the NVIDIA Blackwell architecture, both prefill and decode evaluations are consolidated into a single directory. After launching the container and navigating to /workspace, the general workflow proceeds as follows:

- 1. Navigate to the specific subdirectory corresponding to the available architecture and desired evaluation phase (e.g., hopper\_prefill, hopper\_decode, or blackwell).
- 2. Follow the steps in the README.md file of that specific subdirectory to compile the kernels and initiate the automated benchmarks.
- 3. The script will automatically sweep across various threshold scale factors, executing both the BLASST kernels and the dense SOTA baselines.
- 4. Collected measurements for sparsity, execution time, and memory bandwidth, will be logged directly to standard output.

## C.6 Evaluation and expected result

We expect our results to align with what's shown in Table [5.](#page-9-0) Each README.md file in the folders of our repository contain expected outputs of their associated scripts.