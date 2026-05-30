# Contents

In this artifact, we will reproduce the [Figures 12](#page-9-2)[–23.](#page-13-1) Each figure has a shell script to reproduce and visualize the evaluation results automatically. In addition, we also provide a pre-built Docker image hosted on [Github Container Registry.](https://github.com/Raphael-Hao/brainstorm/pkgs/container/brt) Users can quickly initiate a container with this image, which has preconfigured experimental environments.

### Hosting

The artifact is hosted at [https://github.com/Raphael-](https://github.com/Raphael-Hao/brainstorm/tree/osdi2023ae)[Hao/brainstorm/tree/osdi2023ae.](https://github.com/Raphael-Hao/brainstorm/tree/osdi2023ae) To get the code, please git clone the Brainstorm repository and checkout to the osdi2023ae branch.

### Requirements

- 1. Hardware Requirements: [Figures 13,](#page-9-3) [15–](#page-10-2)[17](#page-11-0) and [21–](#page-12-2) [23](#page-13-1) requires a server with a NVIDIA A100 (80GB) GPU, [Figures 12,](#page-9-2) [14](#page-10-1) and [18–](#page-11-1)[20](#page-12-1) requires a server with eight NVIDIA V100 GPUs.
- 2. Software Requirements: Please use docker to build the docker/Dockerfile.update to setup the environment for single and multiple-GPU experiments. A oneclick script python scripts/docker\_gh\_build.py - -type latest is also provided to build the image.
- 3. CUDA Driver: Larger than 11.3

