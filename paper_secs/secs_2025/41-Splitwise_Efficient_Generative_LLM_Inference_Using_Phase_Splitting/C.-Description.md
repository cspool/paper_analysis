# *C. Description*

How to access. The entire artifact is available as an archive on Zenodo: https://doi.org/10.5281/zenodo.11003049. Individual components are also available online as follows:

- The production traces can be downloaded from the Azure Public Dataset GitHub repository [4].
- The KV-cache transfer prototype can be downloaded from the vLLM GitHub repository, currently available as a pull request [1].

● SplitwiseSim, and the associated experiment and plotting scripts, can be downloaded from a separate GitHub repository [20].

Hardware dependencies. The KV-cache transfer prototype requires two GPU machines connected over Infiniband, such as NVIDIA DGX-A100s or NVIDIA DGX-H100s. SplitwiseSim requires a standard x86-64 CPU machine; multiple machines may be used to parallelize simulation runs.

Software dependencies. The KV-cache transfer prototype is built on top of vLLM [51] and MSCCL++ [11]. SplitwiseSim depends on a small set of publicly available Python packages, which can be installed via the included requirements.txt. Data sets. Coding and conversation traces from Microsoft Azure are available online as a part of the artifact release [4].

