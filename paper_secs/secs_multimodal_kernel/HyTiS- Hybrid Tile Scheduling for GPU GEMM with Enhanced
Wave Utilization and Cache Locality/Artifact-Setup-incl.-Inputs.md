# Artifact Setup (incl. Inputs)

Hardware. We conducted experiments on two servers: one equipped with an NVIDIA H100-PCIE GPU with 80 GB of memory, and the other with an NVIDIA A100-PCIE GPU with 40 GB of memory.

Software. The primary software libraries used by HyTiS and the baseline implementations are PyTorch 2.3, CUDA 12.6, CUTLASS 3.4, and Triton 3.2.0.

Datasets / Inputs. <sup>1</sup> and 1.<sup>3</sup> require no data downloads, as all experiments are conducted using randomly generated data at runtime.

Installation and Deployment. First, unpack the artifact from Zenodo, then execute the script install.sh

