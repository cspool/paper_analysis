# A Artifact Appendix

#### A.1 Abstract

MuxWise is an LLM serving framework adopting intra-GPU prefill-decode multiplexing, which is built on the top of SGLang[52]. We provide the source code of MuxWise and scripts to reproduce comparison of chunked-prefill. This appendix includes instructions for reproducing similar data in Figure 16 and Figure 17.

#### A.2 Artifact check-list (meta-information)

- Model: CodeLlama-34b-Instruct-hf.
- Data set: ShareGPT [4] and LooGLE [25].
- Hardware:

NVIDIA H200 NVL (140 GB, 132 SMs)

NVIDIA driver: 580.65.06 (must be greater than 570)

- Experiments: This appendix provides instrucitons for comparing 99%-ile TTFT and 99%-ile TBT MuxWise between MuxWise and chunked-prefill under various workload.
- Metrics: 99%-ile TTFT, 99%-ile TBT
- Output: Jsonl files containing metrics from MuxWise and chunked-prefill with different chunk size.
- How much disk space required (approximately)?: Approximately 200GB
- How much time is needed to prepare workflow (approximately)?: About 10 minutes to build from source code
- How much time is needed to complete experiments (approximately)?: About 2 hours for ShareGPT workload and 4 hours for LooGLE workload.
- Publicly available?: Yes.

#### A.3 Description

**A.3.1 How to access.** The source code of MuxWise is available for download on Zenodo: https://zenodo.org/records/18062118. The pre-built Docker image can be found in: https://hub.docker.com/layers/combathhhhhh/pdmux/sglpr\_torch2.6 bench

**A.3.2** Hardware dependencies. Requires an x86-64 Linux host with at least 200 GB of free disk space, and an NVIDIA H200 NVL GPU (140 GB, 132 SMs).

**A.3.3 Software dependencies.** NVIDIA driver: 580.65.06 (must be greater than 570).

**A.3.4 Data sets.** ShareGPT: chatbot tasks, with an average input length of 226 and average output length of 195.

LooGLE: long-context understanding tasks, with an average input length of 30k and average output length of 15.

A.3.5 Models. CodeLlama-34b-Instruct-hf.

#### A.4 Installation

Please follow the instructions below, which are adapted from our GitHub repository (https://github.com/ykcombat/sglang/tree/slo\_config):

 $\ensuremath{\text{\# 1.}}$  Clone the repository and switch to the slo\_config branch

```
git clone https://github.com/ykcombat/sglang.git
cd sglang
git checkout slo_config

# 2. Build SGLang
pip install --upgrade pip
pip install -e "python"
```

