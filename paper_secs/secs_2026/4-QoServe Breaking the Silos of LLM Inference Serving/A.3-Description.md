# A.3 Description

**A.3.1 How to access.** The artifact is publicly available on GitHub and archived on Zenodo with DOI: 10.5281/zenodo.18218177.

A.3.2 Hardware dependencies. The evaluation scripts assume a 4×A100 80GB node with NVLink. Qwen-7B requires minimum 2×A100 with pairwise NVLink (TP2). Llama3-70B requires 4×H100 80GB GPUs with NVLink (TP4). For resource-constrained environments, tiny scripts are provided that run on a single A100 GPU with reduced execution times while preserving core trends.

**A.3.3 Software dependencies.** Python 3.10, PyTorch 2.3.0, CUDA 12.1, FlashInfer 0.1.1. See README for complete dependency list and installation instructions.

<span id="page-15-0"></span>A.3.4 Data sets. ShareGPT [\[19\]](#page-14-4) and Azure production traces (Code, Conversation) [\[15\]](#page-14-5). Scripts automatically download datasets from public Azure blob storage.

A.3.5 Models. Llama3-8B, Qwen-7B, Llama3-70B accessed via Hugging Face (requires HF token with model access).

#### A.4 Installation

Refer to README for detailed installation instructions. Summary: create conda environment, install dependencies via pip, download datasets, export HF\_TOKEN, and configure GPU clocks for reproducibility.

## A.5 Experiment workflow

The artifact provides automated bash scripts for each figure:

- tester.sh: Quick validation run (∼5 minutes)
- fig7.sh: Goodput with PD colocation (13 hours, Llama3- 8B TP1 only)
- fig7\_tiny.sh: Goodput with PD colocation (8 hours, single A100)
- fig8.sh: Goodput with PD disaggregation (8 hours, Llama3-8B TP1 only)
- fig10\_11.sh: Latency and violations under load (40 hours)
- fig10\_11\_tiny.sh: Latency and violations under load (11 hours, single A100)

Results are saved to benchmark\_output/ (raw logs) and paper\_plots/ (plots and graphs).

#### A.6 Evaluation and expected results

Figure 7 (Goodput, PD colocation): QoServe achieves 1.5–2.4× higher goodput than Sarathi-FCFS and 20–40% over Sarathi-EDF. Current artifact reproduces Llama3-8B TP1 on Azure Code Trace.

Figure 8 (Goodput, PD disaggregation): QoServe shows consistent goodput improvements across disaggregated serving. Current artifact reproduces Llama3-8B TP1 on Azure Conv. Trace.

Figures 10–11 (Latency and violations): QoServe is capable of handling significantly higher load while meeting tail latency SLOs and reduces deadline violations by an order of magnitude under overload. Note: artifact uses fewer requests and coarser QPS sweep than paper for manageable runtime; trends and relative performance remain consistent. The sweep is done on Llama3-8B TP1 for the Azure Code Trace.

