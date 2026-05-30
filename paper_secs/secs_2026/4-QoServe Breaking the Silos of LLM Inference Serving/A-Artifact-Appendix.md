# **A** Artifact Appendix

#### A.1 Abstract

QoServe is a QoS-driven LLM inference serving framework that enables efficient co-scheduling of requests across multiple QoS classes on shared infrastructure. This artifact contains the source code, datasets, and scripts to reproduce key results from our paper (Figures 7, 8, 10, 11). QoServe is built on Sarathi-Serve [4], extending vLLM [1] with deadline-aware scheduling capabilities.

#### A.2 Artifact check-list (meta-information)

- Algorithm: Hybrid prioritization, dynamic chunking, eager relegation for multi-SLO LLM serving
- Model: Llama3-8B [16], Llama3-70B [16], Qwen-7B [7]
- Data set: ShareGPT [19], Azure Conversation traces [15], Azure Code traces [15]
- Run-time environment: Ubuntu 20.04+, CUDA 12.1, Python 3.10, PyTorch 2.3.0
- Hardware: 4×A100 80GB GPUs (preferred for Llama3-8B and Qwen-7B experiments). Minimum: 2×A100 with pairwise NVLink for Qwen-7B (TP2). For Llama3-70B: 4×H100 80GB GPUs with NVLink (TP4). Tiny scripts available for single A100 GPU.
- Metrics: TTFT, TBT, TTLT, goodput (QPS), deadline violations (%)
- Experiments: Maximum goodput per replica, latency and deadline violations under varying load
- How much disk space required?: ~150 GB
- How much time to prepare workflow?: 20-60 minutes
- How much time to complete experiments?: ~61 hours on 4×A100 (Figure 7: 13h, Figure 8: 8h, Figures 10–11: 40h)
- Publicly available?: Yes, on GitHub and Zenodo.
- Archived (DOI)?: Yes, DOI: 10.5281/zenodo.18218177

