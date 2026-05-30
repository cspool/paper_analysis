# *A.1 Abstract*

This artifact packages the code, traces, scripts, and plotting utilities for reproducing the paper's main results across the two case studies.

Case Study 1 is a CPU-runnable wafer-scale GPU simulator for MoE inference. It evaluates our expert allocation and prediction strategies across four large-scale MoE models and two chiplet topologies, reproducing Figure [12.](#page-9-2)

Case Study 2 contains the real-GPU expert placement experiments and reproduces Figure [17](#page-12-0) on an 8×H100 system. It requires a specialized GPU software stack. Both artifacts provide a main\_ae.py workflow for downloading traces, running experiments, and generating figures.

