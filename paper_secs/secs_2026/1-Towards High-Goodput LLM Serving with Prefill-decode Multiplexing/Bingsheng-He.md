# Bingsheng He

dcsheb@nus.edu.sg National University of Singapore Singapore

### **Abstract**

Large Language Model (LLM) serving must meet stringent Service Level Objectives (SLOs) for both the prefill and decode phases. Some existing solutions disaggregate the two phases, causing potential resource idleness or compute redundancy. Others split the prefill phase into chunks and fuse it with decode iteration, creating a dilemma between SLO compliance and high utilization. To address these issues, an efficient serving system should dynamically adapt compute allocation, decouple compute from memory management, and execute prefill and decode independently. We present MuxWise, an LLM serving framework that adopts a new paradigm, intra-GPU prefill-decode multiplexing, to meet these requirements. To fully exploit the paradigm, MuxWise integrates a bubble-less multiplex engine, a contention-tolerant estimator, and an SLO-aware dispatcher. Evaluation shows

<sup>&</sup>lt;sup>†</sup>Corresponding author.

![](_page_0_Picture_26.jpeg)

This work is licensed under a Creative Commons Attribution 4.0 International License.

ASPLOS '26, Pittsburgh, PA, USA
© 2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2359-9/2026/03
https://doi.org/10.1145/3779212.3790236

that MuxWise improves peak throughput under SLO guarantees by an average of  $2.20\times$  (up to  $3.06\times$ ) over state-of-the-art baselines

CCS Concepts: • Computer systems organization  $\rightarrow$  Single instruction, multiple data; Cloud computing; • Software and its engineering  $\rightarrow$  Process management.

Keywords: LLM Serving, PD-Multiplexing, Goodput

