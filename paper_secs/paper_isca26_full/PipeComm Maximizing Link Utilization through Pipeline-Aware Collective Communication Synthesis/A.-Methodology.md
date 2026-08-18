# *A. Methodology*

Experiment Setup: To evaluate the effectiveness of PipeComm, we evaluate using both simulation and measurement on real machine. For simulation, we utilize ASTRAsim [64], [65], a distributed machine learning simulator that supports various network topologies. ASTRA-sim's congestion-aware analytical backend models message transfers at link granularity, simulating send and receive operations in a first-come, first-served manner. We extend ASTRA-sim to implement and test different communication algorithms. To validate PipeComm 's effectiveness in real-world settings, we also conduct measurements on a two-node multi-GPU system.

We evaluate two versions of PipeComm: Pipe-Sol, which uses Gurobi [16] as the MILP solver for optimal synthesis, and Pipe-Ict, which adopts an incremental strategy. Our evaluation covers multiple perspectives. First, we assess the performance gains of the synthesized AllReduce algorithms compared to multiple baseline approaches. Second, we evaluate performance on the AllGather and AlltoAll operation to further validate PipeComm 's effectiveness across different collective operations. Third, we perform an end-to-end evaluation of training performance across various machine learning workloads to demonstrate the practical benefits. Fourth, we conduct

TABLE III NETWORK TOPOLOGY CONFIGURATIONS. FOR HETEROGENEOUS SETUPS, LINKS ALONG DIFFERENT DIMENSIONS HAVE DISTINCT LATENCY (α) AND

BANDWIDTH (1/β).

Topology Scale Dim-1 Config Dim-2 Config *Homogeneous Evaluation* Hypercube3D 5 × 5 × 5 0.2 µs, 50 GB/s Mesh2D 8 × 8 0.2 µs, 50 GB/s *Comparison with Baseline (MultiTree)* Mesh2D 8 × 8 0.15 µs, 16 GB/s Torus2D 8 × 8 0.15 µs, 16 GB/s *Heterogeneous Evaluation* Mesh2D 8 × 8 0.2 µs, 50 GB/s 0.15 µs, 100 GB/s Switch2D 8 × 8 0.2 µs, 50 GB/s 0.05 µs, 200 GB/s

a case study to analyze the scalability of PipeComm, highlighting the efficiency of the autonomous incremental strategy. Finally, we report results on real GPU platforms to validate performance under real-world deployment.

The baselines we compare fall into two categories: heuristic approaches (TACOS [63], Themis [46], BlueConnect [10], MultiTree [20]) and solver-based approaches (TACCL [51], TE-CCL [31]). Due to their limited scalability, the solverbased methods are evaluated only on small-scale topologies.

