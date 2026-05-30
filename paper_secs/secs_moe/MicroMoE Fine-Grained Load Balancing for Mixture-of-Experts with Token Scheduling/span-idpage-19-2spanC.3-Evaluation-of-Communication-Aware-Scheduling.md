# <span id="page-19-2"></span>C.3 Evaluation of Communication-Aware Scheduling

We evaluate the performance of the communication-aware scheduling in Appendix [A.1.](#page-16-6) The communication-aware scheduling considers two levels of locality in token dispatching: GPU-level (intra-node) locality and node-level (internode) locality. We set α<sup>1</sup> = 0.1,α<sup>2</sup> = 1.0 as the weights of intra-node and inter-node communication in Problem [4.](#page-16-5) We use DeepEP as the communication backend due to its superior performance and reduced system overhead compared to NCCL. We believe that the dispatch time of DeepEP provides a more accurate reflection of the communication volume. For other parameters, we use 16 GPUs, 32 experts, hidden\_size=2048, sequence\_length=4, micro\_batch\_size=4. We use randomly generated tokens as input.

We compare the execution time of an MoE layer while enabling/disabling the GPU-level/node-level locality in the communication-aware scheduling. As shown in Figure [17,](#page-19-1) the overall execution time decreases as we consider more levels of locality during scheduling.

