# A Appendix / Memory Efficiency through Parallelism Redesign

We calculate the memory usage and the memory savings achieved using IFMoE's parallelism mechanism under the condition of bfloat16 precision.

<span id="page-5-5"></span>Table 2: Memory Usage and Memory Optimization with IFMoE. #Expert and #Machine represents the number of global experts in a single layer and the number of parallel machine during inference. M(Attention) and M(Experts) represents the memory usage of attention parameters and expert parameters in a single layer. M(Optimization) represents the memory savings with IFMoE on a single machine.

| Model          | #Expert | #Machine | M(Attention) | M(Experts) | M(Optimization) |
|----------------|---------|----------|--------------|------------|-----------------|
| Deepseek-Lite  | 64      | 2        | 28MB         | 1.1GB      | 4.6GB           |
| Qwen2-57B-A14B | 64      | 4        | 66MB         | 3.5GB      | 10GB            |
| Deepseek-v2    | 160     | 8        | 360MB        | 7.4GB      | 23GB            |

Table [2](#page-5-5) presents the basic memory usage and optimization across various fine-grained expert architectures. Although the memory consumption of attention mechanisms is significantly smaller compared to expert parameters, the replication of shared memory accounts for a substantial portion. By leveraging tensor parallelism on these parameters, the optimized memory usage is remarkable, allowing the freed memory to be reallocated for computation and KV-cache storage, thereby enabling larger batch sizes and longer context generation.

