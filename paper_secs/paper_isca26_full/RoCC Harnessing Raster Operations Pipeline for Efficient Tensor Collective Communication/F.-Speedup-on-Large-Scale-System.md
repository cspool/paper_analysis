# *F. Speedup on Large-Scale System*

We evaluate scalability with Astra-Sim [49], scaling the number of GPUs to 32, 64, 128, and 256. We use combined tensor and data parallelism, where a varying number of groups of 8 GPUs run on different parts of the data in parallel and the 8 GPUs in each group run tensor parallelism. We test with GPT-3 to ensure sufficient per-GPU workload. We generate a Chakra [51] trace and rewrite the execution graph to enable fine-grained overlap with RoCC. As shown in Figure 31, RoCC achieves 20%, 21%, 13%, and 13% speedups, respectively, demonstrating robust scaling.

