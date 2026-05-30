# **7 Conclusion**

In this paper, we propose Comet, a MoE system that aims to achieve fine-grained communication and computation overlapping for MoE. Comet features two key designs to achieve seamless overlapping without impact the computational efficiency: Shared tensor based dependency resolving that enables fine-grained overlapping, while eliminating the bottleneck caused by fine-grained communication I/O; The workload assignment mechanism that promises precise and adaptive overlapping of operators, inducing maximal latency concealing. Comet achieves 1.96× speedup in a single MoE layer and 1.71× speedup in the end-to-end execution of MoE models, compared with existing literature.

