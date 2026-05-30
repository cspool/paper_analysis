# Abstract

Mixture-of-Experts (MoE) has emerged as a promising approach to scale up deep learning models due to its significant reduction in computational resources. However, the dynamic nature of MoE leads to load imbalance among experts, severely impacting training efficiency. While previous research has attempted to address the load balancing challenge, existing solutions either compromise model accuracy or introduce additional system overhead. As a result, they fail to achieve efficient and fine-grained load balancing, which is crucial to optimizing training efficiency.

We propose FineEP, a novel parallelization strategy to achieve fine-grained load balancing in MoE systems. FineEP is capable of achieving complete load balancing in every micro-batch through efficient token scheduling across GPUs. Furthermore, we propose FineMoE, an efficient distributed MoE training system with FineEP's load balancing capabilities. Our experimental results demonstrate that FineMoE improves the end-to-end training throughput by up to 47.6% compared with the state-of-the-art system, and almost consistently achieves complete load balance among GPUs.

