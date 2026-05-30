# Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference

Haiyang Huang\*† , Newsha Ardalani\*, Anna Sun\*, Liu Ke\*‡, Hsien-Hsin S. Lee\*, Anjali Sridhar\*, Shruti Bhosale\*, Carole-Jean Wu\*, Benjamin Lee\*\$

<sup>∗</sup>Meta AI †Duke University \$University of Pennsylvania ‡Washington University in St. Louis

*Abstract*—Mixture-of-Experts (MoE) models have gained popularity in achieving state-of-the-art performance in a wide range of tasks in computer vision and natural language processing. They effectively expand the model capacity while incurring a minimal increase in computation cost during training. However, deploying such models for inference is difficult due to their large size and complex communication pattern. In this work, we provide a characterization of two MoE workloads, namely Language Modeling (LM) and Machine Translation (MT) and identify their sources of inefficiencies at deployment.

We propose three optimization techniques to mitigate sources of inefficiencies, namely (1) Dynamic gating, (2) Expert Buffering, and (3) Expert load balancing. We show that dynamic gating improves maximum throughput by 6.21-11.23× for LM, 5.75- 10.98× for MT Encoder and 2.58-5.71× for MT Decoder. It also reduces memory usage by up to 1.36× for LM and up to 1.1× for MT. We further propose Expert Buffering, a new caching mechanism that only keeps hot, active experts in GPU memory while buffering the rest in CPU memory. This reduces static memory allocation by up to 1.47×. We finally propose a load balancing methodology that provides additional scalability to the workload.

