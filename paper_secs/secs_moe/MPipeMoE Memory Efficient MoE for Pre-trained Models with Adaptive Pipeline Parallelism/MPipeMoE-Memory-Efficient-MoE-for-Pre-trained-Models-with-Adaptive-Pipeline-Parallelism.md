# MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism

Zheng Zhang *School of Computer Science WuHan University* zzhang3031@whu.edu.cn

Donglin Yang *Nvidia Corp.* dongliny@nvidia.com

Yaqi Xia *School of Computer Science WuHan University* yaqixia@whu.edu.cn

Liang Ding *JD Explore Academy JD.com Inc.* liangding.liam@gmail.com

Dacheng Tao *JD Explore Academy JD.com Inc.* dacheng.tao@gmail.com

Xiaobo Zhou *University of Macau* waynexzhou@um.edu.mo

Dazhao Cheng *School of Computer Science WuHan University* dcheng@whu.edu.cn

*Abstract*—Recently, Mixture-of-Experts (MoE) has become one of the most popular techniques to scale pre-trained models to extraordinarily large sizes. Dynamic activation of experts allows for conditional computation, increasing the number of parameters of neural networks, which is critical for absorbing the vast amounts of knowledge available in many deep learning areas. However, despite the existing system and algorithm optimizations, there are significant challenges to be tackled when it comes to the inefficiencies of communication and memory consumption.

In this paper, we present the design and implementation of MPipeMoE, a high-performance library that accelerates MoE training with adaptive and memory-efficient pipeline parallelism. Inspired by that the MoE training procedure can be divided into multiple independent sub-stages, we design adaptive pipeline parallelism with an online algorithm to configure the granularity of the pipelining. Further, we analyze the memory footprint breakdown of MoE training and identify that activations and temporary buffers are the primary contributors to the overall memory footprint. Toward memory efficiency, we propose memory reusing strategies to reduce memory requirements by eliminating memory redundancies, and develop an adaptive selection component to determine the optimal strategy that considers both hardware capacities and model characteristics at runtime. We implement MPipeMoE upon PyTorch and evaluate it with common MoE models in a physical cluster consisting of 8 NVIDIA DGX A100 servers. Compared with the state-of-art approach, MPipeMoE achieves up to 2.8× speedup and reduces memory footprint by up to 47% in training large models.

*Index Terms*—Mixture of Experts, Pipeline Parallelism, Distributed Training, Memory Efficiency

