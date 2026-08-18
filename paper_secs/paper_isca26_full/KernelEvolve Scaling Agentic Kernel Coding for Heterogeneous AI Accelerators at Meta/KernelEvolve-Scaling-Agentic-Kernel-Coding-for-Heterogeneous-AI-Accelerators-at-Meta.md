# **KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta**

#### **KernelEvolve Team, Meta Platforms**

Making deep learning recommendation model (DLRM) training and inference fast and efficient is important. However, this presents three key system challenges – model architecture diversity, kernel primitive diversity, and hardware generation and architecture heterogeneity. The combination of the three diversity dimensions leads to a complex optimization space.

This paper presents KernelEvolve – an agentic kernel coding framework – to tackle heterogeneity at–scale for DLRM training and inference. KernelEvolve is designed to take kernel specifications as input and automate the process of kernel generation and optimization for recommendation model across heterogeneous hardware architectures through multiple programming abstractions, including Triton, CuTe DSL, and low-level hardware diagnostic languages, spanning the full hardware-software optimization stack. The kernel optimization process is described as graph-based search with selection policy, universal operator, fitness function, and termination rule, dynamically adapts to runtime execution context through retrieval-augmented prompt synthesis. The system integrates a persistent knowledge base encoding hardware-specific constraints for heterogeneous AI accelerators, enabling effective kernel generation even for proprietary architectures absent from LLM training corpora.

We designed, implemented, and deployed KernelEvolve to optimize a wide variety of production recommendation models across generations of NVIDIA and AMD GPUs, as well as Meta's latestgeneration AI accelerators (MTIA v3). We validate KernelEvolve on the publicly-available KernelBench suite, achieving 100% pass rate on all 250 problems across three difficulty levels, and 160 PyTorch ATen operators across three heterogeneous hardware platforms, demonstrating 100% correctness over all 480 operator-platform configurations. KernelEvolve reduces development time from weeks to hours and achieves substantial performance improvements by up to 17 times over PyTorch baselines across diverse production use cases and for heterogeneous AI systems at-scale. Beyond performance efficiency improvements, KernelEvolve significantly mitigates the programmability barrier for new AI hardware by enabling automated kernel generation for proprietary accelerators. We hope the insights and deployment experience presented in this paper will shed new light on the design of AI systems and optimization at-scale.

**Date:** July 8, 2026

**ISCA 2026 Paper:** [https://gangliao.me/assets/pdf/kernelevolve\\_isca26\\_paper.pdf](https://gangliao.me/assets/pdf/kernelevolve_isca26_paper.pdf)

**Correspondence:** Gang Liao: [gangliao@meta.com](mailto:gangliao@meta.com), Carole-Jean Wu: [carolejeanwu@meta.com](mailto:carolejeanwu@meta.com), Gaoxiang Liu:

[gaoxiang@meta.com](mailto:gaoxiang@meta.com\)

