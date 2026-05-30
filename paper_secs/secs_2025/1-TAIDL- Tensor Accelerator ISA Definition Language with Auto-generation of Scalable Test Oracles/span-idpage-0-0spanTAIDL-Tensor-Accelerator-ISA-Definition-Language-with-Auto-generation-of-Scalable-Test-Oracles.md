# <span id="page-0-0"></span>TAIDL: Tensor Accelerator ISA Definition Language with Auto-generation of Scalable Test Oracles

Devansh Jain University of Illinois Urbana-Champaign, USA devansh9@illinois.edu

Akash Pardeshi University of Illinois Urbana-Champaign, USA pardesh2@illinois.edu Marco Frigo University of Illinois Urbana-Champaign, USA mfrigo3@illinois.edu

Zhihao Wang University of Illinois Urbana-Champaign, USA zhihaow6@illinois.edu

Charith Mendis University of Illinois Urbana-Champaign, USA charithm@illinois.edu Jai Arora University of Illinois Urbana-Champaign, USA jaia3@illinois.edu

Krut Patel University of Illinois Urbana-Champaign, USA ksp8@illinois.edu

#### Abstract

With the increasing importance of deep learning workloads, many hardware accelerators have been proposed in both academia and industry. However, software tooling for the vast majority of them does not exist compared to the software ecosystem and innovations proposed for established platforms such as CPUs and GPUs. We observed that the lack of well-defined hardware-software interfaces and correctness testing tools like fast and scalable *test oracles* (also known as functional simulators) act as significant barriers to adopting these emerging accelerators in the software community. These interfaces and tools are essential in building software such as retargetable compilers and optimized kernels.

To bridge these gaps, we first present TAIDL, an instruction specification language that provides novel constructs to describe the instruction set architectures (ISAs) of tensor accelerators. Next, given ISA definitions in TAIDL, we introduce techniques to automatically generate fast and scalable test oracles for diverse sets of accelerators, which are needed for testing software correctness of code that targets pre-silicon hardware designs. Automated generation of such tools reduces the burden on hardware architects and the repeated development efforts required across different accelerator platforms. Further, our techniques allow us to execute these simulators on GPUs, leading to highly scalable simulations. To demonstrate the expressivity of TAIDL, we instantiated several tensor accelerator ISAs with different compute capabilities and memory hierarchies. Further, we show that test oracles generated using TAIDL definitions are orders of magnitude faster and more scalable than existing instruction-level functional simulators, making them suitable for integration into software development cycles. TAIDL is available at https://github.com/act-compiler/taidl.

![](_page_0_Picture_12.jpeg)

This work is licensed under a Creative Commons Attribution 4.0 International License. MICRO '25, Seoul, Republic of Korea © 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-1573-0/25/10 https://doi.org/10.1145/3725843.3756075

