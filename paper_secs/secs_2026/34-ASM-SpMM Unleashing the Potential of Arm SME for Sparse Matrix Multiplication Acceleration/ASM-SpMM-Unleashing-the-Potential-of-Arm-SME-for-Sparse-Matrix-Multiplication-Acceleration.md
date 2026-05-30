# ASM-SpMM: Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration

## Jiazhi Jiang

Sun Yat-sen University Guangzhou, China jiangjzh6@mail2.sysu.edu.cn

#### Jinhui Wei

Sun Yat-sen University Guangzhou, China weijh28@mail2.sysu.edu.cn

#### Xijia Yao

Sun Yat-sen University Guangzhou, China yaoxj9@mail2.sysu.edu.cn

## Dan Huang\*

Sun Yat-sen University Guangzhou, China huangd79@mail.sysu.edu.cn

#### Jiayu Chen

Sun Yat-sen University Guangzhou, China chenjy666@mail2.sysu.edu.cn

#### Yutong Lu\*

Sun Yat-sen University Guangzhou, China luyutong@mail.sysu.edu.cn

### **Abstract**

Sparse Matrix—Matrix Multiplication (SpMM) is a core kernel in scientific computing, data analytics, and artificial intelligence, supporting applications such as linear solvers and Graph Neural Networks (GNNs). The Scalable Matrix Extension (SME) in Armv9 introduces dedicated matrix acceleration for ARM CPUs, but exploiting its full potential for SpMM requires architecture-aware optimizations to address irregular sparsity and hardware constraints.

We present ASM-SpMM, a high-performance SpMM library co-designed with ARM SME. ASM-SpMM combines a memory-efficient compression format, an SME-aware kernel optimized for outer-product execution, a hybrid matrix-vector execution strategy, and work-stealing-based dynamic load balancing across heterogeneous cores. Experiments on emerging Armv9 platforms demonstrate up to 7.9× speedup over state-of-the-art SpMM libraries across diverse matrices. A GNN inference case study further shows that ASM-SpMM significantly improves end-to-end performance, highlighting the effectiveness of SME-aware SpMM optimization on ARM CPUs.

CCS Concepts: • Computer systems organization  $\rightarrow$  Parallel architectures.

Keywords: SpMM, ARM SME, SVE, Outer Product

#### **ACM Reference Format:**

Jiazhi Jiang, Xijia Yao, Jiayu Chen, Jinhui Wei, Dan Huang, and Yutong Lu. 2026. ASM-SpMM: Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration. In *Proceedings of* 

<sup>\*</sup>Corresponding author

![](_page_0_Picture_21.jpeg)

This work is licensed under a Creative Commons Attribution 4.0 International License.

PPoPP '26, Sydney, NSW, Australia
© 2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2310-0/2026/01
https://doi.org/10.1145/3774934.3786422

the 31st ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming (PPoPP '26), January 31 – February 4, 2026, Sydney, NSW, Australia. ACM, New York, NY, USA, 13 pages. https://doi.org/10.1145/3774934.3786422

#### 1 Introduction

General Sparse Matrix-Matrix Multiplication(SpMM) serves as a critical and computationally intensive kernel underpinning a broad spectrum of application domains, including scientific computing, data analytics, and modern artificial intelligence [3, 5, 11]. Its pervasive role extends from the core of linear algebra solvers to large-scale graph analytics and Graph Neural Networks (GNNs) [4, 16]. As such, advancements in SpMM optimization are poised to deliver substantial improvements across diverse applications.

The ARM architecture, long dominant in mobile computing, is rapidly expanding into desktop and high-performance systems [9, 10]. To meet the growing demands of AI workloads, vendors have introduced dedicated matrix acceleration units such as NVIDIA Tensor Cores [14] and Intel AMX [8], delivering substantial performance gains for matrix operations[29]. Following this trend, ARM integrated the Scalable Matrix Extension (SME) into Armv9, offering specialized hardware support for matrix multiplication. Distinct from other matrix units that implement GEMM with inner products, SME adopts the outer product as its building block. This operation takes two input vectors and generates a matrix via elementwise multiplication (Figure 1). With ARM's growing role in performance-critical domains, optimizing SpMM on SME is timely and essential to fully unleash its potential for scientific and AI applications.

<span id="page-0-0"></span>
$$\begin{split} \mathbf{C} &= \mathbf{A} \times \mathbf{B} = \begin{bmatrix} a_{00} & a_{01} \\ a_{10} & a_{11} \end{bmatrix} \begin{bmatrix} b_{00} & b_{01} \\ b_{10} & b_{11} \end{bmatrix} = \begin{bmatrix} a_{00} \\ a_{10} \end{bmatrix} \begin{bmatrix} b_{00} & b_{01} \end{bmatrix} + \begin{bmatrix} a_{01} \\ a_{11} \end{bmatrix} \begin{bmatrix} b_{10} & b_{11} \end{bmatrix} \\ &= \begin{bmatrix} a_{00}b_{00} & a_{00}b_{01} \\ a_{10}b_{00} & a_{10}b_{01} \end{bmatrix} + \begin{bmatrix} a_{01}b_{10} & a_{01}b_{11} \\ a_{11}b_{10} & a_{11}b_{11} \end{bmatrix} = \begin{bmatrix} a_{00}b_{00} + a_{01}b_{10} & a_{00}b_{01} + a_{01}b_{11} \\ a_{10}b_{00} + a_{11}b_{10} & a_{10}b_{01} + a_{11}b_{11} \end{bmatrix} \end{split}$$

Figure 1. Outer-product execution of Matrix Multiplication.

However, fully unleashing SME's potential for SpMM acceleration presents unique challenges. Designed primarily

for dense matrix operations, SME's dense-oriented architecture does not inherently align with irregular structures encountered in sparse matrix computations. Integrating SME with SpMM thus demands careful consideration and innovative solutions. Previous research on GPU-based Tensor Core acceleration for sparse deep learning workloads provides some insights[1, 15, 18, 22, 23, 28]. Nevertheless, ARM SME diverges significantly from other matrix multiplication accelerators (e.g., GPU Tensor Cores) by adopting vector outer-product instructions and a dedicated Z Array (ZA) register, which serves as a large matrix accumulator for highthroughput operations. However, SME currently exposes only low-level primitives without compiler or programming model support, thereby shifting the complexity of kernel construction to the software stack and requiring novel algorithmic and system-level designs.

In this work, we target SpMM optimization for GNN and scientific computing workloads on ARM CPUs with SME. Our analysis reveals four key challenges: inefficient sparse storage, the mismatch between sparsity and SME's dense-oriented outer-product primitives, limited coordination between matrix and vector units, and workload imbalance across heterogeneous cores. To address these issues, we propose ASM-SpMM, which integrates SME-aware kernels, a new sparse compression format, and dynamic scheduling mechanisms tailored to ARM SME.

To the best of our knowledge, ASM-SpMM represents the first effort to accelerate sparse matrix multiplication with ARM SME. We perform comprehensive evaluations of ASM-SpMM using large-scale power-law graph matrices representative of GNN workloads, as well as a wide spectrum of sparse matrices from the SuiteSparse Matrix Collection. Our experimental study includes rigorous comparisons against state-ofthe-art SpMM kernels, including those from the Armadillo library, Eigen library, the ARM Performance library (ArmPL), Cholmod, and MP-SpMM[30] on the Apple M4 ARM processor, and the newly released LX2 ARM processor. The results demonstrate that ASM-SpMM consistently achieves substantial average speedups over these baselines across diverse, real-world sparse matrix benchmarks, highlighting its effectiveness and the potential of architecture-aware optimization for ARM SME. A case study illustrates that ASM-SpMM significantly accelerates end-to-end GNN inference compared to DGL[20] and PyG[2], two widely used GNN frameworks. We summarize our contributions as follows:

- We introduce a novel, memory-efficient sparse matrix compression format tailored to ARM SME's outer-product execution model, enabling efficient use of matrix primitives.
- We implement a highly optimized SME-aware SpMM kernel with specialized outer-product designs that coordinate prefetching, multi-tile parallelism, and pipelined execution to improve register utilization and fully exploit SME's instruction-level parallelism.

- Leveraging instruction-level scheduling, we propose sophisticated hybrid kernel strategies that coordinate SME with vector units on ARM CPUs, thereby unlocking computation performance across heterogeneous resources.
- We propose a dynamic inter-thread work-stealing scheme to achieve load-balanced execution, enabling sparse-aware workload distribution across heterogeneous cores and further improving SpMM scalability on multi-core ARM CPU.

