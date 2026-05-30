# I. INTRODUCTION

Sparse-Dense Matrix Multiplication (SpMM) is a critical kernel in sparse operations. It plays an important role in many areas such as machine learning, data analytics, and high-performance computing. With the rise of deep learning, the importance of SpMM has become more pronounced. As a fundamental component in deep learning [1]–[3] and graph neural networks [4]–[7], the performance of SpMM directly impacts that of these applications. Therefore, optimizing SpMM across various platforms [8], [9], particularly in Graphics Processing Unit (GPU) [10], has become an important research topic.

GPUs, empowered by their data-parallel processing capabilities, are utilized across diverse application domains [11], [12]. Their exceptional throughput underscores the importance of efficient memory data loading [13]. To achieve high throughput, GPU are equipped with coalesced units that detect continuous

memory access patterns, reducing the number of memory transactions.

Optimizing SpMM on GPUs presents two main challenges: how to efficiently handle sparse matrices and dense matrices respectively? For sparse matrices, the primary bottleneck arises from their inherent characteristics of sparsity and irregular distribution of non-zero elements. To mitigate these issues, various storage formats have been developed to reduce storage requirements and enhance implementation efficiency. The most widely used formats include Compressed Sparse Row (CSR), Compressed Sparse Column (CSC), Coordinate (COO), Diagonal (DIA), and Ellpack (ELL) [14]. Moreover, some simple optimization techniques like tiling and reordering apply to the basic formats [15]. For dense matrices, the most common storage formats include row-major and column-major. In addition to storage format, most research focuses on improving data reuse via techniques like blocking, packing, and other microkernel optimizations [16]. However, the fundamental formats for sparse/dense matrix and simple optimization techniques cannot handle the load imbalance of SpMM among threads on GPUs. The load imbalance is triggered by the irregular distribution of non-zero elements of the sparse matrix.

SOTA methods offer various solutions for accelerating SpMM via improved load balance. RoDe [10] divides the rows of the sparse matrix into a block part (fixed length) and a residual part based on their length. Additionally, RoDe optimizes the sub-block pipeline to enhance performance. ASpT [17] employs an adaptive tiling technique to improve performance. Sputnik [18] leverages vector memory instructions and load balancing techniques to accelerate SpMM. However, these methods, leveraging various formats and optimization techniques, cannot effectively support the coalesced memory access of both sparse and dense matrices simultaneously, as elaborated with an analysis in Section III.

Despite the relevance of memory access coalescence for GPU performance, a reduced amount of work has exploit it to accelerate SpMM on GPU devices. The most relevant previous proposal, GE-SpMM [19], leverages shared memory using the CSR format. It enables coalesced memory access for sparse matrices from device memory to shared memory and dense matrices from device memory to threads in two steps instead of loading them simultaneously. However, since the storage capacity of shared memory is relatively small, the performance benefits of memory access coalescence at the shared memory

<sup>\*</sup>Corresponding author (luohuizhang@hnu.edu.cn)

<sup>&</sup>lt;sup>1</sup>Swift is available at https://github.com/MinttHu/Swift.git

level diminish as the size of the sparse and dense matrices increases.

To mitigate these limitations, this paper introduces Swift, a novel approach to enhance data loading efficiency by achieving highly coalesced memory access for both the sparse and dense matrices of SpMM between global memory and threads on the GPU. Swift uses a coordinated sparsity-based sorting of the columns of the sparse matrix and their corresponding dense matrix columns, combined with a blocking strategy detailed in Section IV. After sparsity-based sorting, the address gap between elements of the dense matrix array accessed by neighborhood threads within a warp is eliminated. Consequently, it can ensure highly coalesced memory access to both the sparse and dense matrices. On top of sparsity-based sorting, Swift uses a blocking strategy, which generates regular (fix-length) and irregular (residual, variable-sized) blocks, to balance the load among threads. The regular blocks are efficiently handled by the GPU warps without incurring load imbalance. For the irregular blocks, Swift uses a batching strategy to gather and allocate them to GPU warps while maintaining load balance.

We evaluate the performance of Swift when running the SpMM product C=AB, where A is a  $M\times K$  sparse matrix and B is a  $K\times N$  dense matrix. We consider the complete SuiteSparse matrix collection [20], both single- and double-precisions, and two different N of values of B. Our experimental results indicate the significant performance improvements offered by Swift over SOTA methods (ASpT [17], cuSPARSE [21], RoDe [10], and Sputnik [18]) across the entire SuiteSparse Matrix Collection on RTX 4080s, 3090Ti, A100, and V100. For example, when N=128, Swift achieves an average speedup of  $1.79\times$ ,  $27.02\times$ ,  $3.62\times$ , and  $6.53\times$ , respectively on RTX 4080s with double-point precision.

This paper makes the following contributions:

- It demonstrates the performance benefits of exploiting coalesced memory access for SpMM on GPUs.
- It introduces Swift, a novel approach that achieves coalesced memory access for both sparse and dense matrices, ensures load balance across threads, and supports efficient segmented summation for SpMM on GPUs.
- It evaluates the performance of Swift considering the complete SuiteSparse matrix collection, and we demonstrate that Swift delivers significant performance speedups with respect to state-of-the-art approaches like ASpT [17], cuSPARSE [20], RoDe [10], and Sputnik [18].

The rest of the paper is organized as follows: Section II presents background information directly related to this work. Section III presents the motivation for the Swift research. Section IV presents the Swift algorithm with its process and implementation details. Section V presents the results of the performance evaluation. Section VI presents the generally related work on SpMM. Section VII concludes this work and discusses future work.

![](_page_1_Figure_8.jpeg)

Fig. 1: The illustration of running SpMM on GPUs.

#### II. BACKGROUND

#### A. Sparse-Dense Matrix Multiplication

The importance of the SpMM kernel for a variety of scientific and engineering domains has motivated the development of methods to efficiently run SpMM on GPU devices [18], [22], [23]. Figure 1 illustrates the workflow of executing SpMM on a modern GPU. The process can be divided into four steps. Step 1: Transfer the data of the sparse and dense matrices from the host memory to the device memory. Step 2: Load the data from the global memory to threads. Step 3: Multiply the corresponding elements and add them to the corresponding result matrix. Step 4: Store the result back to the global memory. Algorithm 1 shows an SpMM kernel based on the CSC format. In this algorithm, individual columns of the sparse matrix are processed in parallel (Lines 1-10). Within each column, a for-loop (Lines 3-8) computes the corresponding elements in the dense matrix. The results are then stored back into the result matrix using the appropriate indices.

