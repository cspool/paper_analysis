# V. UNI-STC DATAFLOW

This section details the software-hardware co-design of Uni-STC, focusing on the dataflow from both the software and hardware perspectives.

## *A. Software Dataflow*

Based on the BBC format and the UWMMA instruction set, we design the four sparse kernels at the software level. The implementation of SpMV and SpMSpV is presented in Algorithm 1. During execution, Uni-STC computes the multiplication of two blocks of matrix A and corresponding vectors, accumulating the results in each thread's ry register. Finally, shfl\_gather is used to accumulate the results in the first 16 threads, and then written back to global memory. For SpMM and SpGEMM, detailed in Algorithm 2, the dataflow leverages the first-level CSR structure within the BBC format. This structure facilitates the scheduling of T1 tasks through a rowby-row outer product formulation (Ci<sup>∗</sup>+ = Aik × Bk<sup>∗</sup>).

![](_page_8_Figure_0.jpeg)

Fig. 14: Comparison of DS-STC, RM-STC and our Uni-STC on a downsized  $8(M) \times 8(N) \times 8(K)$  T1 task.

The 'warpRow', 'warpIndex', and 'warpRowId' variables are used in the preceding algorithms to implement a static load-balancing scheme, which configures the data processing range of each warp.

For dense computations, the structural information of dense vector and matrix is stored in GPU memory (a total of 96B). This information is loaded into registers with a single read operation at the start of the computation.

