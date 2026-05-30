# B Appendix / GroupedGEMM Kernel

GroupedGEMM(Grouped General Matrix Multiplication) operation can be viewed as a generalization of the batched APIs that enable different matrix sizes, transpositions, and scaling factors to be grouped and parallelized in one kernel launch.

In the scenario of fine-grained MoE service, the computation for each expert can be small, making the workload of a single GEMM operation less efficient. Grouped GEMM allows multiple smaller matrix multiplications (for different experts) to be processed in parallel, increasing computational efficiency by better utilizing hardware resources. At the same time, the application of GroupedGEMM could reduce the kernel launch overhead and combines these operations into a single kernel launch.

Currently, there are three main implementations of the GroupedGEMM kernel. The first is designed with Triton[\[16\]](#page-5-6), the second with Cutlass[\[4\]](#page-4-11), and most recently, cuBLAS[\[3\]](#page-4-12) introduced a new GroupedGEMM kernel in CUDA version 12.5. However, due to compatibility issues between PyTorch and CUDA versions, we selected the Cutlass implementation for IFMoE.

