# Abstract

We introduce Stream-K, a work-centric parallelization of matrix multiplication (GEMM) and related computations in dense linear algebra. Whereas contemporary decompositions are primarily tile-based, our method operates by partitioning an even share of the aggregate inner loop iterations among physical processing elements. This provides a near-perfect utilization of computing resources, regardless of how efficiently the output tiling for any given problem quantizes across the underlying processing elements.

On GPU processors, our Stream-K parallelization of GEMM produces a peak speedup of up to 14× and 6.7×, and an average performance response that is both higher and more consistent across 32,824 GEMM problem geometries than state-of-the-art math libraries such as CUTLASS and cuBLAS. Furthermore, we achieve this performance from a single tile size configuration per floating-point precision, whereas today's math libraries employ complex kernel-selection heuristics to select from a large ensemble of kernel variants.

Keywords: Matrix-Multiplication, GPU, Load-Balancing

