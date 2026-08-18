# I. INTRODUCTION

Fully Homomorphic Encryption over the Torus (TFHE) enables computation directly on encrypted data, eliminating the need for decryption at any stage of the computation.

<sup>\*</sup>Both authors contributed equally to this research.

<sup>†</sup>Corresponding authors.

This strong security guarantee, however, comes at the cost of substantial computational overhead compared with operations on plaintext. Prior work has addressed this challenge by accelerating TFHE with ASIC-based accelerators [6, 27, 28], achieving impressive performance improvements. Nevertheless, designing and fabricating large-scale ASICs is extremely costly and time-consuming, which limits their practicality and widespread adoption in rapidly evolving application scenarios.

Modern Graphics Processing Units (GPUs), with thousands of parallel computing cores and high memory bandwidth, have demonstrated significant performance advantages for highly parallel workloads such as neural networks [16]. In addition to conventional CUDA Cores, modern GPUs incorporate specialized Tensor Cores designed to accelerate dense matrix and tensor operations using mixed-precision arithmetic (e.g., FP64, FP16, INT8). By offering extremely high throughput for matrix-multiply-accumulate (MMA) operations and exposing these capabilities through warp-level primitives, Tensor Cores further enhance the GPU's suitability for workloads that exhibit large-scale vectorization, fine-grained parallelism, and regular memory access patterns. Motivated by these architectural strengths, prior research has leveraged GPUs to accelerate CKKS-based FHE schemes, achieving substantial speedups [8, 10, 12, 38].

In contrast, TFHE-based schemes are particularly well-suited for general-purpose computation over arbitrary-precision or unbounded integers, without being restricted to fixed-point representations. This makes TFHE a natural fit for control-intensive, bit-level, and logic-heavy workloads, such as privacy-preserving quantized neural networks that require exact, rather than approximate, computation. However, these characteristics also mean that TFHE often incurs significantly higher computational and memory demands, which in turn makes GPUs a particularly suitable platform for accelerating TFHE applications.

In this paper, we systematically analyze ZAMA's existing GPU implementation of TFHE-based applications. Our analysis uncovers that the current design does not fully exploit batching opportunities inherent in realistic TFHE workloads and pays insufficient attention to data reuse, both of which limit the achievable performance limits. Furthermore, to the best of our knowledge, this is the first work to investigate mapping high-precision TFHE-related FFT operations onto Tensor Cores and to critically examine the limitations of directly adopting Tensor Core mapping strategies developed for CKKS-based schemes [8, 10] in the TFHE context.

To overcome these limitations, we propose memory-aware algorithmic optimizations for TFHE computations that significantly improve data reuse and reduce memory pipeline stalls by decreasing expensive accesses to L2 cache and global memory. We further design and implement a Tensor Core-based FFT kernel tailored to TFHE, together with an optimized memory access pattern that mitigates the overhead of shared-memory accesses and better exploits warp-level parallelism. Finally, we perform an in-depth analysis of the numerical precision requirements of floating-point arithmetic in PBS,

TABLE I: Summary of TFHE parameter notation.

| Symbols        | Description                     |
|----------------|---------------------------------|
| $\overline{N}$ | Size of polynomial              |
| n              | Dimension of LWE Ciphertext     |
| k              | Dimension of GLWE Ciphertext    |
| $\beta$        | bit width of Decomposition base |
| $\ell$         | Bootstrapping key level         |
| $\lambda$      | Security level                  |

characterizing the minimum bit-width necessary to guarantee correctness and exploring the potential for leveraging lower-precision Tensor Core operations.

This work makes the following major contributions:

- We propose a novel reuse method for executing TFHE applications on GPUs, which reduces redundant memory accesses and improves computational efficiency.
- We design a new mapping method and memory access format for executing the FFT on Tensor Cores, and further analyze the precision requirements of FFT computation for TFHE applications.
- We apply kernel fusion using a cross-iteration kernel, which further enhances data reuse for FFT and IFFT operations and alleviates memory access pressure.
- MNEMOS improves TFHE application performance by  $1.96\times$  on average and up to  $2.23\times$  in the best case, and improves the throughput of PBS by up to  $3.01\times$ .

#### II. BACKGROUND

