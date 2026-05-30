# Relation To Contributions

1.1: We evaluate the overall performance of HyTiS compared to baselines, including cuBLAS, Inductor-Triton, Split-K, and Stream-K. Our test corpus is designed to cover a wide range of GEMM

problems commonly used in deep learning workloads and scientific applications. The performance gains over the baselines highlight the effectiveness of HyTiS's system design (1).

1.2: We perform a breakdown analysis including three metrics: execution speedup, SM balance, and DRAM read data volume. To assess workload balance across SMs (contribution 2), we collect three metrics by NSight Compute: \_\_\_., \_\_\_., and \_\_\_.. For simplicity, we define a composite metric B = ( − )/ to represent SM workload balance. Furthermore, to assess the contribution of HyTiS's adaptive tile layout scheduling method (3), which enhances L2 cache affinity, we use the \_\_ . metric from the NSight Compute profiler to measure the volume of data transferred from DRAM to the L2 cache.

1.3: To demonstrate the effectiveness of HyTiS in addressing the wave quantization problem, we compare its performance with cuBLAS and Inductor-Triton in experiments that vary while keeping and fixed. The experiments results are divided into two regions: the quantization-prominent region (highlighted in orange) and the non-quantization-prominent region. The dominant performance in the quantization-prominent region validates the effectiveness of HyTiS in mitigating the wave quantization issue and improving SM workload balance (2).

1.4: We provides a detailed analysis of the sensitivity of HyTiS with respect to the hyperparameters 1 and 2, as well as the effect of input tensor size of the tuning search space. It supports Claim4, which highlights the importance of adaptive search space design and hyperparameter robustness in HyTiS.

