# Expected Results

1.1: On H100, HyTiS outperforms cuBLAS in over 90% of cases and consistently outperforms Stream-K, Split-K, and Inductor-Triton. On A100, HyTiS exceeds cuBLAS performance in more than 50% of cases and maintains its advantage over Stream-K, Split-K, and Inductor-Triton.

1.2: To isolate the performance impact of two-level tile scheduling, we developed a variant of HyTiS, called HyTiS(L1), which implements only level 1 tile scheduling. Comparing HyTiS with HyTiS(L1), we observe a significant improvement in SM workload balance.

1.3: We developed a variant of HyTiS, HyTiS(STL), which omits adaptive tile layout optimization and instead uses a static group-M tile layout with a fixed group size of 8. Comparing HyTiS with HyTiS(STL), the volume of data read from DRAM to the L2 cache is significantly reduced in many cases.

1.<sup>4</sup> As shown in Figure 11, as the number of virtual tiles ("#vtile") increases, corresponding to larger input tensor sizes, the optimal value of 1 tends to converge toward 1. This indicates that a smaller search space is sufficient for the first-level tile scheduling under large problem sizes. In contrast, 2 fluctuates within a narrow range between 1.0 and 1.3, suggesting that the second-level search space remains relatively stable across different tensor scales.

