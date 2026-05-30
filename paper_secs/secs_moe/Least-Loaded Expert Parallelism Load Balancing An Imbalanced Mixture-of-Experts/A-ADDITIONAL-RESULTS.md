# A ADDITIONAL RESULTS

### A.1 SEPARATE VS FUSED GROUPED-GEMM

Fig. [8](#page-13-0) shows the compute time between a naive for-loop of GEMMs using cuBLAS implementation vs a fused optimized Grouped-GEMM kernel written in Triton, with adoption of Tensor Memory Accelerator (TMA). The cuBLAS version launches N GPU kernel launches, causing high overhead, while the Triton version launches only one. However, as shown, the cuBLAS version still outperforms Triton counterpart because each cuBLAS GEMM kernel is hardware-specific and highly optimized at architecture level, while the Triton version is a generic implementation. In addition, despite all computations have the same FLOPs, the compute time dramatically increases the more experts are present. Therefore, it is better to compute a few giant GEMMs with few experts than to compute many tiny GEMMs with many experts. Both expert parallelism and our method (LLEP) leverage this principle by spreading expert weights across EP ranks, allowing each rank to compute only a handful of experts.

<span id="page-13-0"></span>![](_page_13_Figure_4.jpeg)

Figure 8: Grouped-GEMM benchmark (lower is better): execution time vs. number of experts under balanced workload with the same total FLOPs. Specifically, B<sup>i</sup> = 65536 tokens are evenly distributed across N experts, with H = D = 8192.

