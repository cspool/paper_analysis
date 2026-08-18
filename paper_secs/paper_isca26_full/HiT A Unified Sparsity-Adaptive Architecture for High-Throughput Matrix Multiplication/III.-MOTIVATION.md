# III. MOTIVATION

Modern workloads exhibit diverse sparsity patterns and are often executed on the same accelerator platforms in datacenters and high-performance computing systems. As a result, accelerators must efficiently support the full sparsity spectrum. However, most prior designs target only a narrow sparsity range, resulting in poor performance outside their target regime. In particular, prior accelerators supporting HS

![](_page_2_Figure_7.jpeg)

Fig. 4: (a) Analysis of the Gustavson dataflow as the number of PE rows increases (x-axis). The left y-axis shows normalized speedup, and the right y-axis shows normalized total read conflicts and normalized MAC utilization. (b) Average onchip storage required for unmerged partial sums per tile as the number of PE rows increases in outer-product dataflow.

workloads often rely on Gustavson-based dataflow, which becomes a bottleneck in large designs that require highly parallel computation.

