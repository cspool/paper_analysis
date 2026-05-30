# <span id="page-6-0"></span>3.3 Tile Layout Scheduling

After determining the tile size as outlined in the previous section, the next challenge is selecting an optimal tile layout. To address this, we consider two fundamental tile layout patterns: group-M (denoted GM) and group-N (denoted GN), each characterized by a group size parameter s. Consequently, we represent the tile layout as (tl, s), where  $tl \in \{GM, GN\}$ . Notably, column-major and rowmajor layouts are special cases of these patterns:  $(GM, \lceil M/bM \rceil)$  for column-major, and  $(GN, \lceil N/bN \rceil)$  for row-major.

As discussed in Section 2.3, the tile layout affects both the volume of data transferred from DRAM to the L2 cache and the overall performance. As illustrated on the right side of Figure 8, the L2 cache is shared among multiple SMs, allowing data requested by different SMs within the same wave to be reused. We denote the data volume needed for the input tensors of the i-th wave as  $V_i$ . A lower value of  $V_i$  corresponds to better L2 cache data locality within a wave. In practice, the first wave plays a critical role, as the L2 cache is initially empty and cannot benefit from previously cached data. Therefore, for a given tile layout pattern, either GM or GN, we select the optimal group size by minimizing  $V_1$ . Through analytical derivation, the optimal group size for group-M  $s_{opt}^{GM}$  equals  $min(\lceil \sqrt{N_{SM} \cdot bN/bM} \rceil, \lceil M/bM \rceil)$  or  $min(\lfloor \sqrt{N_{SM} \cdot bN/bM} \rfloor, \lceil M/bM \rceil)$ , and the result of  $s_{opt}^{GN}$  is analogous. To further determine the optimal tile layout between GM and *GN*, we use the total data required volume  $V_{tol} = \sum_{i} V_i$  across all waves as a metric, which reflects the overall L2 cache data locality. The layout with the smaller  $V_{tol}$  is selected as the optimal layout, denoted as  $tl_{opt}$ , as shown in the left of Figure 8. The computation of the tile layout  $(tl_{opt}, s_{opt})$  involves only simple mathematical operations and can be performed efficiently at runtime.

#### <span id="page-6-2"></span>Algorithm 1 Implementation of HyTiS

```
1: function HyTiS_GEMM(a, b, c, \mathcal{K}_1, \mathcal{K}_2, n1_wave, n2_tiles)
         pid = blockIdx.x
         k \ tiles = \lceil P.K/\mathcal{K}_1.bK \rceil
 3:
         for i = 0 to k\_tiles \times n1\_wave do
 4:
 5:
             ki, tid = i\%k\_tiles, pid
             if ki == 0 then
 6:
                 offs\_m, offs\_n = l1\_offset\_fn(tid)
 7:
                 ta, tb = Load(a, offs_m, ...), Load(b, offs_n, ...)
 8:
 9:
             tc+ = \mathcal{K}_1.compute(ta, tb, tc)
 10:
             if ki == k\_tiles - 1 then
 11:
                 store(tc, offs\_m, offs\_n)
 12:
                 tid+=N_{SM}
 13:
             end if
         end for
 15:
         if pid >= n2\_tiles then
 16:
 17:
             return
         end if
         for i = 0 to \lceil K/\mathcal{K}_2.bK \rceil do
20:
             offs_m, offs_n = l2\_offset\_fn(tid)
             ta, tb = Load(a, offs_m, ...), Load(b, offs_n, ...)
21:
             tc = \mathcal{K}_1.compute(ta, tb)
22:
             store(tc, offs_m, offs_n)
 24:
         end for
    end function
 25:
26:
 27: function _MAIN(P, a, b, c)
         ts = HyTiScheduler(P.M, P.N, P.K)
         K_1, K_2, n1_wave, n2_tiles, grid_size = ts.autotune()
         l1\_offset\_fn = ts.emit\_l1\_offset\_fn()
         l2\_offset\_fn = ts.emit\_l2\_offset\_fn()
31:
         HyTiS\_GEMM < grid\_size > (a, \quad b, \quad c, \quad \mathcal{K}_1, \mathcal{K}_2, \quad n1\_wave,
     n2_tiles)
33: end function
```

According to our analytical model, the volume of global-to-L2 memory traffic, denoted as  $\mathcal{V}(tl,s)$ , remains constant across different tile layouts when the workload comprises only a single wave. This observation implies that tile layout scheduling offers no performance benefit at the second level of scheduling. As a result, adaptive tile layout optimization is applied exclusively at the first level, where multiple waves are present and layout decisions have a measurable impact. For simplicity, the second level adopts a fixed column-major layout.

#### 4 Implementation

HyTiS is implemented on top of Triton [37], an open-source language and compiler framework designed for expressing and compiling tiled neural network computations into highly optimized machine code. By leveraging Triton's high-level, user-friendly programming interfaces, HyTiS concentrates on optimizing matrix multiplication at the tile scheduling level, while relying on Triton's robust infrastructure for intra-tile optimizations. Triton offers a comprehensive set of low-level optimizations, including automatic memory coalescing, thread swizzling, vectorization, efficient shared

memory allocation, and synchronization. This division of responsibilities simplifies the overall optimization workflow in HyTiS and ensures the generation of highly efficient GPU kernels.

Kernel Design. To realize the proposed hybrid tile scheduling, we implement a two-level scheduling GEMM kernel, as illustrated in the \_ function of Algorithm [1.](#page-6-2) This kernel consists of two main phases. The first (lines 4–15) implements level-1 tile scheduling using micro-kernel K1, responsible for executing 1\_ full waves. And the second part performs level-2 tile scheduling to process the remaining partial wave using micro-kernel (line 19-24). Each scheduling phase consists of four key primitive operations: , , and and \_ . The first three are adopted from Triton, while \_ is generated by HyTiS in two variants: 1\_ \_ and 2\_ \_ , which map tile IDs to corresponding address offsets in the output tensor for the two scheduling levels. On NVIDIA Hopper architecture, we leverage persistent kernel execution to eliminate CTA launch overhead, and TMA instructions to accelerate global memory loading. In contrast, on the Ampere architecture, TMA instructions are not supported and persistent kernels incur excessive register file usage. Therefore, for Ampere, we adopt a traditional data-parallel launch strategy while preserving the same tile scheduling order used on Hopper to maintain consistency across architectures.

User Interface. We integrate the core design components of HyTiS into a module named ℎ, which accepts the GEMM problem shape as input. The implementation details described in Section [3](#page-4-1) are encapsulated within the method, which returns the selected throughput-oriented micro-kernel K1, the latency-oriented micro-kernel K2, the number of full waves 1\_, and the number of tiles in the partial wave 2\_. As is the most time-consuming operation in the scheduling pipeline, its results are cached, similar to the approach used in Py-Torch Inductor, to eliminate runtime overhead. The returned parameters serve as input to the main execution function \_. Additionally, ℎ constructs separate \_ functions for each scheduling level, encapsulating the tile-to-thread block mapping logic required to coordinate TO and LO scheduling efficiently.

