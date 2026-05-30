# <span id="page-3-3"></span>3.2 Preloading More Operators Improves HBM Bandwidth Utilization

Operators in a DL model have different *compute intensities* (i.e., number of floating-point operations, or FLOPs, performed per byte). While some are compute-intensive due to more on-chip data reuse (e.g., operators that use model parameters, which are reused by all input requests in a batch), others are memory-intensive (e.g., the KV cache [45], which has no data reuse among requests in a batch).

The diverse HBM access and execution time across operators cause sub-optimal computation and HBM bandwidth utilization. If the currently executing operator has a short execution time while the next operator has a long HBM time, the current operator finishes before the next operator completes preloading, and the computation stalls. Similarly, if the next operator finishes preloading before the current operator completes, the HBM bandwidth is underutilized.

<span id="page-4-2"></span>![](_page_4_Figure_2.jpeg)

Figure 7: The inter-core bandwidth demand of each core across time, with different preload settings. The demand does not count HBM controller-to-core traffic.

<span id="page-4-1"></span>![](_page_4_Figure_4.jpeg)

Figure 8: The total per-core interconnect bandwidth demand.

To improve HBM bandwidth utilization, we can preload more operators. This also improves compute utilization, as more data will be ready on-chip, so future execution is less likely to stall. However, preloading more operators requires a larger preload space. Figure 6 shows how the HBM bandwidth demand varies over time for LLM inference with different per-core preload space sizes. The bandwidth demand is quantified as the minimum HBM bandwidth to prevent on-chip execution from stalling. With small preload space, the bandwidth demand fluctuates drastically due to insufficient preload opportunities. With larger preload space, more operators can be preloaded. This smooths out the bandwidth demand, reduces the compute/memory idleness, and enhances the overall performance.

