# <span id="page-11-0"></span>F. Sensitivity to L2C size

This section evaluates the effectiveness of Bumper across a range of L2C capacities, from 1MB to 16MB. Figure 18

<span id="page-11-3"></span>![](_page_11_Figure_10.jpeg)

Fig. 19: Sensitivity of the performance of the baseline and Bumper (left y-axis) and the baseline BPU MPKI (right y-axis) to increasing BTB size.

reports the corresponding speedups across the evaluated workloads. A key take-away is that Bumper consistently improves performance across all considered L2C sizes. For small L2Cs (1–2MB), the observed speedups are lower (2.9%-3.6%, on average) because the large instruction working sets of mobile applications limit temporal locality in small caches, thus reducing code pollution alone cannot prevent useful lines from being evicted before reuse. As the L2C size increases (4–8MB), Bumper becomes more effective (average speedups are 5.1%-6.8%) by retaining useful code lines longer and reducing premature evictions. Beyond 16MB, however, the benefits taper off, as most of the active instruction and data working set already fits within the L2C. One notable exception is app6, for which Bumper provides the highest benefits when L2C is 16MB; this happens because app6 has very poor temporal locality and its combined instruction and data working set does not fit in a 16MB L2C.

# <span id="page-11-0"></span>F. Sensitivity to L2C size

This section evaluates the effectiveness of Bumper across a range of L2C capacities, from 1MB to 16MB. Figure 18

<span id="page-11-3"></span>![](_page_11_Figure_10.jpeg)

Fig. 19: Sensitivity of the performance of the baseline and Bumper (left y-axis) and the baseline BPU MPKI (right y-axis) to increasing BTB size.

reports the corresponding speedups across the evaluated workloads. A key take-away is that Bumper consistently improves performance across all considered L2C sizes. For small L2Cs (1–2MB), the observed speedups are lower (2.9%-3.6%, on average) because the large instruction working sets of mobile applications limit temporal locality in small caches, thus reducing code pollution alone cannot prevent useful lines from being evicted before reuse. As the L2C size increases (4–8MB), Bumper becomes more effective (average speedups are 5.1%-6.8%) by retaining useful code lines longer and reducing premature evictions. Beyond 16MB, however, the benefits taper off, as most of the active instruction and data working set already fits within the L2C. One notable exception is app6, for which Bumper provides the highest benefits when L2C is 16MB; this happens because app6 has very poor temporal locality and its combined instruction and data working set does not fit in a 16MB L2C.

