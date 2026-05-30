# *F. Overheads*

ARIADNE uses additional metadata to track Sharing Degree and perform dynamic Zero-copy at runtime UVM driver. Importantly, ARIADNE's implementation requires no additional GPU resources. Also, as additional host driver metadata is less than 70 B per VABlock and less than 100 B per GPU, our implementation adds approximately 560 KB of memory for a 16 GB application; this spatial overhead is negligible given the tens of gigabytes of host memory.

The tracking and computation of Sharing Degree and WCSS, along with the Zero-copy process, introduce additional latency. However, our experiments show that these processes introduce a latency of at most 100 ns, which is negligible within the context of the roughly 20 µs fault handling process for a single VABlock. Consequently, ARIADNE's operations do not introduce any considerable overhead.

#### VIII. RELATED WORK

Characteristics of GPU UVM system Diverse studies have been conducted to analyze the performance characteristics of UVM systems, across various hardware and workloads [1]–[4], [11], [14], [16], [19], [29], [30], [45], [46], [48], [58]. Allen et al. [2], Gayatri et al. [19], and Zheng et al. [58] have validated the UVM's performance characteristics and effectiveness in various scenarios. Wang et al. [51], Vijaykumar et al. [50], and Jablin et al. [26] conduct in-depth analysis of the access patterns of GPU. Chien et al. [11], Agarwal et al. [1] and Shao et al. [46] evaluated the efficacy of various UVM memory policies. Kim et al. [30] analyze the inefficiency of fault handling process, and suggest thread oversubscription.

Memory management for GPU systems Various studies have been proposed to analyze the memory characteristics of GPGPU workloads and optimize its memory management policies. Prefetching prevents future faults and increases the copy efficiency by enlarging the data transferred at once [16], [19], [20], [28], [34], [36], [47], [54]. However, prefetchers alone cannot resolve thrashing when the actual working-set size exceeds GPU memory. Other studies have leveraged hardware [32], [44], [52] or code-level analysis [7], [9], [26], [29], [33], [35] to understand GPU's thread execution architecture-based memory access characteristics [29], [44] and adjust memory policies [5], [13], [24], [25], [31], [37], [38], [44], [44], [53], [55], [59] and residencies [7]–[9], [42]. For instance, LAMAR [44] identified the inefficiencies caused by fragmentation that depend on the locality of GPGPU workloads and proposed an effective method to dynamically determine the optimal data granularity. Dynamap [9] inserts additional instructions to utilize memory access information, while Li et al. and SUV [7] perform detailed static analysis at the code level. ETC [32], Choukse et al. [13], and Nihaal et al. [38] increase the effective memory size by compressing cold data on the GPU. Kim et al. [31] and ETC [32] introduce thread throttling to prevent thrashing by limiting memory demands below memory capacity. Notably, ARIADNE's design is distinguished from previous work by its ability to be implemented solely through kernel module modifications, without requiring any changes to hardware, the compiler, or application code. By leveraging the GPU thread execution architecture, ARIADNE quantifies the migration suitability as a Sharing Degree at runtime, then dynamically adjusts the page to its optimal location.

#### IX. CONCLUSION

By analyzing both the GPU's hardware and software thread execution structures, we devise the Sharing Degree, a metric of spatial locality of pages within the UVM driver. Leveraging the Sharing Degree, we propose ARIADNE, a dynamic runtime UVM management system. Through optimal page placement, ARIADNE achieves an average performance improvement of

2.91× over two SOTA methods and exhibits linear performance degradation even under significant GPU memory oversubscription. Notably, to the best of our knowledge, ARIADNE is the first runtime-only UVM management framework that requires no modifications to the hardware, compiler, or application code. Unlike prior approaches, ARIADNE preserves the GPU abstraction of UVM, making ARIADNE directly applicable to binary, closed-source applications.

#### ACKNOWLEDGEMENTS

This work was supported by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (RS-2025-25433771), Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (RS-2018-II180503, RS-2024-00396013, RS-2024-00459797, RS-2025-09942968, and RS-2025-02263869).

#### **APPENDIX**

#### A. Abstract

This artifact comprises ARIADNE, SOTA solutions for comparison, and the associated GPGPU benchmark suites, such as polybench, rodinia, hecbench, and XSBench. ARIADNE is implemented exclusively through modifications to the kernel module driver. The comparative SOTA, AC, refers to the access-counter based migration mechanism within the NVIDIA UVM driver. Additionally, SUV, a SOTA of compiler-assisted method, is included as an optional component, as it requires a different environmental configuration compared to ARIADNE. All basic experiments are executed via shell scripts, allowing for the reproduction of perbenchmark execution times for each configuration, as well as Figures 9, 11, and 13. Given that the results are normalized comparisons, variations in absolute execution times on different systems may alter the exact magnitude of performance gaps. Nevertheless, the general trends observed in the paper remain valid. The artifact is publicly available on Zenodo(https://doi.org/10.5281/zenodo.17829999).

#### B. Artifact check-list (meta-information)

- Compilation: GCC, NVCC
- Data set: Polybench, Rodinia, Hecbench, XSBench
- Run-time environment: linux 6.0, CUDA 12.1
- Hardware: NVIDIA RTX A5000
- Run-time state: NVIDIA open-kernel modules driver
- Execution: Shell scripts
- Metrics: Benchmark execution time
- · Output: Figures, raw data
- Experiments: Final result reproduction
- How much disk space required (approximately)?: 15 GB (210 GB for comparison with SUV [7])
- How much time is needed to prepare workflow (approximately)?: 30 minutes (6 hours for comparison with SUV [7])
- How much time is needed to complete experiments (approximately)?: 2 hours
- Publicly available?: Yes
- Workflow automation framework used?: Shell scripts
- Archived (provide DOI)?: Yes. (10.5281/zenodo.17829999)

