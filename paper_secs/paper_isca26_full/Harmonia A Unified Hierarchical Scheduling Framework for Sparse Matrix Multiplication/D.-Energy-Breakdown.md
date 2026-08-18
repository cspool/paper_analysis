# *D. Energy Breakdown*

Fig.13 presents the energy consumption of Harmonia compared to the static Vesper baseline, along with an internal breakdown across computation, data routing and sparsity control, and SRAM accesses. SRAM accesses dominate the energy budget at roughly 50%, confirming that Harmonia's focus on minimizing on-chip memory traffic directly targets the primary energy bottleneck of sparse accelerators. Data routing and sparsity control contribute approximately 20%, indicating that the added Reconfiguration Engine and Tiling Controller introduce minimal power overhead. Finally, actual computation accounts for the remaining 20% to 30%, depending on the inherent sparsity of the workload.

Overall, Harmonia reduces total energy consumption by 40% on average. Workloads that experience massive SRAMtraffic reductions naturally benefit the most. For instance, the highly irregular matrix rajat19 achieves over 60% energy savings after Harmonia successfully halves its SRAM accesses through coordinated tile-shape and dataflow adaptation. These results demonstrate that Harmonia's hierarchical scheduling improvements translate directly into significant energy efficiency, especially for highly skewed matrices.

![](_page_11_Figure_8.jpeg)

![](_page_11_Figure_9.jpeg)

Fig. 14. Scalability and Robustness analysis. (a) Speedups over the staticscheduling baseline across different compute-array scales. Harmonia leverages an expanded spatial canvas to smooth out local sparsity variations, delivering stable and increasing performance gains. (b) Normalized SRAM accesses across varying on-chip buffer capacities. Through dynamic micro-retiling, Harmonia consistently minimizes memory traffic and sustains superior buffer utilization, even in capacity-abundant environments.

It is crucial to note that because Harmonia strictly preserves the versatile datapath and introduces only lightweight feedback counters, the baseline compute and routing energy remain largely unaffected. The system completely avoids the heavy structural power overheads typically associated with highly flexible spatial architectures. Instead, the overwhelming majority of the energy savings stems from the software–hardware co-design: unified inter-tile shape adaptation and intra-tile dataflow switching eliminate redundant partial-sum movement and repeated operand reloads. This proves that integrating intelligent, lightweight runtime feedback is a far more energyefficient strategy for handling irregular sparsity than simply over-provisioning hardware buffers.

#### *E. Adaptivity and Robustness*

Harmonia's robust adaptivity stems from its hierarchical codesign, which seamlessly integrates offline analytical planning with real-time hardware feedback.

Compute Scalability. Fig.14(a) illustrates Harmonia's scalability across varying compute-array sizes under a fixed 64 KB local buffer capacity. For smaller arrays (e.g., 16 × 16), we observe slightly higher speedup variance. This occurs because fine-grained, tile-level sparsity fluctuations have a disproportionately large impact on merge depth and buffer pressure when computational resources are constrained. Additionally, a smaller array artificially inflates the buffer-to-compute ratio, which somewhat masks the inefficiencies of static scheduling and narrows the performance gap. However, as the PE array scales up (e.g., to 64 × 64), spatial aggregation smooths out local sparsity variations. Harmonia's dynamic tuning leverages this expanded spatial canvas to balance workloads more effectively, yielding steadily increasing and highly stable speedups over the baseline.

Memory Scalability. Fig. 14(b) evaluates the impact of memory scaling on a fixed 32×32 PE array. A larger on-chip SRAM naturally reduces spill-induced stalls and provides a more forgiving environment for static schedulers. Yet, even with ample capacity, Harmonia's ability to perform microretiling and dataflow switching ensures vastly superior buffer utilization. By dynamically expanding reuse windows and actively alleviating merge pressure at runtime, Harmonia sustains consistently lower normalized SRAM traffic across all evaluated workloads, regardless of the available capacity.

Ultimately, these scaling studies confirm that Harmonia is not overfitted to a specific hardware dimension, but instead offers a robust, future-proof scheduling substrate that scales gracefully alongside both compute and memory resources.

