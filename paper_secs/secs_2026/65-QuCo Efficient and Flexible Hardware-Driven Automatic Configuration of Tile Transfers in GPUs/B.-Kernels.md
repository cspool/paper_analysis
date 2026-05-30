# *B. Kernels*

For the eight kernels listed in Table I, Figure 6 compares the performance obtained for six execution cases:

7For kernels like *Elementwise* or *Dot-Product*, the tile size range is reduced, yielding 5 discrete options.

i) *NoATT/Non-Tuned*; ii) *NoATT/Fine-Tuned*; iii) *ATT/Non-Tuned*; iv) *ATT/Informed-Tuned*; v) *ATT/Fine-Tuned*; and vi) QuCo. As detailed below, each case represents a different level of optimization and complexity in kernel execution. All the results are normalized to an ideal ATT implementation, where the ATT operates with an unlimited LDS, allowing all data to fit into the LDS and enabling continuous tile loading without LDS constraints (an *ideal-scenario* performance bound).

The first two cases, *NoATT/Non-Tuned* and *NoATT/Fine-Tuned*, evaluate kernels that do not take advantage of the ATT. *NoATT/Non-Tuned* corresponds to a naive implementation, where memory operations and computations are poorly optimized, using small tile sizes, and as a result, issuing more memory requests, leading to suboptimal application performance (see Figure 6). In contrast, *NoATT/Fine-Tuned* is the configuration obtained through extensive design space exploration to optimize kernel parameters—such as tile size and queue slots—resulting in significantly improved performance across all cases, particularly for simpler workloads such as *ElementwiseK* and *Dot-Product*.

Among the ATT-based implementations, *ATT/Non-Tuned* serves as a baseline case where the ATT unit is used without proper tuning of tile sizes and queue slots. This lack of optimization leads to poor performance across all kernels. Without informed parameter selection, tile sizes may be too small to leverage memory bandwidth or too large to fit efficiently into LDS memory, causing stalls. Similarly, the number of queue slots may be insufficient to overlap memory transfers and compute, leading to idle cycles and creating substantial performance gaps compared to the upper bound. While this approach achieves competitive results on lightweight workloads, it struggles with more sensitive kernels that require precise alignment between memory transfers and computation. For example, in *Dot-Product* and *Elementwise*, the lack of tuning not only prevents overlap but can introduce wavefront scheduling stalls or memory contention.

*ATT/Informed-Tuned* incorporates heuristic-based configurations inspired by NVIDIA guidelines, using tile sizes between 64 and 256 elements and queue slots between 2 and 4 (double or quadruple buffering) [30]. This approach delivers strong performance for simpler kernels like *ElementwiseK*, *Elementwise*, *Sumvectors*, and *Dot-Product*, but its performance

![](_page_8_Figure_0.jpeg)

Fig. 7: Ablation study showing kernel execution normalized to *ATT/Fine-Tuned*, see Section V-C for details.

for *Matrix-Vector*, *Matrix-Matrix*, *Matrix-Matrix+Reduction* and *Batched-Matrix-Matrix* remains suboptimal due to the increased complexity and resource demands of these workloads.

The *ATT/Fine-Tuned* represents an exhaustive design space exploration for each kernel, identifying the best possible tile and slot configurations through repeated profiling. This approach requires substantial computational effort, with the GPU kernel executed once per configuration, and requiring manual tuning. Obviously, this optimized execution case provides the best performance, particularly for *Matrix-Vector*, *Matrix-Matrix*, *Matrix-Matrix+Reduction* and *Batched-Matrix-Matrix* workloads, since these kernels can now make better use of GPU resources, achieving performance closer to the ideal. However, the complexity of this approach makes it impractical for complex kernels like these four (as shown in Table I, 2.6e+14 kernel launches would be required for *Matrix-Matrix* or *Batched-Matrix-Matrix* workloads). This situation is even worse for real-world applications (e.g., *Whisper Tiny* with 2.1e+17 kernel launches).

In contrast, the ability of the QuCo unit to automatically select values for the configuration parameters of the queues based on architectural characteristics and kernel properties, fully eliminates the need for this impractical exhaustive tuning. As shown in Figure 6, QuCo achieves performance that is slightly below *ATT/Fine-Tuned* but consistently outperforms *NoATT/Fine-Tuned*, *ATT/Non-Tuned*, and *ATT/Informed-Tuned* across all the kernels. Without requiring any manual tuning or host-side intervention, QuCo provides near-optimal configurations with significantly reduced complexity.

For the challenging matrix kernels (*Matrix-Matrix*, *Matrix-Matrix + Reduction* and *Batched-Matrix-Matrix*), all methods—QuCo included—fall significantly short of the ideal performance due to limited data reuse and large working set sizes that exceed the LDS capacity. In this case, the *K* tile dimension cannot fully reside in LDS, requiring frequent re-fetches from global memory and increasing pressure on the L2 cache. This effect is especially pronounced when compared to the theoretical unlimited-LDS baseline, which manages to retain all tile fragments in on-chip memory.

Interestingly, the *Batched-Matrix-Matrix* kernel presents a special case where *ATT/Fine-Tuned* and QuCo slightly outperforms the *NoATT/Fine-Tuned* implementation. We observe this is largely due to the overhead of managing a high number of asynchronous barriers. This behavior is consistent with prior work [47], where the authors report achieving performance

![](_page_8_Figure_7.jpeg)

Fig. 8: Speedup over the *ATT/Fine-Tuned* baseline for several DNN models and composite kernel workloads

on par with optimized cuBLAS and Triton implementations. This benchmark serves as a representative case illustrating that while QuCo enables automated configuration, not all workloads benefit equally from the ATT.

To understand the memory-level effects of QuCo's automated configuration, we analyzed DRAM request activity during kernel execution. Figure 5 shows a complete trace of DRAM requests for the kernels. In the *NoATT/Fine-Tuned* case (blue line), memory accesses occur abruptly and irregularly, with idle periods between request spikes. This behavior indicates poor overlap between memory access and computation, as global memory loads are issued synchronously by the kernel.

In contrast, the QuCo-enabled configuration (red line) maintains a consistently high level of DRAM activity throughout execution. As previously discussed, the *Batched-Matrix-Matrix* kernel reflects the synchronization overhead and its impact at the memory level.

This sustained throughput is the result of asynchronous tile transfers using Operand Queues, which are configured and allocated by QuCo, to later load data tiles into the LDS. Because memory and compute are overlapped more effectively, the kernel completes significantly earlier than its no ATT counterpart. This result demonstrates QuCo's ability to exploit the available DRAM bandwidth and better hide memory latency even without programmer intervention.

