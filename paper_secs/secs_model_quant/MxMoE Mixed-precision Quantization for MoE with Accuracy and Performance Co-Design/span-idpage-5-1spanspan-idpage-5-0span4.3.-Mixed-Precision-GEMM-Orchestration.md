# <span id="page-5-1"></span><span id="page-5-0"></span>4.3. Mixed-Precision GEMM Orchestration

![](_page_5_Figure_3.jpeg)

Figure 4. MxMoE ensures that tile configurations for different precisions have the same number of warps.

As discussed in Section 3.3, sequentially processing computations for each expert is inefficient. CUTLASS does not support heterogeneous precisions. Fusing heterogeneous-precision GEMMs introduces two fundamental challenges:

1) Optimal tile sizes and warp layouts vary across precisions, making a unified kernel for all possible precisions inherently suboptimal, and 2) The large number of possible precision combinations makes developing a custom kernel for each combination prohibitively costly. MxMoE addresses these challenges through an automated kernel generation framework that consists of three key components: micro-kernel specialization, resource configuration, and tile scheduling.

Micro-Kernel Specialization. We introduce configurable CTA-level micro-kernels implemented as CUDA device functions, designed with Cooperative-Thread-Group (CTA) index independence to enable subsequent horizontal fusion. Each micro-kernel's resources are specified via C++ template parameters, while memory access patterns are optimized for specific quantization schemes through meticulous hand-tuning of compute-to-memory access pipelines.

For instance, the W2A16 micro-kernel integrates fused de-

quantization with bit manipulation techniques for optimized integer-to-float conversion (Kim et al., 2022b), while the W4A4-g128 variant employs multistage software pipelining that enforces strict adherence to 128 quantization group constraints. Following bitwidth allocation, MxMoE generates a tile scheduler with a precision-aware routing logic, composing heterogeneous GEMM operations into unified kernel execution streams. We discuss in detail the advantages of micro kernel specilization over other possible approaches in App. A.2.

Resource Configuration. We next address the configuration of computational resources for horizontally fused mixed-precision Group-GEMM kernels. Building on the hardware-aware bitwidth allocation in Eq. 7, MxMoE derives tile configurations optimized for each quantization method under two critical constraints. First, warp count consistency is enforced across all micro-kernel tile configurations as shown in Fig. 4. Second, shared memory allocation follows the maximum requirement among fused operations. These two constraints ensures compliance with the CUDA programming model's requirement for uniform resources across CTAs (as shown in Fig. 3).

To mitigate shared memory waste from divergent tile sizes, we employ k-dimension tiling (slice-K). As illustrated in Fig. 4, the tile size for W4A16 is substantially smaller than that of W8A8. This disparity results in shared memory under-utilization for W4A16 micro-kernel. Our solution introduces additional parallelism along the k-dimension for W4A16 configurations through strategic tile partitioning. This dual-purpose optimization simultaneously reduces warp under-utilization while increasing shared memory utilization.

Tile Schedule. Finally, MxMoE optimizes the scheduling of tiles with heterogeneous precision requirements. The execution time of tiles varies significantly across different precision and tile shape configuration, making the scheduling order a critical determinant of overall completion time. This is a classic makespan minimization problem. While dynamic programming can achieve optimal solution, MxMoE implements an efficient greedy heuristic that prioritizes computationally intensive tiles. Given that the number of tiles in MoE blocks typically exceeds the available SM count by a substantial margin, this approach achieves near-optimal performance (Graham, 1966) while significantly reducing the scheduling overhead compared to dynamic programming solutions.

