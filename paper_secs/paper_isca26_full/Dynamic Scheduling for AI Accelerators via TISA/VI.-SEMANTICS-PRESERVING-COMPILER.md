# VI. SEMANTICS-PRESERVING COMPILER

The TISA abstraction not only enables dynamic tile scheduling at runtime but also offers a unifying intermediate representation that allows compilers to preserve semantic information down to the hardware interface. Our compiler stack mirrors the traditional hierarchical decomposition flow of deep learning compilation while maintaining operator context, dependencies, and resource semantics until TISA generation. The end-toend flow consists of a framework bridge, a graph compiler, a fusion compiler, the TISA generator, and backend-specific code generation. Together, these components progressively lower models from high-level operator graphs to hardwareexecutable TISA binaries while preserving semantic metadata used by the dynamic scheduler.

- *a) Framework bridge:* The bridge layer ingests models from PyTorch [3], JAX [6], and TensorFlow [1] using the torchxla [36] frontend, exporting framework graphs to XLA or StableHLO dialects. By aligning TISA's OpType taxonomy with StableHLO operator abstractions, we ensure a consistent mapping of operator semantics across frameworks, simplifying subsequent dependency and resource analyses.
- *b) Graph compiler (GC):* Our MLIR-based [24] GC consumes StableHLO IR and performs architecture-aware optimizations, including fusion, tiling, and locality-driven reordering. It emits a software-scheduled tile graph e.g., Figure 2, that exposes legal overlap opportunities across heterogeneous units (tensor, vector, DMA) while minimizing off-chip communication. Unlike conventional graph optimizers, GC explicitly preserves operator boundaries and typed dependency edges in a custom MLIR dialect, forming a semantically rich intermediate representation that serves as input to the fusion compiler.

Tile dimensions maximize arithmetic intensity subject to SRAM capacity constraints (e.g., 64×64 for the Epoch 256 KB staging buffer). Non-divisible tensor boundaries trigger edge tiles. Instead of relying on padding overhead, TISA directly encodes the precise Shape and TileMem ranges for these boundaries, generating tailored instructions that seamlessly dispatch smaller edge tiles without unaligned penalties. For workloads comprising thousands of concurrent tiles, *hierarchical scheduling* ensures scalability: each core manages 256 tiles locally, with the compiler performing global coordination via static tile-to-core assignment.

*c) Fusion compiler (FC):* FC specializes fused subgraphs produced by GC into TISA-compatible operators. Built atop MLIR, FC defines a custom TISA dialect whose operations (e.g., tisa.gemm, tisa.softmax) encode operator semantics through OpType and dependency descriptors, resource intents (mappings to execution unit classes) via UnitMap, and memory access patterns in terms of symbolic TileMem ranges and scopes. Through this dialect, the compiler translates the software-scheduled tile graph into a stream of TISA instructions that preserve operator identity, data dependencies, and resource affinity. These attributes constitute the semantic contract later consumed by the runtime scheduler.

Conventional compilers flatten operators into loop nests, discarding tensor-level boundaries. The TISA compiler truncates lowering at tile granularity–it need not lower to fine-grained ISA instructions, as the hardware scheduler consumes tilelevel semantics directly. This simplifies optimizations: e.g., ping-pong buffering requires only allocating two buffers and emitting alternating TISA tiles; no loop unrolling or instruction reordering is needed, as the runtime scheduler handles overlap. The semantic triad is identically codified from high-level graph components into binary output, seamlessly transitioning context into hardware.

*d) TISA generator and backends:* The TISA generator provides a virtual tile-level instruction set that unifies multiple hardware backends. Its operation semantics mirror StableHLO operators, while its data semantics are defined on tiles sized to fit L1/L2 or shared SRAM capacity. OpType is statically bound to accelerator unit classes (tensor, vector, DMA), which allows the runtime to perform legality checks and enable crossunit overlap.

Currently, two backends are implemented: (1) TISA-NPU backend, which targets our Epoch hardware with full dynamic scheduling support. It uses a custom LLVM-based lowering path that embeds TISA metadata into the final binary, consumed by the hardware scheduler. (2) TISA-CPU backend, which emits optimized CPU kernels for functional validation and reference execution. On CPU, overlapping tiles execute serially, but this backend retains identical TISA semantics, enabling end-to-end verification. Both backends preserve identical semantic descriptors to guarantee consistent scheduling behavior across platforms.

*e) Runtime interface:* During execution, the compiled binary emits per-tile descriptors that encode all required scheduling attributes, including OpType, UnitMap, and TileMem. These descriptors form ready sets that populate the runtime's waiting queues (WQs) and issue queues (IQs), where arbitration and dispatch occur under the dynamic tile scheduler described in Section V. Each tile executes in a run-to-complete, non-preemptive fashion, with scheduling decisions made at tile boundaries to balance adaptivity and low hardware overhead.

*f) Discussion:* This compiler–runtime co-design closes the loop between semantic preservation and dynamic execution. By carrying operator context and dependency metadata down to TISA, the compiler enables the runtime to make legality and overlap decisions based directly on semantics rather than opaque instruction streams. In turn, the dynamic scheduler translates these semantics into runtime performance, achieving adaptivity without sacrificing correctness.

## VII. IMPLEMENTATION

## *A. Implementation on Epoch*

We implement our framework on both a CPU backend and an AI accelerator Epoch. The CPU implementation serves as a functional and accuracy reference by providing a TISAsemantic operator library, while the Epoch backend fully exploits TISA's dynamic scheduling and heterogeneous execution capabilities through native ISA support.

- *a) Hardware overview:* Epoch is a throughput-oriented AI accelerator that offloads compute-intensive kernels from a host CPU via a high-bandwidth interconnect and shares a 48 GB DDR memory. The chip has been successfully taped out at 1 GHz and is currently in commercialization. All Epoch performance results presented in this paper are measured on this taped-out physical silicon with W = 8. It is organized to expose abundant tile-level parallelism with 32 cores. Each core integrates three specialized engines: a Matrix Engine (ME) for tensor arithmetic, a Vector Engine (VE) for elementwise and reduction operations, and a Data Engine (DE) for DMA and asynchronous data movement. This heterogeneous structure closely resembles the architectures summarized in Table I, Therefore, porting this framework to other accelerators requires adding the hardware scheduler and TISA to it.
- *b) Memory hierarchy:* Each core provides 1.5 MB of local memory, and cores communicate via on-chip shared SRAM, enabling inter-core tile reuse. An on-chip NoC connects the system, and parameters and activations are exchanged with the host via the 48 GB global DDR memory.
- *c) TISA integration:* The TISA instructions act as the software–hardware contract on Epoch, and each core integrates a hardware scheduler (Section V) that consumes TISA descriptors and orchestrates heterogeneous execution without explicit software barriers. On ME, we introduce custom tensor instructions for block matrix/tensor arithmetic. On VE, we extend the vector ISA with tile-friendly operations. On DE, we expose DMA-style descriptors to support asynchronous, non-blocking transfers. All components adhere to the TISA interface, which conveys semantic context (OpType), resource affinity (UnitMap), and tile memory descriptors (TileMem) to the scheduler for legality checks and dynamic overlap.
- *d) Kernel library and compiler integration:* On top of these hardware extensions, we implement a high-performance operator library where kernels are expressed at tile granularity and executed through double-buffered pipelines across

```
1 // CUDA fa3 pseudocode
 2 // Load Q K data
 3 tma_load_q ( s_Q ,Q) ;
 4 tma_load_k_transpose ( s_K ,K) ;
 5 warpgroup_fence_producer () ;
 6 // Matrix multiply (P=Q *K)
 7 wgmma :: mma_sync ( s_P , s_Q , s_K );
 8 // Softmax compute (S= softmax (P) )
 9 wgmma :: wait () ; // Wait s_P
10 softmax_warpgroup ( s_S , s_P , state );
11 for ( int j = 0; j < Tc ; j ++) {
12 if (j < Tc - 1) {
13 // Load K data ( next tile )
14 tma_load_k_transpose ( s_K_next ,
           K_next ) ;
15 warpgroup_barrier_arrive () ;
16 // Matrix multiply ( next tile )
17 wgmma :: mma_async ( s_S_next , s_Q ,
           s_K_next ) ;
18 }
19 // Load V data
20 tma_load_v ( s_V , V);
21 warpgroup_barrier_wait () ;
22 // Matrix multiply (R =S*V )
23 wgmma :: mma_sync ( s_R , s_S , s_V );
24 if (j < Tc - 1) {
25 // Softmax compute ( next tile )
26 wgmma :: wait () ; // Wait s_S_next
27 softmax_warpgroup ( s_S_next ,
           s_S_next , state_next );
28 }
29 // Rescale R data ( O= Rescale (R))
30 wgmma :: wait () ; // Wait s_R
31 rescale_warpgroup ( s_O , s_R , state ,
          state_next );
32 warpgroup_commit_batch () ;
33 // Update next index
34 update_carousel_index () ;
35 }
36 // Store O data
37 tma_store_o (O , s_O ) ;
38 warpgroup_epilogue () ;
                                            1 // TISA fa3 pseudocode
                                            2 // Load Q K data
                                            3 tisa :: load <de >( s_Q ,Q);
                                            4 tisa :: load_transpose <de >( s_K ,K) ;
                                            5 // Matrix multiply (P=Q *K)
                                            6 tisa :: gemm <me >( s_P , s_Q , s_K ) ;
                                            7 // Softmax compute (S= softmax (P) )
                                            8 tisa :: softmax <ve >( s_S , s_P , state );
                                            9 for ( int j = 0; j < Tc ; j ++) {
                                           10 if (j < Tc - 1) {
                                           11 // Load K data ( next tile )
                                           12 tisa :: load_transpose <de >(
                                                       s_K_next , K_next ) ;
                                           13 // Matrix multiply ( next tile )
                                           14 tisa :: gemm <me >( s_S_next , s_Q ,
                                                       s_K_next ) ;
                                           15 }
                                           16 // Load V data
                                           17 tisa :: load < de >( s_V ,V );
                                           18 // Matrix multiply (R =S*V )
                                           19 tisa :: gemm < me >( s_R , s_S , s_V );
                                           20 if (j < Tc - 1) {
                                           21 // Softmax compute ( next tile )
                                           22 tisa :: softmax <ve >( s_S_next ,
                                                       s_S_next , state_next ) ;
                                           23 }
                                           24 // Rescale R data ( O= Rescale (R))
                                           25 tisa :: rescale <ve >( s_O , s_R , state ,
                                                      state_next );
                                           26 // Update next index
                                           27 update_next_index () ;
                                           28 }
                                           29 // Store O data
                                           30 tisa :: store <de >( O , s_O ) ;
```

Fig. 5: Comparison of FlashAttention-3 CUDA and TISA pseudocode. CUDA explicitly manages synchronization (lines 5, 9, 15, 21, 26, 30, 32, and 38), while TISA eliminates all barriers via semantics-aware dependency resolution.

ME/VE/DE. Kernels are shape-parametric and mapped directly to units indicated by OpType, while the upstream compiler (Section VI) automatically generates the corresponding TISA instructions. This separation decouples TISA semantics from hardware-specific implementations, allowing runtime scheduling to remain hardware-agnostic while still leveraging optimized kernels per operator type.

*e) Multi-core execution:* For multi-core execution, the compiler employs spatial partitioning: independent tile groups (e.g., attention heads, batch dimensions) are statically assigned to cores. Each core's local TISA scheduler operates independently via its in-flight semantic tables. Inter-core synchronization uses lightweight NoC signals triggered by shared SRAM bank updates. Runtime load balancing occurs at the software level between kernel invocations, not within the tile scheduler.

# VI. SEMANTICS-PRESERVING COMPILER

The TISA abstraction not only enables dynamic tile scheduling at runtime but also offers a unifying intermediate representation that allows compilers to preserve semantic information down to the hardware interface. Our compiler stack mirrors the traditional hierarchical decomposition flow of deep learning compilation while maintaining operator context, dependencies, and resource semantics until TISA generation. The end-toend flow consists of a framework bridge, a graph compiler, a fusion compiler, the TISA generator, and backend-specific code generation. Together, these components progressively lower models from high-level operator graphs to hardwareexecutable TISA binaries while preserving semantic metadata used by the dynamic scheduler.

- *a) Framework bridge:* The bridge layer ingests models from PyTorch [3], JAX [6], and TensorFlow [1] using the torchxla [36] frontend, exporting framework graphs to XLA or StableHLO dialects. By aligning TISA's OpType taxonomy with StableHLO operator abstractions, we ensure a consistent mapping of operator semantics across frameworks, simplifying subsequent dependency and resource analyses.
- *b) Graph compiler (GC):* Our MLIR-based [24] GC consumes StableHLO IR and performs architecture-aware optimizations, including fusion, tiling, and locality-driven reordering. It emits a software-scheduled tile graph e.g., Figure 2, that exposes legal overlap opportunities across heterogeneous units (tensor, vector, DMA) while minimizing off-chip communication. Unlike conventional graph optimizers, GC explicitly preserves operator boundaries and typed dependency edges in a custom MLIR dialect, forming a semantically rich intermediate representation that serves as input to the fusion compiler.

Tile dimensions maximize arithmetic intensity subject to SRAM capacity constraints (e.g., 64×64 for the Epoch 256 KB staging buffer). Non-divisible tensor boundaries trigger edge tiles. Instead of relying on padding overhead, TISA directly encodes the precise Shape and TileMem ranges for these boundaries, generating tailored instructions that seamlessly dispatch smaller edge tiles without unaligned penalties. For workloads comprising thousands of concurrent tiles, *hierarchical scheduling* ensures scalability: each core manages 256 tiles locally, with the compiler performing global coordination via static tile-to-core assignment.

*c) Fusion compiler (FC):* FC specializes fused subgraphs produced by GC into TISA-compatible operators. Built atop MLIR, FC defines a custom TISA dialect whose operations (e.g., tisa.gemm, tisa.softmax) encode operator semantics through OpType and dependency descriptors, resource intents (mappings to execution unit classes) via UnitMap, and memory access patterns in terms of symbolic TileMem ranges and scopes. Through this dialect, the compiler translates the software-scheduled tile graph into a stream of TISA instructions that preserve operator identity, data dependencies, and resource affinity. These attributes constitute the semantic contract later consumed by the runtime scheduler.

Conventional compilers flatten operators into loop nests, discarding tensor-level boundaries. The TISA compiler truncates lowering at tile granularity–it need not lower to fine-grained ISA instructions, as the hardware scheduler consumes tilelevel semantics directly. This simplifies optimizations: e.g., ping-pong buffering requires only allocating two buffers and emitting alternating TISA tiles; no loop unrolling or instruction reordering is needed, as the runtime scheduler handles overlap. The semantic triad is identically codified from high-level graph components into binary output, seamlessly transitioning context into hardware.

*d) TISA generator and backends:* The TISA generator provides a virtual tile-level instruction set that unifies multiple hardware backends. Its operation semantics mirror StableHLO operators, while its data semantics are defined on tiles sized to fit L1/L2 or shared SRAM capacity. OpType is statically bound to accelerator unit classes (tensor, vector, DMA), which allows the runtime to perform legality checks and enable crossunit overlap.

Currently, two backends are implemented: (1) TISA-NPU backend, which targets our Epoch hardware with full dynamic scheduling support. It uses a custom LLVM-based lowering path that embeds TISA metadata into the final binary, consumed by the hardware scheduler. (2) TISA-CPU backend, which emits optimized CPU kernels for functional validation and reference execution. On CPU, overlapping tiles execute serially, but this backend retains identical TISA semantics, enabling end-to-end verification. Both backends preserve identical semantic descriptors to guarantee consistent scheduling behavior across platforms.

*e) Runtime interface:* During execution, the compiled binary emits per-tile descriptors that encode all required scheduling attributes, including OpType, UnitMap, and TileMem. These descriptors form ready sets that populate the runtime's waiting queues (WQs) and issue queues (IQs), where arbitration and dispatch occur under the dynamic tile scheduler described in Section V. Each tile executes in a run-to-complete, non-preemptive fashion, with scheduling decisions made at tile boundaries to balance adaptivity and low hardware overhead.

*f) Discussion:* This compiler–runtime co-design closes the loop between semantic preservation and dynamic execution. By carrying operator context and dependency metadata down to TISA, the compiler enables the runtime to make legality and overlap decisions based directly on semantics rather than opaque instruction streams. In turn, the dynamic scheduler translates these semantics into runtime performance, achieving adaptivity without sacrificing correctness.

## VII. IMPLEMENTATION

## *A. Implementation on Epoch*

We implement our framework on both a CPU backend and an AI accelerator Epoch. The CPU implementation serves as a functional and accuracy reference by providing a TISAsemantic operator library, while the Epoch backend fully exploits TISA's dynamic scheduling and heterogeneous execution capabilities through native ISA support.

- *a) Hardware overview:* Epoch is a throughput-oriented AI accelerator that offloads compute-intensive kernels from a host CPU via a high-bandwidth interconnect and shares a 48 GB DDR memory. The chip has been successfully taped out at 1 GHz and is currently in commercialization. All Epoch performance results presented in this paper are measured on this taped-out physical silicon with W = 8. It is organized to expose abundant tile-level parallelism with 32 cores. Each core integrates three specialized engines: a Matrix Engine (ME) for tensor arithmetic, a Vector Engine (VE) for elementwise and reduction operations, and a Data Engine (DE) for DMA and asynchronous data movement. This heterogeneous structure closely resembles the architectures summarized in Table I, Therefore, porting this framework to other accelerators requires adding the hardware scheduler and TISA to it.
- *b) Memory hierarchy:* Each core provides 1.5 MB of local memory, and cores communicate via on-chip shared SRAM, enabling inter-core tile reuse. An on-chip NoC connects the system, and parameters and activations are exchanged with the host via the 48 GB global DDR memory.
- *c) TISA integration:* The TISA instructions act as the software–hardware contract on Epoch, and each core integrates a hardware scheduler (Section V) that consumes TISA descriptors and orchestrates heterogeneous execution without explicit software barriers. On ME, we introduce custom tensor instructions for block matrix/tensor arithmetic. On VE, we extend the vector ISA with tile-friendly operations. On DE, we expose DMA-style descriptors to support asynchronous, non-blocking transfers. All components adhere to the TISA interface, which conveys semantic context (OpType), resource affinity (UnitMap), and tile memory descriptors (TileMem) to the scheduler for legality checks and dynamic overlap.
- *d) Kernel library and compiler integration:* On top of these hardware extensions, we implement a high-performance operator library where kernels are expressed at tile granularity and executed through double-buffered pipelines across

```
1 // CUDA fa3 pseudocode
 2 // Load Q K data
 3 tma_load_q ( s_Q ,Q) ;
 4 tma_load_k_transpose ( s_K ,K) ;
 5 warpgroup_fence_producer () ;
 6 // Matrix multiply (P=Q *K)
 7 wgmma :: mma_sync ( s_P , s_Q , s_K );
 8 // Softmax compute (S= softmax (P) )
 9 wgmma :: wait () ; // Wait s_P
10 softmax_warpgroup ( s_S , s_P , state );
11 for ( int j = 0; j < Tc ; j ++) {
12 if (j < Tc - 1) {
13 // Load K data ( next tile )
14 tma_load_k_transpose ( s_K_next ,
           K_next ) ;
15 warpgroup_barrier_arrive () ;
16 // Matrix multiply ( next tile )
17 wgmma :: mma_async ( s_S_next , s_Q ,
           s_K_next ) ;
18 }
19 // Load V data
20 tma_load_v ( s_V , V);
21 warpgroup_barrier_wait () ;
22 // Matrix multiply (R =S*V )
23 wgmma :: mma_sync ( s_R , s_S , s_V );
24 if (j < Tc - 1) {
25 // Softmax compute ( next tile )
26 wgmma :: wait () ; // Wait s_S_next
27 softmax_warpgroup ( s_S_next ,
           s_S_next , state_next );
28 }
29 // Rescale R data ( O= Rescale (R))
30 wgmma :: wait () ; // Wait s_R
31 rescale_warpgroup ( s_O , s_R , state ,
          state_next );
32 warpgroup_commit_batch () ;
33 // Update next index
34 update_carousel_index () ;
35 }
36 // Store O data
37 tma_store_o (O , s_O ) ;
38 warpgroup_epilogue () ;
                                            1 // TISA fa3 pseudocode
                                            2 // Load Q K data
                                            3 tisa :: load <de >( s_Q ,Q);
                                            4 tisa :: load_transpose <de >( s_K ,K) ;
                                            5 // Matrix multiply (P=Q *K)
                                            6 tisa :: gemm <me >( s_P , s_Q , s_K ) ;
                                            7 // Softmax compute (S= softmax (P) )
                                            8 tisa :: softmax <ve >( s_S , s_P , state );
                                            9 for ( int j = 0; j < Tc ; j ++) {
                                           10 if (j < Tc - 1) {
                                           11 // Load K data ( next tile )
                                           12 tisa :: load_transpose <de >(
                                                       s_K_next , K_next ) ;
                                           13 // Matrix multiply ( next tile )
                                           14 tisa :: gemm <me >( s_S_next , s_Q ,
                                                       s_K_next ) ;
                                           15 }
                                           16 // Load V data
                                           17 tisa :: load < de >( s_V ,V );
                                           18 // Matrix multiply (R =S*V )
                                           19 tisa :: gemm < me >( s_R , s_S , s_V );
                                           20 if (j < Tc - 1) {
                                           21 // Softmax compute ( next tile )
                                           22 tisa :: softmax <ve >( s_S_next ,
                                                       s_S_next , state_next ) ;
                                           23 }
                                           24 // Rescale R data ( O= Rescale (R))
                                           25 tisa :: rescale <ve >( s_O , s_R , state ,
                                                      state_next );
                                           26 // Update next index
                                           27 update_next_index () ;
                                           28 }
                                           29 // Store O data
                                           30 tisa :: store <de >( O , s_O ) ;
```

Fig. 5: Comparison of FlashAttention-3 CUDA and TISA pseudocode. CUDA explicitly manages synchronization (lines 5, 9, 15, 21, 26, 30, 32, and 38), while TISA eliminates all barriers via semantics-aware dependency resolution.

ME/VE/DE. Kernels are shape-parametric and mapped directly to units indicated by OpType, while the upstream compiler (Section VI) automatically generates the corresponding TISA instructions. This separation decouples TISA semantics from hardware-specific implementations, allowing runtime scheduling to remain hardware-agnostic while still leveraging optimized kernels per operator type.

*e) Multi-core execution:* For multi-core execution, the compiler employs spatial partitioning: independent tile groups (e.g., attention heads, batch dimensions) are statically assigned to cores. Each core's local TISA scheduler operates independently via its in-flight semantic tables. Inter-core synchronization uses lightweight NoC signals triggered by shared SRAM bank updates. Runtime load balancing occurs at the software level between kernel invocations, not within the tile scheduler.

