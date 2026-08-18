# Spatial Organization of Compute and Distillation Blocks.

To achieve a target throughput of one magic state per cycle per compute block, we group compute blocks with multiple distillation units to form a "super-block." This organization is motivated by the fact that individual distillation protocols often exhibit lower throughput than the consumption rate of a compute block [52]. These super-blocks are distributed around the central memory qubits to minimize movement overhead and routing congestion, while ensuring every compute block

can access any logical qubit via shared ancilla paths. Figure 16 demonstrates this layout for the 18-qubit OFT.

This strategy is informed by the broader principle that modern architectural optimizations have significantly reduced the resource cost per magic state [52]. By treating the factory cost as a primary variable in our volume-minimization strategy rather than a static overhead, TACO can adaptively scale distillation resources. This enables higher throughput and shorter execution times while maintaining a favorable balance between hardware investment and operational speed.

#### F. Physical Realization of the Proposed Architecture

The proposed architecture can be physically realized as a specialized lattice-surgery organization built on standard surface-code patches, similar to prior compact, intermediate, and fast FTQC organizations [51]. The key difference is that our organization is tailored to the high locality exposed by TACO-optimized circuits.

We distinguish between *compute* regions and *memory* regions. This distinction is architectural rather than technological: both regions use the same surface-code substrate, physical qubits, and couplers, but serve different logical roles during execution. Compute regions handle active non-Clifford processing, while memory regions primarily store logical states and support Clifford operations. To support arbitrary  $\pi/4$  rotations, compute regions employ a modified logical patch that has the same code distance d while expanding the physical footprint from the standard  $d \times d$  patch to approximately  $2d \times d$ , requiring roughly  $2 \times$  more physical qubits per logical patch. In contrast, memory regions retain the standard patch structure.

Under this organization, qubits are moved between memory and compute regions by teleporting or swapping the logical state between physical patch locations, while the underlying physical qubits remain unchanged. This can be implemented using standard lattice-surgery primitives.

## V. EVALUATION METHODOLOGY

