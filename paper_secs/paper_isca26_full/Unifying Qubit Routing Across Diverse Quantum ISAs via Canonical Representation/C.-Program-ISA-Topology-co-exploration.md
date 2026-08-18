# *C. Program-ISA-Topology co-exploration*

Our evaluation also systematically explores how program patterns, ISA selection, and hardware topologies impact each other. We highlight some co-design guidelines particularly according to results achieved by CANOPUS (Table [IV,](#page-9-0) Fig. [11\)](#page-10-0):

- *Topology-program affinity matters more than raw connectivity:* Heavy-hex topology consistently incurs higher routing overhead across all ISAs, despite having higher average connectivity. This is because most quantum algorithms are constructed in a subroutine-unrolling approach, naturally more friendly to chain topology. The QFT kernel detailed in Section [V-A](#page-7-5) is a thorough good example.
- *Heterogeneous ISAs yield disproportionate gains:* Combining CX-family and iSWAP-family gates into Het provides

![](_page_10_Figure_0.jpeg)

<span id="page-10-0"></span>Fig. 11. Routing overhead in terms of (a) C<sub>count</sub> and (b) C<sub>depth</sub> for different compilers across various device topologies and quantum ISAs.

substantially greater routing overhead reduction than either family alone. On 1D chain under Canopus, ZZPhase reduces count overhead by 9.6% and SQiSW by 7.9% relative to CX, while Het achieves a 23.9% reduction. The same amplified effect holds across other topologies, indicating that the two gate families address complementary routing scenarios, enabling Canopus to select the most efficient decomposition in each SWAP insertion context. This benefit is more pronounced for circuits largely containing CX/CZ as 2Q blocks, such as qec9.

- Gate mirroring is another approach to designing powerful quantum ISAs: Both ZZPhase and SQiSW achieve comparable results to Het, since mirror gates naturally enable low-overhead SWAP absorption, that is, SWAP mirroring.
- ISA selection should be program-aware: For Hamiltonian simulation programs like ising, ZZPhase ISA is essential to improve execution performance. Therein multiple Ising gates (i.e., 2-local Pauli rotations equivalent to XX(θ)) are included. As a discrete fractional XX(θ) basis gate set, ZZPhase ISA inherently aligns better with these workloads than other gate families, significantly boosting execution performance. Besides, the commutation patterns (the fourth pattern in Fig. 7(b)) occurring in ising can be effectively identified in the canonical form and the commutativity-optimization mechanism plays a critical role in routing (see Fig. 13, Table V and Section VI-F for further discussion). While circuits dominated by CX/CZ blocks (e.g., qec9) benefit more from heterogeneous ISAs in which both

