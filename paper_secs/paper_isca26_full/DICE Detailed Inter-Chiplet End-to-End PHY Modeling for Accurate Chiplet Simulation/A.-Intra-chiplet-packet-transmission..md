# A. Intra-chiplet packet transmission.

We modify gem5's Python configuration to implement the system layout in Figure 1, modeling an AMD EPYC-style architecture.

CCD/IOD. Each CCD in DICE integrates multiple cores with private L1/L2 caches and a shared LLC. Core-to-core communication within a CCD uses a CCD-local NoC over intra-die links. The default intra-CCD topology is a 2×4 mesh with 8-core (aligned with EPYC "Genoa"). DICE models an IOD that aggregates memory controllers, DMA engines, and I/O. Placed in center (Figure 1), the IOD provides uniform access from CCDs to memory. Internally, we instantiate 4 PHY routers in the IOD in a 2 × 2 mesh, with each PHY connected to 2 memory controllers (in total 8).

Reflecting the use of a less advanced process node for the IOD (*e.g.*, 14 nm *vs.* 5 nm for CCDs), we model a higher NoC frequency for the CCDs than for the IOD. In DICE, intra-CCD links run at 2.0 GHz, are 128 bits wide, and incur 1-cycle router latency and 2-cycle link latency. In contrast, IOD links and routers operate at 1.0 GHz, with the same 128-bit link width, 1-cycle router latency, and 2-cycle link transmission latency. In summary, Table I lists the default parameters of the CCD and IOD models in DICE.

TABLE I: Default for intra-CCD/IOD NoC.

<span id="page-2-0"></span>

|     | Topology                                                                          | Link width | Freq    | Router lat | Link lat |
|-----|-----------------------------------------------------------------------------------|------------|---------|------------|----------|
| CCD | $\begin{array}{c} \text{mesh } 2 \times 4 \\ \text{mesh } 2 \times 2 \end{array}$ | 128-bit    | 2.0 GHz | 1 cycle    | 2 cycles |
| IOD |                                                                                   | 128-bit    | 1.0 GHz | 1 cycle    | 2 cycles |

# A. Intra-chiplet packet transmission.

We modify gem5's Python configuration to implement the system layout in Figure 1, modeling an AMD EPYC-style architecture.

CCD/IOD. Each CCD in DICE integrates multiple cores with private L1/L2 caches and a shared LLC. Core-to-core communication within a CCD uses a CCD-local NoC over intra-die links. The default intra-CCD topology is a 2×4 mesh with 8-core (aligned with EPYC "Genoa"). DICE models an IOD that aggregates memory controllers, DMA engines, and I/O. Placed in center (Figure 1), the IOD provides uniform access from CCDs to memory. Internally, we instantiate 4 PHY routers in the IOD in a 2 × 2 mesh, with each PHY connected to 2 memory controllers (in total 8).

Reflecting the use of a less advanced process node for the IOD (*e.g.*, 14 nm *vs.* 5 nm for CCDs), we model a higher NoC frequency for the CCDs than for the IOD. In DICE, intra-CCD links run at 2.0 GHz, are 128 bits wide, and incur 1-cycle router latency and 2-cycle link latency. In contrast, IOD links and routers operate at 1.0 GHz, with the same 128-bit link width, 1-cycle router latency, and 2-cycle link transmission latency. In summary, Table I lists the default parameters of the CCD and IOD models in DICE.

TABLE I: Default for intra-CCD/IOD NoC.

<span id="page-2-0"></span>

|     | Topology                                                                          | Link width | Freq    | Router lat | Link lat |
|-----|-----------------------------------------------------------------------------------|------------|---------|------------|----------|
| CCD | $\begin{array}{c} \text{mesh } 2 \times 4 \\ \text{mesh } 2 \times 2 \end{array}$ | 128-bit    | 2.0 GHz | 1 cycle    | 2 cycles |
| IOD |                                                                                   | 128-bit    | 1.0 GHz | 1 cycle    | 2 cycles |

