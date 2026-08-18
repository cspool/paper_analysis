# Omelet: A Packaging-Aware Hierarchical Interconnect Simulator for 2.5D/3D Chiplet Architectures

Jiho Kim, Danish Baig, Faaiq Waqar, Ashita Victor, Shimeng Yu, Muhannad Bakir, Cong Hao *School of Electrical and Computer Engineering Georgia Institute of Technology*, Atlanta, USA

{jiho.kim, dbaig3, faaiq.waqar, avictor8}@gatech.edu, {shimeng.yu, mbakir, callie.hao}@ece.gatech.edu

*Abstract*—Monolithic integration of SoCs is reaching practical and economic limits, motivating the transition to disaggregated chiplet-based 2.5D/3D systems. In these systems, system performance depends not only on microarchitecture, but is tightly coupled to integration decisions such as stacking, interposers, and bonding methods. These packaging decisions directly shape communication characteristics, including link latency, bandwidth, energy, and I/O density. At the same time, communication is no longer confined to a uniform on-chip network. Instead, data movement traverses multiple interconnect domains, including on-chip wires, lateral interposer links, and vertical die-to-die connections. Since each domain is implemented using different physical technologies, they exhibit distinct electrical and communication characteristics, naturally forming a hierarchical interconnect structure. However, existing simulators rely on packaging-agnostic, decoupled abstractions that fail to capture technology-specific link behavior and cross-layer interactions. This disconnect obscures integration-driven bottlenecks and limits accurate system-level design space exploration.

We present **Omelet**, a packaging-aware hierarchical interconnect simulator for 2.5D/3D chiplet systems. The simulator integrates intra-chiplet networks (Network-on-Chip, NoC), interposer-level communication (Network-on-Interposer, NoI), and vertical tier-to-tier links (Network-on-Layer, NoL) within a unified cycle-level framework, with link parameters derived from electromagnetic extraction and compact SPICE models of advanced packaging interconnects. This unified hierarchy enables architects to evaluate how packaging technologies, network architectures, and chiplet organizations jointly influence latency, congestion, and bottleneck formation in multi-chiplet 2.5D/3D systems. **Omelet**, with documentation and examples, is publicly available on GitHub<sup>1</sup> .

*Index Terms*—chiplet-based systems, 2.5D/3D integration, hierarchical interconnects, packaging-aware simulation, technology-aware modeling, design space exploration

## I. INTRODUCTION

Modern HPC and AI workloads continue to demand increasing compute density and bandwidth, placing sustained pressure on conventional monolithic system-on-chip (SoC) scaling. However, continued scaling faces multiple constraints: transistor scaling is slowing, reticle limits cap maximum die size, and yield degrades sharply with increasing area, collectively limiting further monolithic integration. In addition, integrating all system functions on a single process node introduces fundamental inefficiencies: while digital logic benefits from aggressively scaled nodes, analog and mixed-signal circuits typically favor mature or specialty processes with higher

supply voltages [11], [22], and DRAM is fabricated using processes optimized for dense memory arrays (e.g., HBM). These conflicting requirements make it increasingly impractical to realize high-performance systems within a single monolithic die.

These constraints have driven a shift toward disaggregated chiplet-based integration, where chips are partitioned into multiple dies and interconnected through advanced packaging technologies such as 2.5D and 3D stacking<sup>2</sup> . While this approach improves scalability and enables heterogeneous integration, it also substantially expands the design space. Efficient system design now requires jointly considering chiplet partitioning, logical placement, and the underlying packaging technology that defines interconnect characteristics, making it significantly more challenging to identify optimal design points.

▶ Advanced Packaging Technologies Within this expanded design space, packaging introduces significant variation, with no single solution universally adopted. Instead, a wide range of packaging technologies coexist, each offering distinct trade-offs in electrical reach, I/O density, and vertical connectivity. Commercial 2.5D approaches include TSMC CoWoS [12] and Intel EMIB [31], as well as organic fan-out packages like TSMC InFO [44]. On the 3D side, multi-tier stacks including Intel Foveros [15], AMD 3D V-Cache [50], and Samsung X-Cube [33] employ micro-bump or hybrid bonding to enable dense vertical integration.

While this growing diversity presents new opportunities, it also complicates architectural modeling. Different packaging choices impose distinct electrical and physical constraints, directly shaping achievable bandwidth, latency, and communication behavior at the system level [23], [29]. As a result, architects can no longer rely on a single interconnect

<sup>2</sup>We use '3D' to denote vertically integrated chiplet systems, including '3.5D' configurations where stacked chiplets are integrated on an interposer.

![](_page_0_Figure_16.jpeg)

Fig. 1: Hierarchical communication in 3D structure.

<sup>1</sup>https://github.com/sharc-lab/Omelet

TABLE I: Feature comparison across chiplet and interconnect simulators. Missing features are annotated with L1–L3.

| Feature                       | Omelet<br>(This work) | HISIM [49] | CIMlet [7] | RapidChiplet [13] | SIAM [24] | Gemini [6] | MuchiSim [36] | Kite [3] | CNSim [8] |    |
|-------------------------------|-----------------------|------------|------------|-------------------|-----------|------------|---------------|----------|-----------|----|
| Integration Support           | 2.5D/3D               | 2.5D/3D    | 2.5D/3D    | 2.5D              | 2.5D      | 2.5D       | 2.5D          | 2.5D     | 2.5D      |    |
| Unified Hierarchical Network  | Ë                     | é          | é          | é                 | é         | é          | Partial       | Partial  | Partial   | L1 |
| Physically-Derived Inputs     | Ë                     | é          | é          | é                 | é         | é          | é             | é        | é         | L2 |
| Component-based Link Modeling | Ë                     | é          | é          | é                 | é         | é          | é             | é        | é         | L3 |

L1: NoC, NoI, NoL evaluated independently and summed post hoc; L2: hand-tuned constants or parameters assembled from inconsistent sources; L3: Link modeled as a single abstract link rather than a chain of components

abstraction; instead, system behavior must be understood in the context of the specific integration technology. This challenge becomes more pronounced in emerging open chiplet ecosystems, where a single chiplet design may be deployed across multiple packaging platforms [28]. In such settings, the same chiplet can exhibit different performance characteristics depending on the underlying packaging technology, as interconnect properties fundamentally alter communication bottlenecks and system-level behavior.

▶ Hierarchical Communication in 2.5D/3D Systems These packaging choices directly translate into a multi-layer interconnect hierarchy at the system level, as illustrated in Fig. 1. Within each chiplet, communication is handled by the on-die network-on-chip (NoC), which leverages dense metal routing and short, low-latency wires. Cross-chiplet communication over an interposer is carried by the network-on-interposer (NoI), where signals traverse longer redistribution layers (RDLs) with coarser-pitch wiring. In vertically stacked designs, communication occurs through TSVs or bonding technologies via a network-on-layer (NoL). These layers meet at chiplet boundaries, where traffic transitions across domains with fundamentally different latency, bandwidth, and physical connectivity. This mismatch introduces asymmetric communication paths and creates localized pressure at boundaries, where traffic concentrates and delays emerge. Critically, these effects are not isolated: backpressure and congestion propagate across layers, influencing communication beyond the immediate boundary. As a result, overall system behavior is governed by cross-layer interactions rather than any single interconnect domain in isolation, and models that treat these layers independently fail to capture these system-level effects.

