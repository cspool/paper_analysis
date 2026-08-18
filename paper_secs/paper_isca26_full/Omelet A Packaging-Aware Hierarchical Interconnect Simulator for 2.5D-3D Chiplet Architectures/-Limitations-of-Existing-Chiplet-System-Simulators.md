# ▶ Limitations of Existing Chiplet System Simulators

Despite the tightly coupled and packaging-dependent nature of communication in 2.5D/3D systems, existing chiplet simulators rely on abstractions that overlook critical system-level effects:

- L1) Decoupled Cross-Layer Communication Modeling: Existing simulators treat NoC, NoI, and NoL as separate communication domains, evaluating them independently and combining their delays post hoc. This fragmented approach ignores the fact that communication spans these layers continuously, missing congestion and interaction effects that arise at chiplet boundaries.
- L2) Technology-Unconstrained Manual Parameterization: Many simulators require users to specify latency and bandwidth as hand-tuned constants, or assemble link parameters from disparate published sources, such as wire RC from one reference and driver energy from another,

that correspond to different physical structures and assumptions. In both cases, the resulting link configurations are not internally consistent and may not correspond to any realizable physical implementation. This limits meaningful design comparisons and places a significant burden on architects to determine physically plausible values.

L3) Monolithic Assumptions of Die-to-Die Interconnects : Existing simulators model off-chip communication as uniform links with latency scaling only with distance, effectively extending on-chip assumptions across chiplet boundaries. In practice, die-to-die communication is composed of multiple physical components, including I/O drivers, interposer wiring, TSVs, and bonding interfaces, each contributing distinct delay and bandwidth constraints. Collapsing these components into a single link abstraction and modeling latency as a function of distance alone masks their non-uniform contributions and misrepresents latency and bandwidth.

These limitations confine architectural exploration to simplified or unrealizable configurations, preventing accurate evaluation of system behavior under realistic packaging constraints.

