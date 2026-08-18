# *C. Routing Evolution from PCIe Compatibility to CXL Fabric*

From its inception, CXL was designed to extend the PCIe ecosystem while enabling future scalability. As shown in Figure 2b, the standard preserves backward compatibility by supporting two routing modes: *Hierarchy-Based Routing* (HBR), which mirrors PCIe's traditional host-centered tree structure, and *Port-Based Routing* (PBR), which provides a scalable foundation for fabric-wide routing.

In HBR, routing follows a PCIe-like hierarchy in which each host manages a local *Virtual CXL Switch* (VCS) containing devices under its own *Root Complex* (RC). Each host maintains an independent physical address space, preventing direct sharing of memory across hosts. This design simplifies hardware and software integration and remains compatible with PCIe enumeration, but it confines data to a single host domain and does not support inter-host communication or memory sharing.

PBR fundamentally changes this model. Instead of grouping devices under hosts, each switch port is assigned a unique *Port ID* (PID) and operates as an independent routing endpoint. The switch fabric forms a unified VCS that spans hosts, accelerators, and memory devices, and routing depends on destination PIDs rather than host hierarchy. A *Fabric Manager* (FM) maps PIDs to coherence identifiers and memory regions, creating a global

address and coherence domain that enables direct host-to-host and host-to-memory access at hardware speed. This increases control complexity, but PBR is essential for building large-scale, disaggregated, composable systems.

Mixed HBR/PBR switch configurations were first formalized in the CXL 3.1 specification [50] and remain part of subsequent CXL generations. This coexistence reflects CXL's dual strategy: HBR preserves compatibility with PCIe-based system software, while PBR enables multi-host, fabric-level composition that defines the long-term trajectory of CXL. Although most controllers and switches available in 2024∼2025 still operate primarily in HBR-only mode, this remains a transitional phase. PBR-capable switches will form the basis for future memorycentric and fully disaggregated infrastructures, allowing CXL to evolve from a host-expansion interface into a scalable interconnect fabric.

# *C. Routing Evolution from PCIe Compatibility to CXL Fabric*

From its inception, CXL was designed to extend the PCIe ecosystem while enabling future scalability. As shown in Figure 2b, the standard preserves backward compatibility by supporting two routing modes: *Hierarchy-Based Routing* (HBR), which mirrors PCIe's traditional host-centered tree structure, and *Port-Based Routing* (PBR), which provides a scalable foundation for fabric-wide routing.

In HBR, routing follows a PCIe-like hierarchy in which each host manages a local *Virtual CXL Switch* (VCS) containing devices under its own *Root Complex* (RC). Each host maintains an independent physical address space, preventing direct sharing of memory across hosts. This design simplifies hardware and software integration and remains compatible with PCIe enumeration, but it confines data to a single host domain and does not support inter-host communication or memory sharing.

PBR fundamentally changes this model. Instead of grouping devices under hosts, each switch port is assigned a unique *Port ID* (PID) and operates as an independent routing endpoint. The switch fabric forms a unified VCS that spans hosts, accelerators, and memory devices, and routing depends on destination PIDs rather than host hierarchy. A *Fabric Manager* (FM) maps PIDs to coherence identifiers and memory regions, creating a global

address and coherence domain that enables direct host-to-host and host-to-memory access at hardware speed. This increases control complexity, but PBR is essential for building large-scale, disaggregated, composable systems.

Mixed HBR/PBR switch configurations were first formalized in the CXL 3.1 specification [50] and remain part of subsequent CXL generations. This coexistence reflects CXL's dual strategy: HBR preserves compatibility with PCIe-based system software, while PBR enables multi-host, fabric-level composition that defines the long-term trajectory of CXL. Although most controllers and switches available in 2024∼2025 still operate primarily in HBR-only mode, this remains a transitional phase. PBR-capable switches will form the basis for future memorycentric and fully disaggregated infrastructures, allowing CXL to evolve from a host-expansion interface into a scalable interconnect fabric.

