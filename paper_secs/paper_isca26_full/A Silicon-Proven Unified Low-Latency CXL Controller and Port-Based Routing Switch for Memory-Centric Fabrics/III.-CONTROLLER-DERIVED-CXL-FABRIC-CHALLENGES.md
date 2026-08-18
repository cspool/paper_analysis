# III. CONTROLLER-DERIVED CXL FABRIC CHALLENGES

This section discusses how legacy controller architectures introduce latency and scalability bottlenecks in CXL fabrics. It then examines switchless, controller-driven designs proposed to mitigate these issues and analyzes remaining limitations.

## *A. Prevailing View on CXL Performance*

Recent CXL memory expanders are implemented using controller silicon derived from early PCIe-based IP libraries. These IPs accelerated initial adoption and remain the basis for commercially deployed controllers, switches, memory expanders, and CPUs. However, their PCIe lineage imposes hierarchical routing and deep transaction pipelines that diverge from a system-bus-oriented design philosophy, introducing additional latency and limiting topological flexibility.

These architectural constraints appear consistently across published measurements. Because CXL expanders operate within the memory hierarchy, reported latencies vary with

|                | 2025<br>[10] | IPDPS SIGMOD<br>2025<br>[11] | ASPLOS<br>2025<br>[12] | 2025<br>[13] | POMACS ICCE-Asia<br>2024<br>[14] |
|----------------|--------------|------------------------------|------------------------|--------------|----------------------------------|
| HW             | DDR5         | DDR5                         | DDR5                   | DDR5         | DDR5                             |
| Config.        | x2           | N/A                          | x1                     | x1           | N/A                              |
| Test<br>Method | MLC          | MLC                          | MLC, MIO               | MLC          | MLC                              |
| RTT<br>Latency | 266 ns       | 265.2 ns                     | 271–442 ns             | 268 ns       | 285.2 ns                         |

TABLE I: CXL latency.

methodology, yet independent studies [10,12,51] observe substantially high access latency. Table I summarizes prior results [10-14], highlighting variation due to differences in platform, firmware maturity, and measurement approach. Among these studies, [12] in particular analyzed the impact of various realsystem factors on CXL memory latency, including DRAM behaviors such as row buffer misses and host cache capacity. In this work, we performed a projection based on values extracted through digitization of the graphs presented in [12]. The results suggest that, in a 4th Gen Intel Xeon Scalable Processors (Sapphire Rapids, SPR) environment, the p99.9 latency can increase up to 442 ns. This projection reflects the fact that the LLC capacity of SPR is approximately 2.7× smaller than that of 5th Gen Intel Xeon Scalable Processors (Emerald Rapids, EMR). Although these results do not capture all system configurations, they reflect the characteristics of currently available controllers and align across multiple independent evaluations.

When CXL is accessed through switches, these effects become more pronounced. A single expander traverses one controller stack per request and response, whereas a switch traversal typically involves at least two controller pipelines plus routing and fabric-management operations, increasing round-trip latency even without backend DRAM effects. In addition, current commercial controllers predominantly support HBR routing, limiting data sharing and preventing multihost composition. Consequently, most existing systems adopt single-host topologies that use CXL links either to extend local memory capacity [10,14,15,52] or to repurpose stranded memory in virtualized environments [37,53,54].

#### *B. Switchless Design for Multi-Host Memory Expansion*

To address the limitations of single-host memory expansion, the CXL MHD has gained attention as a practical option for multi-host attachment. An MHD integrates several logical interfaces, or *heads*, into a single physical device, with each head connecting independently to different hosts or fabrics. This design removes the earlier CXL restriction that a device could attach to only one host. By allowing multiple hosts to access the same device in parallel, an MHD enables concurrent access and resource pooling, where pooling refers to capacity aggregation rather than data sharing, which is discussed later.

MHDs also offer advantages in multi-tenant environments by simplifying resource isolation and security management [55-

![](_page_3_Figure_7.jpeg)

- (a) Single-headed expansion. (b) Multi-headed expansion.

Fig. 3: Comparison of memory expansion configurations.

57]. Each head operates as an independent logical resource, preventing interference among tenants and maintaining clear management boundaries. This isolation has motivated academic and industrial efforts [56,58,59] to bypass high-latency controller traversal in conventional CXL designs. Originally explored in early switchless configurations, MHD-based designs enable direct host-device paths without intermediate switches, aiming for lower latency and higher aggregate bandwidth across hosts. In such settings, dynamic memory pooling offers capacity flexibility for workload needs while supporting scalable, bandwidth-oriented processing in parallel systems.

Figure 3 compares a conventional *single-headed device* (SHD) with an MHD in a compute node with up to four processors connected through a *non-uniform memory access* (NUMA) topology. As shown in Figure 3a, an SHD allows only one processor to attach directly to the memory expander, forcing others to reach it indirectly through NUMA links.

In contrast, an MHD assigns dedicated heads to individual processors, enabling concurrent access without requiring an intermediate switch (Figure 3b). It also supports dynamic redistribution of memory capacity across processors; memoryintensive workloads can borrow unused capacity from others, reducing localized bottlenecks.

# III. CONTROLLER-DERIVED CXL FABRIC CHALLENGES

This section discusses how legacy controller architectures introduce latency and scalability bottlenecks in CXL fabrics. It then examines switchless, controller-driven designs proposed to mitigate these issues and analyzes remaining limitations.

## *A. Prevailing View on CXL Performance*

Recent CXL memory expanders are implemented using controller silicon derived from early PCIe-based IP libraries. These IPs accelerated initial adoption and remain the basis for commercially deployed controllers, switches, memory expanders, and CPUs. However, their PCIe lineage imposes hierarchical routing and deep transaction pipelines that diverge from a system-bus-oriented design philosophy, introducing additional latency and limiting topological flexibility.

These architectural constraints appear consistently across published measurements. Because CXL expanders operate within the memory hierarchy, reported latencies vary with

|                | 2025<br>[10] | IPDPS SIGMOD<br>2025<br>[11] | ASPLOS<br>2025<br>[12] | 2025<br>[13] | POMACS ICCE-Asia<br>2024<br>[14] |
|----------------|--------------|------------------------------|------------------------|--------------|----------------------------------|
| HW             | DDR5         | DDR5                         | DDR5                   | DDR5         | DDR5                             |
| Config.        | x2           | N/A                          | x1                     | x1           | N/A                              |
| Test<br>Method | MLC          | MLC                          | MLC, MIO               | MLC          | MLC                              |
| RTT<br>Latency | 266 ns       | 265.2 ns                     | 271–442 ns             | 268 ns       | 285.2 ns                         |

TABLE I: CXL latency.

methodology, yet independent studies [10,12,51] observe substantially high access latency. Table I summarizes prior results [10-14], highlighting variation due to differences in platform, firmware maturity, and measurement approach. Among these studies, [12] in particular analyzed the impact of various realsystem factors on CXL memory latency, including DRAM behaviors such as row buffer misses and host cache capacity. In this work, we performed a projection based on values extracted through digitization of the graphs presented in [12]. The results suggest that, in a 4th Gen Intel Xeon Scalable Processors (Sapphire Rapids, SPR) environment, the p99.9 latency can increase up to 442 ns. This projection reflects the fact that the LLC capacity of SPR is approximately 2.7× smaller than that of 5th Gen Intel Xeon Scalable Processors (Emerald Rapids, EMR). Although these results do not capture all system configurations, they reflect the characteristics of currently available controllers and align across multiple independent evaluations.

When CXL is accessed through switches, these effects become more pronounced. A single expander traverses one controller stack per request and response, whereas a switch traversal typically involves at least two controller pipelines plus routing and fabric-management operations, increasing round-trip latency even without backend DRAM effects. In addition, current commercial controllers predominantly support HBR routing, limiting data sharing and preventing multihost composition. Consequently, most existing systems adopt single-host topologies that use CXL links either to extend local memory capacity [10,14,15,52] or to repurpose stranded memory in virtualized environments [37,53,54].

#### *B. Switchless Design for Multi-Host Memory Expansion*

To address the limitations of single-host memory expansion, the CXL MHD has gained attention as a practical option for multi-host attachment. An MHD integrates several logical interfaces, or *heads*, into a single physical device, with each head connecting independently to different hosts or fabrics. This design removes the earlier CXL restriction that a device could attach to only one host. By allowing multiple hosts to access the same device in parallel, an MHD enables concurrent access and resource pooling, where pooling refers to capacity aggregation rather than data sharing, which is discussed later.

MHDs also offer advantages in multi-tenant environments by simplifying resource isolation and security management [55-

![](_page_3_Figure_7.jpeg)

- (a) Single-headed expansion. (b) Multi-headed expansion.

Fig. 3: Comparison of memory expansion configurations.

57]. Each head operates as an independent logical resource, preventing interference among tenants and maintaining clear management boundaries. This isolation has motivated academic and industrial efforts [56,58,59] to bypass high-latency controller traversal in conventional CXL designs. Originally explored in early switchless configurations, MHD-based designs enable direct host-device paths without intermediate switches, aiming for lower latency and higher aggregate bandwidth across hosts. In such settings, dynamic memory pooling offers capacity flexibility for workload needs while supporting scalable, bandwidth-oriented processing in parallel systems.

Figure 3 compares a conventional *single-headed device* (SHD) with an MHD in a compute node with up to four processors connected through a *non-uniform memory access* (NUMA) topology. As shown in Figure 3a, an SHD allows only one processor to attach directly to the memory expander, forcing others to reach it indirectly through NUMA links.

In contrast, an MHD assigns dedicated heads to individual processors, enabling concurrent access without requiring an intermediate switch (Figure 3b). It also supports dynamic redistribution of memory capacity across processors; memoryintensive workloads can borrow unused capacity from others, reducing localized bottlenecks.

