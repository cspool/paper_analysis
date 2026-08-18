# *A. Hardware Architecture with Redundant Interconnects*

Basic Wafer-Scale Architecture and Component-Level Faults. The basic wafer-scale architecture is fabricated as a near-wafer-sized chip by printing multiple reticle-scale dies and stitching them across reticle boundaries, enabling die-todie communication and forming a regular 2D grid of cores separated by scribe lines. Fig.7 illustrates this hierarchy, where each core consists of a PE and a router, multiple cores form a die, and stitched dies together constitute a wafer-scale chip on a single wafer. We consider component-level faults at the granularity of PEs (affecting computation), routers, and links (affecting communication).

Redundant Interconnect Architecture and Routing Assumption. ConBIN automatically selects a redundant interconnect design at design time to best approximate the expected topology under various wafer-scale fault distributions (Sec.V-C), and this design is bounded by process-feasible wiring and layout constraints to ensure manufacturability. Fig.8(a) shows the microarchitecture that implements this redundancy template, which is uniformly applied to all cores. Each router in ConBIN's redundant architecture retains its five logical ports but extends their physical connectivity to support multiple candidate interconnects per direction. For the four cardinal directions, redundant interconnects can connect a router not only to its immediate neighbor but also to routers located several hops away, depending on the redundancy configuration via multiplexers. Similarly, the router-toPE interface is also enhanced with redundancy. In contrast to the baseline design where each PE is only attached to a single router, our redundant architecture allows a PE to have multiple candidate router connections for fault tolerance. After postfabrication repair, however, exactly one PE–router connection is activated, and the selected pair forms the active core used for routing and scheduling.

Once links are configured after fabrication, routing paths are determined using a shortest path algorithm with the constraint of prohibited turns [50], which ensures cycle-free and globally connected networks even under irregular topologies caused by defects.

### *B. Design Metrics for Approximating Mesh Topology*

Despite redundant interconnects, spatially correlated defects make it infeasible to maintain a perfect mesh topology at wafer scale. ConBIN instead targets a near-mesh structure and defines quantitative metrics to assess router connectivity uniformity and PE accessibility relative to the ideal mesh.

Metric 1: Lower-Tail Average Router Degree. In a faultfree mesh, each router connects to four neighbors, but faults may remove links and create uneven connectivity. While the average router degree reflects global connectivity, it fails to capture localized regions with the lowest connectivity, where traffic is forced to detour and concentrate. These regions form structural bottlenecks, and communication delay grows rapidly as contention intensifies. To enhance the resilience of the redundant architecture under diverse fault distributions, we model the weakest regions that most affect communication and define the lower-tail average router degree Dlow, which measures the average connectivity among the least connected α fraction of fault-free routers (α ∈ (0, 1]):

$$D_{low} = \frac{1}{\alpha N_r} \sum_{i \in R_{low}} d_i \tag{2}$$

where N<sup>r</sup> is the number of fault-free routers, d<sup>i</sup> is the degree (number of fault-free inter-router links) of router i, and Rlow represents the bottom αN<sup>r</sup> routers by degree. A higher Dlow indicates that even the weakest routers retain sufficient connectivity, suggesting a topology that more closely approximates the uniformity of a regular mesh.

Metric 2: Accessible PE Ratio. Beyond router connectivity, each PE must remain attached to at least one reliable router. We therefore define the accessible PE ratio RP E, which quantifies the fraction of fault-free PEs connected to a router whose degree exceeds a threshold β:

$$R_{PE} = \frac{1}{N_{PE}} \sum_{j=1}^{N_{PE}} \mathbf{1} \left( \exists r \in \mathcal{N}(j) \text{ s.t. } d_r > \beta \right)$$
 (3)

where NP E is the number of fault-free PEs and N (j) is the set of routers reachable by PE j. A higher RP E indicates a higher potential of more usable compute resources and higher available parallelism after repaired.

Together, Dlow and RP E provide a structural abstraction of the redundant design after repair: Dlow reflects its communication bottleneck resilience and RP E reflects its compute

![](_page_6_Figure_0.jpeg)

Fig. 8. (a) Microarchitecture of a router with redundant interconnects. (b) Redundancy design space with illustrative examples. (c) Eventual redundancy design derived by ConBIN for 128×136 chip scale.

salvage capacity. Both metrics serve as optimization objectives in ConBIN's automated redundancy design.

# *A. Hardware Architecture with Redundant Interconnects*

Basic Wafer-Scale Architecture and Component-Level Faults. The basic wafer-scale architecture is fabricated as a near-wafer-sized chip by printing multiple reticle-scale dies and stitching them across reticle boundaries, enabling die-todie communication and forming a regular 2D grid of cores separated by scribe lines. Fig.7 illustrates this hierarchy, where each core consists of a PE and a router, multiple cores form a die, and stitched dies together constitute a wafer-scale chip on a single wafer. We consider component-level faults at the granularity of PEs (affecting computation), routers, and links (affecting communication).

Redundant Interconnect Architecture and Routing Assumption. ConBIN automatically selects a redundant interconnect design at design time to best approximate the expected topology under various wafer-scale fault distributions (Sec.V-C), and this design is bounded by process-feasible wiring and layout constraints to ensure manufacturability. Fig.8(a) shows the microarchitecture that implements this redundancy template, which is uniformly applied to all cores. Each router in ConBIN's redundant architecture retains its five logical ports but extends their physical connectivity to support multiple candidate interconnects per direction. For the four cardinal directions, redundant interconnects can connect a router not only to its immediate neighbor but also to routers located several hops away, depending on the redundancy configuration via multiplexers. Similarly, the router-toPE interface is also enhanced with redundancy. In contrast to the baseline design where each PE is only attached to a single router, our redundant architecture allows a PE to have multiple candidate router connections for fault tolerance. After postfabrication repair, however, exactly one PE–router connection is activated, and the selected pair forms the active core used for routing and scheduling.

Once links are configured after fabrication, routing paths are determined using a shortest path algorithm with the constraint of prohibited turns [50], which ensures cycle-free and globally connected networks even under irregular topologies caused by defects.

### *B. Design Metrics for Approximating Mesh Topology*

Despite redundant interconnects, spatially correlated defects make it infeasible to maintain a perfect mesh topology at wafer scale. ConBIN instead targets a near-mesh structure and defines quantitative metrics to assess router connectivity uniformity and PE accessibility relative to the ideal mesh.

Metric 1: Lower-Tail Average Router Degree. In a faultfree mesh, each router connects to four neighbors, but faults may remove links and create uneven connectivity. While the average router degree reflects global connectivity, it fails to capture localized regions with the lowest connectivity, where traffic is forced to detour and concentrate. These regions form structural bottlenecks, and communication delay grows rapidly as contention intensifies. To enhance the resilience of the redundant architecture under diverse fault distributions, we model the weakest regions that most affect communication and define the lower-tail average router degree Dlow, which measures the average connectivity among the least connected α fraction of fault-free routers (α ∈ (0, 1]):

$$D_{low} = \frac{1}{\alpha N_r} \sum_{i \in R_{low}} d_i \tag{2}$$

where N<sup>r</sup> is the number of fault-free routers, d<sup>i</sup> is the degree (number of fault-free inter-router links) of router i, and Rlow represents the bottom αN<sup>r</sup> routers by degree. A higher Dlow indicates that even the weakest routers retain sufficient connectivity, suggesting a topology that more closely approximates the uniformity of a regular mesh.

Metric 2: Accessible PE Ratio. Beyond router connectivity, each PE must remain attached to at least one reliable router. We therefore define the accessible PE ratio RP E, which quantifies the fraction of fault-free PEs connected to a router whose degree exceeds a threshold β:

$$R_{PE} = \frac{1}{N_{PE}} \sum_{j=1}^{N_{PE}} \mathbf{1} \left( \exists r \in \mathcal{N}(j) \text{ s.t. } d_r > \beta \right)$$
 (3)

where NP E is the number of fault-free PEs and N (j) is the set of routers reachable by PE j. A higher RP E indicates a higher potential of more usable compute resources and higher available parallelism after repaired.

Together, Dlow and RP E provide a structural abstraction of the redundant design after repair: Dlow reflects its communication bottleneck resilience and RP E reflects its compute

![](_page_6_Figure_0.jpeg)

Fig. 8. (a) Microarchitecture of a router with redundant interconnects. (b) Redundancy design space with illustrative examples. (c) Eventual redundancy design derived by ConBIN for 128×136 chip scale.

salvage capacity. Both metrics serve as optimization objectives in ConBIN's automated redundancy design.

