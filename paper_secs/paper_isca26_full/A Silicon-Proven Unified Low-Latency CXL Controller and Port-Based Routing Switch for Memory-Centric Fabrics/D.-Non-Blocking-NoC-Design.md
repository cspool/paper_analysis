# *D. Non-Blocking NoC Design*

Figure 14b illustrates the internal *Network-on-Chip* (NoC) that enables non-blocking data transfers among all ports. Each

![](_page_8_Figure_11.jpeg)

- (a) Controller. (b) Pipeline. (c) Chip micrograph of Switch.

Fig. 15: Floorplan and chip micrograph of CXL Switch.

port can issue ingress and egress transactions concurrently, allowing the switch to sustain parallel traffic without contention.

The NoC provides a high-bandwidth on-chip communication fabric that dynamically connects ports according to routing decisions. It maintains timing consistency across processing paths and preserves predictable behavior under high port density and bursty workloads. Operating within a unified timing domain ensures that concurrent traffic does not cause misalignment, keeping latency stable under varying load.

By combining this NoC fabric with the hardware-automated routing pipeline described earlier, the switch maintains uniform per-hop delay and avoids the latency growth commonly observed in multi-port designs. As a result, the switch operates as a fixed-latency datapath across large, multi-tier CXL fabrics, advancing the goal of a composable, memory-centric interconnect with deterministic performance and seamless scalability.

# *D. Non-Blocking NoC Design*

Figure 14b illustrates the internal *Network-on-Chip* (NoC) that enables non-blocking data transfers among all ports. Each

![](_page_8_Figure_11.jpeg)

- (a) Controller. (b) Pipeline. (c) Chip micrograph of Switch.

Fig. 15: Floorplan and chip micrograph of CXL Switch.

port can issue ingress and egress transactions concurrently, allowing the switch to sustain parallel traffic without contention.

The NoC provides a high-bandwidth on-chip communication fabric that dynamically connects ports according to routing decisions. It maintains timing consistency across processing paths and preserves predictable behavior under high port density and bursty workloads. Operating within a unified timing domain ensures that concurrent traffic does not cause misalignment, keeping latency stable under varying load.

By combining this NoC fabric with the hardware-automated routing pipeline described earlier, the switch maintains uniform per-hop delay and avoids the latency growth commonly observed in multi-port designs. As a result, the switch operates as a fixed-latency datapath across large, multi-tier CXL fabrics, advancing the goal of a composable, memory-centric interconnect with deterministic performance and seamless scalability.

