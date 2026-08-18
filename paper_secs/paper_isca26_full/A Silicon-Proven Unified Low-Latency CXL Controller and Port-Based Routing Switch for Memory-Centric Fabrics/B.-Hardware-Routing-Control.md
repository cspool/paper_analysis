# *B. Hardware Routing Control*

After format translation, the routing stage determines the egress path for each packet, as illustrated in Figure 11. The switch uses two hardware-managed structures to guide forwarding decisions: a routing table that resolves the DPID to an output port, and a grouping structure that coordinates load distribution among ports that share routing equivalence. Together, these structures form a localized hardware routing plane that operates independently of firmware.

The DPID-based routing table provides fixed-latency nexthop lookup and incorporates link-status information so that forwarding reflects current connectivity. Entries update through internal topology synchronization, allowing the routing logic to handle link additions or failures without software.

Above this, the routing-group structure monitors traffic conditions across ports within the same group and selects among them according to congestion-aware policies. By referencing lightweight hardware metrics, the logic balances utilization and preserves throughput under varying load patterns.

Both structures support concurrent lookup and update operations, enabling forwarding decisions to be made without stalling the datapath. Routing resolution and group arbitration complete within a tightly bounded latency budget, ensuring predictable behavior even under high traffic. Combined with the upstream conversion pipeline, this hardware-managed routing stage maintains consistent per-hop delay and stable throughput across multi-hop CXL fabric configurations.

#### *C. Hardware Scheduling and Deterministic Operation*

The conversion and routing flow is realized as a unified hardware pipeline with deterministic processing stages. As illustrated in Figure 12a, each incoming packet carries minimal routing context that enables the hardware scheduler to coordinate subsequent stages. The scheduling logic manages egress selection and routing decisions according to the availability of downstream resources, allowing packet forwarding to proceed without firmware or interrupt-driven control.

Synchronization across adjacent CXL switches is supported by a fabric-level timing mechanism, shown in Figure 12b,

![](_page_7_Figure_0.jpeg)

- (a) Deterministic scheduling. (b) Fabric synchronization.

Fig. 12: Hardware scheduling for conversion and routing flow.

which maintains alignment for multi-hop operation. Routing structures and timing state update dynamically without halting in-flight traffic, enabling runtime topology adjustments and link recovery while sustaining continuous packet flow.

From ingress to egress, the sequence of decoding, translation, lookup, and arbitration completes within a tightly bounded latency window. This predictable timing behavior ensures stable performance under full load and maintains congestion tolerance in multi-hop, multi-switch configurations. Because all scheduling and arbitration occur in hardware, the switch operates as a deterministic element within the CXL fabric rather than relying on software intervention.

Independent processing stages handle header interpretation, routing resolution, and port assignment concurrently without stalling. Hardware evaluation shows that the scheduler dispatches packets as soon as resources become available, reinforcing the pipeline's deterministic behavior.

# *B. Hardware Routing Control*

After format translation, the routing stage determines the egress path for each packet, as illustrated in Figure 11. The switch uses two hardware-managed structures to guide forwarding decisions: a routing table that resolves the DPID to an output port, and a grouping structure that coordinates load distribution among ports that share routing equivalence. Together, these structures form a localized hardware routing plane that operates independently of firmware.

The DPID-based routing table provides fixed-latency nexthop lookup and incorporates link-status information so that forwarding reflects current connectivity. Entries update through internal topology synchronization, allowing the routing logic to handle link additions or failures without software.

Above this, the routing-group structure monitors traffic conditions across ports within the same group and selects among them according to congestion-aware policies. By referencing lightweight hardware metrics, the logic balances utilization and preserves throughput under varying load patterns.

Both structures support concurrent lookup and update operations, enabling forwarding decisions to be made without stalling the datapath. Routing resolution and group arbitration complete within a tightly bounded latency budget, ensuring predictable behavior even under high traffic. Combined with the upstream conversion pipeline, this hardware-managed routing stage maintains consistent per-hop delay and stable throughput across multi-hop CXL fabric configurations.

#### *C. Hardware Scheduling and Deterministic Operation*

The conversion and routing flow is realized as a unified hardware pipeline with deterministic processing stages. As illustrated in Figure 12a, each incoming packet carries minimal routing context that enables the hardware scheduler to coordinate subsequent stages. The scheduling logic manages egress selection and routing decisions according to the availability of downstream resources, allowing packet forwarding to proceed without firmware or interrupt-driven control.

Synchronization across adjacent CXL switches is supported by a fabric-level timing mechanism, shown in Figure 12b,

![](_page_7_Figure_0.jpeg)

- (a) Deterministic scheduling. (b) Fabric synchronization.

Fig. 12: Hardware scheduling for conversion and routing flow.

which maintains alignment for multi-hop operation. Routing structures and timing state update dynamically without halting in-flight traffic, enabling runtime topology adjustments and link recovery while sustaining continuous packet flow.

From ingress to egress, the sequence of decoding, translation, lookup, and arbitration completes within a tightly bounded latency window. This predictable timing behavior ensures stable performance under full load and maintains congestion tolerance in multi-hop, multi-switch configurations. Because all scheduling and arbitration occur in hardware, the switch operates as a deterministic element within the CXL fabric rather than relying on software intervention.

Independent processing stages handle header interpretation, routing resolution, and port assignment concurrently without stalling. Hardware evaluation shows that the scheduler dispatches packets as soon as resources become available, reinforcing the pipeline's deterministic behavior.

