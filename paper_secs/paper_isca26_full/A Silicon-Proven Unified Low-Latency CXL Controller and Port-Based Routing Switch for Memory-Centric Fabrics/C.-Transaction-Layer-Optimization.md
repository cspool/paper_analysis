# *C. Transaction Layer Optimization*

The transaction layer orchestrates requests across *CXL.io*, *CXL.cache*, and *CXL.mem*, ensuring semantic consistency while determining which messages are issued to the link layer. Whereas the link layer packages and transmits flits under predefined rules, this layer governs message ordering and grouping to sustain continuous data flow through the pipeline.

In PCIe-derived architectures, each protocol maintains its own queue structure. Although modular, this separation leads to queue contention, uneven priority handling, and long processing delays under mixed traffic. As shown in Figure 7, the proposed controller replaces this fragmented organization with a unified flow-control and scheduling engine that coordinates dispatch across all protocol domains. The scheduler can consider traffic patterns to regulate dispatch behavior, while ensuring fairness is provided between each message class. This adaptivity minimizes idle cycles and maintains consistent utilization of the downstream pipeline.

Silicon evaluation shows up to 1.3× higher throughput compared with conventional per-protocol queuing. These indicate that the transaction layer acts not only as a command

![](_page_5_Figure_10.jpeg)

- (a) PCIe-based integration. (b) Unified integration.

Fig. 8: Comparison of cross-layer integration.

processor but as a coordinated control point for fabric-wide data movement, and together with the streamlined link layer forms a synchronized pipeline enabling deterministic, low-latency communication.

## *D. Cross-Layer Integration*

While per-layer refinements improve localized behavior, overall controller latency is influenced by interface delays between protocol stages. As shown in Figure 8a, conventional PCIe-derived designs treat each layer as an isolated pipeline, requiring synchronization and handshaking at every boundary. These transitions introduce idle cycles and timing misalignment, contributing a substantial portion of round-trip delays.

In contrast, the proposed controller reduces this overhead by redefining layer boundaries and forming a unified data path across the physical, link, and transaction layers (Figure 8b). Each layer maintains its protocol-specific responsibilities but operates under a shared buffering and timing framework, allowing data to advance without explicit inter-layer synchronization. This structure removes redundant staging and handshake delays that limited throughput. Control metadata and packet data are processed in parallel, and a unified timing reference ensures stable propagation across the pipeline.

The controller also incorporates cooperative feedback among layers. The physical layer tracks link activity to regulate data release, and the transaction layer considers link utilization when selecting messages, enabling the overall pipeline to self-regulate under varying traffic conditions.

Silicon evaluation shows that this cross-layer integration reduces round-trip latency to below 50 ns, improves link bandwidth by twenty five percent, and decreases latency variation under bursty workloads. These results demonstrate a shift from a layered protocol stack to a unified communication fabric that delivers deterministic, low-latency operation and enhances scalability for memory-centric systems.

# *C. Transaction Layer Optimization*

The transaction layer orchestrates requests across *CXL.io*, *CXL.cache*, and *CXL.mem*, ensuring semantic consistency while determining which messages are issued to the link layer. Whereas the link layer packages and transmits flits under predefined rules, this layer governs message ordering and grouping to sustain continuous data flow through the pipeline.

In PCIe-derived architectures, each protocol maintains its own queue structure. Although modular, this separation leads to queue contention, uneven priority handling, and long processing delays under mixed traffic. As shown in Figure 7, the proposed controller replaces this fragmented organization with a unified flow-control and scheduling engine that coordinates dispatch across all protocol domains. The scheduler can consider traffic patterns to regulate dispatch behavior, while ensuring fairness is provided between each message class. This adaptivity minimizes idle cycles and maintains consistent utilization of the downstream pipeline.

Silicon evaluation shows up to 1.3× higher throughput compared with conventional per-protocol queuing. These indicate that the transaction layer acts not only as a command

![](_page_5_Figure_10.jpeg)

- (a) PCIe-based integration. (b) Unified integration.

Fig. 8: Comparison of cross-layer integration.

processor but as a coordinated control point for fabric-wide data movement, and together with the streamlined link layer forms a synchronized pipeline enabling deterministic, low-latency communication.

## *D. Cross-Layer Integration*

While per-layer refinements improve localized behavior, overall controller latency is influenced by interface delays between protocol stages. As shown in Figure 8a, conventional PCIe-derived designs treat each layer as an isolated pipeline, requiring synchronization and handshaking at every boundary. These transitions introduce idle cycles and timing misalignment, contributing a substantial portion of round-trip delays.

In contrast, the proposed controller reduces this overhead by redefining layer boundaries and forming a unified data path across the physical, link, and transaction layers (Figure 8b). Each layer maintains its protocol-specific responsibilities but operates under a shared buffering and timing framework, allowing data to advance without explicit inter-layer synchronization. This structure removes redundant staging and handshake delays that limited throughput. Control metadata and packet data are processed in parallel, and a unified timing reference ensures stable propagation across the pipeline.

The controller also incorporates cooperative feedback among layers. The physical layer tracks link activity to regulate data release, and the transaction layer considers link utilization when selecting messages, enabling the overall pipeline to self-regulate under varying traffic conditions.

Silicon evaluation shows that this cross-layer integration reduces round-trip latency to below 50 ns, improves link bandwidth by twenty five percent, and decreases latency variation under bursty workloads. These results demonstrate a shift from a layered protocol stack to a unified communication fabric that delivers deterministic, low-latency operation and enhances scalability for memory-centric systems.

