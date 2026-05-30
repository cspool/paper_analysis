# F. How to Use DFBM?

DFBM supports two deployment scenarios. *First*, the chiplet supplier integrates DFBM into the chiplet's internal NoC. This enables deadlock-free interconnection with other chiplets

across any interconnect floorplan. Since DFBM is implemented as a standalone module, suppliers save significant development time and engineering labor costs—eliminating the need to redesign tightly integrated chiplet architectures during modifications. *Second*, DFBM acts as a standalone bridging module, inserted by chip designers between target chiplets to enable required deadlock-free connectivity. Here, the chiplet supplier should provide a subset of the configuration parameters mentioned in Section III-E.

#### IV. CREDIT MANAGEMENT

The Credit Management (CM) guarantees deadlock avoidance under worst-case traffic conditions by guaranteeing complete absorption of all packets traversing from chiplets to the interposer. As illustrated in Fig. 5, the CM executes a two-stage process.

![](_page_5_Figure_11.jpeg)

Fig. 5. CM implements a two-stage flow control. In stage 1, the CM confirms the expected credit value. In stage 2, admission arbitration is executed based on credits in stage 1.

In the first stage, the Expected Credit Table decodes coherence transaction types and maps them to predefined credit values. In the second stage, an admission arbitration mechanism compares available credits in the deadlock buffer against the expected credit values. Based on this comparison, the CM either grants permission for packet injection or blocks the packet, ensuring NoC resource allocation aligns with predicted traffic demands. The central challenge resides in constructing the expected credit table, which must encode predictive relationships between coherence transaction types and NoC transmission behaviors.

