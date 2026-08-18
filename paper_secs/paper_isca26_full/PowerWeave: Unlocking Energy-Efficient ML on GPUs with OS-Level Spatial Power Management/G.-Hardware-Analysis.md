# *G. Hardware Analysis*

We quantify the silicon area overhead of PowerWeave's fine-grained voltage domains using the cost models detailed in Section VI. Table IV presents the area footprint for a single instance of the required per-domain components, synthesized and scaled to a 5nm process node. These results show that voltage-domain boundary synchronization dominates area overhead at approximately 0.0359 mm<sup>2</sup> per additional domain, about an order of magnitude larger than the Digital LDO controller and clock-generation logic combined (0.0059 mm<sup>2</sup> ). This disparity highlights that the primary cost of spatial DVFS lies not in the regulation or clocking circuits, but in the isolation and synchronization logic required to maintain data integrity between independent frequency domains.

![](_page_12_Figure_0.jpeg)

Fig. 12: Observed vs. requested performance degradation for the full system, without sensitivity, and without live weights.

![](_page_12_Figure_2.jpeg)

Fig. 13: Component area by number of domains.

Figure 13 illustrates how overheads scale as DVFS granularity is increased from coarse (per-GPC) to fine (per-SM). Assuming a silicon area of 1600 mm<sup>2</sup> for a datacenter-class dual-die GPU, even at the finest granularity evaluated (per-SM, 148 domains), the total implementation overhead remains below 0.5% of the total chip area. This analysis confirms that the hardware cost of implementing PowerWeave is minimal.

