# B. Research Questions

original rule.

Our goal is to turn the classical rule from a heuristic into a basis for concrete and actionable framework, guided by the following three questions:

denote amortized capital cost (CapEx), consistent with Gray's

- **RQ1** (calibrated economics). Retain the economic view but make it realistic by explicitly modeling host resource usage and first-principles SSD behavior. How does the break-even interval change under this calibrated model?
- **RQ2** (constraint-aware refinement). Add feasibility constraints (i.e., processor I/O capacity and application latency targets). How do they reshape the break-even interval, and when do they become the primary constraint?
- **RQ3** (platform viability and guidance). Integrate DRAM bandwidth/capacity limits and workload's access-interval profile. Can a unified framework fusing *economics*, *workload*, and *hardware constraints* assess viability and optimality, and, when needed, recommend upgrades?

Findings from these questions indicate that the DRAM-flash caching threshold has collapsed into the seconds regime due to

the drastic elevation of IOPS/\$ of storage drives. As a result, the long-standing boundary between memory and storage has blurred, leading to the following research question:

• **RQ4** (software re-think). As the DRAM-flash threshold drops to seconds, how should we rethink data-intensive software, and what principles should guide the redesign of data structures, access paths, and scheduling to fully exploit this new regime for throughput, efficiency, scalability, and cost?

Addressing these four questions shapes the remainder of this paper. It establishes a unified economics/feasibility framework with interpretable metrics for provisioning and upgrades, and it opens a principled design-space exploration under seconds-scale DRAM-flash caching.

#### C. Discussion of Assumptions and Scope

All the modeling parameters in this study are derived from mature NAND flash technology and established roadmaps, in contrast to prior explorations that hypothesize active-memory roles for emerging NVMs. We model controller translation bandwidth and PCIe packet/bandwidth limits explicitly; in our evaluated configurations we provision these to be non-limiting, so the dominant bounds arise from NAND/channel physics and host capacity. Our goal is not to forecast product specifics, but to examine how feasibility and provisioning change once the full IOPS potential of NAND flash is unleashed. The framework is forward-looking yet physically grounded, and can be re-parameterized as devices and standards evolve.

Analytical vs. simulation components. For clarity, we explicitly distinguish the analytical and simulation roles in this work. Sections III, IV, and V develop a closed-form, first-principles framework that derives break-even intervals, feasibility bounds, and platform-viability thresholds from device timing, host IOPS limits, DRAM bandwidth/capacity constraints, and workload access-interval profiles. Section VI presents MQSim-Next, which models NAND timing, multiplane concurrency, ECC behavior, and channel scheduling to characterize realistic device-level IOPS and latency trends. Section VII then integrates these components: MQSim-Next provides calibrated device behavior, while the analytical framework determines usable IOPS, break-even thresholds, and system-level feasibility for the case-study evaluations.

