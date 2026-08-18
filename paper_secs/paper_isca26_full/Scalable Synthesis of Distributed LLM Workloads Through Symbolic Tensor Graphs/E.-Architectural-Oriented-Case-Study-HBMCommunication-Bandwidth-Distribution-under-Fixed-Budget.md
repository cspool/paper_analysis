# E. Architectural-Oriented Case Study: HBM/Communication Bandwidth Distribution under Fixed Budget

We demonstrate how STAGE supports architectural design exploration by studying bandwidth partitioning under a fixed off-chip bandwidth budget per accelerator. The total budget is divided between HBM and scale-up interconnect bandwidth. Using STAGE-generated workloads, we sweep HBM bandwidth shares and assign the remaining budget to interconnects.

Fig. 15 reports normalized runtime across multiple total bandwidth budgets for four workload classes: communicationheavy, balanced, memory-heavy, and compute-heavy.

The results highlight three key insights. First, bandwidth provisioning should be workload-aware, as different workloads may prefer different bandwidth distributions. Second, the preferred split is primarily determined by workload characteristics and is relatively insensitive to the total bandwidth budget: while changing the total budget affects overall runtime, the preferred split remains stable. Third, the optimal HBM share consistently exceeds 50%, as most interconnect traffic originates from HBM, while direct communication from on-chip memory is rare due to limited on-chip capacity and ML workload compute patterns.

