# *E. Risk Mitigation for Non-Migratable Memory*

The introduction of a tiered memory architecture with CXL expansion brings new challenges in managing non-migratable memory allocations. Certain types of kernel data, such as slab caches, page tables, and other immovable structures, must remain accessible at all times and cannot be safely migrated between memory tiers. Placing such allocations in CXL memory, which may be subject to higher latency or transient errors, poses a risk to system stability and reliability.

To mitigate this risk, we configure CXL memory as zone\_movable within the Linux kernel. This restricts the placement of non-migratable pages: only user-space memory and migratable kernel data are allocated in CXL-attached memory. All critical kernel structures remain in local DRAM, where they benefit from lower latency and higher reliability.

This design decision enhances the effectiveness of TPP, as it helps ensure the OS can freely migrate hot user pages from CXL to local memory without concern for immovable kernel data being stranded in the slower tier. It also simplifies memory management, as the kernel can enforce clear boundaries between migratable and non-migratable allocations.

The software stack further supports validation and monitoring of memory placement policies, providing visibility into the distribution of kernel and user pages across memory tiers. This enables operators to verify that non-migratable allocations are correctly confined to local DRAM and to detect any anomalies that may arise due to misconfiguration or software bugs.

