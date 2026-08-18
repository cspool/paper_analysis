# *A. UVM-Enabled Multi-GPU System*

This paper targets UVM-managed, discrete multi-GPU systems in which several GPUs are connected by high-bandwidth links such as PCIe [31] and NVLink [15]. Figure 1 (a) illustrates the baseline architecture. Each GPU comprises multiple streaming multiprocessors (SMs). The L1 TLB is shared between the two SMs within each Texture Processing Cluster (TPC), and the L2 TLB is shared across the SMs of a GPU Processing Cluster (GPC). The L3 TLB is shared across all GPCs. A discrete GPU typically maintains its own local memory and page tables, with page-table walks handled by the GPU Memory Management Unit (GMMU). In UVM-managed multi-GPU systems, the CPU-side UVM driver handles GPU far-faults and maintains a centralized page table, supplying up-to-date translations to all GPUs.

![](_page_1_Figure_10.jpeg)

Fig. 1. UVM-managed Multi-GPU overview

Figure 1 (a) illustrates the address translation process. Upon a memory request, the L1 TLB performs a lookup 1 . A miss in the L1 TLB leads to a check in the L1 Miss Status Holding Register (MSHR), and if missing there, the request advances to the L2 TLB 2 . A miss at this level sends the request to the L3 TLB 3 . When an L3 TLB miss happens, GMMU will do page table walks 4 . A failed page walk results in a farfault, reported to the host's UVM driver via interrupt 5 . The driver's centralized page table manages far-faults, ensuring data consistency and tracking pages across GPUs and CPUs.

# *A. UVM-Enabled Multi-GPU System*

This paper targets UVM-managed, discrete multi-GPU systems in which several GPUs are connected by high-bandwidth links such as PCIe [31] and NVLink [15]. Figure 1 (a) illustrates the baseline architecture. Each GPU comprises multiple streaming multiprocessors (SMs). The L1 TLB is shared between the two SMs within each Texture Processing Cluster (TPC), and the L2 TLB is shared across the SMs of a GPU Processing Cluster (GPC). The L3 TLB is shared across all GPCs. A discrete GPU typically maintains its own local memory and page tables, with page-table walks handled by the GPU Memory Management Unit (GMMU). In UVM-managed multi-GPU systems, the CPU-side UVM driver handles GPU far-faults and maintains a centralized page table, supplying up-to-date translations to all GPUs.

![](_page_1_Figure_10.jpeg)

Fig. 1. UVM-managed Multi-GPU overview

Figure 1 (a) illustrates the address translation process. Upon a memory request, the L1 TLB performs a lookup 1 . A miss in the L1 TLB leads to a check in the L1 Miss Status Holding Register (MSHR), and if missing there, the request advances to the L2 TLB 2 . A miss at this level sends the request to the L3 TLB 3 . When an L3 TLB miss happens, GMMU will do page table walks 4 . A failed page walk results in a farfault, reported to the host's UVM driver via interrupt 5 . The driver's centralized page table manages far-faults, ensuring data consistency and tracking pages across GPUs and CPUs.

