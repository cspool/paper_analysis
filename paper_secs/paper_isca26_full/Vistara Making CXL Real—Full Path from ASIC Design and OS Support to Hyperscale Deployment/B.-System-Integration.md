# *B. System Integration*

Having developed the Vistara ASIC as a scalable CXL memory expander, we then integrate it into our memoryoptimized server platform, *MemServer*. This platform is designed to fully leverage Vistara's capabilities, enabling efficient and reliable memory expansion at scale. Figure 6 shows the server's high-level architecture.

![](_page_4_Figure_4.jpeg)

Fig. 6. High-level architecture of a *MemServer*.

Compute and Memory Architecture. Each MemServer node has a single-socket AMD Turin processor, built on the Zen5 microarchitecture [3]. CPU has 158 cores per node, where each core is 2-way SMT and operates at ≈3GHz.

The memory subsystem in MemServer provides high capacity and high bandwidth. Local DRAM is provisioned at 768GB of DDR5-6400 per node, organized across 12 memory channels. Each channel operates at 6400 mega-transfers per second (MT/s), thus, the system delivers up to 614 GB/s of aggregate bandwidth. The idle latency for local DRAM is ≈130 ns, and the system is capable of sustaining peak bandwidths exceeding 400 GB/s under mixed read/write workloads.

CXL Integration. CXL memory expansion is enabled by two Vistara ASICs per node, each connected via a PCIe Gen5 x8 interface. Vistara CXL ASIC supports two independent DDR4 memory channels (72 bits wide), with two 32GB DIMMs per channel, resulting in a total of 256GB of CXL-attached memory per server. Hence, in total, the server has 158 cores or 316 hyperthreads with 1TB of memory.

Each DDR4 channel operates at 2400 MT/s, delivering sufficient aggregate bandwidth to support capacity-bound workloads at low power and cost. By engineering Vistara to accept industry-standard DDR4 RDIMMs, including modules reclaimed from decommissioned servers, we efficiently reuse a substantial pool of DDR4 memory. This approach maximizes resource utilization and enables a sustainable and costeffective path to large-scale memory expansion as the fleet transitions to DDR5-native CPUs.

CXL Access Interleaving. To ensure balanced utilization of the CXL links and minimize contention, memory accesses are interleaved across Vistara devices at a 256-byte granularity at the CPU host-bridge. This means consecutive memory blocks are alternately mapped across both Vistara CXL modules. Within each Vistara device, accesses are further interleaved across DIMMs, ranks, and banks to maximize parallelism and ensure a fair distribution of traffic across all memory channels and components. Hence, the system maximizes aggregate bandwidth by allowing simultaneous access to both devices, and helps to evenly distribute traffic, reducing the risk of bottlenecks or contention on any single CXL link.

Platform Configuration Summary. Table VI consolidates the MemServer platform configuration for reference.

TABLE VI MEMSERVER PLATFORM CONFIGURATION.

| Component          | Specification                            |
|--------------------|------------------------------------------|
| CPU                | AMD Turin, 158/316 cores/threads, ≈3 GHz |
| CPU TDP            | 300 W                                    |
| Local Memory       | 768 GB DDR5-6400, 12 channels            |
| Local Peak BW      | 614 GB/s                                 |
| Local Idle Latency | ≈130 ns                                  |
| CXL Devices        | 2× Vistara ASICs, PCIe Gen5 x8 each      |
| CXL Memory         | 256 GB DDR4-2400 (8×32 GB RDIMMs)        |
| CXL Peak BW        | ≈76 GB/s                                 |
| CXL Idle Latency   | ≈250 ns                                  |
| CXL Interleave     | 256 B granularity across 2 devices       |
| CXL + DIMM Power   | ≈50 W total (2 ASICs + 8 DIMMs)          |
| Total Memory       | 1024 GB (1 TB)                           |
| Total Server Power | 450–560 W                                |

Power and Thermal Management. In the memory-optimized configuration, the processor operates at a TDP of 300W, while the local DDR5 memory subsystem contributes an additional 50–60W per node under load. Each Vistara CXL ASIC draws up to 9W, and the associated CXL-attached DDR4 DIMMs add approximately 32W, resulting in a total CXL memory expansion power budget of ≈50W per server. Overall, the total server power consumption at high utilization is between 450W and 560W, depending on workload mix and turbo settings.

The Vistara CXL cards are installed in dedicated rearaccessible slots within each MemServer chassis. To manage the increased thermal load from high-density memory and CXL devices, the chassis employs directed airflow with highcapacity fans that channel cool air directly across the Vistara modules, for stable operation under heavy workloads.

