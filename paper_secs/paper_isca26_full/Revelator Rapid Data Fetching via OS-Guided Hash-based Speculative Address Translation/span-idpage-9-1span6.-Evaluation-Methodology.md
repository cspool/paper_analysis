# <span id="page-9-1"></span>6. Evaluation Methodology

We evaluate Revelator using Virtuoso [13, 96], a validated simulation methodology for accurately modeling the overheads of address translation, ported on top of Sniper [97, 98], an event-driven multicore simulator. We extend Virtuoso to model Revelator's OS-level tiered hash-based allocation policy and Sniper to model Revelator's hardware speculation engine. We use CityHash [115] as the hash function for both the OS and hardware components and charge a 2-cycle latency to produce the

<span id="page-9-3"></span>hash result. Table 2 shows the details of the baseline simulated system configuration.

**Table 2: Simulated Baseline System Configurations** 

| Core     | 8-way OoO; ROB: 300 entries; x86-64 2.9 GHz                                                         |  |
|----------|-----------------------------------------------------------------------------------------------------|--|
| MMU      | L1 I-TLB: 128-entry, 8-way assoc, 1-cycle                                                           |  |
|          | L1 D-TLB (4 KB): 64-entry, 4-way assoc, 1-cycle;<br>L1 D-TLB (2 MB): 32-entry, 4-way assoc, 1-cycle |  |
|          | L2 TLB: 2048-entry, 16-way assoc, 12-cycle                                                          |  |
|          | 3 Page Structure Caches: 4,8,32-entry; 2-cycle                                                      |  |
| L1 Cache | 64 KB, 8-way, 4-cycle; pLRU;                                                                        |  |
| L2 Cache | 1 MB, 16-way, 12-cycle; pLRU;                                                                       |  |
| L3 Cache | 2 MB/core, 16-way assoc, 35-cycle; SRRIP [143]                                                      |  |
| DRAM     | 128GB-DDR4-2400, 4 channels; 2 ranks/channel; 16 banks/rank; tCL = tRCD = tRP = 14.17 ns            |  |

Workloads. Table 3 lists all the workloads used in our evaluation. All evaluated workloads exhibit a high number of page table walks per kilo-instruction (PTWPKI) (>5), similar to prior studies [17, 19, 23, 144–146]. Each benchmark is executed for 300M instructions. In single-core systems, we evaluate the 11 most translation-intensive workloads from the GraphBIG [99], XSBench [103], GUPS [100], DLRM [101], and GenomicsBench [102] suites. In multi-core and NUMA systems, we evaluate 30 mixes from 140 workloads from Google's server suite [104, 147].

**Evaluated Systems in Native Execution.** We evaluate: (i) **THP**: Baseline system that utilizes Transparent Huge Pages (4KB/2MB). (ii) **ReserveTHP** [105]: reserves 2MB regions and promotes them based on the memory utilization of their resident 4KB pages, (iii) SpecTLB [88]: speculates physical addresses based on reserved large page regions, as described in §2, (iv) **POM-TLB** [107]: uses a 16MB software-managed L3 TLB, (v) *L2 TLB-64K*: uses a large 64K-entry L2 TLB, (vi) **ASAP** [108]: lays out page table data contiguously to enable page table prefetching, (vii) **DMT** [16]: establishes a direct mapping from virtual addresses to last-level PT entries, (viii) **ECH** [19]: employs the elastic cuckoo hash-based page table to improve parallelism in page table walks, (ix) Mosaic-Pages [106]: employs Iceberg hashing [148] to drastically increase the TLB reach, (x) **SpOT** [21]: a state-of-the-art speculative address translation scheme, described in detail in §2, (xi) **Revelator**: employs tiered hash-based allocation (§4.1) and speculation engine (§4.3) with 3 hashes and degree filtering (unless otherwise specified). (xii) *Revelator+THP*: Revelator combined with THP (§4.4). THP is enabled in L2 TLB-64K, SpOT, POM-TLB, ASAP, DMT, and ECH.

Memory Fragmentation & Utilization. We model fragmentation by controlling how many 2MB pages are available compared to the total possible 2MB pages in the system. We use three levels: high, medium, and low fragmentation, which correspond to 10%, 50%, and 90% of the total possible 2MB pages being available, respectively. These levels match measurements from our lab's 22-node cluster (each node has 64 cores [149] and 256GB of memory), after approximately one month of uptime, one hour of uptime, and a fresh reboot, respectively. Fragmentation controls the available contiguity in free memory, whereas memory utilization controls how much memory is already occupied regardless of the data layout. We initialize each experiment with 20% base memory utilization by randomly marking 20% of the total physical pages as occupied.

Table 3: Evaluated Workloads

<span id="page-10-2"></span>

| Suite                                   | Workload                                                                                                                                                                                       | PTWPKI |
|-----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|
| GraphBIG [99]                           | Betweenness Centrality (BC), Breadth-first<br>search (BFS), Connected components (CC),<br>Graph coloring (GC), PageRank (PR), Triangle<br>counting (TC), Single-source shortest path<br>(SSSP) | 10-45  |
| XSBench [103]                           | Particle Simulation (XS)                                                                                                                                                                       | 12     |
| GUPS [100]                              | Random-access (RND)                                                                                                                                                                            | 25     |
| DLRM [101]                              | Sparse-length sum (DLRM)                                                                                                                                                                       | 18     |
| GenomicsBench [102]                     | k-mer counting (GEN)                                                                                                                                                                           | 35     |
| Google<br>Workload<br>Traces [104, 147] | 140 traces from Sierra, Bravo, Tango, Tahoe,<br>Merced, Yankee, Delta, Whiskey                                                                                                                 | 5-22   |

