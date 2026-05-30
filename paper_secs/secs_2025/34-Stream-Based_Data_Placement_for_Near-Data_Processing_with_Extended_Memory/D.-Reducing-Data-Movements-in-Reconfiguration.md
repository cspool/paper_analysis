# D. Reducing Data Movements in Reconfiguration

After each periodical reconfiguration, the cache configuration is sent to the NDP stacks and applied. Prior work [6], [7] employed bulk invalidation to invalidate all cached data if the space was reassigned. Because the reconfiguration epoch is sufficiently long, e.g., 50 million cycles, even though the bulk invalidation is slow and takes up to 300k cycles [6], it does not significantly impact the overall performance.

Nevertheless, to reduce data movements and cache misses, we discuss an optimization in NDPExt that adopts consistent hashing [40] to keep as many cached data unchanged as possible. For each stream, we consider each possible DRAM row location (RRowBase) in each NDP unit as an individual spot in the circular space of consistent hashing, in total  $65536 \times 64$  spots. During reconfiguration, NDPExt remaps the stream data to the nearest spot leveraging consistent hashing, therefore saving data movements. In our experiments, consistent hashing

TABLE II SYSTEM CONFIGURATIONS.

| NDP system                                             | 4×2 inter-stack mesh, 16 NDP cores per stack;<br>128 NDP cores in total                                                                                |  |
|--------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|--|
| NDP core<br>L1I<br>L1D                                 | 2 GHz, in-order<br>2-way, 32 kB per core, 64 B cachelines, LRU<br>4-way, 64 kB per core, 64 B cachelines, LRU                                          |  |
| NDP HBM                                                | 16 GB HBM 3.0, 1600 MHz, 256 MB/unit;<br>RCD-CAS-RP: 24-24-24;<br>RD/WR: 1.7 pJ/bit, ACT/PRE: 0.6 nJ                                                   |  |
| NDP HMC                                                | 16 GB HMC 2.1, 1250 MHz, 256 MB/unit;<br>RCD-CAS-RP: 14-14-14;                                                                                         |  |
| Extended memory                                        | DDR5-4800, 4 channels × 2 ranks × 16 banks;<br>RCD-CAS-RP: 40-40-40;<br>RD/WR: 3.2 pJ/bit, ACT/PRE: 3.3 nJ                                             |  |
| Intra-stack network<br>Inter-stack network<br>CXL link | 128-bit link, 1.5 ns/hop [65], [69]; 0.4 pJ/bit<br>32 GB/s per dir., 10 ns/hop [20], [22], [69]; 4 pJ/bit<br>16-lane; 200 ns link latency; 11.4 pJ/bit |  |

reduces the invalidation traffic by 9.4% on average, and brings a 3.7% speedup compared to bulk invalidation.

