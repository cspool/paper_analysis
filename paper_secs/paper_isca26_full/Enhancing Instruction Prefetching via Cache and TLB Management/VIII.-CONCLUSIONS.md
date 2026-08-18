# VIII. CONCLUSIONS

This work demonstrates that the address translation latency of L1I prefetches that cross page boundaries and the variable behavior of lines fetched in L2C by L1I prefetches undermines the benefits of modern L1I prefetchers. To address these limitations, this work proposes *Instruction Prefetch Centric Cache and TLB Management (IP-CaT)*, the first microarchitectural scheme to orchestrate TLB and cache management to maximize the benefits of L1I prefetching for applications with large code footprints. Our evaluation shows that IP-CaT significantly enhances the performance of state-of-theart L1I prefetchers and outperforms leading TLB and cache management policies across 105 single-core and 160 multicore server workloads, with only 0.79KB of storage overhead.

