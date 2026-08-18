# *F. Iterations*

Due to the nature of Out-of-Order processors, as well as the filtering caused by the L1 cache, the performance variations induced by running R-Max (or any prefetcher) can cause the memory access pattern, as seen at the given level of the cache, to change versus the initial case where no prefetcher is run. Counters and timestamps collected in previous iterations may not match exactly with those observed in subsequent iterations. To address this, R-Max uses the aforementioned data structures and operations to adapt to the re-ordering.

Each simulation iteration generates a memory access file that is used in the next iteration. As this process iterates, useless prefetches are filtered out and missing accesses are inserted to the file. The simulation will eventually converge to a point where memory accesses to L2 will be primarily served by prefetches, yielding the highest possible hit rate and IPC.

