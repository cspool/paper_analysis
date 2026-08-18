# *B. Recording of L2 Memory Accesses*

In our baseline, L2 R-Max implementation, R-Max must prefetch within the physical address space, since the L2 is physically addressed.

Demand accesses are captured with timestamps for postprocessing. As caches are usually physically tagged, physical addresses are recorded. If R-Max begins to diverge from previous iterations (due to core OoO or L1 filtering changes), such as memory access reordering, the replacement policy falls back to LRU because we lose the knowledge of the future. We address this issue in Section IV-F.

