# <span id="page-3-0"></span>3 Performance Tradeoffs in ELK

As discussed in §2.3, to efficiently use ICCA chips with HBM, we must trade-off multiple performance factors. We summarize how each performance factor is mapped to a compiler decision in Figure 4. First, increasing per-core *execution space* enables faster percore execution with a larger tile size (§3.1). Second, increasing *number of preloaded operators* can better overlap on-chip execution and off-chip HBM load, improving HBM bandwidth utilization (§3.2). Third, increasing *preload space* for each preloaded operator

<span id="page-3-4"></span>![](_page_3_Figure_11.jpeg)

Figure 6: HBM bandwidth demands of models across time, given different preload spaces. The legend shows per-core preload space size in KB (same for all cores).

reduces the inter-core data exchange overhead and the memory access contention, since the shared data can be duplicated on cores in advance to reduce the overhead of on-demand accesses to other cores (§3.3). We validate the insights with experiments on our ICCA chip emulator (see implementation in §5) as follows.

