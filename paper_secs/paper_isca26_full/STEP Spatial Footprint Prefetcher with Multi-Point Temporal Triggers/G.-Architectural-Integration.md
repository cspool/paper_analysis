# *G. Architectural Integration*

The STEP prefetcher is integrated as an add-on hardware entity, located in parallel to the L2 cache of our MPSoC architecture. It snoops on the bus between the L1 and L2 cache and inserts prefetch requests into the prefetch queue (PQ) of the L2 cache. The latter will then take care of prefetching the corresponding cachelines into the L2 cache. Although we mainly prototype STEP as an L2C prefetcher, the principles apply at other cache levels, which is also shown in Section V-C and Section V-D.

#### IV. EXPERIMENTAL METHODOLOGY

