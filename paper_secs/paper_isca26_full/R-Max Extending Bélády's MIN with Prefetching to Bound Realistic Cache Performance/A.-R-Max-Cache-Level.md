# *A. R-Max Cache Level*

R-Max can be placed individually or simultaneously in multiple levels of cache(L1D/L1I, L2 or LLC). Many recent prefetchers use L2 for placement [11], [21], [25]. The L2 is more capacity constrained than the LLC, making prefetching decisions more impactful. By contrast to L1, the L2 suffers from processor requests being filtered by L1, making predicting useful prefetches more difficult. Thus, prefetching to the L2 is both more challenging and potentially more beneficial than prefetching to other levels in the cache. As a result, for the remainder of this discussion, we focus on an R-Max placed in the L2, however, we also present results for R-Max in other levels of the cache in Section VI.

