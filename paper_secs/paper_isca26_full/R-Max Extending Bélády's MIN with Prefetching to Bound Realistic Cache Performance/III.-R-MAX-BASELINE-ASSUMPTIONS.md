# III. R-MAX BASELINE ASSUMPTIONS

The goal of R-Max is to study the realistic upper bounds of prefetching, given perfect prediction of future references (omniscience), but realistic physical constraints (not omnipotence), thus R-Max follows the logic and constraints posed by the memory system as described here.

We assume a typical 3-level cache hierarchy. Because R-Max follows the constraints of a realistic cache design, we have a finite number of MSHRs at each cache level, and perform a finite number of tag checks per machine cycle, enforcing a tag bandwidth constraint. When a miss occurs at a particular cache level, an MSHR entry is allocated and the memory request is forwarded to the next, lower, level of the memory hierarchy. In the event that all levels miss, the request goes to the DRAM for resolution. After the block is found in DRAM and the data is returned, MSHRs are deallocated sequentially at the cache levels that have seen the miss. The process of handling accesses takes time.

Prefetches issued by R-Max go through the same steps as demand fetches. If a prefetch misses, it is forwarded to the next cache level after some latency. R-Max prefetches must be placed in a given cache level upon a cache fill, resulting in cache replacement, thus R-Max does not violate the capacity constraint of the cache. The caches in our system are setassociative with a finite number of sets and ways. Because R-Max is omniscient to future memory accesses, upon initial insertion into a given level, we allow R-Max to directly allocate an MSHR entry if the prefetch is not found in the cache. We allow R-Max to know what is currently in the cache without going through tag check because R-Max monitors the traffic going into and out of the cache and is fully aware of the cache state. However, the prefetches issued by R-Max must still pay bandwidth and latency costs without violating capacity constraints. Thus, while omniscient, R-Max is not omnipotent and cannot magically place blocks into the cache without delay or without replacing other data in the cache.

Since R-Max is an exploration of the upper limits of prefetching given perfect knowledge, we impose no particular prefetcher metadata state limits, and of course have devised

![](_page_4_Figure_0.jpeg)

Fig. 2. R-Max high level overview.

no actual prediction algorithm, rather we assume we know the future reference stream perfectly.

