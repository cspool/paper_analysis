# *A. tak¨ o Hardware Overview ¯*

As Figure 3 shows, tak¨ o is a tiled chip. Each tile has a core, ¯ an L1, a private L2, and a shard of an L3 bank which is shared across tiles. Each tile also has an engine that runs callbacks.

tak¨ o allows programmers to register ¯ OnMiss, OnEvict, and OnWriteback (henceforth OnWB) callbacks for virtual address ranges. These callbacks run on the tile's engine during the corresponding cache events. Figure 3 depicts an OnMiss workflow. When an address with a registered OnMiss misses in the cache, a hardware thread runs the OnMiss on the corresponding engine to calculate the cache line's contents.

Similarly, when a line in the address range is evicted from a cache, an OnEvict or OnWB is invoked to run, depending on whether the data is clean or dirty respectively. Together, tak¨ o's three callback types allow for custom calculations and ¯ behavior to run as part of the cache's handling of these address ranges. This functionality allows data transformation to occur as part of data movement instead of occurring once the data has been loaded, and for any clean-up to happen as part of data eviction. Moreover, as the computation results are cached, redundant work is avoided if the same transformation needs to be run again, improving performance on certain workloads. For addresses with no callbacks registered, the semantics of the conventional load/store interface are preserved.

