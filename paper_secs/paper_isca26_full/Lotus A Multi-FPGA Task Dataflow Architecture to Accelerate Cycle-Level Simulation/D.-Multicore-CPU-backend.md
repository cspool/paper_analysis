# *D. Multicore CPU backend*

We also implement an multicore CPU backend for Lotus DSL programs. This backend implements RepCut-style [41] partitioning to map tasks to threads while avoiding singlecycle cross-thread communication. We use PaToH to perform hypergraph partitioning. We also perform several optimizations: we place values in memory to avoid false sharing, implement cross-cycle synchronization using a scalable tree barrier, and pin threads to CPU cores to place them in nearby cores, which reduces communication cost.

