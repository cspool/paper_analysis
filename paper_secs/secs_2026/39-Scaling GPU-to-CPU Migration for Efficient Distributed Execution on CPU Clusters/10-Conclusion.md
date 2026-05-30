# 10 Conclusion

We introduce CuCC, a framework for migrating GPU programs to CPU clusters. CuCC incorporates compiler analysis and transformations to generate CPU cluster programs with low communication overhead. Our evaluation demonstrates that GPU programs can be efficiently executed on CPU clusters, achieving execution times within the same order of magnitude as GPUs. Furthermore, since modern data centers typically contain far more CPU nodes than GPUs, clusterwide throughput analysis shows that CPUs achieve 2.59× higher throughput compared to GPU execution.

As the industry moves toward higher-bandwidth networks such as 400 Gbps and 800 Gbps, the performance of clustered CPUs will continue to improve. Hence, we expect that running GPU programs on clustered CPUs will become even more compelling.

