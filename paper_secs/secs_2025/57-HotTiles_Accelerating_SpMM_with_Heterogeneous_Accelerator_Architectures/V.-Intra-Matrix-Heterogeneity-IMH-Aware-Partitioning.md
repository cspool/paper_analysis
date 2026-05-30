# V. Intra Matrix Heterogeneity (IMH) Aware Partitioning

Figure 7 illustrates the preprocessing steps of our IMH-aware modeling and partitioning method, which we call *Hot-Tiles*. First, the matrix tiles are scanned and fed to the cold and hot performance models of Section IV to extract, for each tile and worker type, the estimated execution time and the estimated number of bytes accessed from main memory. Then, a partitioning heuristic is used to split the sparse matrix into cold and hot tiles. Finally, the cold and hot sections of the initial sparse matrix are stored in the appropriate sparse compression format as required by each heterogeneous accelerator. In this section, we focus on the partitioning heuristic.

