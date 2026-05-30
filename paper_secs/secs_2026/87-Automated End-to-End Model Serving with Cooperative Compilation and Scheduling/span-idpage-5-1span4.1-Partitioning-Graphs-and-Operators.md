# <span id="page-5-1"></span>4.1 Partitioning Graphs and Operators

Per § [2.4,](#page-4-2) the compiler treats DNN operators as concatenated operator tiles, which are named micro operators. The tile size of micro operators is determined alongside the tile size of kernels (§ [4.2\)](#page-5-2). For small operators with little computation, the compiler opts for "merge" rather than "tile", which merges a subgraph and creates a virtual operator named "shepherd operator" for the subgraph. This is used to avoid the substantial overhead caused by frequently scheduling small operators.

