# 1 Introduction

The sparse matrix-sparse vector multiplication (SpMSpV) computes  $\mathbf{y} = \mathbf{A}\mathbf{x}$  with both the matrix  $\mathbf{A}$  and the input vector  $\mathbf{x}$  being sparse. SpMSpV is frequently used in machine learning [8] and scientific computing [3, 5, 32, 41]. It also serves as a fundamental primitive in graph analytics, underlying core algorithms such as breadth-first search (BFS), PageRank and personalized PageRank, and it is the algebraic backbone of many graph frameworks including GraphBLAS [15], Gunrock [34], GraphBLAST [39], GraphMat [33].

Beyond these domains, SpMSpV also appears in event-driven workloads such as spiking neural networks (SNNs), where spike delivery can be naturally formulated as sparse matrix-sparse vector multiplication [9]. Moreover, both brain-inspired neural models and real-world graphs (e.g., social networks) are known to exhibit highly clustered, small-world connectivity patterns [35], which create opportunities for exploiting locality in SpMSpV execution.

SpMSpV can be implemented under two execution paradigms: Row-major methods traverse CSR rows and are

naturally aligned with CSR-based SpMV. Some implementations can be regarded as direct extensions of SpMV, obtained by adding value validation [33]. Alternatively, methods such as tileSpMSpV [18] and BerryBees [26] adopt bitmap-compressed frontiers together with masking of visited nodes, and are specifically optimized for unweighted BFS-style traversals. However, for *weighted* SpMSpV, row-major traversal still scans all matrix rows regardless of vector sparsity, and the bitmask cannot avoid loading matrix indices. As a result, row-major methods cannot fully exploit vector sparsity.

Column-major SpMSpV, in contrast, follows a vector-driven paradigm: the computation consists of a *fetch phase*, which gathers matrix columns corresponding to nonzeros in the vector, and a *write-back phase*, which uses column indices to update the result vector, essentially an index-scatter where multiple entries may map to the same position.

On CPUs, representative studies include fgSpMSpV [10], work-efficient SpMSpV [2], and HAM-SpMSpV [37]. On GPUs, prior work has explored different *write-back* methods and kernel selection: graph analytics frameworks (e.g., Gunrock [34]) use atomic instructions to directly handle write conflicts; FastSpMSpV [40] adopts a *sort-reduce* approach to avoid conflicts, and Adaptive SpMSpV [20] selects write-back strategies (atomic vs. sort) based on matrix characteristics, while also adapting load-balancing granularity and switching to row-major SpMSpV or SpMV under dense vectors.

However, we observe that in some cases the two prevalent write-back strategies—atomic updates and sort-based updates—both fail to achieve satisfactory bandwidth utilization: the former suffers from scattered index updates and frequent conflicts, while the latter relies on costly global sorting (see Section 3).

Similar challenges arise in SpGEMM, where hashing has been used effectively to aggregate partial products and eliminate intra-row conflicts [12, 13, 25, 28, 36]. However, SpMSpV lacks the natural row partitioning of SpGEMM: instead of resolving conflicts only within a single row, it must handle all intermediate updates across the matrix. As a result, a hash table can only eliminate a portion of the write conflicts, while the remaining updates still require global atomic writes. This leads to two key questions: whether SpMSpV provides sufficient locality for hash aggregation, and whether the benefit of fewer write conflicts can outweigh the overhead of the hash table.

To address these challenges, we propose a vector-driven hash-aggregation (VDHA) algorithm for *weighted* SpM-SpV (both the matrix and the input vector contain general weights) on GPUs. VDHA reduces write-back conflicts via local aggregation in shared memory, enhances locality through column decomposition with reordering, and reduces hash cost by pipelining computation with memory access. Concretely, we propose **VDHA**:

- Shared-memory hash aggregation. Intermediate results are first accumulated in a shared-memory hash table and flushed only when the table becomes sufficiently full, reducing the write-back conflicts and promoting coalesced writes.
- Short/long-column decomposition with reordering.
   We first classify columns by their length (the number of nonzeros) into short and long categories. Long columns are further split into smaller segments and reordered to improve locality and raise aggregation density, thereby maximizing the benefit of shared-memory accumulation.
- Overlapping memory and computation. We design a
  pipeline that overlaps irregular global memory accesses
  with hash computation, effectively hiding hash computation latency behind memory stalls and making aggregation nearly free.

To systematically evaluate VDHA, we consider two benchmarks. The first consists of over 100 large-scale web graphs from the Konect [19] and LAW [6, 7] collections, which are representative of *graph analytics workloads* where weighted SpMSpV is most critical (e.g., PageRank and Personalized PageRank on web graphs). The second includes over 200 matrices from the SuiteSparse [11] collection, a widely used benchmark that covers diverse domains such as scientific computing, engineering, and optimization. Both benchmarks contain only matrices with at least 5 million nonzeros. Together, these datasets allow us to assess both the practical impact on real graph workloads and the generality across broader application scenarios.

Across four vector sparsity levels (0.01, 0.05, 0.10, 0.20; defined as the fraction of nonzeros in the input vector), VDHA outperforms the *best-of-seven* baselines (including cuSPARSE, two row-major SpMSpV kernels using value validation [30, 31], and the four representative column-major SpMSpV kernels from [20, 34, 40]), achieving geometric-mean speedups of **1.41**× on Konect/LAW (up to **3.42**×) and **1.13**× on SuiteSparse (up to **2.55**×).

**Contributions.** This paper makes the following contributions:

- VDHA algorithm. By enhancing locality and reducing hashing overhead, we realize a practical and efficient hashbased solution for weighted SpMSpV on GPUs
- **Systematic comparison.** We conduct a comprehensive evaluation against SOTA baselines, across over 100 realworld network graphs and over 200 scientific graphs with a wide range of vector sparsities, demonstrating consistent speedups.
- Lightweight Performance Prediction. We provide a lightweight analysis method to quickly assess whether a matrix benefits from VDHA, facilitating its integration into adaptive frameworks.

