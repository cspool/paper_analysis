# <span id="page-10-1"></span>6.4 Performance Characterization and Predictive Modeling

To better understand when VDHA performs well—and why it may underperform on certain matrices—we conduct a matrix-level performance characterization on representative matrices. Based on these observations, we further develop a lightweight predictive model to quickly determine whether or not VDHA is beneficial for a given input. Table 3 summarizes the selected datasets, including their dimensions, number of nonzeros (nnz), and structural characteristics( $\rho$  and  $\gamma$ ).

Impact of Locality and Coalescing. We jointly analyze the role of the local aggregation rate  $\rho$  and the coalescing factor  $\gamma$ . A larger  $\rho$  means more updates can be absorbed into the shared-memory hash table, reducing the number of global writes. At the same time, a larger  $\gamma$  reflects more coalesced memory transactions, lowering bandwidth waste. Together, high  $\rho$  and high  $\gamma$  mean that many updates are aggregated and written back in contiguous sections, which directly translates into higher effective bandwidth.

Table 3 shows representative cases. Web graphs such as it-2004 and sk-2005 exhibit strong temporal locality ( $\rho$ ) and irregular accesses that are greatly improved after reordering (better  $\gamma$ ), leading to substantial speedups.

In contrast, some matrices exhibit near-diagonal nonzero structures, such as *atmosmodl* and *G3\_circuit*. For these matrices, VDHA expansion generates few overlapping updates, resulting in low  $\rho$  and limited improvement in  $\gamma$ . Moreover, when nonzeros are distributed in a highly regular fashion,

our vector processing introduces little benefit but adds overhead, making VDHA less effective. Therefore, our approach is less advantageous on diagonal-like structure.

**Toward Fast Predictors.** While  $\rho$  and  $\gamma$  explain performance behavior well, they are runtime metrics and cannot be obtained without executing the kernel. To enable lightweight prediction, we introduce two simple structural statistics that can be computed directly from the matrix.

We denote the two structural indicators as (i) *bandwidth index B*, defined as the average distance between the topmost and bottommost nonzeros per column in the CSC matrix, and (ii) *variance index V*, defined as the column-wise variance of nonzeros. Together with the number of rows, the total number of nonzeros, and the input vector sparsity, we train a simple decision tree classifier following the methodology of Adaptive SpMSpV [20]. We use 70% of matrices for training and 30% for testing. On the test set, the decision tree achieves **91.3%** accuracy (measured by F1 score).

Furthermore, if we fall back to a naive implementation (BlockAtomic) when the predictor estimates VDHA to be suboptimal, the geometric-mean speedup on the SuiteSparse dataset across all four vector sparsities improves from **1.13**× to **1.16**×. If fallback to the best among all seven baselines (best-of-7), the adaptive scheme further achieves **1.22**× speedup.

Our predictor uses five lightweight structural features (num\_rows, num\_nnzs, vector sparsity, bandwidth index *B*, and variance index *V*) and achieves good accuracy. Adding more features (e.g., Adaptive SpMSpV uses 13 matrix and vector features) may improve accuracy further but increases extraction overhead, presenting a trade-off for future work.

### 7 Related Works

The optimization of SpMSpV has been driven primarily by graph analytics workloads. GPU frameworks such as Gunrock [34], GraphBLAST [39], push-pull library [38], GraphLab [22], MultiGraph [17], Graphpad [1], Ligra [29], and GSwitch [23] incorporate SpMSpV as a core primitive to accelerate fundamental applications including BFS, PageRank, and personalized PageRank. Beyond frameworks, dedicated GPU kernels have also been proposed. Approaches such as TileSpMSpV [18] and BerryBees [26] specifically target unweighted BFS by employing a tiled format with output masking, in which frontier vectors are binary and results are accumulated using atomicOr. BerryBees further exploits bit-level tensor cores on recent GPUs to accelerate these operations.

Other works developed more general-purpose kernels: FastSpMSpV [40] introduced a reduce-based method to avoid atomics via global reducing, while Adaptive SpMSpV [20] selected among multiple kernels (row/col-major, atomic- or

<span id="page-11-2"></span>

| Metric         | it-2004 | sk-2005 | mycielskian19 | inline_1  | delaunay_n24 | roadNet-CA | atmosmodl  | G3_circuit |
|----------------|---------|---------|---------------|-----------|--------------|------------|------------|------------|
| Group          | LAW     | LAW     | Mycielski     | GHS_psdef | DIMACS10     | SNAP       | Bourchtein | AMD        |
| Rows           | 41.2M   | 50.6M   | 393K          | 503K      | 16.7M        | 1.97M      | 1.49M      | 1.59M      |
| NNZs           | 1.15B   | 1.95B   | 903M          | 36.8M     | 101M         | 5.53M      | 10.3M      | 7.66M      |
| (2048)<br>𝜌0.1 | 0.665   | 0.664   | 0.491         | 0.472     | 0.137        | 0.085      | 0.126      | 0.131      |
| 𝛾0.1<br>(2048) | 0.793   | 0.617   | 0.399         | 0.490     | 0.199        | 0.149      | 0.196      | 0.189      |
| Speedup        | 1.69×   | 1.92×   | 1.59×         | 1.74×     | 0.91×        | 0.68×      | 0.65×      | 0.65×      |
| Thumbnail      |         |         |               |           |              |            |            |            |

Table 3. Statistics and locality/coalescing metrics of representative datasets.

<span id="page-11-0"></span>Table 4. Ablation study of individual optimization components. Performance is normalized to the full VDHA pipeline (hash + split + reorder).

| Method                 | Normalized Performance |  |  |
|------------------------|------------------------|--|--|
| Hash only              | 0.689×                 |  |  |
| Hash + split           | 0.947×                 |  |  |
| Hash + split + reorder | 1.000×                 |  |  |

<span id="page-11-1"></span>Table 5. Parameter sensitivity of VDHA under different hash-table and split sizes (normalized average speedup).

|            | Hash-table size |        |        |        |  |
|------------|-----------------|--------|--------|--------|--|
| Split size | 1024            | 2048   | 3072   | 4096   |  |
| 64         | 0.9779          | 0.9346 | 0.8831 | 0.8215 |  |
| 128        | 0.9254          | 0.9449 | 0.9109 | 0.8818 |  |
| 256        | 0.9784          | 1.0000 | 0.9848 | 0.9704 |  |
| 512        | 0.8953          | 0.9225 | 0.9031 | 0.9329 |  |

sort-based, different load-balancing strategies) using heuristics on matrix statistics. These efforts highlight the challenges of avoiding write conflicts and balancing workloads.

On CPUs, related efforts such as HAM-SpMSpV [\[37\]](#page-13-7), workefficient SpMSpV [\[2\]](#page-12-8), and Regu2D-SpMV [\[14\]](#page-12-27) demonstrate efficient sparse computations by leveraging cache locality, vectorization, and work-efficient load balancing.

Besides being studied directly, SpMSpV can also be viewed as a special case of SpMV, SpMM, or SpGEMM. Graph-Mat [\[33\]](#page-13-5) observed that SpMSpV can be implemented on top of SpMV by adding a lightweight bitmask to validate vector entries before reading values, thereby reducing some unnecessary memory accesses. HOLA-SpMV [\[31\]](#page-12-19) and Naive-SpMV [\[30\]](#page-12-18) employ two fundamentally different loadbalancing strategies. HOLA-SpMV uses lightweight global balancing to equalize CTA workloads, whereas Naive-SpMV avoids balancing entirely, incurring no overhead but relying on GPU parallelism to mask imbalance.

RoDe [\[27\]](#page-12-21) highlighted that real-world matrices often exhibit highly skewed nonzero distributions, and addressed this by decomposing matrices into a regular part and a residual part processed separately.

A large body of SpGEMM research has focused on handling the many-to-one write-back of intermediate products. Hash-based methods [\[12,](#page-12-10) [13,](#page-12-11) [25,](#page-12-12) [28,](#page-12-13) [36\]](#page-13-9) use shared-memory hash tables to temporarily store and combine intermediate products before writing them back. Sort-based [\[4,](#page-12-28) [21\]](#page-12-29) or merge-based approaches [\[16\]](#page-12-30) first generate candidate triples and then sort or merge them to accumulate results in order.

