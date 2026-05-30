# <span id="page-6-0"></span>C. Fusion Search Engine

Our search engine is designed to efficiently explore the vast search space composed of loop schedules, tiling sizes, and resource mapping to find the optimal fusion plan. Its core principle is to leverage an analytical cost model and pruning strategies to rapidly filter out a large number of inefficient or incorrect candidates.

<span id="page-6-2"></span>1) Cost Model: Our performance model is inspired by the analytical model in Chimera [60]. We model the data movement cost across the L levels of the memory hierarchy. The cost  $C_l$  of transferring data to level l is determined by the required data volume  $V_l$  for a given tiling strategy  $\mathcal{T}_l$ , and the memory bandwidth  $B_l$  of that level.

$$C_l(\mathcal{T}_l) = \frac{V_l(\mathcal{T}_l)}{B_l} \tag{1}$$

To optimize the overall performance, we aim to minimize the bottleneck, which is the slowest data movement stage among all memory levels. This is formulated as a minimax optimization problem:

$$\min_{\mathcal{I}_1, \dots, \mathcal{I}_L} \left\{ \max_{l=1, \dots, L} \left( C_l(\mathcal{T}_l) \right) \right\}$$
(2)

The optimization is subject to memory capacity constraints of each level, where the memory usage  $U_l$  dictated by the tiling strategy  $\mathcal{I}_l$  cannot exceed the available capacity Cap<sub>l</sub>.

s.t. 
$$U_l(\mathcal{I}_l) \le \operatorname{Cap}_l, \quad \forall l \in \{1, \dots, L\}$$
 (3)

2) Pruning Strategies: While prior work has established pruning principles for kernel fusion, these do not address the vast search space introduced by clusters and are thus insufficient for our needs. Building upon these foundations, we propose the following pruning strategies:

- Initial Search Space: We construct our initial search space starting from the loop schedule and tile size. Drawing from methodologies in existing work, the minimum block size is set to that of a single MMA operation, i.e.,  $16 \times 16 \times 16$ . The cluster dimension can be chosen from one of five values  $\{1, 2, 4, 8, 16\}$ . Since there are 4 independent dimensions, this results in 5<sup>4</sup> possibilities for the cluster configuration. For a model like GPT-6.7B, we consider a problem size with M = 256, N = 16384, and K = T = 4096. The number of valid tile choices is thus  $(256/16) \times (16384/16) \times (4096/16) \times (4096/16)$ . As shown in Table IV, there are a total of (24 + 12 +4+1) = 41 possible combinations for spatial and temporal partitioning. Therefore, the initial search space contains  $(24+12+4+1) \times 5^4 \times (256/16) \times (16384/16) \times$  $(4096/16) \times (4096/16) \approx 2.75 \times 10^{13}$  possibilities.
- Rule 1, Divisible Tile Sizes: This is a pruning strategy from prior work [55], which mandates that the selected tile sizes should be hardware-aware and the problem size dimensions are evenly divisible by them.
- Rule 2, Cluster Size Constraint: The product of cluster dimensions for each GEMM across M, N, and K must be less than the hardware limit (for H100, it is 16), and the cluster dimensions of consecutive GEMMs must be identical to ensure feasibility.
- Rule 3, Activation constraint: To ensure the correctness
  of the activation between consecutive GEMMs, the accumulation dimension of preceding GEMM must be placed\nin the innermost loop. Otherwise, partial sums would be
  computed, which cannot be used by the activation and
  would lead to incorrect results in the subsequent GEMM.
- Rule 4, Dependency constraint: If L dimension is set as spatial, given the dependency of GEMM, all spatial tile in L dimension will need intermediate tensor of C, but different tiles can not communicate with each other directly, therefore the fusion will fail.
- Rule 5, Memory Capacity Limit: A tensor cannot exceed the capacity of the lowest-level cache to which it can spill.

Among the rules above, only Rule 1 is derived from prior work [55]; the rest are novel strategies specific to this paper for handling the search space introduced by clusters. Following the analysis of prior work, the pruned search space has 11,550 ( $\sim 10^4$ ) possibilities. In contrast, our work, which considers

<span id="page-6-1"></span>TABLE IV: Possible partitions for Spatial (S) and Temporal (T) dimensions. The letter combinations in the S and T columns are examples only.

| Num of dim in S | S (Spatial) | T (Temporal) | Num of schedules                                                                                           |
|-----------------|-------------|--------------|------------------------------------------------------------------------------------------------------------|
| 1               | M           | NKL          | $(C_4^1 \times 3! = 24)$<br>$(C_4^2 \times 2! = 12)$<br>$(C_4^3 \times 1! = 4)$<br>$(C_4^4 \times 0! = 1)$ |
| 2               | MN          | KL           | $(C_4^2 \times 2! = 12)$                                                                                   |
| 3               | MNK         | L            | $(C_4^3 \times 1! = 4)$                                                                                    |
| 4               | MNKL        | Ø            | $(C_4^4 \times 0! = 1)$                                                                                    |

## Algorithm 2: Fusion Search Algorithm

```
Input: Graph g, Device d, Top-k count k
   Output: The best execution plan p_{best}
1 Function SearchEngine(g, d, k)
2
       all\_candidates \leftarrow EnumerateAllCandidates(g, d);
       pruned candidates \leftarrow
         PruneCandidates(all candidates);
       top k list \leftarrow [];
5
       foreach (s,t,r) in pruned_candidates do
            (D_v, plan) \leftarrow \text{DataflowAnalyzer}(g, d, s, t, r);
 6
            est\_cost \leftarrow CalculateCost(D_v);
            top\_k\_list \leftarrow
 8
             UpdateTopKList(top_k_list, (est_cost, plan), k);
       p_{best} \leftarrow \text{ProfileBestFromList}(top\_k\_list, d);
10
       return p_{best};
```

the use of clusters, addresses a much larger search space. Therefore, a cost model is required for further analysis.

3) Search Algorithm: Algorithm 2 details our fusion search method. This algorithm takes a DNN graph g, device information d, and the top-k count k as input. We first employ the pruning strategies mentioned in the previous section to filter the search space (line 3). Then, the legal candidates are fed into the DataflowAnalyzer for detailed analysis. As depicted in Algorithm 1, we analyze and obtain the specific dataflow details under the current parameters, namely the placement of each reused tensor within the cache hierarchy and the concrete data movement volume (line 5-6). Subsequently, using the cost model described in Section IV-C1, we iteratively evaluate each configuration to maintain a list of top-k candidates. Finally, these candidates are profiled on hardware to determine the ultimate execution plan (line 7-9). This entire search is performed offline; at runtime, kernel selection is achieved by using binning and table look-ups for the varying M dimension to select from our pre-compiled kernels. This is efficient because in FFN/conv scenarios, only the M dimension varies dynamically while N, K, and L are fixed.

#### V. IMPLEMENTATION

FlashFuser is a code generation framework built upon NVIDIA CUTLASS [41]. It takes a high-level DNN model description as input and utilizes our three core components—the Fusion Search Engine, Dataflow Analyzer, and dsm\_comm primitive—to generate high-performance fused kernels, separating the implementation into a front-end for search and a back-end for code generation.

### A. Front-End: The Fusion Search Engine

Our front-end is a Python-based search engine that explores the space of LoopSchedules, TilingSizes and ResourceMapping(with DSM, the lowest-level cache, selected by default). For each configuration, it invokes our Dataflow Analyzer to heuristically determine the memory

mapping for intermediate results and compute the data movement volume. It then uses a cost model and pruning rules to filter candidates. The back-end is subsequently invoked to generate code. Finally, the top-*K* configurations are passed to the hardware for on-device measurement to identify the fused kernel with the optimal performance.

#### B. Back-End: Code Generation and Primitive Implementation

The back-end translates the optimal plan from the frontend into high-performance CUDA code, leveraging the highlyoptimized components of CUTLASS.

- a) Realizing the Dataflow Analyzer: Our heuristic plan is realized during code generation. The decision between register and smem is made by calculating the theoretical register usage for a given tile size to avoid performance-degrading spills to global memory. If SMEM is still not large enough, the data must be placed in DSM.
- b) Implementing the dsm\_comm Primitive: We implemented SHUFFLE, MUL, and REDUCE operations for the dsm\_comm primitive using a fine-grained data exchange mechanism built on TMA for data movement and the mbarrier intrinsic for many-to-many synchronization. Unlike the native all-to-one cluster-sync in CUTLASS, our mbarrier-based approach allows us to synchronize only the necessary groups of CTAs for a given exchange, enabling the construction of higher-level collectives like ring communication for SHUFFLE.
- c) Integrating Primitives into Kernel: Our code generator extends the CUTLASS kernel structure—prologue, mainloop, and epilogue-to orchestrate the cluster-level dataflow prescribed by the front-end. In the prologue, semaphore initialization is extended to the DSM to prepare it for inter-CTA communication. The mainloop is augmented with our dsm\_comm operations. For instance, upon completion of the producer's accumulation loop, a DSM mul is performed for GatedFFN variants to exchange and apply computation. Within the consumer's accumulation loop, a DSM shuffle implements a ring communication pattern to exchange intermediate results among CTAs. Finally, in the epilogue, a DSM reduce accumulates partial sums from different CTAs using a scatterreduce scheme before storing the final result to global memory. This design maps the problem's spatial dimensions to the grid, while the temporal dimension to the nested execution loop within the kernel's mainloop.

## VI. EVALUATION

