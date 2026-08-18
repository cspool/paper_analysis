# F. Data Reuse Analysis

We compute on-chip data reuse efficiency as the percentage of actual reuse achieved relative to the theoretical maximum, where the theoretical maximum is defined as the total number of times each input and output tensor entry could be reused before eviction under an unlimited on-chip buffer. As shown in Figure 13, TensorPrism achieves an average reuse efficiency

![](_page_11_Figure_7.jpeg)

Fig. 14. Sensitivity study on the parameter choices in the Flickr dataset.

![](_page_11_Figure_9.jpeg)

Fig. 15. Normalized speedup of TensorPrism over SPADE in LLM-related datasets across various densities.

of 67.86% across all four tensors. It improves reuse efficiency over prior work on average, outperforming SPADE, HotTiles, GSpTC, TCP, and HyperSB by 56.5%, 57.4%, 33.8%, 42.6%, and 23.7%, respectively. This stems from the intra-tile data reuse mechanism proposed by TensorPrism, which both exploit temporal locality in input fibers and output accumulators. On the other hand, nel1 exhibits extreme irregularity, introducing minimal reuse efficiency.

#### G. Sensitivity Study

**CoGTP Parameter Setup Influence Analysis.** Figure 14 presents a sensitivity analysis of CoGTP's objective function parameters across two parameter spaces: (a) Intra-tile data reuse coefficient ( $\alpha$ ) versus inter-tile communication penalty ( $\lambda_{cut}$ ), and (b) reuse coefficient versus imbalance penalty ( $\lambda_b$ ). The 3D surfaces reveal how parameter choices impact partitioning quality, measured by normalized speedup on the Flickr dataset. We normalized speedup over the slowest parameter setup, respectively. To be noted, the static parameter( $\lambda_b$  in (a) and  $\lambda_{cut}$  in (b)) is all set to 1.0. The results expose three critical insights into CoGTP's design space.

First, when  $\alpha \leq 3$ , the performance remains relatively stable, exhibiting only smooth variations. However, the performance collapses transparently when  $\alpha > 3$ , indicating saturation where most high-degree vertices are already optimally placed. Moreover, an excessively large reuse-benefit coefficient can conflict with workload balancing, which further contributes to the performance decline.

Second, penalty terms provide coupled refinement. As shown in Figure 14(a), within each  $\alpha$  level, performance gradually decreases as  $\lambda_{cut}$  increases, with a noticeably sharper decline beyond 1.5. This demonstrates that a larger intertile communication penalty could not guarantee performance improvement, proving that the TCP method may not be a wise choice all the time. Similarly, the imbalance penalty

![](_page_12_Figure_0.jpeg)

Fig. 16. Practical overhead analysis: (a) Preprocessing overhead of TensorPrism and baselines in the nel1 dataset. (b) Graph storage overhead of hypergraph (HyperG) and co-occurrence graph (CoG).

λ<sup>b</sup> ensures vertices distribute evenly across PEs, preventing pathological cases where high-degree vertex concentration in a few partitions starves other PEs. Figure 14(b) indicates that with the increase of λb, the performance improves, reaching the peak at 1.0-1.5, but beyond this limit, the performance improvement is weakened. This illustrates the importance of workload balancing in tensor contractions. Moreover, due to the inherent power-law distribution of the Flickr dataset, it highlights the effect of workload balancing.

Third, the selected configuration (α = 2, λcut = 1, λ<sup>b</sup> = 1) represents a conservative sweet spot. It achieves 93.8% of peak performance while avoiding aggressive reuse-only strategies (α ≥ 4) that overfit to specific tensor structures. It also steers clear of overly vigorous workload-balancing schemes that benefit only certain datasets, such as power-law datasets. This choice prioritizes robustness across diverse sparsity patterns in the benchmark suite over maximum performance on individual datasets, consistent with the observation that memory-intensive tensor workloads benefit more from moderate reuse improvements applied consistently than from extreme optimization that may fail on irregular tensors. The surface topology in Figure 14 confirms that CoGTP's three-term objective function requires joint optimization, where no single parameter dominates across all regimes, validating the coupled formulation in Equation 6.

Performance Evaluation in Mildly Sparse Datasets. Further, to assess TensorPrism at midly sparsity (Density ≥ 1%), we include two sparse tensors from intermediate attention maps of LLaMA models [74]. Each tensor has a batch size of 2, 32 attention heads, a sequence length of 512, and a perhead dimension of 128 for LLaMA-8B or 80 for LLaMA-2.7B, consistent with the official architecture specifications. Tensors are sparsified to three density levels (1%, 10%, and 20% nnzs) via magnitude-based pruning, following the methodology adopted in state-of-the-art LLM compression work such as SparseGPT [75]. As shown in Figure 15, TensorPrism achieves up to 2.41× speedup over SPADE, demonstrating its effectiveness in mildly sparse regimes.

