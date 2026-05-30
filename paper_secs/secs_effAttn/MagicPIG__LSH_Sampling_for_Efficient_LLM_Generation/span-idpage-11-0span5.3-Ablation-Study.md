# <span id="page-11-0"></span>5.3 Ablation Study

In this section, we empirically validate our two previous observations.

Centering is important for good performance. In Section 4.3, we use a translation to center the keys before applying LSH sampling. Empirical results show this to be important for downstream tasks as shown in Figure 9a. Without centering, the accuracy drops to almost zero in retrieval (NIAH) and degrades to 65% in FWE. We find almost no keys (less than 0.1%) can be sampled by the query without centering, as their orientation is almost opposite, as shown in Figure 2c.

Sampling goes beyond TopK. In Figures 9b and 9c, We compare the performance of MAGICPIG and TopK attention in two aggregated tasks (CWE, FWE) where TopK attention experiences significant performance degradation (Figure 1). MAGICPIG can even beat exact TopK attention in these two tasks by a margin up to 3% and 8% respectively, demonstrating that sampling improves the ceiling of TopK, which is impossible for a search-only algorithm.

## 6 Conclusion

In this work, we first present the limitation of TopK attention approximation for addressing the computational and memory challenges of long-context LLM generation. Then we show oracle sampling can go beyond TopK and introduce MagicPig, a novel approach that leverages LSH sampling to approximate the oracle sampling. MagicPig significantly reduces the workload of attention computation while preserving high accuracy across diverse tasks. MagicPig relies on LSH sampling and a system co-design that offloads hash tables and reduced attention computation to the CPU. Our experimental results demonstrate that MagicPig substantially improves throughput and latency across multiple hardware configurations, outperforming traditional TopK attention mechanisms. The theoretical soundness, robustness, and scalability of MagicPig open up new opportunities in both attention approximation methods and algorithm-hardware co-design.

