# E. Tensor Algebra Accelerators

In recent years, the popularity of domain-specific tensor algebra accelerators has increased. A typical accelerator based on a spatial architecture consists of off-chip main memory, an on-chip shared global buffer, various scratchpads, and a 1D and/or 2D processing engine (PE) array where each PE contains compute units [9], [27], [28], [38], [62]. This design minimizes memory transfer latency while maximizing compute utilization [7]–[9], [11], [27]. Various tools enable the quick modeling and design space exploration of tensor algebra accelerators, including Timeloop [41] and Accelergy [56], GAMMA [61], and DOSA [23].

## III. PASSES PERFORMED BY A CASCADE OF EINSUMS

Our first contribution is to demonstrate a novel analysis that can be applied to a cascade of Einsums. The key insight is that cascades of Einsums provide a precise description of the iteration space for each Einsum and the data space for each constituent tensor, enabling us to derive the algorithmic minimum live footprint for each tensor, with implications for the allowed fusion schedules and required buffer capacity/memory traffic. Because this analysis relies only on the cascade of Einsums, it holds for any choice of mapping and binding.

