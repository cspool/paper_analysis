# C. Automated Weight Quantization via Greedy Search

A single tensor may require many distinct DynFP configurations across its constituent groups, and the corresponding

search space is prohibitively large, spanning multiple exponent, mantissa layouts, optional gap-insertion variants, and numerous candidate Z-values. Exhaustive or manual exploration is therefore impractical. To navigate this combinatorial space efficiently, we introduce an automated search framework that systematically synthesizes the optimal DynFP configuration for each group. Unlike prior approaches that rely on a fixed or globally uniform format, our method constructs a datadriven numerical representation that adapts to each group's local distribution within a compact format set, preserving pergroup flexibility while reducing quantization error.

We describe the procedure using DynFP-4 as an example. The algorithm builds a palette of k formats (e.g., k = 16) for each tensor through an iterative, greedy selection process. The workflow comprises three phases.

Candidate Pool Generation. The framework first enumerates all viable DynFP-4 configurations, generating a pool of 96 candidate formats. These candidates arise from four base E/M layouts (E3M0, E2M1, E1M2, and E1M2I) combined with 24 possible Z-value assignments. To avoid reintroducing subnormals, Z is restricted to values within the normal region of the internal E3M2 domain (Z ≥ 0.5).

Iterative Greedy Format Search. Instead of allowing each group to choose freely among all 96 candidates—which would be costly in metadata and search time—the algorithm constructs a compact tensor-level palette P of size k. Initially, P is empty. In the initialization step (t = 1), the algorithm evaluates every candidate format and selects the one that minimizes the global mean squared error (MSE) across all groups. This format becomes f<sup>1</sup> in the palette. In the subsequent greedy iterations (t = 2 to k), each remaining candidate f<sup>t</sup> is evaluated by computing the global MSE obtained when every group is allowed to choose its best format from P ∪ ft. The candidate that yields the largest marginal reduction in global MSE is appended to the palette. The process repeats until P contains k formats.

Final Format Assignment. Once the palette is finalized, the algorithm performs a final pass over the tensor. Each group selects the single format in P that minimizes its local quantization error. The index of the chosen format (e.g., a 4 bit value for a 16-entry palette) is then stored alongside the group's scaling factor. And the Z of each format is loaded into UNICORE's Unified Format Converter when computing a new weight tensor.

This greedy search procedure behaves similarly to clustering the weights in representation space, requires no activation calibration, does not introduce distribution bias, and is highly efficient as a one-time offline step per checkpoint. For instance, quantizing Llama-2-7B with this method completes in about two minutes on a single RTX 6000 Ada GPU, after which inference only uses the stored format indices and scales, requiring no runtime format search.

