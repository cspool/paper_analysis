# D. Pareto-Optimality of FuseMax

We further observe that by varying the size of the PE array (between  $16 \times 16$  and  $512 \times 512$ ) and setting the global and per-

<sup>6</sup>FLAT reports larger energy savings over the unfused baseline because it only reports energy associated with DRAM traffic during the tensor products.

PE buffers to accommodate the resulting pipelined/interleaved binding, we generate a family of designs for efficient transformer inference.

## VII. RELATED WORK

Spatial architectures have been applied successfully to a variety of domains in academia [9], [10], [40], [45] and industry [3], [27]. Beyond FLAT [28] (discussed in the main body of the paper), TileFlow [62] is a framework for modeling and searching for efficient fused dataflows (including for attention) on spatial architectures. Though TileFlow does explore a broader space of dataflows than FLAT, even implementing the 2-pass softmax cascade (Section IV-E2), its dataflows remain softmax-compute limited. Recent work has explored the scheduling/compilation of a multi-Einsum kernels [21], [54], [62]. However, these works explore a limited set of transformations, making FuseMax's inter-Einsum interleaving not discoverable.

Quantization and sparsity have also been successfully applied to reduce the transformer inference compute and live footprint. We view these schemes as complementary to our work. GPTQ [20], AWQ [32], and LLM.int8() [16] quantize model weights to 4 or 8 bits without significant accuracy degradation. Outlier-aware quantization schemes like GOBO [60] and OliVe [22] quantize both weights and activations to a low-bit precision on specific hardware designs. SpAtten [53] prunes entire tokens and heads, while Sanger [34] and DOTA [46] use quantized or low-rank projected Q and K tensors to estimate which values of QK and A can be safely pruned. All of these algorithms are expressible as cascades of Einsums, and therefore, may be combined with FuseMax to improve performance and energy efficiency, though we leave their specification and implementation to future work.

## VIII. CONCLUSION

This paper advanced the state of the art in spatial accelerator design for transformer inference. To do so, we expressed attention and its variants as cascades of Einsums. We used these cascades to reason about attention's characteristics, independent of its mapping/scheduling. Using these principles, we proposed FuseMax—an accelerator that uses deep fusion and fine-grain pipelining to map attention onto a spatial architecture. FuseMax achieves  $\sim 100\%$  utilization of both PE arrays, demonstrating  $6.7\times$  speedup over the prior state-of-the-art (FLAT) using 79% of the energy on attention and  $5.3\times$  speedup over FLAT using 83% of the energy on end-to-end inference.

Our work shows that cascades of Einsums provide a powerful abstraction for representing and analyzing domain-specific kernels. Future work may explore their application to other attention variants (e.g., those exploiting quantization and sparsity) or even other domains (e.g., fully homomorphic encryption, scientific computing, relational algebra, etc.). Doing so enables mapping-agnostic analysis and may elucidate previously undiscovered cascades and schedules for these algorithms.

#### ACKNOWLEDGMENT

We thank the anonymous MICRO and MLArchSys reviewers for their feedback on submitted versions of the work; Abhimanyu Bambhaniya, Sheng-Chun Kao and Tushar Krishna for discussions on FLAT; and finally Tanner Andrulis and Angshuman Parashar for help with Timeloop. We would also like to thank Nafea Bshara, Ron Diamant, Serina Tan, Stephen Neuendorffer, Yakun Sophia Shao, Hongbin Zheng, and others at Amazon, AMD, and NVIDIA for helpful discussions about the work as it matured.

## APPENDIX

#### A. Abstract

In this artifact, we provide Timeloop and Accelergy models of the accelerator FuseMax, an accelerator for encoder-style transformer inference. For ease-of-use, we provide a Docker container and a set of Jupyter notebooks through which to run the experiments. This artifact can be evaluated on an x86-84 machine with 5 GB of disk space.

## *B.* Artifact check-list (meta-information)

- Algorithm: Timeloop/Accelergy model of the FuseMax accelerator and the baselines it was evaluated against
- Program: Python, Timeloop, Accelergy
- Run-time environment: Docker, Jupyter
- Hardware: x86-64 machine
- Output: Plots generated from scripts
- Experiments: Modeling of the five different accelerator design points via Timeloop and Accelergy models
- How much disk space required (approximately)?: 5GB
- How much time is needed to prepare workflow (approximately)?: 20 minutes
- How much time is needed to complete experiments (approximately)?: 9 hours
- Publicly available?: Yes
- Archived (provide DOI)?: Provided after evaluation

