# Abstract

Sparse matrix-matrix multiplication (SpMM) is a critical computational kernel in numerous scientific and machine learning applications. SpMM involves massive irregular memory accesses and poses great challenges to conventional cachebased computer architectures. Recently dedicated SpMM accelerators have been proposed to enhance SpMM performance. However, current SpMM accelerators still face challenges in adapting to varied sparse patterns, fully exploiting inherent parallelism, and optimizing cache performance. To address these issues, we introduce ACES, a novel SpMM accelerator in this study. First, ACES features an adaptive execution flow that dynamically adjusts to diverse sparse patterns. The adaptive execution flow balances parallel computing efficiency an d da ta re use. Se cond, ACES incorporates locality-concurrency co-optimizations within the global cache. ACES utilizes a concurrency-aware cache management policy, which considers data locality and concurrency for optimalreplacement decisions. Additionally, the integration of a non-blocking buffer with the global cache enhances concurrency and reduces computational stalls. Third, the hardware architecture of ACES is designed to integrate

<sup>†</sup>Corresponding authors.

![](_page_0_Picture_9.jpeg)

[This work is licensed under a Creative Commons Attribution International](https://creativecommons.org/licenses/by/4.0/)  4.0 License.

ASPLOS '24, April 27-May 1, 2024, La Jolla, CA, USA © 2024 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-0386-7/24/04.

<https://doi.org/10.1145/3620666.3651381>

Xian-He Sun† sun@iit.edu Department of Computer Science, Illinois Institute of Technology USA

all innovations. The architecture ensures efficient support across the adaptive execution flow, advanced cache optimizations, and fine-grained parallel processing. Our performance evaluation demonstrates that ACES significantly outperforms existing solutions, providing a 2.1× speedup and marking a substantial advancement in SpMM acceleration.

#### ACM Reference Format:

Xiaoyang Lu, Boyu Long, Xiaoming Chen, Yinhe Han, and Xian-He Sun. 2024. ACES: Accelerating Sparse Matrix Multiplication with Adaptive Execution Flow and Concurrency-Aware Cache Optimizations. In 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3 (ASP-LOS '24), April 27-May 1, 2024, La Jolla, CA, USA. ACM, New York, NY, USA, 15 pages. <https://doi.org/10.1145/3620666.3651381>

