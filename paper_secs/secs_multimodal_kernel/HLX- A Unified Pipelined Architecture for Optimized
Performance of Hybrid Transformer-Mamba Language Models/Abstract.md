# Abstract

The rapid increase in demand for long-context language models has revealed fundamental performance limitations in conventional Transformer architectures, particularly their quadratic computational complexity. Hybrid Transformer-Mamba models, which interleave attention layers with efficient state-space model layers such as Mamba-2, have emerged as promising solutions combining the strengths of both Transformer and Mamba. However, maintaining a high compute utilization and performance across workloads (e.g., varying sequence length and batch size) in the Hybrid models is challenging due to their heterogeneous compute patterns and shifting performance bottlenecks between the two key computational kernels: FlashAttention-2 (FA-2) and State-Space Duality (SSD).

In this paper, we introduce HLX, a unified pipelined architecture designed to ensure optimized performance across workloads for Hybrid models. Through detailed kernel-level analysis, we identify two key blockers that limit compute utilization: inter-operation dependencies in FA-2 and excessive memory traffic in SSD. To overcome these hurdles, we propose two novel fine-grained pipelined dataflows named PipeFlash and PipeSSD. PipeFlash effectively hides operational dependencies in attention computations, while PipeSSD firstly introduces the fused pipelined execution for SSD computations, substantially enhancing data reuse and reducing memory traffic. In addition, we propose a unified hardware architecture that can process both PipeFlash and PipeSSD in an efficient pipelining scheme to maximize the compute utilization. Finally, across sequence lengths from 1K to 128K, the proposed HLX architecture achieves up to 97.5% and 78.4% compute utilization for FA-2 and SSD, respectively, resulting in an average speedup of 1.75× and 2.91× over A100, and an average 2.78× (FA-2), 1.84× (FA-3), and 4.95× speedups over H100. For end-to-end latency and batching, HLX achieves a 1.56× and 1.38× speedup over A100 and a 2.08× and 1.76× (1.84× and 1.72×) speedup when running FA-2 (FA-3) on H100. It also significantly reduces area and power consumption by up to 89.8% and 63.8% compared to GPU baselines.

![](_page_0_Picture_6.jpeg)

[This work is licensed under a Creative Commons Attribution 4.0 International License.](https://creativecommons.org/licenses/by/4.0) MICRO '25, Seoul, Republic of Korea © 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-1573-0/25/10 <https://doi.org/10.1145/3725843.3756115>

[Gyeongrok Yang](https://orcid.org/0009-0009-3263-0705) KAIST Daejeon, Republic of Korea toddlerf@kaist.ac.kr

[Joo-Young Kim](https://orcid.org/0000-0003-1099-1496) KAIST Daejeon, Republic of Korea jooyoung1203@kaist.ac.kr

