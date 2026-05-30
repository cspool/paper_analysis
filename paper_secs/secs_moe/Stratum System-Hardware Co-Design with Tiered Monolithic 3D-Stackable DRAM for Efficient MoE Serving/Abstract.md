# Abstract

As Large Language Models (LLMs) continue to evolve, Mixture of Experts (MoE) architecture has emerged as a prevailing design for achieving state-of-the-art performance across a wide range of tasks. MoE models use sparse gating to activate only a handful of expert sub-networks per input, achieving billion-parameter capacity with inference costs akin to much smaller models. However, such models often pose challenges for hardware deployment due to the massive data volume introduced by the MoE layers. To address the challenges of serving MoE models, we propose Stratum, a system–hardware co-design approach that combines the novel memory technology Monolithic 3D-Stackable DRAM (Mono3D DRAM), near-memory processing (NMP), and GPU acceleration. The logic and Mono3D DRAM dies are connected through hybrid bonding, whereas the Mono3D DRAM stack and GPU are interconnected via silicon interposer. Mono3D DRAM offers higher internal bandwidth than HBM thanks to the dense vertical interconnect pitch enabled by its monolithic structure, which supports implementations of higher-performance near-memory processing. Furthermore, we tackle the latency differences introduced by aggressive vertical scaling of Mono3D DRAM along the -dimension by constructing internal memory tiers and assigning data across layers based on access likelihood, guided by topic-based expert usage prediction to boost NMP throughput. The Stratum system achieves up to 8.29×

<sup>∗</sup>Equal contribution

![](_page_0_Picture_15.jpeg)

[This work is licensed under a Creative Commons Attribution 4.0 International License.](https://creativecommons.org/licenses/by/4.0) MICRO '25, Seoul, Republic of Korea © 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-1573-0/2025/10 <https://doi.org/10.1145/3725843.3756043>

Mingu Kang University of California, San Diego La Jolla, United States mingu@ucsd.edu

improvement in decoding throughput and 7.66× better energy efficiency across various benchmarks compared to GPU baselines.

