# [Yiming Zhang](https://orcid.org/0000-0001-6450-8485)†

NICE Lab, Shanghai Jiao Tong University Shanghai, China Alibaba Cloud Beijing, China sdiris@gmail.com

## Abstract

This paper presents zBuffer, a zero-copy and metadata-free serialization library for high-performance and low-cost RPCs. At the core of zBuffer is scatter-gather reflection, a novel technique that collaboratively (i) leverages the NIC scattergather hardware feature to offload the costly data coalescing, and (ii) utilizes the static reflection mechanism of modern programming languages to enable type queries on complex data objects without requiring explicit metadata construction. We leverage C++ language features, mainly including template meta-programming and macros, to realize static reflection at compile time. Based on zBuffer, we design a fast RPC system (called zRPC) which eliminates all RPC memory copy overheads not only in (de)serialization but also in network transmission. Extensive evaluation shows that zBuffer/zRPC significantly outperforms state-of-the-art serialization/RPC mechanisms: zBuffer is approximately 7× faster than Cornflakes in serialization for complex data objects; and zRPC reduces 99th percentile latency by 21% and achieves 62% higher throughput than eRPC on the Masstree key-value (KV) store with the YCSB benchmark.

CCS Concepts: • Computing methodologies → Parallel programming languages; • Networks → Programming interfaces.

<sup>†</sup>Yiming Zhang and Youmin Chen are the corresponding authors.

![](_page_0_Picture_16.jpeg)

[This work is licensed under a Creative Commons Attribution 4.0 Interna](https://creativecommons.org/licenses/by/4.0)[tional License.](https://creativecommons.org/licenses/by/4.0)

PPoPP '26, Sydney, NSW, Australia © 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2310-0/2026/01 <https://doi.org/10.1145/3774934.3786426>

Keywords: Zero-Copy, Data Serialization, Reflection

#### ACM Reference Format:

Xiangyu Liu, Huiba Li, Shun Gai, Youmin Chen, and Yiming Zhang. 2026. zBuffer: Zero-Copy and Metadata-Free Serialization for Fast RPC with Scatter-Gather Reflection. In Proceedings of the 31st ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming (PPoPP '26), January 31 – February 4, 2026, Sydney, NSW, Australia. ACM, New York, NY, USA, [13](#page-12-0) pages. [https://doi.](https://doi.org/10.1145/3774934.3786426) [org/10.1145/3774934.3786426](https://doi.org/10.1145/3774934.3786426)

