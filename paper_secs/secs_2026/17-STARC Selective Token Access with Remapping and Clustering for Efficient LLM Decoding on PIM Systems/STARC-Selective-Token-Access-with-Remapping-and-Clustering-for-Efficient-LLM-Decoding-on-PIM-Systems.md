# STARC: Selective Token Access with Remapping and Clustering for Efficient LLM Decoding on PIM Systems

Zehao Fan Rensselaer Polytechnic Institute Troy, NY, USA fanz2@rpi.edu

Zhenyu Liu Rensselaer Polytechnic Institute Troy, NY, USA liuz32@rpi.edu Yunzhen Liu University of Massachusetts, Amherst Amherst, MA, USA yunzhenliu@umass.edu

Yayue Hou Rensselaer Polytechnic Institute Troy, NY, USA houy4@rpi.edu Garrett Gagnon Rensselaer Polytechnic Institute Troy, NY, USA gagnog@rpi.edu

Hadjer Benmeziane IBM Research – Ruschlikon Ruschlikon, Switzerland hadjer.benmeziane@ibm.com

Kaoutar El Maghraoui IBM T. J. Watson Research Center Yorktown Heights, NY, USA kelmaghr@us.ibm.com

#### **Abstract**

Serving large language models (LLMs) places significant pressure on memory systems due to frequent accesses and growing key-value (KV) caches as context lengths increase. Processing-in-memory (PIM) architectures offer high internal bandwidth and near-data compute parallelism, but current designs target dense attention and perform poorly under the irregular access patterns of dynamic KV cache sparsity. To mitigate this limitation, we propose STARC, a sparsityoptimized data mapping scheme for efficient LLM decoding on PIM. STARC clusters semantically similar KV pairs and co-locates them contiguously within PIM banks, enabling retrieval at cluster granularity by matching queries against precomputed centroids. This bridges the gap between finegrained sparse attention and row-level PIM operations, improving utilization while minimizing overhead. On a simulated HBM-PIM system, under constrained KV budgets, STARC achieves up to 78% and 65% reductions in attentionlayer latency and energy over token-wise sparsity methods, and up to 93% and 92% reductions relative to full attention, while preserving model accuracy.

*CCS Concepts:* • Computer systems organization  $\rightarrow$  *Architectures*; • Computing methodologies  $\rightarrow$  Machine learning.

*Keywords:* Processing-in-memory (PIM); Large language model (LLM); Sparse attention; KV clustering; KV cache

![](_page_0_Picture_13.jpeg)

This work is licensed under a Creative Commons Attribution 4.0 International License.

ASPLOS '26, Pittsburgh, PA, USA
© 2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2359-9/2026/03
https://doi.org/10.1145/3779212.3790226

Liu Liu Rensselaer Polytechnic Institute Troy, NY, USA liu.liu@rpi.edu

#### **ACM Reference Format:**

Zehao Fan, Yunzhen Liu, Garrett Gagnon, Zhenyu Liu, Yayue Hou, Hadjer Benmeziane, Kaoutar El Maghraoui, and Liu Liu. 2026. STARC: Selective Token Access with Remapping and Clustering for Efficient LLM Decoding on PIM Systems. In *Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '26), March 22–26, 2026, Pittsburgh, PA, USA.* ACM, New York, NY, USA, 17 pages. https://doi.org/10.1145/3779212.3790226

#### 1 Introduction

Large language models (LLMs) have demonstrated exceptional capabilities across a wide range of natural language processing tasks and are increasingly deployed in real-world applications such as interactive chat systems [1, 57, 61], code generation tools [38, 45, 48], and decision support [29, 47, 55]. During decoding, however, LLMs operate auto-regressively, requiring repeated attention over a growing key-value (KV) cache [41]. As context lengths scale, the KV cache expands proportionally, leading to frequent and large memory accesses. Despite high computational throughput, modern GPUs are constrained by limited memory bandwidth, making attention layers predominantly memory-bound [25]. Processingin-memory (PIM) architectures [8, 13, 19, 20, 39] offer a promising solution by alleviating bandwidth bottlenecks and enabling efficient in-memory computation. Recent work has explored heterogeneous designs (e.g., GPU-PIM, NPU-PIM) that offload memory-bound attention layers to PIM while leveraging traditional accelerators (xPUs) for computeintensive feed-forward networks (FFNs) and Query-Key-Value (QKV) generation [15, 40].

However, the trend toward longer contexts continues to impose substantial computation and memory costs, driven by the quadratic complexity of attention. Recent methods alleviate this by introducing **KV cache sparsity** through

<span id="page-1-0"></span>![](_page_1_Figure_2.jpeg)

Figure 1. Enhanced PIM execution efficiency through STARC. Due to the coarse row-level access granularity of PIM, directly applying sparsity to KV caches often fails to skip computation. STARC addresses this by clustering keys and values such that selected tokens are physically co-located, enabling effective computation skipping and realizing the speedup benefits of sparsity on PIM.

selective retrieval or compression, retrieving only a subset of tokens to approximate full attention. While such methods can reduce retrieval by over 90% with minimal accuracy loss, they introduce irregular and dynamic access patterns that traditional PIM designs—optimized for dense, row-level accesses—struggle to support. Most existing PIM-enabled systems largely assume full KV cache attention, leading to underutilization when sparsity is applied. Techniques such as Quest [\[49\]](#page-16-11) address this by retrieving at page granularity, aligning with memory row organization and improving bandwidth efficiency. Yet, page-based layouts remain coarsegrained, often fetching semantically irrelevant tokens, which wastes compute and undermines accuracy. This mismatch between dynamic sparsity and rigid PIM data layouts remains a fundamental barrier to efficient LLM decoding. To address this challenge, we propose STARC, a sparsityoptimized data mapping scheme designed specifically for PIM architectures. The key idea, illustrated in Figure [1,](#page-1-0) is to cluster semantically similar tokens and physically co-locate their KV pairs in memory. Our design aligns sparse attention with PIM's row-level organization.

To overcome the mismatch between dynamic sparsity and rigid PIM layouts, we propose STARC, a sparsity-optimized data mapping scheme for LLM decoding on PIM. STARC clusters semantically similar tokens and co-locates their KV entries contiguously in memory, enabling sparse attention to align with row-level PIM operations. Queries retrieve clusters by matching against precomputed centroids, ensuring most fetched vectors are relevant and improving hardware utilization. By performing lightweight clustering directly within PIM and fixing clusters across decoding steps, STARC

achieves efficient support for sparse attention in LLM serving. This paper makes the following contributions:

- We analyze the challenges of applying KV cache sparsity to PIM-enabled LLM inference and identify the mismatch between dynamic sparse retrieval and rigid row-level PIM data layouts.
- We propose STARC, a novel clustering-based data mapping scheme that co-locates semantically similar KV entries to align sparse attention with PIM bank organization.
- We introduce efficient in-memory designs that directly leverage existing PIM primitives and hardware to implement cosine-based K-means clustering for KV clustering, avoiding additional area overhead while minimizing GPU involvement and exploiting near-data compute.
- We demonstrate that STARC significantly improves throughput, utilization, and energy efficiency over state-of-the-art PIM system baselines while preserving model accuracy under sparse attention. It reduces attention-layer latency and energy by up to 78% and 65% compared to token-wise sparsity, and under a KVcache budget of 1024, achieves up to 93% latency and 92% energy reduction relative to full KV retrieval.

