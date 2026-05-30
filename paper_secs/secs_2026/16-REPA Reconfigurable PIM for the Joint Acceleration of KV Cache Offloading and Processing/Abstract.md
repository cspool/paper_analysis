# Abstract

The use of KV cache in LLM inference leads to large memory footprint and sub-optimal decoding performance. Prior studies typically address one of these two limitations by either offloading or stage-split inference. In this paper, we explore and reveal the possibility of a joint solution, and propose REPA, a GPU-PIM hybrid system to prototype this idea. We leverage reconfigurable ReRAM PIM to achieve fast KV cache persistence, and balance the requirement of processing speed and memory capacity. To fully unleash the parallelization potential of REPA, we propose optimizations in (1) architecture, (2) data mapping and (3) pipelining: (1) We propose bulk-wise memory instructions and multi-level controllers to enable finer-grained parallelism in the PIM device. (2) We propose locality-aware data mapping to make the best of the aforementioned architectural optimization, and reduce long-range data transfer on chip. (3) We adopt subbatch pipelining to reduce idleness in batches, and propose transfer overlapping to shadow the KV cache transfer by computation. Experimental results show that REPA exhibits high inference speed, energy efficiency and integratability. It is 1.5–6.5× faster, and 8–10× more efficient than NVIDIA A100. It also outperforms state-of-the-art DRAM PIM systems by up to 1.4× for long context inference. When integrated into existing offloading systems, REPA achieves 1.4–2.0× offloading speed, and 1.2–1.4× end-to-end speedup, showcasing its high potential for fast KV cache offloading and processing.

CCS Concepts: • Computer systems organization → Heterogeneous (hybrid) systems.

<sup>∗</sup>Corresponding author.

![](_page_0_Picture_9.jpeg)

[This work is licensed under a Creative Commons Attribution 4.0 Interna](https://creativecommons.org/licenses/by/4.0)[tional License.](https://creativecommons.org/licenses/by/4.0)

ASPLOS '26, March 22–26, 2026, Pittsburgh, PA, USA. © 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2359-9/2026/03 <https://doi.org/10.1145/3779212.3790212>

