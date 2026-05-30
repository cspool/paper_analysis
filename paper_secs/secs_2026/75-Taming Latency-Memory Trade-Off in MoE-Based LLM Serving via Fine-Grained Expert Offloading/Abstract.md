# Abstract

Large Language Models (LLMs) have gained immense success in revolutionizing various applications, including content generation, search and recommendation, and AI-assisted operations. To reduce high training costs, Mixture-of-Experts (MoE) architecture has become a popular backbone for modern LLMs. However, despite the benefits, serving MoE-based LLMs experience severe memory inefficiency due to sparsely activated experts. Recent studies propose to offload inactive experts from GPU memory to CPU memory to improve the serving efficiency of MoE models. However, they either incur high inference latency or high model memory footprints due to coarse-grained designs.

To tame the latency-memory trade-off in MoE serving, we present *FineMoE*, a fine-grained expert offloading system for MoE serving that achieves low inference latency with memory efficiency. We design *FineMoE* to extract fine-grained expert selection patterns from MoE models and semantic hints from input prompts to efficiently guide expert prefetching, caching, and offloading decisions. *FineMoE* is prototyped on top of HuggingFace Transformers and deployed on a six-GPU testbed. Experiments with open-source MoE models and real-world workloads show that *FineMoE* reduces inference latency by 47% and improves expert hit rate by 39% over state-of-the-art solutions.

*CCS Concepts:* • Computing methodologies → Distributed algorithms; Artificial intelligence; Machine learning.

<sup>\*</sup>This work was conducted while Xingqi Cui was a remote intern student, advised by Dr. Hao Wang at the IntelliSys Lab, Stevens Institute of Technology.

![](_page_0_Picture_11.jpeg)

[This work is licensed under a Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode) [License.](https://creativecommons.org/licenses/by/4.0/legalcode)

*EUROSYS '26, April 27–30, 2026, Edinburgh, Scotland Uk* © 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2212-7/26/04 <https://doi.org/10.1145/3767295.3769319>

*Keywords:* Artificial Intelligence, Large Language Model, Mixture-of-Experts, Model Serving, Offloading

#### ACM Reference Format:

Hanfei Yu, Xingqi Cui, Hong Zhang, Hao Wang, and Hao Wang. 2026. Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading. In *21st European Conference on Computer Systems (EUROSYS '26), April 27–30, 2026, Edinburgh, Scotland Uk.* ACM, New York, NY, USA, [16](#page-15-0) pages. <https://doi.org/10.1145/3767295.3769319>

