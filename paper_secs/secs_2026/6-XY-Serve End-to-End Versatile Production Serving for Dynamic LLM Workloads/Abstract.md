# Abstract

Meeting growing demands for low latency and cost efficiency in production-grade large language model (LLM) serving systems requires integrating advanced optimization techniques. However, dynamic and unpredictable input-output lengths of LLM, compounded by these optimizations, exacerbate the issues of workload variability, making it difficult to maintain high efficiency on AI accelerators, especially DSAs with tile-based programming models. To address this challenge, we introduce XY-Serve, a versatile, Ascend NPU native, end-to-end production LLM-serving system. The core idea is an abstraction mechanism that smooths out the workload variability by decomposing computations into unified, hardware-friendly, fine-grained meta primitives. Then, kernels can efficiently execute without concerning the irregularity of workload. After this abstraction mechanism, for Attention, we propose a meta-kernel that computes the basic

<sup>∗</sup>Both authors contributed equally to this research.

<sup>†</sup>Corresponding author

![](_page_0_Picture_17.jpeg)

[This work is licensed under a Creative Commons Attribution 4.0 Interna](https://creativecommons.org/licenses/by/4.0)[tional License.](https://creativecommons.org/licenses/by/4.0)

ASPLOS '26, Pittsburgh, PA, USA © 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2165-6/26/03 <https://doi.org/10.1145/3760250.3762228>

pattern of GEMM-Softmax-GEMM with architectural-aware tile sizes. For Linear, we introduce a virtual padding scheme that adapts to dynamic shape changes while using highly efficient GEMM primitives with assorted fixed tile sizes. XY-Serve sits harmoniously with vLLM. Experimental results show up to 95% end-to-end throughput improvement compared with current publicly available baselines on Ascend NPUs. We also set a new performance record for Linear (average 14.6% faster) and Attention (average 21.5% faster) kernels relative to existing libraries. Lastly, we demonstrate the generality of our technologies on GPU platform.

CCS Concepts: • Hardware → Emerging architectures; • Computing methodologies → Parallel algorithms.

Keywords: Inference System; AI Accelerator; Large Language Model;

#### ACM Reference Format:

Mingcong Song, Xinru Tang, Fengfan Hou, Jing Li, Wei Wei, Yipeng Ma, Runqiu Xiao, Hongjie Si, Dingcheng Jiang, Shouyi Yin, Yang Hu, and Guoping Long. 2026. XY-Serve: End-to-End Versatile Production Serving for Dynamic LLM Workloads. In Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1 (ASPLOS '26), March 22–26, 2026, Pittsburgh, PA, USA. ACM, New York, NY, USA, [16](#page-15-0) pages. <https://doi.org/10.1145/3760250.3762228>

