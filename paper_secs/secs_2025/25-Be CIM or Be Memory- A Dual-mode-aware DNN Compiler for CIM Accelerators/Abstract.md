# Abstract

Computing-in-memory (CIM) architectures demonstrate superior performance over traditional architectures. To unleash the potential of CIM accelerators, many compilation methods have been proposed, focusing on application scheduling optimization specific to CIM. However, existing compilation methods often overlook CIM's capability to switch dynamically between compute and memory modes, which is crucial for accommodating the diverse memory and computational needs of real-world deep neural network architectures, especially the emerging large language models. To fill this gap, we introduce CMSwitch, a novel compiler to optimize resource allocation for CIM accelerators with adaptive modeswitching capabilities, thereby enhancing the performance of DNN applications. Specifically, our approach integrates the compute-memory mode switch into the CIM compilation optimization space by introducing a new hardware abstraction

<sup>∗</sup>Corresponding author.

![](_page_0_Picture_17.jpeg)

[This work is licensed under a Creative Commons Attribution 4.0 Interna](https://creativecommons.org/licenses/by/4.0/legalcode)[tional License.](https://creativecommons.org/licenses/by/4.0/legalcode)

ASPLOS '25, March 30-April 3, 2025, Rotterdam, Netherlands © 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-1079-7/25/03. <https://doi.org/10.1145/3676641.3716248>

attribute. Then, we propose a novel compilation optimization pass that identifies the optimal network segment and the corresponding mode resource allocations using dynamic programming and mixed-integer programming. CMSwitch uses the tailored meta-operator to express the compilation result in a generalized manner. Evaluation results demonstrate that CMSwitch achieves an average speedup of 1.31× compared to existing SOTA CIM compilation works, highlighting CM-Switch's effectiveness in fully exploiting the potential of CIM processors for a wide range of real-world DNN applications.

CCS Concepts: • Hardware → Memory and dense storage; • Software and its engineering → Compilers.

Keywords: Compute-in-memory (CIM), Compilation, Deep Neural Network (DNN)

### ACM Reference Format:

Shixin Zhao, Yuming Li, Bing Li, Yintao He, Mengdi Wang, Yinhe Han, and Ying Wang. 2025. Be CIM or Be Memory: A Dual-modeaware DNN Compiler for CIM Accelerators. In Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '25), March 30-April 3, 2025, Rotterdam, Netherlands. ACM, New York, NY, USA, [16](#page-15-0) pages. <https://doi.org/10.1145/3676641.3716248>

