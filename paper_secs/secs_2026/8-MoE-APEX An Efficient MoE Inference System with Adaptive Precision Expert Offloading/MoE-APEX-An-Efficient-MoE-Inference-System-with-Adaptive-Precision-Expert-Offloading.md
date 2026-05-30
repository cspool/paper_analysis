# MoE-APEX: An Efficient MoE Inference System with Adaptive Precision Expert Offloading

Peng Tang<sup>∗</sup> Shanghai Jiao Tong University Shanghai, China tttppp@sjtu.edu.cn

Yifei Pu Shanghai Jiao Tong University Shanghai, China pkq2006@sjtu.edu.cn

Jiacheng Liu<sup>∗</sup> The Chinese University of Hong Kong Hong Kong, China liujiacheng@ieee.org

Jing Wang Shanghai Jiao Tong University Shanghai, China jing618@sjtu.edu.cn

Xiaofeng Hou† Shanghai Jiao Tong University Shanghai, China hou-xf@cs.sjtu.edu.cn

Pheng-Ann Heng The Chinese University of Hong Kong Hong Kong, China pheng@cse.cuhk.edu.hk

Chao Li† Shanghai Jiao Tong University Shanghai, China lichao@cs.sjtu.edu.cn

## Abstract

Mixture-of-experts (MoE) architectures enable scalable Large Language Models (LLMs) with reduced computational overhead, yet their deployment on memory-constrained edge devices is hindered by substantial memory demands. Traditional expert-offloading techniques mitigate memory constraints but often significantly increase inference latency. We introduce MoE-APEX, an Adaptive Precision EXpert offloading system that optimizes MoE inference for edge architectures by dynamically managing expert precision. Our core innovation is to replace less critical cache-miss experts with low-precision variants, reducing loading latency while maintaining accuracy. MoE-APEX introduces three innovative techniques that map the natural hierarchy of MoE computation: (1) a token-level dynamic expert loading mechanism, (2) a layer-level adaptive expert prefetching technique, and (3) a sequence-level cost-aware expert caching policy. These innovations enable MoE-APEX to leverage the benefits of mixed-precision expert inference fully. Implemented atop Llama.cpp, MoE-APEX achieves decoding speedups ranging from 1.34× to 9.75× compared to state-of-the-art MoE offloading systems across diverse edge devices, offering a robust solution for efficient MoE deployment in resourceconstrained environments.

<sup>†</sup>Co-corresponding authors.

![](_page_0_Picture_12.jpeg)

[This work is licensed under a Creative Commons Attribution 4.0 Interna](https://creativecommons.org/licenses/by/4.0)[tional License.](https://creativecommons.org/licenses/by/4.0)

ASPLOS '26, Pittsburgh, PA, USA. © 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2359-9/2026/03 <https://doi.org/10.1145/3779212.3790187>

Minyi Guo Shanghai Jiao Tong University Shanghai, China guo-my@cs.sjtu.edu.cn

CCS Concepts: • Computer systems organization → Real-time system architecture.

Keywords: Edge Computing; Inference Acceleration; Parameter Offloading

#### ACM Reference Format:

Peng Tang, Jiacheng Liu, Xiaofeng Hou, Yifei Pu, Jing Wang, Pheng-Ann Heng, Chao Li, and Minyi Guo. 2026. MoE-APEX: An Efficient MoE Inference System with Adaptive Precision Expert Offloading. In Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '26), March 21–26, 2026, Pittsburgh, PA, USA. ACM, New York, NY, USA, [16](#page-15-0) pages. [https://doi.org/10.1145/](https://doi.org/10.1145/3779212.3790187) [3779212.3790187](https://doi.org/10.1145/3779212.3790187)

