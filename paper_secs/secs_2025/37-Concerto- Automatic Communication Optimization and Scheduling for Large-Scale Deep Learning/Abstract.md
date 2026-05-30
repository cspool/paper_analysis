# Abstract

With the exponential growth of deep learning (DL), there arises an escalating need for scalability. Despite significant advancements in communication hardware capabilities, the time consumed by communication remains a bottleneck during training. The existing various optimizations are coupled within parallel systems to implement specific computationcommunication overlap. These approaches pose challenges in terms of performance, programmability, and generality. In this paper, we introduce Concerto, a compiler framework designed to address these challenges by automatically optimizing and scheduling communication. We formulate the scheduling problem as a resource-constrained project scheduling problem and use off-the-shelf solver to get the near-optimal scheduling. And use auto-decomposition to create overlap opportunity for critical (synchronous) communication. Our evaluation shows Concerto can match or outperform state-of-the-art parallel frameworks, including Megatron-LM, JAX/XLA, DeepSpeed, and Alpa, all of which include extensive hand-crafted optimization. Unlike previous works, Concerto decouples the parallel approach and

<sup>∗</sup>Shenggan and Shengjie contributed equally. † Corresponding Author.

![](_page_0_Picture_24.jpeg)

[This work is licensed under a Creative Commons](https://creativecommons.org/licenses/by/4.0/) [Attribution International 4.0 License.](https://creativecommons.org/licenses/by/4.0/)

ASPLOS '25, March 30–April 3, 2025, Rotterdam, Netherlands © 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-0698-1/25/03 <https://doi.org/10.1145/3669940.3707223>

