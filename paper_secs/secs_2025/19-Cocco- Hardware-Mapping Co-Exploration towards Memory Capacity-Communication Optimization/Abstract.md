# Abstract

Memory is a critical design consideration in current dataintensive DNN accelerators, as it profoundly determines energy consumption, bandwidth requirements, and area costs. As DNN structures become more complex, a larger on-chip memory capacity is required to reduce data movement overhead, but at the expense of silicon costs. Some previous works have proposed memory-oriented optimizations, such as different data reuse and layer fusion schemes. However, these methods are not general and potent enough to cope with various graph structures.

In this paper, we explore the intrinsic connection between network structures and memory features to optimize both hardware and mapping. First, we introduce a graph-level execution scheme with a corresponding dataflow and memory management method. This scheme enables the execution of arbitrary graph patterns with high data reuse and low hardware overhead. Subsequently, we propose Cocco, a hardware-mapping co-exploration framework leveraging graph-level features of networks. It aims to minimize communication overhead, such as energy consumption and bandwidth requirements, with a smaller memory capacity. We formulate the graph-partition scheduling and memory configuration search as an optimization problem and employ a genetic-based method to achieve efficient co-exploration for large and irregular networks. Experiments demonstrate that Cocco obtains lower external memory access, lower bandwidth requirements, and more stable optimization for graph partition compared to the greedy algorithm and dynamic programming introduced in prior works. Cocco also reduces the costs by 1.89% to 50.33% using co-exploration compared to other typical methods.

<sup>∗</sup>Corresponding author.

![](_page_0_Picture_8.jpeg)

[This work is licensed under a Creative Commons Attribution International](https://creativecommons.org/licenses/by/4.0/)  4.0 License.

ASPLOS '24, April 27-May 1, 2024, La Jolla, CA, USA © 2024 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-0372-0/24/04.

<https://doi.org/10.1145/3617232.3624865>

CCS Concepts: • Hardware → Design reuse and communication-based design; On-chip resource management; • Computer systems organization → Parallel architectures; • Software and its engineering → Compilers.

Keywords: Design space exploration, Memory, Graph analysis, Subgraph, Genetic algorithm, Deep learning accelerator

#### ACM Reference Format:

Zhanhong Tan, Zijian Zhu, and Kaisheng Ma. 2024. Cocco: Hardware-Mapping Co-Exploration towards Memory Capacity-Communication Optimization. In 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1 (ASPLOS '24), April 27-May 1, 2024, La Jolla, CA, USA. ACM, New York, NY, USA, 16 pages. <https://doi.org/10.1145/3617232.3624865>

