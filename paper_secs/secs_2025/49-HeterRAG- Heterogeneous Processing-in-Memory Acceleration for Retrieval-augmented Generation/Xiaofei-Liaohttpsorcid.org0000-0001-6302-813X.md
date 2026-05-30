# [Xiaofei Liao](https://orcid.org/0000-0001-6302-813X)∗†

Huazhong University of Science and Technology Wuhan, China xfliao@hust.edu.cn

### Abstract

By integrating external knowledge bases, Retrieval-augmented Generation (RAG) enhances natural language generation for knowledgeintensive scenarios and specialized domains, producing content that is both more informative and personalized. RAG systems typically consist of two fundamental stages: retrieval and generation. The retrieval stage experiences low bandwidth utilization due to its random and irregular memory access patterns. Meanwhile, the generation stage is also constrained by memory bandwidth limitations, which arise from involving a significant number of General Matrix-Vector Multiplications (GEMV) operations. These two stages collectively lead to memory bottlenecks within RAG systems. Recent efforts leverage HBM-based Processing-in-Memory (PIM) to accelerate conventional Large Language Models (LLMs). However, the retrieval stage incurs substantial storage overhead due to the need to maintain large-scale knowledge bases, resulting in a capacity bottleneck. Solely relying on HBM-based PIM in RAG is both costly and insufficient to meet the capacity demands. Fortunately, DIMM-based PIM provides a low-cost, high-capacity alternative that complements HBM. In this work, we propose HeterRAG, a novel heterogeneous PIM acceleration system for RAG. It combines

<sup>†</sup>Cluster and Grid Computing Lab, School of Computer Science and Technology, HUST. School of Software Engineering, HUST.

![](_page_0_Picture_19.jpeg)

[This work is licensed under a Creative Commons Attribution 4.0 International License.](https://creativecommons.org/licenses/by/4.0) ISCA '25, Tokyo, Japan © 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-1261-6/25/06 <https://doi.org/10.1145/3695053.3731089>

