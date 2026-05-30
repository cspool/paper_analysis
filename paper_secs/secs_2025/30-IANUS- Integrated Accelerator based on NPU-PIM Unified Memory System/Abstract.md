# Abstract

Accelerating end-to-end inference of transformer-based large language models (LLMs) is a critical component of AI services in datacenters. However, the diverse compute characteristics of LLMs' end-to-end inference present challenges as previously proposed accelerators only address certain operations or stages (e.g., self-attention, generation stage, etc.). To address the unique challenges of accelerating end-toend inference, we propose IANUS – Integrated Accelerator based on NPU-PIM Unified Memory System. IANUS is a domain-specific system architecture that combines a Neural Processing Unit (NPU) with a Processing-in-Memory (PIM) to leverage both theNPU's high computation throughput and the PIM's high effective memory bandwidth. In particular, IANUS employs a unified main memory system where the PIM memory is used both for PIM operations and for NPU's main memory. The unified main memory system ensures that memory capacity is efficiently utilized and the movement of shared data between NPU and PIM is minimized. However, it introduces new challenges since normal memory accesses and PIM computations cannot be performed simultaneously.

![](_page_0_Picture_4.jpeg)

[This work is licensed under a Creative Commons Attribution International](https://creativecommons.org/licenses/by/4.0/)  4.0 License.

ASPLOS '24, April 27-May 1, 2024, La Jolla, CA, USA © 2024 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-0386-7/24/04. <https://doi.org/10.1145/3620666.3651324>

Thus, we propose novel PIM Access Scheduling that manages not only the scheduling of normal memory accesses and PIM computations but also workload mapping across the PIM and the NPU. Our detailed simulation evaluations show that IANUS improves the performance of GPT-2 by 6.2× and 3.2×, on average, compared to the NVIDIA A100 GPU and the state-of-the-art accelerator. As a proof-of-concept, we develop a prototype of IANUS with a commercial PIM, NPU, and an FPGA-based PIM controller to demonstrate the feasibility of IANUS.

CCS Concepts: • Computer systems organization → Heterogeneous (hybrid) systems; • Computing methodologies → Planning and scheduling.

Keywords: Accelerators, Heterogeneous Architectures, Neural Processing Unit, Processing-in-memory, Large Language Model, Workload Mapping, Scheduling

## ACM Reference Format:

Minseok Seo, Xuan Truong Nguyen, Seok Joong Hwang, Yongkee Kwon, Guhyun Kim, Chanwook Park, Ilkon Kim, Jaehan Park, Jeongbin Kim, Woojae Shin, Jongsoon Won, Haerang Choi, Kyuyoung Kim, Daehan Kwon, Chunseok Jeong, Sangheon Lee, Yongseok Choi, Wooseok Byun, Seungcheol Baek, Hyuk-Jae Lee, and John Kim. 2024. IANUS: Integrated Accelerator based on NPU-PIM Unified Memory System. In 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3 (ASPLOS '24), April 27-May 1, 2024, La Jolla, CA, USA. ACM, New York, NY, USA, 16 pages. [https://doi.org/10.1145/](https://doi.org/10.1145/3620666.3651324) [3620666.3651324](https://doi.org/10.1145/3620666.3651324)

