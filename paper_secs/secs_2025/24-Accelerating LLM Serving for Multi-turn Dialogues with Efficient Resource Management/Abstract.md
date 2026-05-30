# Abstract

Although there have been significant efforts to make LLM serving efficient, we observe two limitations of current stateof-the-art serving frameworks in handling multi-turn dialogues between users and assistants, particularly in chat scenarios. First, existing LLM frameworks incur substantial computational overhead in recomputing attention keys and values (KVs) for understanding context across multiple turns of user queries. Second, as the prompt length of user queries is amplified due to multi-turns, a first-come-firstserved (FCFS) scheduling policy often causes head-of-line blocking issues, leading to underutilization of GPU resources.

To address these limitations, we present FlashGen to rapidly complete multi-turn queries by efficiently utilizing the compute and memory resources of GPUs as well as the host hardware (e.g., DRAM and SSD). We introduce a multi-level KV cache comprised of GPU, CPU, and SSD, to efficiently retain attention KVs from prior turns. Our approach employs low-cost cache restoration techniques to avoid the recomputation burden. Further, we propose a request reordering technique to effectively utilize GPU memory. This scheduling technique carefully adjusts the request order without compromising fairness. Our proposed techniques outperform the vLLM framework in terms of both latency and throughput. For OPT 30B and Llama-2 70B models with the ShareGPT dataset, we achieve 1.63× and 2.85× better throughput, respectively while in a similar latency boundary.

CCS Concepts: • Computer systems organization; • Software and its engineering → Memory management; Scheduling;

Keywords: LLM Serving; Multi-turn Dialogues; KV Cache Management; Request Reordering

#### ACM Reference Format:

Jinwoo Jeong and Jeongseob Ahn. 2025. Accelerating LLM Serving for Multi-turn Dialogues with Efficient Resource Management. In Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems,

![](_page_0_Picture_10.jpeg)

[This work is licensed under a Creative Commons Attribution-](https://creativecommons.org/licenses/by-nc-nd/4.0)[NonCommercial-NoDerivatives 4.0 International License.](https://creativecommons.org/licenses/by-nc-nd/4.0)

ASPLOS '25, Rotterdam, Netherlands © 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-1079-7/25/03 <https://doi.org/10.1145/3676641.3716245>

Volume 2 (ASPLOS '25), March 30-April 3, 2025, Rotterdam, Netherlands. ACM, New York, NY, USA, [15](#page-14-0) pages. [https://doi.org/10.1145/](https://doi.org/10.1145/3676641.3716245) [3676641.3716245](https://doi.org/10.1145/3676641.3716245)

