# Abstract

As large language models continue to scale up, distributed training systems have expanded beyond 10k nodes, intensifying the importance of fault tolerance. Checkpoint has emerged as the predominant fault tolerance strategy, with extensive studies dedicated to optimizing its efficiency. However, the advent of the sparse Mixture-of-Experts (MoE) model presents new challenges due to the substantial increase in model size, despite comparable computational demands to dense models.

In this work, we propose the Mixture-of-Checkpoint System (MoC-System) to orchestrate the vast array of checkpoint shards produced in distributed training systems. MoC-System features a novel Partial Experts Checkpointing (PEC) mechanism, an algorithm-system co-design that strategically saves a selected subset of experts, effectively reducing the MoE checkpoint size to levels comparable with dense models. Incorporating hybrid parallel strategies, MoC-System involves fully sharded checkpointing strategies to evenly distribute the workload across distributed ranks. Furthermore, MoC-System introduces a two-level checkpointing management method that asynchronously handles in-memory snapshots and persistence processes.

We build MoC-System upon the Megatron-DeepSpeed framework, achieving up to a 98.9% reduction in overhead for each checkpointing process compared to the original method, during MoE model training with ZeRO-2 data parallelism and expert parallelism. Additionally, extensive empirical analyses substantiate that our methods enhance efficiency while

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. ASPLOS '25, Rotterdam, Netherlands.

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.

ACM ISBN 979-8-4007-1079-7/25/03

<https://doi.org/10.1145/3676641.3716006>

maintaining comparable model accuracy, even achieving an average accuracy increase of 1.08% on downstream tasks.

CCS Concepts: • Computer systems organization → Reliability.

Keywords: Fault Tolerance, Checkpoint, Mixture of Experts, Large Language Models, Training

#### ACM Reference Format:

Weilin Cai, Le Qin, and Jiayi Huang. 2025. MoC-System: Efficient Fault Tolerance for Sparse Mixture-of-Experts Model Training. In Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '25), March 30–April 3, 2025, Rotterdam, Netherlands. ACM, New York, NY, USA, [17](#page-16-0) pages. [https://doi.org/10.1145/](https://doi.org/10.1145/3676641.3716006) [3676641.3716006](https://doi.org/10.1145/3676641.3716006)

