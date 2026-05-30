# Abstract

Recent foundation models are capable of handling multiple tasks and multiple data modalities with the unified base model structure and several specialized model components. However, efficient training of such multi-task (MT) multimodal (MM) models poses significant system challenges due to the sophisticated model architecture and the heterogeneous workloads of different tasks and modalities.

In this paper, we propose Spindle, a brand new training system tailored for resource-efficient and high-performance training of MT MM models via wavefront scheduling. The key idea of Spindle is to decompose the model execution into waves and address the joint optimization problem sequentially, including both heterogeneity-aware workload parallelization and dependency-driven execution scheduling. We build our system and evaluate it on various MT MM models. Experiments demonstrate the superior performance and efficiency of Spindle, with speedup ratio up to 71% compared to state-of-the-art training systems.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. ASPLOS '25, Rotterdam, Netherlands

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.

ACM ISBN 979-8-4007-1079-7/2025/03 <https://doi.org/10.1145/3676641.3715992> CCS Concepts: • Computer systems organization → Cloud computing; • Computing methodologies → Artificial intelligence; Parallel computing methodologies.

Keywords: Multi-Task Large Models; Distributed Training; Workload Heterogeneity

### ACM Reference Format:

Yujie Wang, Shenhan Zhu, Fangcheng Fu, Xupeng Miao, Jie Zhang, Juan Zhu, Fan Hong, Yong Li, and Bin Cui. 2025. Spindle: Efficient Distributed Training of Multi-Task Large Models via Wavefront Scheduling. In Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '25), March 30-April 3, 2025, Rotterdam, Netherlands. ACM, New York, NY, USA, [17](#page-16-0) pages. [https://doi.org/](https://doi.org/10.1145/3676641.3715992) [10.1145/3676641.3715992](https://doi.org/10.1145/3676641.3715992)

