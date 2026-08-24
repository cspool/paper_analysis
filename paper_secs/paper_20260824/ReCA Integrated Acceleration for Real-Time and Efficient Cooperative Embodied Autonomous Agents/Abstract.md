# Abstract

Cooperative embodied systems, where multiple agents collaborate through integrated perception, planning, action, and advanced reasoning powered by large language models (LLMs), show great potential for tackling complex, longhorizon, multi-objective tasks in real-world environments. Despite these algorithmic advancements, deploying embodied agents on current systems remains challenging due to prolonged planning and communication latency, limited scalability, and heightened sensitivity in low-level execution, all of which lead to significant system inefficiencies.

This work proposes ReCA, a characterization and co-design framework dedicated to cooperative embodied agent system acceleration, aiming to enhance both task efficiency and system scalability. On the algorithm level, ReCA enables efficient local model processing to alleviate the substantial model costs. On the system level, ReCA presents a dual-memory structure with integrated long-term and short-term memory, hierarchical cooperative planning scheme with centralized and decentralized cooperation, and planning-guided multistep execution for highly efficient and scalable cooperative embodied agent computation. On the hardware level, ReCA employs a heterogeneous hardware system with high-level planning GPU subsystem and low-level planning accelerator subsystem to ensure efficient and robust task execution. Evaluated across long-horizon multi-objective tasks, ReCA generalizes across application scenarios and system scales,

![](_page_0_Picture_13.jpeg)

[This work is licensed under a Creative Commons Attribution 4.0 Interna](https://creativecommons.org/licenses/by/4.0/legalcode)[tional License.](https://creativecommons.org/licenses/by/4.0/legalcode)

ASPLOS '25, March 30-April 3, 2025, Rotterdam, Netherlands © 2025 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-1079-7/2025/03. <https://doi.org/10.1145/3676641.3716016>

achieving a 4.3% increase in successful missions with 10.2× speedup compared to the state-of-the-art cooperative embodied autonomous agent systems.

CCS Concepts: • Computer systems organization → Embedded and cyber-physical systems; Architectures.

Keywords: Autonomous Agents, Embodied Systems, Cooperative Intelligence, Hardware Accelerator

#### ACM Reference Format:

Zishen Wan, Yuhang Du, Mohamed Ibrahim, Jiayi Qian, Jason Jabbour, Yang (Katie) Zhao, Tushar Krishna, Arijit Raychowdhury, and Vijay Janapa Reddi. 2025. ReCA: Integrated Acceleration for Real-Time and Efficient Cooperative EmbodiedAutonomous Agents. In Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '25), March 30-April 3, 2025, Rotterdam, Netherlands. ACM, New York, NY, USA, [16](#page-15-0) pages. [https://doi.org/10.1145/](https://doi.org/10.1145/3676641.3716016) [3676641.3716016](https://doi.org/10.1145/3676641.3716016)

