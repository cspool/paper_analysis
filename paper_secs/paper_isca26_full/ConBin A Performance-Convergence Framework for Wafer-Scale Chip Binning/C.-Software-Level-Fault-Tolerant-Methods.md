# *C. Software-Level Fault-Tolerant Methods*

Beyond hardware redundancy, software-level techniques mitigate the impact of defects and ensure reliable task management through fault-tolerant task scheduling. There are two classifications of fault-tolerant scheduling approaches. The first targets traditional applications through task mapping [12], [42], [43], [57], remapping and migration techniques [2], [13], [14], [16], [41], [62]. These methods maintain reliable execution and restore runtime performance in the presence of faults, commonly modeled by Weibull [26] or Lognormal distributions [23]. The second classification, though less studied, is dedicated to accelerating AI workloads. Si-Kintsugi [27] proposes a workload mapping model paired with a scalable cost function that reflects how faults affect intercore communication distances. The work further evaluates its performance under random and structured wafer-scale fault patterns, including local, center, edge-local, and scratch defects.

Prior MPSoC solutions that focus on communication latency or throughput [2], [13], [14], [43] and wafer-scale work [27] mainly minimize hop count, as accurately modeling contention remains computationally complex [57]. However,

![](_page_3_Figure_7.jpeg)

Fig. 5. Performance variance comparison of 128×136 chips before and after optimization with baseline CB\* (based on the Cerebras redundant architecture [39]) and Si-Kintsugi scheduling (SK) [27]; variance measured on performance normalized to the fault-free chip.

manufacturing defects intensify communication contention on WSCs, which becomes the primary latency source (further elaborated in Sec.VI-A), rendering hop-based optimization insufficient. We therefore propose scalable contention-aware methods to effectively recover performance.

# *C. Software-Level Fault-Tolerant Methods*

Beyond hardware redundancy, software-level techniques mitigate the impact of defects and ensure reliable task management through fault-tolerant task scheduling. There are two classifications of fault-tolerant scheduling approaches. The first targets traditional applications through task mapping [12], [42], [43], [57], remapping and migration techniques [2], [13], [14], [16], [41], [62]. These methods maintain reliable execution and restore runtime performance in the presence of faults, commonly modeled by Weibull [26] or Lognormal distributions [23]. The second classification, though less studied, is dedicated to accelerating AI workloads. Si-Kintsugi [27] proposes a workload mapping model paired with a scalable cost function that reflects how faults affect intercore communication distances. The work further evaluates its performance under random and structured wafer-scale fault patterns, including local, center, edge-local, and scratch defects.

Prior MPSoC solutions that focus on communication latency or throughput [2], [13], [14], [43] and wafer-scale work [27] mainly minimize hop count, as accurately modeling contention remains computationally complex [57]. However,

![](_page_3_Figure_7.jpeg)

Fig. 5. Performance variance comparison of 128×136 chips before and after optimization with baseline CB\* (based on the Cerebras redundant architecture [39]) and Si-Kintsugi scheduling (SK) [27]; variance measured on performance normalized to the fault-free chip.

manufacturing defects intensify communication contention on WSCs, which becomes the primary latency source (further elaborated in Sec.VI-A), rendering hop-based optimization insufficient. We therefore propose scalable contention-aware methods to effectively recover performance.

