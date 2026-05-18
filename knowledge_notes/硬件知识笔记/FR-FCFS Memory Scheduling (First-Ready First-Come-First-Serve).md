## FR-FCFS Memory Scheduling (First-Ready First-Come-First-Serve)

术语是什么？

FR-FCFS (First-Ready First-Come-First-Serve) 是 DRAM memory controller 中广泛使用的命令调度策略。其核心思想是：在每轮调度中，优先选择当前 ready（满足所有 timing constraints 且 bank 可用）的请求中最早到达者。如果当前没有 ready 请求就绪，MC 等待直到有 bank 满足条件。FR-FCFS 在充分利用 DRAM 带宽（通过优先服务 ready 请求最大化 bank-level parallelism 和 row-buffer hit rate）和保证公平性（通过 oldest-first 策略防止 starvation）之间取得平衡。

从硬件架构角度拆解术语：

在传统 HBM4 MC 中，FR-FCFS 调度器每周期执行：扫描 request queue（通常 ~64 entries 深度）→对每个 entry 检查 target bank 状态（Idle/Activating/Active/Reading/Writing/Precharging/Refreshing）→检查 timing constraints（tRCDRD/tRAS/tRP/tCCDS/tRRDS/tFAW 等 15 个参数）→优先选 ready 的 oldest request →若多个 ready，选 bank group/PC 与上一个命令不同的（做 bank group interleaving 和 PC interleaving）→发出命令。RoMe 保留 FR-FCFS 调度哲学但大幅简化：仅需检查 VBA 状态（4 states）、10 timing parameters、且跨 VBA 交错（避免连续访问同 VBA）即可满带宽。request queue 从 ≥45 entries 降到 2 entries，MC scheduling logic 面积仅为 conventional MC 的 9.1%。

术语一般如何实现？如何使用？

FR-FCFS 首次由 Rixner et al. (ISCA 2000) 提出，现已成为现代 DRAM controller 的 baseline scheduling policy（包括 GPU 的 HBM MC）。RoMe 论文的 baseline 和 RoMe MC 均使用 FR-FCFS 策略。在实际硅片中，FR-FCFS 调度器通常以 CAM-based request queue + per-bank FSM + timing parameter checker 组合实现，是 MC 面积的主要贡献者之一。

涉及论文标题：
- RoMe: Row Granularity Access Memory System for Large Language Models

