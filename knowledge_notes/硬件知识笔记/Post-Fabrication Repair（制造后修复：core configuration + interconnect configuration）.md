## Post-Fabrication Repair（制造后修复：core configuration + interconnect configuration）

术语解释
ConBin 在 WSC 制造与测试后，依据每片芯片独有的故障图，把设计期统一冗余模板"实例化"为可用拓扑的两阶段修复流程：先最大化可用 PE 数（core configuration），再恢复近 mesh 连通（interconnect configuration）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
阶段 1 Core Configuration（最大化可用 PE 数）：把核分为四类——全健康 C^h、仅 router 故障 C^r、仅 PE 故障 C^p、全故障 C^f；对每个 P∈C^r，沿冗余 P-R 链 BFS 寻找 replacement chain（重绑定链）：P 的 PE 改挂到健康 router，被挂 router 释放原 PE 改挂下一 router，依此类推，链必须终止于 C^p 核（提供"空 PE 位"），可修复的 C^r 核数 ≤ |C^p|；多候选链时选最短者以最小化重绑定延迟，完成后按新距离更新各 PE-router 延迟。阶段 2 Interconnect Configuration（恢复近 mesh 连通）：对每个 router 的东/南/西/北端口各选一条互连，优先级 (i) 两端点均功能正常则用原 mesh 链；(ii) 否则选最短可用冗余链（R-R）以最小额外延迟逼近 mesh。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
修复把"设计期统一模板"与"每片独特故障图"解耦：模板保证可制造性（全片相同、R_max=6），修复保证每片可用性。运转例子：一个 router-only 故障核 P 的 PE 经 2 跳 P-R 链重绑到健康 router——该 router 释放原 PE 挂到下一核……链尾 C^p 核提供空位；随后每方向端口按优先级定链，聚簇故障区由长程 R-R 链提供旁路。修复后按 turn-prohibition 最短路径路由。开销：128×136 下修复 7.12 s（40×48 为 1.17 s），线性可扩展。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
BFS 最短路链 + 端口优先级规则均为确定性多项式算法，可产线执行；与纯软件容错调度互补（ConBin 用它先恢复结构，再用 bin 感知映射/调度收敛性能）。Web 证据：Cerebras 专利 US11328208B2 的对应物是"用 wafer test/in-situ 缺陷信息配置每核冗余耦合"；学术界 router-level 冗余（spare router）的配置算法（SARA/DAPA）同样是修复期配置问题，但仅限单跳 spare。

涉及论文标题：
- ConBin: A Performance-Convergence Framework for Wafer-Scale Chip Binning
