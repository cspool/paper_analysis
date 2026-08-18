## 全系统模拟：QEMU + SST Ariel + DRAMSim3 + McPAT

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PhaseWeave 的评估基础设施是 full-system 模拟栈：QEMU 捕获用户态+内核态指令、访存与系统调用 → 转发给 SST（Structural Simulation Toolkit）的 Ariel core 模型（论文 modified for high-accuracy）做 cycle 级微架构建模；主存用 DRAMSim3（cycle-accurate DRAM 时序模拟器）；面积/功耗用 McPAT（32nm 技术节点、缩放至 7nm）。模拟环境含完整软件栈（Ubuntu 22.04 + Linux 6.8.0-85 + runtime 库 + DCPerf 负载）。QEMU 提供函数/指令级全系统执行（含 OS），SST Ariel 把指令流喂进可参数化核模型（乱序流水、缓存、TLB），DRAMSim3 提供真实 DRAM 时序（行激活/刷新/带宽），McPAT 从微架构配置估面积功耗。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在该论文中的运转流程：输入 = 微架构参数表（Table III：4 类 chiplet 各 10/9/9/10 核，频率 3.0/2.5/2.5/2.0GHz、发射宽度 6/4/4/2、ROB 512/256/256/128、L1/L2/LLC/TLB 各级容量、Mem Lat 15/22/15/15 cycles、Mem BW 17.06/25.60/17.06/17.06GB/s）+ 互连参数（片内 2D mesh 3 cycles/hop、chiplet 间 all-to-all 60 cycles、1Gbps/link）+ 请求流（DCPerf 五应用、Poisson 到达、25/50/75% 负载）→ QEMU 逐指令执行（含系统调用与内核态）→ SST Ariel 按 chiplet 类型做 cycle 级建模（硬件 RF 预测器每 100µs epoch 采样 15 特征、Load State 报 runqueue、调度器跑 Algorithm 1 决定迁移）→ DRAMSim3 建模异构内存分区 → McPAT 估面积功耗并按 iso-area 缩放核数 → 输出 P99/P50 端到端延迟、SLO（100ms@P99）下吞吐 QPS、Perf/Watt 与预测准确率。真实系统补充：EMR 服务器上把核分组为 pool 模拟 chiplet、按论文参数设频率/带宽并用 affinity 迁移，验证迁移开销（平均 23.8µs）与功耗（-7.2%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
各组件均为开源基础设施：QEMU（https://www.qemu.org/，全系统模拟器）、SST（https://sst-simulator.org/，Ariel 为其中 CPU 模型，与 gem5 并列的主流周期级框架）、DRAMSim3（https://github.com/UMD-MEMS/DRAMsim3，UMD-MEMS 维护）、McPAT（https://github.com/HewlettPackard/mcpat，HP 开源）。用途：异构多 chiplet 服务器尚无商用硬件，full-system 模拟是唯一能在 OS+运行时+负载全栈下评估 phase 级迁移的系统方法；相比纯 trace 驱动模拟（如仅微架构的 gem5 快速模型），QEMU+SST 能捕获系统调用（网络/内存分配分类是预测器特征）与内核态行为，这是本论文 phase 检测的必要输入。局限：论文未开源 PhaseWeave 的修改（联网检索无公开仓库，截至 2026-08），复现需自行在 SST 上实现预测器/迁移/异构 chiplet 建模。

涉及论文标题：
- PhaseWeave Phase-Aware Execution on Heterogeneous Chiplet Architectures for Datacenters
