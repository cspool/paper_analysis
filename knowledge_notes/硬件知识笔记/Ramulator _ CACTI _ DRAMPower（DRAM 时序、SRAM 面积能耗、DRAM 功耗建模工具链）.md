## Ramulator / CACTI / DRAMPower（DRAM 时序、SRAM 面积能耗、DRAM 功耗建模工具链）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
三者是体系结构研究常用的存储层次评估工具：(1) Ramulator——CMU SAFARI 组开源的快速可扩展 DRAM 模拟器（Kim, Yang, Mutlu, IEEE CAL 2015；https://github.com/CMU-SAFARI/ramulator），用配置驱动的协议描述（channel/rank/bank/时序参数如 tRCD/tRP/tRAS）逐命令模拟 DRAM 访问时序，支持 DDR3/4/5/LPDDR/HBM 等，可输出带宽利用率与延迟；(2) CACTI——HP Labs 开源的片上缓存/内存建模工具（Muralimanohar et al., CACTI 6.0；https://github.com/HewlettPackard/cacti），输入容量/工艺/端口/关联度输出面积、访问时间与每访问能耗；(3) DRAMPower——开源 DRAM 功耗/能量估算工具（Chandrasekar et al. 2012；http://www.drampower.info），输入访存命令流（命令时序）输出各 bank/通道的功耗分解。三者常组合使用：Ramulator 提供时序/带宽，CACTI 提供 on-chip 存储开销，DRAMPower 提供 DRAM 能量。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
本论文（VI-A 章）用法：DDR5-4800 DRAM（38.4 GB/s）用 Ramulator 建模；on-chip SRAM（GS Feature 88KB + Output 4KB + Depth 4KB = 96KB）面积/能耗用 CACTI 估算；总能耗（on-chip + off-chip）用 DRAMPower 估算。流程：自研 cycle-accurate 模拟器（与 RTL 仿真交叉验证）产生访存 trace 与延迟 → Ramulator 按 DDR5 协议模拟时序给出 off-chip 访存延迟/带宽受限 → DRAMPower 吃 Ramulator 命令流给 DRAM 功耗 → CACTI 吃 SRAM 容量/28nm 工艺给面积与每访问能耗 → 三项合并得到表 II（PE 阵列 2.958mm²/1.48W、支持模块 0.064mm²/0.02W、片上缓冲 0.826mm²/0.14W，总 3.85mm²/1.64W）与吞吐/能耗对比（Fig.13-18）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
一般用法：研究者在模拟器（如 gem5/Sniper）或自研 cycle-accurate 模拟器中嵌入 Ramulator 做 DRAM 时序精确模拟，用 CACTI 估算片上存储成本，用 DRAMPower 做能耗归因；三者都是学术界标准、开源的"论文评估工具链"。本论文加速器 RTL 与 cycle-accurate 模拟器未开源（仓库仅含 MLP-based_OIT 的 CUDA 算法代码）；Ramulator/CACTI/DRAMPower 本身开源可复现。

> **TensorPrism 集成视角（ISCA'26）**：TensorPrism 的自研 cycle-accurate 模拟器（遵循 [18,49,69] 既有加速器研究方法论）集成 Ramulator 建模 HBM2（307.2 GB/s），周期级刻画全部计算/访存/控制组件；SRAM 能耗与面积用 CACTI 7.0，RTL 用 Synopsys Design Compiler（TSMC 28nm）综合、PrimeTime PX 功耗。与上例不同，TensorPrism 只组合 Ramulator + CACTI（未用 DRAMPower，能耗经 PrimeTime 翻转活动 + CACTI 参数折算）。流程：CoG Scheduler 划分/PE 计算周期级推进 → 访存请求送 Ramulator（HBM2 状态机时序）→ 输出 DRAM 访问次数/带宽/能耗，与理论最小访问（0.11×）对照；DRAM 访问归一化 1/2.18/2.11/1.27/1.53（vs SPADE/HotTiles/GSpTC/TCP）直接证实共现图划分的复用收益（片上命中 94%）；片外能耗占 47.4%（vs TCP 79.1%）支撑能耗对比。

涉及论文标题：
- Optimizing 3D Gaussian Splatting with Axis-Shared Rasterization and Order-independent Transmittance
- TensorPrism: Rethinking Sparse High-order Tensor Acceleration via Co-occurrence Graph
