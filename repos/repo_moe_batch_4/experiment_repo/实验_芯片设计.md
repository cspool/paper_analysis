## Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving

- 属于芯片设计的实现是什么？实验比较什么？
  - 实现：Stratum 的核心芯片设计是 **Monolithic 3D-Stackable DRAM (Mono3D DRAM)** 与 **Logic Die NMP Processor** 通过 Cu-Cu hybrid bonding（1μm pitch）垂直集成，再通过 2.5D silicon interposer 与 GPU（xPU die）互联的异构芯片架构。关键技术：(1) Mono3D DRAM——1024层水平 1T1C DRAM cells，垂直 Bitline 连接 + WL Staircase 结构，内部带宽 19.01-34.34 TB/s，芯片面积 121 mm²，单芯片 32 GB，密度 2.156 Gb/mm²（5.2× 高于 DDR5）；(2) 8-tier 内存分层——利用 WL staircase 导致的访问延迟差异（tRCD=[2.29, 3.92, 5.99, 8.50, 11.44, 14.82, 18.63, 22.88] ns），快 tier 比最慢 tier 1.6× 更快；(3) Logic Die Processor——7nm，121 mm² die area，16 PUs，128 TFLOPS peak，64k MAC units @ 1GHz，43W power budget，on-chip ring network (2.048 TB/s)；(4) 三种系统配置——Stratum-S（RTX A6000 + 1 Mono3D DRAM chip, 32GB）、Stratum-L（H100 + 6 Mono3D DRAM chips, 192GB）、Stratum-XL（2×Stratum-L modules, 384GB）；(5) CMOS-under-array (CUA) 32nm 高电压电路 + 7nm 低电压逻辑 die hybrid bonding。
  - 实验比较：(a) Mono3D DRAM vs HBM 对比：混合 bonding 1μm pitch vs TSV 10μm pitch，Mono3D 内存密度 5.2× DDR5；(b) 解码吞吐量：Stratum tiering vs GPU baseline（vLLM 0.8.1 on H100/RTX A6000），8.29× (OLMoE)、5.39× (Mixtral)、6.13× (Qwen2.5)、4.48× (Llama-4) 加速；(c) 能效：7.66×、2.74×、3.51×、4.87× 改进；(d) Tiering vs No-Tiering：1.45×/1.39×/1.32×/1.34× 吞吐提升；(e) Mono3D DRAM 层数扩展：1024-layer vs 256-layer vs 64-layer，面积归一化吞吐 1.21× 和 2.96× 提升；(f) 512-layer tiering：17.7-18.3% 性能提升；(g) Expert Swap 开销：<0.37% 时间和 <0.03‰ 能量。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  - Coventor SEMulator3D 工艺模拟器（https://www.coventor.com/products/semulator3d/）——用于 Mono3D DRAM 1T1C 结构的 RC 参数提取，构建 3D DRAM array 工艺模型。
  - NeuroSim V1.4（https://github.com/neurosim/NeuroSim）——用于 Mono3D DRAM 外围电路时序和功耗仿真，结合 DDR5 标准时序校准。
  - HotSpot 3D IC 热模拟器（https://lava.cs.virginia.edu/HotSpot/）——用于垂直集成 memory+logic dies 的热分析，确定 logic die 功率预算（45W/chip）。
  - 论文自研 in-house cycle-level simulator——接受 tensor size、parameter tier assignments、attention head mapping、expert IDs 及各组件 delay/energy 参数，输出总体执行时间和 component-level 能耗分解。

- 模拟器模拟什么的性能，修改了什么。
  - Coventor：模拟 Mono3D DRAM 1024 层 3D 结构的 RC 寄生参数，提取不同 WL 层的访问延迟（tRCD）。基于 35nm feature size 的 1T1C DRAM cell 模型。
  - NeuroSim：模拟 DRAM bank 的外围电路性能（sense amplifier、row decoder 等），以 DDR5 标准为参考校准时序参数，评估各 tier 的 tRCD 和 tRP。
  - HotSpot：模拟 3D IC 的稳态热分布，给定 high-end liquid cooling + vapor chamber 散热方案（convection resistance 0.01 W/K, conductivity 5000 J/(m·K)），确定 logic die 峰值功率 45W。
  - In-house simulator：通过输入 workload 参数（矩阵维度、tier assignments、expert routing）模拟 NMP 的执行 cycle 数和通信 overhead，计算 end-to-end 延迟和能耗。SystemVerilog 实现的 NMP 组件经 Cadence Genus 综合（ASAP7 7nm PDK），post-synthesis 网表级仿真获取延迟和能耗参数。

- 开源情况。基于开源文档和论文，使用例子解释模拟器如何使用？作用是什么？
  - 论文未提供独立开源代码仓库。使用的工具链：Coventor（商业许可）、NeuroSim（开源）、HotSpot（开源学术）、Cadence Genus（商业许可）、ASAP7 PDK（开源 https://github.com/The-OpenROAD-Project/asap7）。
  - 模拟器协同使用流程：(1) Coventor 构建 Mono3D DRAM 1T1C 3D 结构模型 → 提取 WL/BL 寄生 RC 参数 → 输出不同 WL layer 的 latency profile；(2) NeuroSim 以 Coventor RC 参数为输入 → 模拟 bank 级外围电路 → 输出各 tier 的 tRCD/tRP timing 参数（Table 1）；(3) HotSpot 以芯片热参数（material properties, heat sink specs）为输入 → 模拟 3D IC 温度分布 → 输出 logic die 功率上限 45W；(4) Cadence Genus 以 SystemVerilog NMP 设计为输入 → 综合到 ASAP7 7nm 库 → 输出面积、功耗、时序报告 → 作为 in-house simulator 的组件参数；(5) In-house simulator 以所有组件参数 + workload（tensor sizes, routing, tier assignments）为输入 → cycle-accurate 模拟执行流程 → 输出 per-layer latency、能耗 breakdown、throughput。系统级 simulator 额外包含 Request Generator（Poisson 到达模型）、SLO-Aware Scheduler、Memory/Computation Mapper，模拟完整 serving 流程。
