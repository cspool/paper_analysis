## ASAP7（7nm 预测性 FinFET PDK）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ASAP7 是 Arizona State University（Lawrence Clark 组）与 ARM Research 合作开发的 7nm 预测性 FinFET 工艺设计套件（PDK），2016 年发布（Clark 等，Microelectronics Journal，DOI 10.1016/j.mejo.2016.04.006），由 OpenROAD Project 开源维护（https://github.com/The-OpenROAD-Project/asap7，BSD-3）。它是学术界最常用的 sub-10nm 开放 PDK：非可制造（无 foundry sign-off/tape-out），但提供 7nm FinFET 的 BSIM-CMG SPICE 模型（0.7V 标称 Vdd）、4 种阈值电压（SLVT/LVT/RVT/SRAM）、标准单元库（6-track asap7sc6t、7.5-track asap7sc7p5t）、SRAM 宏与 Calibre deck，供研究者无 NDA 地跑完整 FinFET 综合/布局流程。Moirai 用它在 4.0GHz 时钟下综合 CaPNet 神经引擎 RTL，得到 1178 μm² 面积与 8.5mW 功耗（相对 Apple A13 7nm 核心 2.61mm²/3W 仅 0.05%/0.28% 开销）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- ASAP7 在 Moirai 的芯片设计流程中的角色：CaPNet RTL（FCC/BCC 位逻辑单元 + 控制单元 + 流水线寄存器）→ 逻辑综合（用 ASAP7 标准单元库映射到 7nm 门级网表）→ 时序收敛（1-cycle-per-layer 流水线在 4.0GHz 收敛；单周期前向设计最优 2.5GHz）→ 得到面积/功耗报告（1178 μm²、8.5mW）。具体流程：RTL 描述 → Design Compiler 类综合工具读 ASAP7 .lib 时序库 → 时钟约束（4.0GHz）→ 优化 → 面积/功耗报告；Moirai 还据此把 3-cycle 前向延迟硬化进 ChampSim 的预取模型（"严格建模神经计算开销"）。ASAP7 提供的 7.5-track 单元与 4 阈值电压让"低电压低功耗位逻辑阵列"的评估在无 NDA 条件下可复现。
- 与其他 PDK 对比：Sky130/GF180MCU 可制造但为成熟节点（130/180nm），ASAP7 不可制造但代表先进节点（7nm FinFET），适合"先进节点面积/功耗/时序可行性"研究（Moirai 论证"预取器能放进高频 L1D 核"）。注：ASAP7 也可经 OpenROAD Flow Scripts（ORFS）跑全流程 P&R。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：git clone --recurse-submodules https://github.com/The-OpenROAD-Project/asap7 获取（asap7_pdk_r1p7 技术文件/HSpice 模型、asap7sc6t_26、asap7sc7p5t_27/28 单元库）；综合工具读 .lib（如 Synopsys Design Compiler、OpenROAD、Yosys+下游），SPICE 用 HSpice/NGSpice，Calibre deck 需另从 http://asap.asu.edu/asap/ 下载。典型用法：写 Verilog RTL → 设时钟约束 → 综合到 ASAP7 门级 → 面积/功耗/时序报告 → （可选）Innovus/OpenROAD P&R 到 GDS 近似。Moirai 只做综合级面积/功耗评估（论文未明确说明综合工具，仅写"使用 ASAP7 7-nm predictive FinFET process library"）。研究场景：先进节点加速器/微架构部件的面积-功耗-时序可行性预研（本论文、IPU_lite/pro 的 AsAP7 7nm 教育 PDK 综合、若干 7nm DAC/反相器教学项目）。

GenZA 补充视角（ISCA'26，ZKP 加速器 RTL 综合）：GenZA 对关键组件（PE 及其可重构多 bitwidth 算术单元）做完整 Verilog RTL 实现，用 ASAP 7nm 综合得到逐级面积/功耗（Table VII：32 个 64-bit 乘法器 31k µm²/25 mW、128-bit KO 级 13k µm²/20 mW、256/384/768-bit KO 级 13–17k µm²/20–21 mW、模乘级 6k µm²/11 mW、crossbar+wires 9k µm²/13 mW，PE 合计 166k µm²/164 mW；并对比 384-bit/768-bit 全流水模乘器 69k/263k µm² 论证多 bitwidth 灵活性的面积代价）；整芯片 128 PEs+NoC+transpose buffer+SHA3+2×HBM2e PHY 共 58.5 mm²/64.1 W @ 1 GHz，与 PipeZK/SZKP/LegoZK/zkSpeed/UniZK 统一缩放到 7nm 后做 ATP 比较。整芯片频率 1 GHz（对比 Moirai 的 4.0 GHz 逻辑）说明不同设计对 ASAP7 时序收敛目标的用法不同。

涉及论文标题：
- From Memorization to Generalization: A Practical Neural Network Prefetching Framework
- GenZA: A General and Efficient Accelerator for Diverse Zero-Knowledge Proof Protocols
