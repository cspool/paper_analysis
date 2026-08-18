## 开源 EDA 流程（Yosys + OpenROAD + SKY130）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
一套完全开源的 RTL-to-GDSII 数字芯片设计工具链：Yosys（YosysHQ 维护的开源 RTL 逻辑综合工具，读入 Verilog/SystemVerilog，经 ABC 做逻辑优化与门级映射，输出网表；本身不做布局布线）、OpenROAD（UC San Diego 主导的开源物理设计引擎，覆盖 floorplan/placement/CTS/global+detailed routing/STA，常以 OpenLane 流程编排）、SKY130（Google 主导的 SkyWater 130nm 开源 PDK，提供标准单元库与工艺参数，经 open-pdks 生成适配库）。三者组合形成免授权的芯片实现路径，被混合信号/模拟研究广泛采用。DS-ISA 论文用它实现控制器：SystemVerilog → Yosys 综合 → OpenROAD 物理设计，target SKY130 130nm，得 4.6W/79.58mm²（32×32 配置）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
流程运转（以 DS-ISA 控制器为例）：SystemVerilog RTL（指令缓冲/解码器/调度器/Data Path/Control Path）→ Yosys 读入并综合成 SKY130 标准单元门级网表（ABC 逻辑优化）→ OpenROAD：init_fp 读网表与工艺库 → ioplacer/pdngen 摆 I/O 与电源网络 → RePLace 全局布局 → Resizer/OpenDP 详细布局与优化 → TritonCTS 时钟树 → FastRoute/TritonRoute 全局/详细布线 → OpenRCX 寄生抽取 + OpenSTA 时序/功耗分析 → 输出 GDSII 与面积/功耗报告 → 得各配置（8×8 到 32×32）功耗面积（Table I）。混合信号接口（DAC/ADC）未在 SKY130 流片，而是按同类 130nm 设计估算（8-bit ADC 1GS/s 0.72mm²/13.3mW、DAC 600MS/s 0.27mm²/2.4mW）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现与使用：按 OpenLane 风格脚本化（设计文件 + 约束 + 工艺库路径 → 一键跑完整流程）；SKY130 尤其适合混合信号研究（DS-ISA 论文即因此选它而非纯数字性能更好的工艺，并列举 [12][20][21] 等 SKY130 混合信号先例）。复现 DS-ISA 需 SystemVerilog 源码 + SKY130 PDK + Yosys/OpenROAD 工具链——论文未开源 RTL，开源工具链可公开获取（Yosys https://github.com/YosysHQ/yosys；OpenROAD https://github.com/The-OpenROAD-Project/OpenROAD；PDK https://github.com/google/skywater-pdk）。层次归类说明：本条目属"硬件实现工具链"，归入硬件架构为最接近层次。

涉及论文标题：
- DS-ISA: Instruction Set Architecture for Dynamical System Units
