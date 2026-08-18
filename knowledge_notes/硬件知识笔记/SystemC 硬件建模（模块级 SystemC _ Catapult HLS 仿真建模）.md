## SystemC 硬件建模（模块级 SystemC / Catapult HLS 仿真建模）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SystemC（Accellera 标准，systemc.org）是基于 C++ 的硬件建模语言/库，用 module/port/channel/process 描述并发硬件，可做 transaction-level 与 cycle-level 仿真，也是 HLS 工具的常用输入。NeRArch-Sim 用 SystemC 按统一分类学实现 20+ 模块化硬件模型（每个模块配可配置头文件指定数据类型/精度/memory binding），再经 Catapult HLS（Siemens）综合——支持两级表征：HLS 综合（快速 PPA 估计）与全 ASIC post-layout 流程（精确 PPA）。仓库中同类用法：RPU（ISCA'26）也用 SystemC + Catapult HLS 实现证明型 RPU（TSMC N16）；知识库硬件笔记的 TMAC/Stream Decoder 条目同为 SystemC + Catapult HLS 建模。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 NeRArch-Sim 中的运转流程：用户给硬件配置文件（hardware JSON，如 icarus_config.json，含各模块参数与 SRAM 绑定）→ 生成各模块 SystemC HLS 模型（S0_scripts 下 run_hls.py/run_fc.py/run_pwr.py 跑 HLS→逻辑综合→功耗分析）→ 与软件侧算子图经调度器绑定后按 memory-aware duration 模型（式 1）算端到端周期/PPA。表 IV 展示了 SystemC 模块暴露的可配置参数：综合指令（pipelining II、unroll、array partition）、精度（任意整型/浮点/定点）、实现（CORDIC、分段线性 exp）、模块规模（systolic 尺寸、PE 数、buffer 深度）、并行因子、通道深度/流宽/握手协议。验证：同一 SystemC 源同时喂 NeRArch-Sim（HLS 估计）与全 ASIC flow（Fusion Compiler + PrimePower，post-layout），17 个模块延迟全部一致、面积/功率相对误差 4.72%~9.33%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：C++/SystemC 写 `SC_MODULE`（含 sc_in/sc_out、SC_CTOR、sensitive 进程），配 CMake（NeRArch-Sim 的 Hardware/A1_cmod 按 Blending/Encode/FieldComp/Sample/SRAMTest 分类组织模块与 testbench.cpp）；安装需 Catapult 2024.1_2、MatchLib、PDK 路径环境变量，`python3 run_hls.py --module Blending/QSU` 等命令逐模块产出 PPA.log/timing.log/failed.log。用途：快速 PPA 估计（分钟级 vs 全 ASIC 数小时）支持 DSE；与 HLS 综合结果一致性高（表 VI），且可部署到 FPGA/ASIC。

涉及论文标题：
- NeRArch-Sim: A Unified Simulator for Benchmarking and DSE of Neural Rendering Accelerators
