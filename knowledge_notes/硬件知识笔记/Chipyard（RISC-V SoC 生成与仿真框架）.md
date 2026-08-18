## Chipyard（RISC-V SoC 生成与仿真框架）

术语解释
UC Berkeley 的开源 SoC 集成/生成/仿真框架（Chisel + Diplomacy/TileLink），把 Rocket/BOOM 核、RoCC 加速器、内存与外设组合成完整 RISC-V SoC，支持 Verilator 仿真、FireSim 与 FPGA 综合；LIPPEN 用它构建指针加密原型。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Chipyard 以参数化生成器方式组合处理器（Rocket 顺序核、BOOM 乱序核、CVA6、Ibex 等）、缓存、RoCC 协处理器、TileLink/Diplomacy 互连与外围，可生成 Verilog（经 Verilator 编译为 C++ 周期精确模型仿真）或 FPGA/ASIC 位流，集成 FireMarshal 生成可引导 Linux 镜像；v1.8 是 LIPPEN 使用的版本（GitHub 子模块）。Web 证据：官方仓库 https://github.com/ucb-bar/chipyard 、论文 Amdid et al. IEEE Micro 2020。
- 在 LIPPEN 中的角色：硬件原型载体——扩展 Rocket/BOOM 核、RoCC 接口挂 PRINCEv2/QARMA 加速器，生成 VCU118 FPGA bitstream；Verilator 仿真用于功能验证与跑小测试程序；FireMarshal 生成 Linux 镜像在 FPGA 上启动跑 benchmark。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 流程：`make verilog` 由 Chisel 配置生成 SoC RTL（含 RoCC 上的 PRINCEv2）→ `make run-rtl` 用 Verilator 周期精确仿真跑编译后的 RISC-V 二进制验证 PTR_SEAL/PTR_UNSEAL 语义 → `make fpga` 用 Vivado 综合出 VCU118 bitstream → FireMarshal 构建 Linux 镜像烧 SD 卡 → 上板运行 microbenchmark/nbench/SPEC CPU2017，perf 计时得到归一化 overhead。
- LIPPEN 对 Chipyard 的修改以补丁形式给出：chipyard.patch、rocket-chip.patch（核 + RoCC 扩展），与 llvm.patch、pactight.patch 一起构成完整可复现 artifact。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：git clone + 子模块初始化（Ubuntu 20.04/22.04、Conda、Java 11）→ 打补丁 → 构建 riscv-tools（2–4 h）→ Verilator 仿真（30–60 min）→ Vivado 2021.2 生成 bitstream（4–6 h）。开源：Chipyard 本身（BSD 类许可）与 LIPPEN 仓库（GPL-3.0）均公开，见 https://github.com/bearhw/LIPPEN。

- LoRA 中的角色（ISCA'26）：SoC 载体——LoRA SoC 基于 Chipyard 构建，包含 5 级 in-order 64-bit RISC-V Rocket CPU + 异构 CGRA + 其他子系统；CPU 管理数据、经 RoCC 调用 CGRA 加速、处理 CGRA 不执行的代码；reservation station 缓存 RoCC 自定义指令并按依赖分发到各控制器；load/store 控制器经 TileLink + DMA 管理 L2↔SPM 数据传输；CGRA 控制器配置并监控执行。除 XCore（Verilog）外所有 SoC 用 Chisel 建模，生成 Verilog 供 ASIC/FPGA；用 Synopsys VCS 仿真、Design Compiler + TSMC 40nm 综合（~475MHz），artifact 提供基于 Chipyard 的 Docker（docker.cnb.cool/fudaneda/docker/chipyard）与 Verilator 仿真流程复现。

涉及论文标题：
- LIPPEN: A Lightweight In-Place Pointer Encryption Architecture for Pointer Integrity
- LoRA: Towards Improved Applicability of Reconfigurable Architecture for Versatile Nonlinear Functions
