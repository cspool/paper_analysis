## 硬件仿真器（Hardware Emulator，逻辑仿真加速器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 硬件仿真器（emulator）是把 RTL 设计综合成门、映射到可重构芯片阵列（FPGA 或定制门处理器）上运行的周期级仿真加速平台，是芯片流片前验证的工业标准，市场约 20 亿美元。三大平台：Cadence Palladium（定制处理器阵列，容量至 48B 门）、Synopsys ZeBu（商用 FPGA，至约 9.6B 门）、Siemens Veloce（定制 FPGA）。Web 证据：平台对比见 ACL Digital 博客（https://www.acldigital.com/blogs/emulation-platform-comparison ）与 Cadence 产品页（https://www.cadence.com/en/US/home/tools/system-design-and-verification/emulation-and-prototyping/palladium.html ）。
- 在 Lotus（ISCA'26）中作为 baseline：emulator 把电路空间映射到 FPGA 上、多芯片间每模拟周期锁步通信。论文总结其缺陷：①编译慢（大设计数天到数周，需网表跨芯片划分+每分区布局布线）；②系统规模限制可仿真电路规模（数十亿门需数千芯片、昂贵）；③效率低——跨芯片每周期通信（几百 ns 延迟）把速度限制到几 MHz，每 FPGA 只用可达速度约 1/100；④只支持 RTL。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程：RTL → 综合成门级网表 → 网表划分到多个 FPGA（或门处理器）→ 每分区布局布线 → 运行时各芯片锁步推进模拟周期，跨芯片信号每周期交换（经虚拟线 Virtual Wires 或专用 I/O 时间复用）→ 受限通信带宽/延迟决定最高频率（FireAxe cycle-exact 模式 800 KHz）。
- Lotus 对比：emulator 因空间映射"系统大小=电路大小"；Lotus 时间映射让 8 FPGA 仿真需 emulator 10–60 FPGA 的设计（NTT 少 7.5×、MatMult 少 3.75× FPGA），且每 FPGA 速度提升最多 23×（NTT）。论文对 emulator 的 FPGAs 数与速度均采用乐观估计（FireAxe 800 KHz、忽略通信瓶颈）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：商用系统（Palladium/ZeBu/Veloce）或开源/学术平台（FireAxe、FireSim、DIABLO）。论文用 FireAxe [42]（开源多 FPGA emulator，支持用户引导划分、Virtual Wires 式 I/O 时间复用）的实测性能作为 emulator 基线。
- 使用：芯片设计公司用于大型 SoC 的软硬件协同验证、固件启动、操作系统 bring-up、功耗分析等流片前活动；受限场景是学术研究（昂贵、不可得），故论文用 FireAxe 估计。

涉及论文标题：
- Lotus A Multi-FPGA Task Dataflow Architecture to Accelerate Cycle-Level Simulation
