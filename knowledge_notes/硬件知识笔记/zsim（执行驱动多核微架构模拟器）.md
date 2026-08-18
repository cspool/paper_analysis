## zsim（执行驱动多核微架构模拟器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- zsim 是 MIT Daniel Sanchez 与 Stanford Christos Kozyrakis 开发的快速可扩展 x86-64 多核微架构模拟器（ISCA'13，论文引用 [40]）：基于 Intel Pin 动态二进制翻译做执行驱动（execution-driven）模拟，用指令级时序模型精确建模乱序（OOO）核心，配合 bound-weave 两阶段并行技术在宿主多核上线性扩展模拟速度（16 核宿主可模拟 1024 核芯片，简单核 1500 MIPS、详细 OOO 核 300 MIPS），并支持无全系统模拟的用户级虚拟化以运行复杂负载。开源地址 https://github.com/s5z/zsim（web 确认），GPLv2，依赖 gcc、Pin、scons、libconfig、libhdf5，可选 DRAMSim2 建模主存。
- 从硬件架构角度拆解术语，给出运转流程具体例子：RoboCortex 用它模拟对标 Nvidia Jetson AGX Orin 的移动多核系统（Table II）：8 核 x86-64 3.5GHz OOO（224-entry ROB、4-wide issue）、L1 32KB/核 2 路 4-cycle、L2 256KB/核私有 8 路 14-cycle（含 16-entry stream/semantic 预取器建模）、L3 共享 8MB 16 路 45-cycle、双控制器 DDR3-1600、RSU 8×8 fabric + 16-entry Path Buffer（LRU）。论文在 zsim 中建模 RSU 数据流阵列、显式栈/优先队列、Path Buffer、RSU 引导预取器，输出各配置延迟与 L1/L2/L3 miss 统计（Fig. 15/17-27）。模拟原理：Pin 逐指令驱动执行 → 每个访问注入 OOO 核时序模型与缓存层次 → 统计周期与 miss；bound-weave 让并行模拟保持准确。
- 术语一般如何实现？如何使用？：构建用 `scons -j16`，运行 `./build/opt/zsim tests/simple.cfg`（二进制 + .cfg 配置：核/缓存/内存/RSU 参数），输出 per-instruction cycle 与分层统计。使用场景：RoboCortex 这类新增缓存旁硬件（预取器、加速器、缓冲区）的系统级评估；与 cycle-accurate RTL（Synopsys Design Compiler 综合）互补——zsim 给系统级延迟/miss，RTL 给面积/开销。依赖 Pin 版本兼容（Linux 3.0+ 需 Pin 2.10+，新 gcc 需 fork 加 ABI 兼容 flag）。

涉及论文标题：
- Optimizing Spatial Data Structure with Near-Cache Acceleration by Exploiting Physical Locality（RoboCortex）
