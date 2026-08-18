## eFPGA（嵌入式 FPGA 软逻辑，soft logic）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
eFPGA（embedded FPGA，嵌入式可编程门阵列）是嵌入在 SoC/ASIC 内部的软逻辑可重构硬件：与外部独立 FPGA 不同，它是芯片设计的一部分，在制造后可运行时重配置，用于需要硬件速度但又要灵活的任务。论文的 IPU_pro 集成一个微型 eFPGA 作为"软逻辑"（soft logic），解决"状态机遍历类 introspection 程序用 RISC-V 软件跑太慢"的问题——如 entangled 预取器是分支密集的历史表/状态转移图遍历，软件实现会丢维持 cache 状态所需的数据。该 eFPGA 含 590 个可配置逻辑块（CLB）、470 个 analytics IO（AIO）tile（与 baseline IO tile 相同但去掉 staging flip-flop，因为 staging FF 已在 IORegs 中）、8 个小 BRAM（64 项深 × 64-bit 宽），经内存映射寄存器与 RISC-V 核接口，支持"introspection 程序 = RTL 逻辑 + 控制代码"的打包模式（开发者提供 Verilog 与 C 控制代码）。用 FABulous 设计流程（https://github.com/FPGA-Research-Manchester/FABulous）生成并评估，soft-logic 与 IPUpro 最高时钟 1.3GHz，面积 0.22 mm²（表 III）。论文的核心 insight 之一：对广泛的内省任务，最小可重构逻辑（这个微型 eFPGA）就足够，纯处理器核不足以线速处理数据，需混合 core-accelerator 架构。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
eFPGA 软逻辑在 IPU_pro 中的运转流程（Capability 1 Stateful Emulation，entangled 预取器）：CPU 前端每周期把 fetch-PC 送入 IPU（64-bit 输入、2 个 HIT 信号）→ eFPGA 上的 entangled 状态机以每地址一个 cycle 的速度处理（历史表更新、状态转移、entangling 对判定，300 行 Verilog，接近满 eFPGA 利用率）→ 预取决策返回、硬件计数器累计覆盖/准确/缺失统计（每 2³¹ cycles 上报主机避免溢出）→ 与原始 entangled 实现对比：only 差异是 always-hit-in-L2 假设（IPU 不能向 HIT 注入信号，故假设所有 L1 miss 都是 L2 hit 来确定 entangling 对；不改变 L1 内容故访存执行仍正确），各统计指标平均优于实际 <5%。IPU_pro 面积 0.22 mm²=CPU 参考（Zen2 4-core 31 mm²）的 0.7%，功耗 20.8mW≈0.5%；单芯片只放一个 IPU_pro 则降到 0.175%/0.125%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：论文用 FABulous（https://github.com/FPGA-Research-Manchester/FABulous，FPGA'21 开源 eFPGA 框架，支持自定义块/BRAM/IO、Yosys/nextpnr 工具链、15+ 流片验证）生成软逻辑并评估面积/功耗/利用率；作为对照，通用 RISC-V+eFPGA 开源先例包括 Greyhound（IHP SG13G2 130nm，FABulous 生成 784 LUT4+FF eFPGA，支持 warmboot 运行时重配置）与 FuseRISC（双 RISC-V + FABulous eFPGA 作为自定义指令接口 CIF）。使用方式：introspection 开发者对 IPUpro 程序同时提供 C 控制代码与 Verilog 软逻辑（经内存映射寄存器接口），运行时加载位流；对 IPU_lite 则不用软逻辑（纯 RISC-V + 直方图/hash 原语）。局限：eFPGA 时钟（1.3GHz）低于 RISC-V 核（2GHz），面积 0.22 mm² 显著大于核本体（0.0011 mm²），只在其"状态机遍历"优势域使用。

涉及论文标题：
- Enabling Continuous, In-Field Introspection: The Programmable IPU Architecture
