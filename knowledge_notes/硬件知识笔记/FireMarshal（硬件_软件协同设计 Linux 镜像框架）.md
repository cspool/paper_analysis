## FireMarshal（硬件/软件协同设计 Linux 镜像框架）

术语解释
Chipyard 生态的 Linux 镜像生成工具（Pemberton & Amid, ISPASS 2021）：为 RISC-V SoC（FPGA/仿真/ASIC）构建可引导的 Linux 用户态环境，LIPPEN 用它为 VCU118 原型准备运行 benchmark 的 Linux 系统。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- FireMarshal 以声明式配置定义内核、rootfs、initramfs 与应用负载，生成可在 Chipyard 目标（Verilator 仿真、FireSim、FPGA）上直接引导的镜像，把"硬件设计 + 软件栈"的复现变成可重复流程。Web 证据：论文 N. Pemberton & A. Amid, ISPASS 2021（FireMarshal: Making HW/SW Co-Design Reproducible and Reliable）。
- 在 LIPPEN 中的角色：构建并管理 VCU118 上运行的 Linux（用户态镜像含 benchmark 二进制），配合 SD 卡工作流上板执行 microbenchmark/nbench/SPEC CPU2017——firemarshal linux 是 artifact 的运行环境要求之一。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 流程：配置 workload（内核版本、rootfs 内容、启动参数）→ FireMarshal 构建镜像 → 写入 SD 卡 → FPGA boot → Linux 启动后执行插桩 benchmark → perf/计时收集结果。与 Verilator 仿真（功能验证）配合：先仿真验证 ISA/加密正确性，再上板实测性能。
- 价值：消除手工搭建嵌入式 Linux 的不可复现性，使"芯片设计 → 系统软件 → 应用基准"整条链可复现（LIPPEN artifact 的核心目标）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：作为 Chipyard 子工具（`./scripts/firemarshal` 系列），配置 YAML 定义镜像内容；LIPPEN artifact 的 FPGA 流程含 firemarshal 构建步骤。开源：随 Chipyard 分发（https://github.com/ucb-bar/chipyard 内 firemarshal 子模块）。

涉及论文标题：
- LIPPEN: A Lightweight In-Place Pointer Encryption Architecture for Pointer Integrity
