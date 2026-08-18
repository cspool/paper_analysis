## Xilinx VCU118 FPGA（Virtex UltraScale+，SoC 原型验证平台）

术语解释
AMD/Xilinx 的 VCU118 评估板（Virtex UltraScale+ VU9P），LIPPEN 用它承载 64-bit RISC-V Rocket/BOOM + RoCC 密码加速器原型，以 100 MHz 实测性能/面积/功耗。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- VCU118 是基于 Virtex UltraScale+ VU9P 的 FPGA 评估板（LUT/FF/BRAM/URAM 资源丰富，支持 PCIe/DDR4/以太网等），配合 Xilinx Vivado 综合布线生成 bitstream；与 Versal（ACAP，含 AI Engine/NoC hardened IP）不同，VCU118 是纯 fabric 的 UltraScale+ 器件，适合处理器原型与 RTL 验证。论文在 VCU118 上综合出 Rocket-LIPPEN（58,137 LUT/42,100 FF/99 MHz/4.035 W）与 BOOM-LIPPEN（248,416 LUT/98,674 FF/90.6 MHz/5.625 W）等配置，100 MHz 运行（基频 Rocket 150 MHz，RoCC 接口是主要频率下降来源）。
- 作用：比 ASIC 快速迭代、比纯仿真（Verilator）提供真实时序与可测量的功耗（后综合分析），是硬件-软件协同设计（RISC-V SoC + 密码加速器 + Linux + 编译器插桩）的标准验证平台。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 流程：Chipyard 生成 SoC RTL → Vivado 2021.2 综合布线（目标 100 MHz）→ 生成 bitstream 烧入 VCU118 → FireMarshal 构建 Linux 镜像写入 SD 卡 → 上板 boot → 把编译插桩后的 benchmark 二进制经 SD 卡拷入并运行 → perf 计时（运行时间、动态指令数）→ 归一化到未插桩 baseline 得 overhead；面积/频率/功耗从 Vivado 后综合报告提取（Table VI）。
- 硬件成本结论：密码引擎（RoCC 1,034 LUT，PRINCEv2）面积/功耗增幅 <4%（vs Rocket-base 3.935 W），证明全指针加密可在真实处理器中实际部署。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：VCU118 板（Virtex UltraScale+）+ Vivado 2021.2 + Chipyard v1.8 bitstream；artifact 要求 Xilinx VCU118 硬件与完整 synthesis 时间（数小时）。开源：bitstream 生成流程在 https://github.com/bearhw/LIPPEN 的 HARDWARE.md 中说明，artifact https://doi.org/10.5281/zenodo.19901476。

涉及论文标题：
- LIPPEN: A Lightweight In-Place Pointer Encryption Architecture for Pointer Integrity
