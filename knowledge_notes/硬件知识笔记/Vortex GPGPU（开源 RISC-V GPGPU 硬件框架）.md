## Vortex GPGPU（开源 RISC-V GPGPU 硬件框架）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Vortex 是开源、完全可综合的 RISC-V GPGPU 框架（https://github.com/vortexgpgpu/vortex），实现 SIMT 执行模型与完整 GPU 软件栈（OpenCL 前端 POCL + LLVM 后端 + RISC-V 工具链），支持 FPGA 与 ASIC 目标，提供纹理单元与 OpenGL（配合 Skybox 扩展）。架构上：RISC-V ISA + 最小扩展（线程控制、发散管理、同步、纹理采样）；层级内存（多 bank cache、虚拟多端口、MSHR 支持并发访存）；六段流水 Schedule→Fetch→Decode→Issue→Execute→Commit；核心按层级聚类（core 组 socket 共享 L1，多 socket 共享 L2）；含 NoC 元素。高可配置性（SM 数、cache 层级与容量、功能单元数）使其成为 GPU 微架构研究的 RTL 平台。逻辑链：此前 GPU 微架构创新（如 OoO 执行）只能在软件仿真器（Accel-Sim 等）中评估，缺 RTL 验证与电路级精度；Vortex 提供可综合 RTL 基础，sCROOGe 即在其上实现并评估 frontend/backend OoO 执行方案。注意与编译框架层"Vortex GPU 编译器与自定义 intrinsics（POCL-Vortex 工具链）"条目的区分：本条目聚焦 Vortex 作为硬件 RTL 平台（sCROOGe 的用法），工具链条目聚焦编译侧。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
sCROOGe 在其上的运转：基准 commit abdea91 的 Vortex RTL 六段流水作为 baseline；frontend OoO 扩展在 Issue 阶段插入 IsB/InFL/Dependence Checker/Issue Arbiter/UUID；backend 扩展删除 Scoreboard 阶段并在 OC 阶段插入 CU/RAT/RRS/broadcast/UUID。单 SM 评估配置：1 个 processing block + 16KB L1，DRAM 用 Ramulator 集成建模（Vortex 与 sCROOGe 均集成 Ramulator）；软件栈用 POCL 编译 OpenCL kernel + LLVM 生成应用二进制，22 个 Vortex benchmark 驱动 Verilator cycle-accurate RTL 仿真输出 per-kernel 指令数与 cycles；同一 RTL 走 Synopsys DC 综合（GF 22nm FDSOI）→ PrimeTime 功耗 → Innovus PnR 验证 → IMEC N2 2nm 扩展，输出面积/功耗/频率（Fig.19-24）。功能验证含 FPGA 仿真（数百 MHz，比 Accel-Sim 的 KHz 级快约 5 个数量级）。Vortex 的可配置性（warp/thread/CU/IsB/cache 尺寸）支撑 {warps=4..64, threads=4..32} 的 DSE。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SystemVerilog（完整可综合，对比 MIAOW 的混合 RTL/行为建模）；软件栈 = POCL（Vortex 分支，OpenCL 1.2 驱动）+ LLVM + riscv-gnu-toolchain；支持 FPGA（Altera Arria 10/Stratix 10、Xilinx Alveo U50/U250/U280、Versal VCK5000）。使用：GitHub 主仓库 + 工具链，sCROOGe artifact（Zenodo DOI 10.5281/zenodo.19453033，Docker 容器）内含修改后的源码与 benchmark，性能实验经 /vortex/ci/blackbox.sh 传入 OoO scheme 选项（frontend/backend/无 RRS 的 backend 变体）运行，最长约 270h（单进程）；ASIC 流程数据因 NDA 只提供 .csv。Web 证据：Vortex（Georgia Tech，https://vortex.cc.gatech.edu/），PipeIMC 亦基于 Vortex 编译器实现其 IMC ISA（本库编译框架层 Vortex 工具链条目）。

涉及论文标题：
- sCROOGe Circuit-level Design and Optimization Framework for RISC-V Out-of-Order GPUs
