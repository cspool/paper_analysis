## Vortex GPU 编译器与自定义 intrinsics（POCL-Vortex 工具链）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Vortex 是开源 RISC-V GPGPU 全栈（https://github.com/vortexgpgpu/vortex），支持 RV32IMAF/RV64IMAFD，可配置微架构（核数、warp、线程、ALU/FPU/LSU/SFU、cache），提供 OpenCL 1.2 支持并面向 FPGA（Altera Arria 10、Stratix 10、Xilinx Alveo U50/U250/U280、Versal VCK5000）。工具链由 POCL（PoCL 的 Vortex 分支，https://github.com/vortexgpgpu/pocl，OpenCL 1.2 驱动）+ LLVM + riscv-gnu-toolchain 组成；prebuilt 工具链见 https://github.com/vortexgpgpu/vortex-toolchain-prebuilt，新一代统一编译器框架 VOLT（https://github.com/vortexgpgpu/Volt）扩展 PoCL/CuPBoP/LLVM 支持 CUDA 与 OpenCL 前端。PipeIMC 基于 Vortex 编译器实现其 IMC ISA：通过**自定义 intrinsics（custom intrinsics）**把每个 in-SRAM 计算操作表达为指令（如 dest := IMC(src1, src2, op)），每操作解释为至多三个 phase（load/compute/store）；操作编码为 R 型/I 型（31:29 memory flags、28:24 rd、23:17/16:12 rs2/func 或 immediate、11:7 rs1、6:0 opcode），memory flags 表示寄存器用作内存地址还是操作数。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
编译流程（POCL-Vortex 工具链）：OpenCL kernel（Rodinia 的 matmul/stencil3d/backprop/bfs/kmeans/pathfinder + 自写 transformer kernel matvec/ffn/attention/layernorm，32-bit 定点，attention/ffn 用 INT8 量化）用自定义 IMC intrinsics 编写 → POCL 前端把 OpenCL C 编译到 LLVM IR → Vortex 的 LLVM-RISC-V 后端生成 RISC-V + IMC 扩展指令 → 预写入 tag array（kernel 执行前由 CPU compute request 配置 PipeIMC 时写入）。IMC 操作覆盖四类：Compute（dest := IMC(src1, src2, op)，源可为内存指针/寄存器/立即数）、Control Flow（BRANCH(cond,tag)、ptr := SPLIT(cond)、JOIN(ptr)）、Synchronization（BARRIER(operand)、FENCE）、Warp Control（WSPAWN(operand)、TSPAWN(operand)）。为公平对比，论文修改了 EVE 与 Duality Cache 的外围电路以支持同一套 Vortex 编译的 SIMT IMC 程序（Vortex toolchain 编译），各架构用不同数据布局（hybrid-4/bit-serial/hybrid-8）映射这些操作。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：Vortex 主仓库 + POCL 分支提供完整编译-运行栈（OpenCL kernel → 可执行 → 在 RTL 模拟/FPGA/ASIC 上运行）；PipeIMC 在 cycle-approximate 模拟器中重建执行（计算 phase 周期由 cycle-accurate 计算 SRAM 模拟器给出），编译产物是模拟器输入的一部分。评估只测 in-SRAM 计算架构性能，忽略 CPU 上的预处理/后处理（执行 SIMT kernel 的代码）。Web 证据：Vortex（https://vortex.cc.gatech.edu/）是 Georgia Tech 的开源 RISC-V GPGPU 项目，论文 [35] 引用；VOLT（CC 2026 接收）为其新一代编译器框架。Vault 笔记（omnisearch 对 Vortex 无命中）无本术语专门笔记证据。

sCROOGe 合并视角（Vortex 作为 RTL 级 OoO 微架构评估平台）：sCROOGe（ISCA'26，RISC-V Vortex GPGPU）沿用 Vortex 软件栈（POCL 做 OpenCL 前端编译、LLVM 生成应用二进制）驱动其 RTL 评估——22 个 Vortex benchmark 经 POCL+LLVM 编译为二进制，经 Verilator cycle-accurate RTL 仿真输出 per-kernel 指令数与 cycles（性能指标），无需新编译器改造（sCROOGe 的修改全部在 RTL 微架构层，ISA 无扩展）。工具链使用方式与 PipeIMC 相同（OpenCL kernel → POCL → LLVM IR → RISC-V 二进制 → 驱动 RTL 仿真/FPGA），区别在于 sCROOGe 用原生 Vortex 指令（无自定义 intrinsics），其实验经 /vortex/ci/blackbox.sh 以 bash 选项选择 OoO scheme（frontend/backend/无 RRS 的 backend 变体）运行。开源 artifact：Zenodo DOI 10.5281/zenodo.19453033（Docker 容器，含全部 SystemVerilog 源码、benchmark 源码与预计算 .csv）。

涉及论文标题：
- PipeIMC a Pipelined In-SRAM Computing Architecture
- sCROOGe Circuit-level Design and Optimization Framework for RISC-V Out-of-Order GPUs
