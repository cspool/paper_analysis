## UPMEM DPU（DRAM Processing Unit）与银行级 PIM 架构

术语解释
把 DRAM 芯片内每个 bank 旁挂一个极简 in-order 标量 CPU（DPU），用两块 SRAM（IRAM/WRAM）桥接 CPU 与 bank，构成商用 DDR4 DIMM 级可编程 PIM；BAAP 以它为底座，把 WRAM 的一部分替换为关联处理器（AP），形成 Scratchpad/SIMD/DirectAP 三模式银行级引擎。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PIM 按集成粒度分层（论文表 I）：channel/3D-stacked（Active Memory Cube、Tesseract，70–140ns）→ rank-level（MCN、Samsung AxDIMM，OoO CPU/FPGA）→ bank-level（UPMEM DDR4-PIM、Samsung FIMDRAM、SK hynix AiM，几十 ns、聚合带宽最高）→ subarray-level in-situ（Fulcrum、Sieve，改 DRAM subarray 本身）。bank-level 在不触碰 DRAM subarray 的前提下拿到最高内部带宽与低延迟，但受 DRAM 工艺约束（约 3 层金属、逻辑密度 1/10、pitch 4×、晶体管慢 3×）逻辑必须极简。UPMEM 规格（论文表 III；web Hot Chips 31 [Devaux 2019]、ETH 基准 [Gómez-Luna 2022]、DaPPA arXiv:2310.10168 一致）：DDR4-2400 8GB DIMM、16 芯片×8 bank=128 DPU；每 DPU = 32-bit in-order CPU @500MHz、14 级 Revolver 流水线、24 tasklets（HW 线程）、无 bypass/stall 信号、2-split 寄存器堆、1 周期 ADD/32 周期 MUL；24KB IRAM + 64KB WRAM + 64MB DRAM bank（1GB/s，聚合 128GB/s；服务器 20 DIMM 聚合约 2TB/s）；DPU 间无直连，跨 DPU 通信必须经 host 中转。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
数据路径例子（论文 §III-A）：host 用 SDK scatter 数据到 bank → DPU 发 DMA（mram_read/mram_write）把数据搬入 WRAM → CPU 从 WRAM 单周期 load/store 执行（WRAM 访存比 bank 快约两个数量级，隐藏 DRAM 延迟）→ 结果 gather 回 host；指令流由 host 编译 DPU 二进制后从 DRAM bank 灌入 IRAM 供 DPU 取指。BAAP 的关键观察：一旦 DMA 消除数据搬移瓶颈，很多内存受限应用在 bank 级重新变成 compute-bound（单 DPU roofline ≈0.35 GOPS，乘法 32 周期、除法等软件模拟）——可编程但算力单薄，这是 BAAP 把 WRAM 换成 AP 的动机。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
UPMEM SDK（https://sdk.upmem.com）：LLVM 编译器 + 功能模拟器 + host 通信库，host/DPU 两级编程、SPMD 模型、tasklet 多线程。开源仿真：uPIMulator（https://github.com/VIA-Research/uPIMulator，HPCA'24，cycle-approximate、单 DPU 相关性 98.4%/MAE 12.0%）。BAAP 的用法：保留 DPU 标量能力（Scratchpad 模式），把 WRAM 25–88% 重配置为 AP（SIMD/DirectAP 模式），面积开销 1.2281× 折算容量、AP 降频 350MHz。DCC（ISCA'26）从 ML 工作负载角度补充定位：UPMEM 基于 DDR4 接口、面向 CPU 系统，无浮点单元、32-bit 乘法需软件模拟（8-bit 乘法器），故 CPU-PIM 协同的 ML 执行通常打不过 GPU-only，不适合需浮点与强乘法单元的 GPU-PIM 异构 ML 场景；现有 PIM 编译工作（SimplePIM/PIM-DL/CINM/PIMFlow/ATiM）多只支持 UPMEM，是其无法覆盖现代 ML kernel 的原因之一（Web 证据 Hot Chips 31、PrIM 基准一致）。

涉及论文标题：
- BAAP: Coupling Compute-in-SRAM with DRAM Banks for Near-Memory Processing
- DCC: Data-Centric Compilation of Machine Learning Kernels for Processing-In-Memory Architectures
- P3-LLM An Integrated NPU-PIM Accelerator for Edge LLM Inference Using Hybrid Numerical Formats
