## Xilinx Alveo U50（中端数据中心 FPGA 加速卡，HBM2 + PCIe）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Alveo U50 是 AMD/Xilinx 的中端数据中心 FPGA 加速卡（2019，~$3,000）：核心为 UltraScale+ **XCU50**（16 nm FinFET+，872K LUT、1,743K FF、5,952 DSP、~5 MB BRAM、~20 MB URAM）；**8 GB HBM2**（2×4 GB 堆栈、32×256 MB pseudo-channel、峰值 ~316 GB/s）；**PCIe Gen3 x16**（支持 Gen4 x8，理论每方向 ~8 GB/s 双工）；75 W TDP、半高半长单槽、被动散热。与 Alveo U280/U55C 同属 Alveo 家族（U280 8 GB HBM/9024 DSP、U55C 16 GB HBM），与 Versal ACAP 家族（PS+PL+AI Engine+NoC）不同——U50 是纯 FPGA + HBM 的 PCIe 卡。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- Lembas（ISCA'26）把 U50 用作三阶段基因组比对的共享加速平台，其 32 个 HBM pseudo-channel 与 PCIe 带宽是设计的硬约束：① seed（外部内存 columnsort）——16 个 16-to-1 merge-sort kernel 各独占一对 HBM pseudo-channel（1 数据列 256 MB + 1 scratchpad），250 MHz/4 GB/s 每 kernel，恰好饱和 ~8 GB/s 双工 PCIe（32 个 kernel 布局布线失败，且 PCIe 已是瓶颈）；② chain——4 个 kernel（每 kernel 单 PE）即饱和 PCIe；③ extend——16 kernel × 16 PE 脉动阵列 + 8×8 tile traceback（480 bit/tile 对齐 512-bit HBM 接口，预取 4 tile 缓解 HBM 延迟）。三阶段各独立综合成 bitfile（Vivado），软件 orchestrator 按时分复用把同一 bitfile 编程进全部 U50（Seed 361,624 LUT/41.53%、Chain 273,951 LUT/31.46%、Extend 431,957 LUT/49.61%）。默认 2 卡 + 4×1TB NVMe；任务级并行下多卡近线性扩展（3 卡时 NVMe 成瓶颈）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：Vivado 综合+布局布线生成 bitfile（xclbin），OpenCL/XRT host 程序加载并调度 kernel（如 `./host.exe kernel.hw.xclbin`）；PCIe DMA 引擎在 host DRAM/NVMe 与 HBM 之间搬数据。Lembas 的 host 是 i7-8700（16 线程/32 GB DDR4），数据流 NVMe ↔（PCIe）↔ HBM；中间数据（minimizer 流/anchors/chains）驻留 NVMe 而非 HBM/DRAM，系统内存恒定 ~8 GB。云上可用 AWS f2.5xlarge（VU47P FPGA）等实例；论文以 $3,000/卡 计成本。典型应用：ML 推理、计算存储、数据分析、金融科技（U50 官方定位）。

涉及论文标题：
- Lembas: Cost-Efficient Genome Alignment with External Memory and FPGA Acceleration
