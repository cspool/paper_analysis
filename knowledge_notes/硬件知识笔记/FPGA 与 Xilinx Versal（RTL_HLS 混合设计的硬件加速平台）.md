## FPGA 与 Xilinx Versal（RTL/HLS 混合设计的硬件加速平台）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- FPGA = 可反复配置逻辑资源的芯片，适合需要快速迭代、无法摊薄 ASIC 流片成本的硬件加速；Versal = AMD/Xilinx 的 ACAP（Adaptive Compute Acceleration Platform）家族，把标量处理核（PS）、可编程逻辑（PL）、AI Engine、片上网络（NoC）、PCIe/CXL 与 DDR 控制器等 hardened IP 集成在单芯片上。
- Versal 的 hardened NoC（Web 证据：AMD PG313）：NMU/NSU 在 AXI 与 NoC 包（NPP）间转换、NPS 4×4 分组交换、NIDB 跨 die 桥接，聚合带宽 Tbps 级；相比 UltraScale+ 的软 AXI 互连可省 30–60% fabric 资源、改善时序收敛。论文选 FPGA 的理由（§V）：对比 ASIC 可快速迭代、赶上时间窗；对比软件仿真，FPGA 原型评估又快又真（§VII Insights）。
- 论文的工程方法论（§VII）：RTL 与 HLS 混合——复用性基础设施 IP（mux/demux、协议 shim）与功能稳定的模块手写 RTL（改善时序收敛与布线质量），复杂算法用 C/HLS 快速迭代；模块化封装 vendor-dependent platform 隔离板级差异；可观测性占 2.35% 逻辑资源。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- NTI 的资源账本（Table II，Versal 900K LUT/1.8M FF/1341 BRAM/677 URAM 总量）：总占用 LUT 520K(57.8%)、FF 954K(53.0%)、BRAM 828(61.7%)、URAM 208(30.7%)；分项 NHI 36K LUT/103K FF/76 BRAM，NVMe/TCP Engine 195K LUT/446K FF/464 BRAM/74 URAM，TOE 225K LUT/332K FF/264 BRAM/132 URAM，外设 63K LUT。NVMe/TCP Engine 的 BRAM 大头来自 PDU metadata table + PRP table 等片上元数据（Virtual buffer 的实现载体）。
- HHHL 卡资源/热约束下的硬件设计：PRP list 留在宿主内存、片上只存头指针 + 预取（省片上存储）；完成的 PDU 元数据实时失效（降驻留）；AMD CIPS IP 内 System Monitor 测温、90% 阈值主动节流 + NVMe admin completion 报警（热管理闭环）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现流程：Vivado IP Integrator 里配 CIPS + NoC（路由与 QoS 编译期固化）→ PL 内放三 IP → Vitis/板载核跑控制面 → 位流烧入卡上 Flash。论文还建议（§VII future）：未来 FPGA 应把整 NIC 子系统（MAC 互连交换）做成 hardened IP，使节点间通信成为一等组件。
- 使用场景：需要"协议定制 + 快速迭代 + 中小批量"的数据面加速（存储、网络、安全）；当 workload 形态稳定后可从 FPGA 收敛到 ASIC。信息缺口：论文未给出各 IP 的 HLS/RTL 划分明细与频率/时序数据。

CODO 的 FPGA 使用视角（ISCA'26）：把 FPGA 作为数据流加速器的部署平台——综合用 Vitis HLS 2023.2 + Vivado 2023.2（全部实验目标频率 300 MHz），上板用 AMD Alveo U280（PCIe 卡：9024 DSP slice、2.6M FF、1.3M LUT、4032 BRAM18K、8GB HBM），OpenCL host 程序加载 xclbin 执行（./host.exe kernel.hw.xclbin）。U280 资源账本在 CODO 中充当 DSE 资源预算（kernel 级实验统一 DSP=900，约单个 super logic region 的 1/3）；对照平台含 Alveo U55C（16GB HBM）。上板结论：CODO 与 HIDA 是唯一能对全部 DNN（除 YOLO——Vitis HLS 资源估计不准无法实现）生成可执行加速器的编译器；FIFO 数据流设计使 BRAM 占用远低于 ping-pong 方案（ScaleHLS 208.7% 溢出无法上板、Allo 细粒度违例死锁、StreamHLS 无法生成有效设计）。

涉及论文标题：
- BoostX™-NTI Fast, Scalable and Flexible Storage Architecture with NVMe-TCP Initiator Acceleration
- CODO: An Automated Compiler for Comprehensive Dataflow Optimization
