## DPU（Data Processing Unit，含 sidecore 与硬件卸载）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- DPU（Data Processing Unit）= 插在服务器 PCIe 槽上的可编程协处理器卡，专用于承接 CPU 不适合做的数据面工作（网络/存储/安全协议处理、虚拟化卸载），内部通常是"嵌入式核（sidecore，ARM 或 RISC-V）+ 专用加速器 + 高带宽 NIC"的组合。代表产品：NVIDIA BlueField-3（ARM 核 + DPA 加速器簇）、AMD Pensando Salina（P4 可编程流水）、Napatech（Xeon-D 级）、MangoBoost NTI（FPGA 硬件流水）。
- 逻辑链：主机 CPU 逐包处理协议栈开销大且随带宽线性增长 → 把数据面搬到卡上的 DPU → 但"搬到 sidecore"只是搬位置不消计算量（论文实测 BlueField-3 SNAP 16 个 ARM 核全开仅达 200 Gbps line-rate 的 9.6%，SNAP+XLIO 达 34.3%）→ 进一步把关键 I/O 做成专用硬件逻辑（NTI 路径）才能解耦性能与核数。
- 训练集合卸载语境（DisDP）：把 MSDP 的集合通信整体卸载到 SmartNIC 时，off-path SoC SmartNIC 无法线速——BlueField-2 每方向网络 200Gbps，并发 push/pull 需 400Gbps 每方向 Arm-switch 带宽（实际仅 250Gbps）；且包在 Arm 内存 staging 使每方向流量需 2× 内存访问，200Gbps 线速需 800Gbps 内存带宽（实际理论仅 204.8Gbps，DisDP 实测 BlueField-2 仅 20% 链路利用率）；BlueField-3 亦不足（400Gbps 线速需 1600Gbps vs 提供 716.8Gbps 理论带宽）。
- NTI 的 DPU 组成（Fig.5/6）：Xilinx Versal FPGA（NHI + NVMe/TCP Engine + TOE 三 IP）+ 板载 sidecore（控制面）+ 双 100GbE，单槽 HHHL、75W。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 三类 DPU 数据面组织对比：(1) 纯 sidecore 数据面（BlueField SNAP）——主机 NVMe 命令经存储仿真栈转块请求，再跑完整 nvme-tcp + TCP 软件栈发网，算力瓶颈（9.6% line-rate）；(2) sidecore + 部分加速（SNAP+XLIO）——TX 零拷贝绕过板载内存直送 NIC + 用户态 TCP，但 RX 与残余栈仍在核上（34.3%，核减半掉到 13.9%/6.1%）；(3) 全硬件数据面（NTI）——I/O 命令/完成/数据全部在 FPGA 流水内，sidecore 只做 admin 队列/配置/错误恢复，带宽增长不加重 sidecore 负担。
- NTI 流程例子：主机发 NVMe 命令 → NHI 硬件轮询取命令 → Command Scheduler 映射队列 → PDU Header Generator 组头 → PDU Stitcher 经 PRP 表按需 DMA 数据 → TOE 发包；回程 TOE → PDU splitter 边收边切 → PRP 查表 → DMA 直写主机 → Decapsulator 出 completion。硬件 I/O 能力 5000 万 IOPS，4KB 小块可饱和 1.6 Tbps。
- on-path FPGA SmartNIC 集合流程例子（DisDP，Xilinx Alveo U50）：host 调 push(handle) → MMIO 写入 push 请求队列 → push processing unit 出队、按 GPU 虚拟地址 DMA 直读 GPU 显存（register_buf 预注册）→ 流水线内 format conversion unit 把 bf16 梯度转整数（SwitchML 策略，固化在硬件）→ 线速发包到 SmartSwitch；pull 为对称反向路径，push/pull 各占独立流水单元互不争用。FPGA 流水级间数据放片上 SRAM 而非 off-chip DRAM，故无内存带宽瓶颈。U50 资源：LUT 135K(15.5%)、FF 225K(12.9%)、BRAM 354(26.3%)、URAM 128(20.0%)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 通用实现要点：卡上多核 SoC（BlueField-3 为 8/16 核 ARMv8 + DPA RISC-V 簇，Web 证据：NVIDIA 文档）或 FPGA fabric（NTI）；主机侧以标准设备呈现（SNAP/NTI 都仿真为 NVMe PCIe 设备，免改主机驱动）；控制面跑容器化服务（SNAP 以容器部署，NTI 控制面用 Mango SDK + SPDK RPC）。
- 使用要点：选型维度 = 数据面是否全硬件（决定 line-rate 可达性）、form factor 与功耗（FHHL 150W vs HHHL 75W）、协议合规与可运维性（NTI 强调 UNH-IOL 认证与 Dynamic handover）。信息缺口：论文未披露 Versal 具体型号与 sidecore 核数。
- DisDP 使用要点：SmartNIC 直访 GPU 显存需 register_buf 预注册 + GPU 虚拟地址寻址（GPUDirect 类机制，沿 FPGA-NIC/StRoM 做法）；实测 PCIe DMA 与 GPU GEMM 并发几乎零干扰，因此「集合整体卸载到 SmartNIC」优于在 GPU 内做 MPS/SM 抢占优化（DisDP 比 ZeRO-Inf+MPS/Preemp 仿真快 3×）。信息缺口：论文未给出 U50 上 DMA 引擎与网络 MAC 的时钟/位宽细节。

涉及论文标题：
- BoostX™-NTI Fast, Scalable and Flexible Storage Architecture with NVMe-TCP Initiator Acceleration
- DisDP: Disaggregating Compute, Network, and Storage for Model-Sharded Data-Parallel Training
