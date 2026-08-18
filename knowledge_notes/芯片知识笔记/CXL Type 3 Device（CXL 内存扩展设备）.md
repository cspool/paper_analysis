## CXL Type 3 Device（CXL 内存扩展设备）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CXL Type 3 设备是只集成 CXL.io + CXL.mem（无 CXL.cache）的 CXL 设备类型，典型形态是内存扩展器：设备本地内存以 Host-managed Device Memory（HDM）暴露给主机，主机用普通 load/store（经 CXL.mem）字节级访问。Type 3 设备是内存池化/解耦内存的主流形态（web：Linux CXL 内核文档 device-types；CXL 3.0 引入 HDM-H 主机一致性、HDM-D 设备一致性与 HDM-DB 带 back-invalidation snoop 的设备一致性模式）。与 Type 1（smart NIC 类，无本地内存的缓存设备）和 Type 2（GPU 类加速器，三协议全开 + 硬件一致性 DCOH）相比，Type 3 硬件成本最低，适合纯容量/带宽扩展。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
AXLE 论文把 Type 3 设备用作 CCM 底座：在 Type 3 内存扩展器上叠加 PNM 计算单元（FPGA PFL 或 µthread 核），并附加 bus-master DMA 引擎实现设备发起的数据搬运。运转流程：主机 load/store 经 CXL.mem 访问设备内存（HDM）→ PNM 处理数据并把结果写入设备内存 → DMA executor 经 CXL.io posted write 把结果写到主机物理地址（设备→主机反向流，HDM 之外的主机 pin 内存）→ 主机本地轮询消费。论文论证 Type 3 优于 Type 2 的原因：Type 2 的 DCOH 需要大 SRAM 目录跟踪一致性状态、面积/功耗超过 PNM 计算单元本身，且页级 bias flip（Host Bias↔Device Bias）要刷主机缓存、带来数百 ns~µs 延迟；而 CCM 结果是只读、无时间局部性的中间数据，硬件一致性纯属浪费。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Type 3 设备以 add-in card 或 DIMM 形态接入，控制器 ASIC 内嵌 CXL.io/CXL.mem 协议栈与内存控制器；SK hynix 的 CCM 原型卡用 Xilinx Versal VP1502 FPGA 集成 CXL 内存控制器 + PNM 引擎 + 4 个 DIMM 槽。系统侧 Linux 经 daxctl + mmap 把设备内存映射为字节可寻址区域（如做缓存/KV Cache），MLD（multi-logical device）模式下每个逻辑设备有独立 DMA 引擎与路径互不干扰。用法：内存容量扩展、多主机内存池化、CCM 部分卸载（KNN/图分析/OLAP/LLM attention/DLRM 等）。

Vistara 补充视角（ISCA'26，Meta 自研 CXL Type-3 扩展器 ASIC）：Vistara 是 Meta 第一代定制 CXL 内存扩展器，专门为"复用退役 DDR4 RDIMM"而设计——市售 CXL 方案多捆绑新 DRAM、不支持 DDR4 且功耗/成本高，无法复用旧内存。关键规格（Table II）：CXL 2.0/1.1 Type-3、PCIe Gen5 x16（生产部署 x8）、2 个独立 72-bit DDR4 通道（最高 3200 MT/s，生产 2400 MT/s 以省功耗并兼容 mixed-vintage）、每芯片最大 256GB（4×64GB；生产 4×32GB=128GB）、RS(36,32) 2-symbol 纠错 + x4 chip-kill、3 个 RISC-V 管理核（secure 安全启动/control 扩展器固件/boot 设备初始化）、CCI/SMBus/PCIe FW 更新接口、ASIC 空闲时延 ≈50ns、功耗 ≈9W、先进工艺 + 激进 clock/power gating。每板 2× Vistara → MemServer 上 256GB CXL DDR4（8×32GB RDIMM）。系统侧：Linux CXL 驱动把 CXL 内存 online 为独立 NUMA 节点上的 ZONE_MOVABLE（确保可迁移、隔离内核分配）；ACPI CEDT/HMAT 表提供拓扑与性能数据供内核分层决策。生产成果：ML 参数服务器服务器数 -25% 且吞吐 +12%、缓存 QPS +33%、查询时延 -29%、CXL Power/GB 0.7× 与 Cost/GB 0.13× 相对本地 DRAM。

涉及论文标题：
- AXLE: Coordinated Offloading with Asynchronous Back-Streaming in Computational Memory Systems
- Vistara: Making CXL Real—Full Path from ASIC Design and OS Support to Hyperscale Deployment
