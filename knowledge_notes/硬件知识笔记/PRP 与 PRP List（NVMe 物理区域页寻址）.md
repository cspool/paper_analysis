## PRP 与 PRP List（NVMe 物理区域页寻址）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PRP（Physical Region Page）= NVMe 规范的数据缓冲寻址机制：命令里携带一个/两个 64-bit PRP 指针，指向主机物理内存中数据所在页；PRP entry = Page Base Address + Offset（页大小由 CC.MPS 决定，如 4KiB 页用 bit 11:0 作 offset，offset 必须 dword 对齐）。规则（Web 证据：NVMe 规范与 SPDK 文档）：只有命令内第一个 PRP 可带非零页内偏移，PRP2 与 list 条目必须页对齐；数据跨多页时，第二个 PRP 位置放 PRP List 指针——list 是宿主内存中一页内的连续 PRP 条目数组，指向后续数据页，必要时 list 页末条指向下一个 list 页。
- 与 SGL 的关系：SGL（NVMe 1.1 引入）是更灵活的替代寻址方式；PRP 仍是 NVMe/TCP 生态最常用的数据缓冲描述。论文中 PRP 是协议转换的关键对象：NVMe Read 命令携带 PRP 告诉"数据回来放哪"，但远端 NVMe/TCP target 看不到发起端地址空间，因此发起端必须自己记录 PRP、在数据 PDU 到达时完成放置。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- NTI 的 PRP table 流程（§V-A/B，Fig.6/7）：PDU Header Generator 组 CapsuleCmd 头时抽取命令的 PRP 条目存入 PRP table（写命令：供 Stitcher 取数据；读命令：供 Parser 放数据）→ TX 侧 PDU payload receiver 查表得主机地址、DMA 取写数据拼接 PDU；RX 侧 PDU header handler 对数据 PDU 查表得目的地址 → splitter DMA 直写主机 data buffer。条目直接是数据缓冲地址则直返；是 PRP list 则硬件执行 list walking（沿宿主内存的 list 页逐页取地址）——list 本身留宿主内存，片上只存头指针并预取所需地址（省片上存储，§V-C）。
- 资源含义：PRP table 与 PDU metadata table 共享于 Stitcher/Parser 两侧，是 Virtual buffer 按需 DMA 的寻址基础；PRP list walking 也是"哪些地址查询必须能出片"的典型硬件访存模式（随机读、可能跨多页）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件实现（Web 证据：Linux/SPDK）：Linux nvme 核心设 virt_boundary_mask 保证除首个 PRP 外均页对齐；SPDK 用户态驱动用 DPDK 大页保证物理连续以便 PRP 描述。硬件实现（NTI）：PRP table + 硬件 list walking + 预取。
- 使用场景：任何 NVMe 数据面（驱动、固件、DPU IP）的 DMA 寻址设计；与虚拟化交互时 PRP 页必须 pin（NTI 的 DMA 区域为宿主 pinned 内存）。信息缺口：论文未说明 PRP table 条目数与预取深度。

涉及论文标题：
- BoostX™-NTI Fast, Scalable and Flexible Storage Architecture with NVMe-TCP Initiator Acceleration
