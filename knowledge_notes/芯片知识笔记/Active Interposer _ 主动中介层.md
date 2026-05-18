## Active Interposer / 主动中介层

术语是什么？

Active Interposer 是 2.5D 多芯片集成中的一种中间层基板，不仅提供 passive 的金属走线连接各 chiplet，还在 interposer 内部嵌入有源逻辑电路（如 NoC 路由器、电压调节器、安全监控器等）。与仅含金属布线的 passive interposer（硅中介层）不同，active interposer 可以执行数据交换、路由仲裁、协议转换等主动功能，使 interposer 由一个纯互联层升级为系统级通信基础设施。论文中的 active interposer 构成了 chiplet 之间的共享 NoC 网络，每个 chiplet 通过边界 bridge 接入 interposer NoC。

从芯片设计角度拆解术语：

在 2.5D 集成中，多个 chiplet 并列放置于 active interposer 之上，通过 microbump 和 TSV 连接到 interposer 中的 NoC 路由器和链路。论文面向的架构是：四个 homogeneous chiplet（各自内含 4×4 mesh NoC）通过 shared active interposer（同样为 4×4 mesh NoC）互连。chiplet 内部 NoC 的包通过 boundary router 经由 vertical channel (TSV-based) 进入 interposer NoC → interposer NoC 按 XY routing 将包送达目标 chiplet 的 vertical channel → 进入目标 chiplet 内部 NoC → 到达目标 core/cache。Active interposer 上的路由器、VC buffer、credit 管理等逻辑位于 interposer die 中（使用 logic process），其面积和功耗开销由成本较低的 interposer side 承担（芯片设计中的关键 trade-off：将 DFBM 逻辑放在 interposer 而非 chiplet 内部，降低了 chiplet 侧成本并保持了模块化）。

术语一般如何实现？如何使用？

Active interposer 通常基于成熟的硅工艺（如 65nm、28nm 或更先进节点），在 silicon interposer 中集成数字逻辑。实现方式包括：(1) 在 interposer 中嵌入完整的 NoC 路由器 mesh/torus 网络（本文方式）；(2) 在 interposer 中集成安全监控器和电压调节器（IEEE TCPMT 2020）；(3) 使用硅光子 waveguides 实现光 NoC（如 CEA-Leti 的 Starac 技术）。本文的 active interposer 使用 4×4 mesh topology + XY routing + 3 VN + per-VN 2/4 VC + virtual cut-through flow control，与 chiplet 内部 NoC 拓扑对等。通过 UCIe 等 die-to-die 标准定义 physical layer，interposer 作为共享背板连接不同供应商的 chiplet。

涉及论文标题：
- Deadlock-Free Bridge Module for Inter-Chiplet Communication in Open Chiplet Ecosystem
