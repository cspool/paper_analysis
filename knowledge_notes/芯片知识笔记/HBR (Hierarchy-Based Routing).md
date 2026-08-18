## HBR (Hierarchy-Based Routing)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
HBR 是 CXL 沿袭 PCIe 的传统路由模式：以主机为根形成树状层级，每个主机在自己 Root Complex（RC）下管理一个 Virtual CXL Switch（VCS），维护独立的物理地址空间；设备按"属于哪个主机"归属，跨主机通信与内存共享必须绕经主机软件栈。HBR 报文用 4-bit LD-ID/CacheID/BI-ID 在域内标识设备（web：CXL 规范），与 PCIe 枚举兼容。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
HBR 的芯片级代价（本论文）：商用 HBR 交换机不支持交换机级联（256-lane 配置最多 8 节点）；单跳内存扩展器 RTT 265–442ns，过交换机路径叠加 ≥2 个控制器流水线 + 固件路由后 RTT 近乎翻倍，故超大规模厂商回避交换机，退化为 1:1 直连或 MHD。数据路径例子：主机 A 访问挂在主机 B 下的设备时，报文必须先按 PCIe 树状层级上行到 B 的 RC 再下行，无法在 fabric 内直连（web：EE Times "CXL Overcomes Hierarchical Routing Limits"：HBR 下设备间通信必须经主机中转）。本论文交换机在每端口硬件做 HBR↔PBR 双向翻译：HBR 域报文被翻译为内部 PBR 表示（SPID/DPID）转发，出域时经拓扑查表映射回 PCIe 兼容标识符。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
2024–2025 商用控制器/交换机仍以 HBR-only 为主（过渡阶段）。CXL 3.1 起规范正式混合 HBR/PBR：HBR 保留 PCIe 系统软件兼容，PBR 承担 fabric 级组合。使用方式：单主机内存扩展、虚拟化环境回收闲置内存（stranded memory）。局限：不支持多主机组合、跨主机数据共享与交换机级联。

涉及论文标题：
- A Silicon-Proven Unified Low-Latency CXL Controller and Port-Based Routing Switch for Memory-Centric Fabrics
