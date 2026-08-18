## PBR (Port-Based Routing)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PBR 是 CXL 3.1 引入的 fabric 级路由模式：每个交换机端口分配唯一 Port ID（PID）作为独立路由端点；路由按目的 PID 而非主机层级。Fabric Manager（FM）把 PID 映射到一致性标识符与内存区域，形成跨主机/加速器/内存设备的全局地址与一致性域，支持主机到主机、主机到内存的硬件速度直连（CXL 3.1 还引入 Global Integrated Memory GIM 支持 host-to-host）。PBR 报文带 SPID/DPID（源/目的端口 ID，12-bit PID 域）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
PBR 支持任意拓扑（web：Cadence/CXL 规范——mesh/ring/spine-leaf，CXL 3.0 至 4096 节点）。本论文芯片内 PBR 数据路径：入口端口硬件分类器判定 HBR/PBR 域 → HBR 流量翻译为 PBR（派生 SPID/DPID、头部重建，单级流水完成）→ DPID 路由表（DRT）固定周期查出口端口 → 路由组表（RGT）按拥塞状态在等价端口间选路 → 仲裁 → 非阻塞 NoC 转发 → 出口端口（反向同理）。全程固定周期硬件流水线、零固件，每跳延迟与端口数/拓扑深度无关；多交换机级联构成 multi-tier fabric，本论文达 64 节点近线性扩展。vault 佐证：human_notes/算法-协议-硬件笔记/CXL分解内存、一致性协议.md——"CXL 3.1 在互联上不局限于树状拓扑的层次化路由(HBR)，而是采用基于端口的路由(PBR)"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FM 在系统初始化时配置全局路由表（PID 映射与端口连通性，MCTP/CCI 管理通道）；链路增删由内部拓扑同步自动更新路由表，无需软件。专利（US20240378161）描述 PBR 交换机用 SPID/DPID 管理主机间缓存一致性（VCS 概念）。商用：Panmnesia 为全球首个 CXL 3.2 PBR ASIC 交换机（预发布硅片 2026 送样、量产目标 2026H2）；Cadence 等提供 PBR 交换机 IP。

涉及论文标题：
- A Silicon-Proven Unified Low-Latency CXL Controller and Port-Based Routing Switch for Memory-Centric Fabrics
