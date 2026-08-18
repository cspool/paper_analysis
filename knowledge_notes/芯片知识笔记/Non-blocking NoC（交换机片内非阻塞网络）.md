## Non-blocking NoC（交换机片内非阻塞网络）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NoC（Network-on-Chip）是片上互联结构，以路由网络（而非全局总线/crossbar 连线）连接芯片内模块。本论文交换芯片内部的 NoC 互连所有端口 bank，支持并行的 ingress/egress 传输，是非阻塞交换（任意输入到任意输出可同时转发）的硬件基础。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
芯片内数据路径：每个端口 bank（统一控制器 + 转换/路由逻辑）作为硬件端点，入包经转换/路由流水线（分类→翻译→DRT/RGT 查表→仲裁）后由 NoC 并行转发到目标端口 bank；NoC 按路由决策动态连接端口、同时收发互不阻塞。因为所有 datapath 共享一个时钟域，并发流量不造成时序错位，端到端延迟保持接近单个控制器、不随端口数增长——这是交换机"每跳恒定延迟"的关键。vault 佐证：knowledge_notes/硬件知识笔记/2D Mesh NoC in 3D Near-Memory Processing.md（XY 路由、Manhattan 距离）；chiplet 生态中 NoC 常需防死锁桥（DFBM）消除跨 chiplet 循环通道依赖。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
常见实现：mesh/torus/crossbar 拓扑 + 虚通道流控（RC→VA→SA→ST→LT 五级流水线、credit-based、wormhole switching）。本论文将 NoC 与固定周期路由流水线绑定，形成"固定延迟 datapath"，支撑多级 CXL fabric 的可组合内存架构；宽端口可聚合为单一高带宽接口或拆分为窄端口，因共享时钟域延迟稳定。评估中 4N4S_SWopt 在 YCSB-A 写密集下达 >95% 带宽利用率。

涉及论文标题：
- A Silicon-Proven Unified Low-Latency CXL Controller and Port-Based Routing Switch for Memory-Centric Fabrics
