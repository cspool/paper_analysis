## ns-3（packet-level 网络模拟器）

术语解释
免费开源（GPLv2）的离散事件网络模拟器，逐包级（packet-level）模拟网络协议与拓扑行为；R2D2 用它建模 400 GbE 网络的带宽与延迟。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ns-3 是 ns 家族第三代（ns-2/GTNetS/yans 的合成），面向研究与教学，由华盛顿大学、Inria、Georgia Tech、INESC TEC 等开发，250+ 贡献者、SIGCOMM 2020 Networking Systems Award。离散事件调度：事件（发包/收包/超时）按时间排序执行，模拟数百节点到百万节点规模；脚本用 C++/Python。Web 证据：nsnam.org 官方定义"discrete-event network simulator"，逐包处理、支持 IP/非 IP 网络、TCP（BBR/DCTCP）与路由模型、实时调度器与 DCE（跑真实 Linux 协议栈）。
- 论文用法：网络带宽与延迟用 packet-level ns-3 模拟，400 GbE、per-hop 500ns 延迟（引用 Tomahawk 4 25.6Tbps switch 数据）；输出 FCT（流完成时间）与吞吐。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 R2D2 评估中 ns-3 负责网络数据面微基准：输入=拓扑（R2D2 直连 / NDP fat-tree / OCS）+ Gao et al. 内存流量 trace（每条目=资源访问流：请求时间+流大小）+ 链路参数（400 GbE、500ns/跳）→ 逐包模拟转发/排队/拥塞（fat-tree 多跳竞争、incast 碰撞；R2D2 单跳直连无中间转发）→ 输出平均/p99 FCT 与平均/p01 吞吐。结果：R2D2 比 fat-tree 平均 FCT 改善 43.3%、吞吐 70.2%，与 OCS 相同。
- 与自定义 discrete-event simulator 的分工：ns-3 管数据面性能，自定义 DES 管 R2D2 runtime 调度（allocation latency、机器人并行度、故障注入）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：写 C++（或 Python）脚本定义 Node/Channel/NetDevice/Application，配置点对点/CSMA/无线信道、TCP/UDP 协议栈，用 Simulator::Run() 驱动离散事件循环，TraceSource 采集统计。论文侧未说明对 ns-3 的具体修改（采用其 packet-level 模型与 500ns 每跳参数）。

涉及论文标题：
- R2D2 Robotized Reconfigurable Network for Disaggregated Datacenters
