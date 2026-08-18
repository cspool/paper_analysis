## Bulk Synchronous Flow（BS，批量同步流卸载）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bulk Synchronous Flow 是内存中心视角（memory-centric）的 CCM 卸载机制，由 M²NDP 提出：把 CCM 当作内存，主机对映射到特定地址范围（uncacheable 内存映射函数区）发单条 CXL.mem store 携带 kernel 信息即可启动远端 kernel（CXL 内存控制器上的自定义 packet filter 区分普通访存与 kernel 启动），同步的 store 响应即 kernel 完成信号，主机用 memory barrier 阻塞后续访存直至结果可取。对比 RP：kernel 启动从多次 CXL.io 往返 + 轮询降为单条 store，轻量 kernel 的周期数只有 RP 的 16.7%，支持细粒度卸载。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
PageRank 迭代的 BS 时序（AXLE Fig.1b/Fig.6）：主机发 CXL.mem store 启动 kernel → 收到 launch ACK → 立即发 CXL.mem load 取结果，load 被硬件 barrier 挂起直至 kernel 完成 → 结果回传后主机才继续执行 → 下一轮迭代。缺陷：主机空闲 = T_CCM + T_data（PageRank 案例约 98%），CCM 空闲 = T_data + T_host（约 50%）；跨迭代依赖（新 frontier 依赖上轮结果）使主机并发无法填补空闲，端到端 pipeline 完全串行化——AXLE 称之为"两个空闲时间"问题。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：M²NDP 的 M²func 机制——主机单条 store 到内存映射函数区 + packet filter 硬件识别 + 同步响应语义；M²NDP 开源模拟器原生支持（M2NDP-public，Ramulator + BookSim2）。使用方式：细粒度/粗粒度通用卸载（OLAP 布尔标记、图分析、DLRM 等），是 AXLE 的 SOTA 基线。局限：同步语义锁死主机（host core stall 达 97.83%，AXLE Fig.13），需异步机制（AXLE）弥补。

涉及论文标题：
- AXLE: Coordinated Offloading with Asynchronous Back-Streaming in Computational Memory Systems
