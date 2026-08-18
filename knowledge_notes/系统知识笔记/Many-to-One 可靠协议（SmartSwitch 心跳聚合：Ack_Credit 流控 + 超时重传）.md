## Many-to-One 可靠协议（SmartSwitch 心跳聚合：Ack/Credit 流控 + 超时重传）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 可靠传输 = 流控（发送速率匹配接收能力）+ 可靠性（丢包/出错检测与恢复）两机制的组合（Web 证据：IB RC 的 PSN/ACK、RPC-RDMA credit 会计）。传统实现为每包确认（per-packet ACK）：PS 与 64 worker 各建 1:1 连接时，确认包淹没 PS 网络 IOPS——DisDP 模拟显示 32 worker 吞吐降至 30Gbps、64 worker 仅 18Gbps（100Gbps 网络）。
- DisDP 方案：周期心跳（heartbeat）代替逐包确认——worker/PS 周期发心跳包，携带 Ack（下一期望序号）与 Credit（可接收的最大序号）两个连接状态；SmartSwitch 把 PS 的心跳广播给所有 worker，并把各 worker 的心跳在 heartbeat table 中 min 聚合成 1 份给 PS。PS 每心跳周期只收发 1 个确认包，与 worker 数无关。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程：worker 发梯度包（带递增序号）→ SmartSwitch 网内聚合 → PS 收聚合梯度；PS 心跳广播 Credit（宣布还能收多少）→ 各 worker 更新 TX Credit 后恢复发送；worker 心跳上行经交换机 min 聚合（取最小 Ack/Credit 为全局进度）→ PS 据此推进 Ack。
- 流控：worker/PS 发出序号等于 TX Credit 的包后停发，等心跳更新 Credit 再恢复。可靠性：发送方检测 TX Ack 超过用户定义时间（如 1s）未增长且仍有未确认包 → 从 Ack 序号重发；交换机把重传包对应的聚合梯度转发给 PS。
- 为什么不在交换机做可靠端点：Tofino ≤20 级流水，而可靠 RDMA/TCP 包处理需 >50 级硬件阶段；交换机只做无状态聚合 + 心跳表，端点逻辑留在 SmartNIC/PS。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现载体：worker/PS 侧 SmartNIC 硬件实现心跳/重传状态机；SmartSwitch 用 TCAM/SRAM 维护 heartbeat table（min 聚合）。使用场景：many-to-one 汇聚型通信（梯度聚合）与 one-to-many 广播（参数分发）；以吞吐为目标、可容忍心跳周期级延迟的训练场景。信息缺口：论文未给出心跳周期与重传定时器的具体取值及参数敏感性。

涉及论文标题：
- DisDP: Disaggregating Compute, Network, and Storage for Model-Sharded Data-Parallel Training
