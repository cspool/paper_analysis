## Regionally Reconfigurable High-Bandwidth Domain

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Regionally Reconfigurable High-Bandwidth Domain（区域可重构高带宽域）是 MixNet 的核心架构设计概念。利用 MoE 训练中 EP 通信的**强局部性**（只有同一 MoE block 内的 expert 层需要 all-to-all，不同 PP stage 的 expert 层不直接通信），将全局 GPU 集群划分为多个较小的区域（region），每个 region 内使用毫秒级可重构 OCS 为 EP 的 sparse all-to-all 提供高带宽直连电路。Region 的大小由 EP group size 决定（通常 ≤ 64-128 GPU），正好落在 commodity OCS 的端口容量范围内（≤ 576 ports）。多个 region 通过去中心化 topology controller 独立管理各自的 OCS 拓扑，避免全局重配置的 scalability 瓶颈。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
区域可重构高带宽域在 MixNet 中的运转（以 4096 GPU cluster, Mixtral 8×22B, EP=8, PP=8 为例）：
- **Region 划分**：每个 PP stage 的 8 个 GPU（EP group）构成一个 region。512 servers × 8 GPU = 4096 GPU，按 PP=8 分 8 个 PP stage，每个 stage 有 512/8 = 64 servers → 64 × 8 = 512 GPU per PP stage → 512/8 = 64 EP groups per PP stage → 64 regions。
- **每 region OCS**：每个 region（64 servers, 512 GPUs）配一个 Polatis OCS（576×576 ports），可覆盖所有 server 的 OCS NIC。实际 OCS 仅需连接同一 EP group 内的 servers（8 servers per EP group，每 server 6 OCS NICs = 48 ports per EP group）。
- **独立重配置**：各 region 的 topology controller 独立收集本 region 的 traffic demand → greedy algorithm 生成本 region 的 OCS 拓扑 → 独立执行 OCS 重配置。不同 region 的重配置时间可重叠（并行执行），不互相阻塞。
- **跨 region 通信**：跨 PP stage 的 PP traffic 和全局 DP all-reduce 走 EPS Fat-tree（与 OCS 物理隔离），不受 OCS 重配置影响。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Region 划分策略：(a) 按 EP group 划分（自然方案，因 EP 通信局限在同一 EP group 内）；(b) 按 rack/row 物理 proximity 划分（减少光纤长度和信号衰减）；(c) 混合划分（同一 region 内的 servers 物理靠近但跨 EP group 时通过 EPS relay）。
- 去中心化控制面：每 region 有独立的 topology controller（运行在 region 内某 server 上），收集 traffic demand 通过 all-gather（数 μs），计算 topology 通过 greedy algorithm（μs 级），下发 OCS 重配置通过 TL1 commands（数十 ms）。无需全局 controller 或全局同步。
- 硬件部署：每 server 分配 2-6 个 NIC 给 OCS（取决于 EP group size 和 OCS port 成本）。NIC 使用 RoCEv2 协议，通过 QSFP28 optical transceiver + duplex LC fiber 连接到 OCS 前面板端口。
- 工程考量：(a) OCS port count 限制 region 大小（Polatis 576 端口 → 约 72 servers × 8 NICs）；(b) 光纤布线复杂度（每个 OCS port 一条 fiber → 大规模 cluster 需要 structured fiber management）；(c) OCS 重配置期间光路中断需要 burst-mode transceiver 快速恢复（当前 commodity transceiver 的 NIC activation time ~5.67s，论文排除此时间）。

涉及论文标题：
- MixNet: A Runtime Reconfigurable Optical-Electrical Fabric for Distributed Mixture-of-Experts Training
