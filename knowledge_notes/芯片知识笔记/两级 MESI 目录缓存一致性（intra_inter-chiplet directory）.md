## 两级 MESI 目录缓存一致性（intra/inter-chiplet directory）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PhaseWeave 服务器为跨 chiplet 的缓存一致性采用标准两级 MESI 协议（类似多路 socket 服务器），其目录组织成两级：每条缓存行有一个全局 home（global home core + LLC slice，用地址哈希决定）并在每个 chiplet 内还有一个 local home；每个 chiplet 维护 intra-chiplet directory（记录本 chiplet 内哪些 core 缓存了该行）与 inter-chiplet directory（记录哪些远端 chiplet 缓存了该行）。这是 MESI（Modified-Exclusive-Shared-Invalid）写失效协议在"物理异构、多 chiplet"组织下的目录分层实现：全局 home 收失效请求，先在 intra 目录内向本 chiplet 内持有该行的 core 发 invalidation，同时按 inter 目录把失效转发给远端 chiplet 的 local home，local home 再向其芯片内 core 广播，保证全服务器所有副本失效。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 chiplet 物理组织中，目录是"本地 vs 远端"一致性事务的天然边界（参见 vault 中 Multi-Chiplet GPU / CDG 笔记：跨 chiplet NoC 上 coherence 事务走 D2D 链路）。PhaseWeave 的具体失效流程：某 chiplet 内 core 写一行 → 请求发往该行的 global home（可能在另一 chiplet）→ global home 查 intra-chiplet directory 向本地持有者发失效（低延迟、不跨片）→ 同时查 inter-chiplet directory 把失效转发到所有远端 local home → 远端 local home 在各自 chiplet 内向本地 core 广播失效 → 全部确认后写者获得所有权。两级目录把"全服务器广播"变成"全局 home 一次广播 + 每 chiplet 本地广播"，用 inter-chiplet directory 的存储换取跨片失效消息数（O(远端chiplet数) 而非 O(核数)），与 Dorado 的"簇是本地 vs 远端一致性事务边界"思路同构。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：每个 chiplet 维护两级目录结构（intra 按 core 位图、inter 按 chiplet 位图），global home 由地址哈希分布在各 chiplet 的 LLC slice 上；SST 全系统模拟中建模该两级目录（论文对 SST Ariel 的修改之一），与 DRAMSim3 的异构内存分区配合。用途：在核可跨 chiplet 迁移（iso-ISA）且热页随主导 chiplet 迁移的场景下维持一致性；相比每 chiplet 独立一致性域（如 AMD EPYC 不维护跨 CCD 一致性），两级 MESI 提供全服务器统一的缓存一致视图，是"单一 OS + 统一物理地址空间"chiplet 服务器的基础设施。与 gem5_Garnet 中 MESI Two Level coherence protocol 在 NoC 上跑 coherence transaction 的机制同源。

涉及论文标题：
- PhaseWeave Phase-Aware Execution on Heterogeneous Chiplet Architectures for Datacenters
