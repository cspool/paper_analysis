## Channel Dependency Graph (CDG) for Multi-Chiplet NoC / 多芯粒NoC的通道依赖图

术语是什么？

Channel Dependency Graph (CDG) 是 Network-on-Chip 死锁分析的核心理论工具，最初由 Dally 和 Seitz 提出。CDG 以 NoC 中的物理或虚拟通道（channel/VC）为节点，以路由函数定义的连续通道间依赖关系为有向边，构成有向图。NoC 死锁的经典充要条件是：路由算法是 deadlock-free 当且仅当 CDG 无环。在 multi-chiplet NoC 场景中，deadlock 环可以跨越 chiplet 内部 NoC 和 interposer NoC 的通道边界——单个 chiplet 的 CDG 无环不保证跨 chiplet 组合后的 CDG 无环。

从芯片设计角度拆解术语：

本文扩展了 CDG 到 multi-chiplet 场景：chiplet 内部 NoC 通道和 interposer NoC 通道共同构成 CDG 的节点集合，依赖关系包括 (1) chiplet 内部相邻路由器的通道间依赖；(2) boundary router 与 vertical channel (TSV) 间的依赖；(3) interposer 上相邻路由器间的通道依赖。跨 chiplet 死锁环的典型形成路径：In-Req 从 interposer vertical channel → chiplet boundary router → chiplet 内部 NoC → cache controller → 触发 Out-Rsp → 返回 boundary router → 但 interposer vertical channel 被其他包占用 → backpressure → chiplet 内部 NoC 通道堵塞 → 间接阻塞后续 In-Req → 形成跨 chiplet-interposer-chiplet 的循环等待。传统的 deadlock avoidance 通过 turn restriction（MTR）剪断 CDG 中可能导致环的边，或通过 VC isolation（DeFT）将 CDG 划分为互不重叠的子图，或通过 injection control（RC/DFBM）从源头阻止会形成环依赖的包进入网络。

术语一般如何实现？如何使用？

CDG 是理论分析工具而非运行时机制。在实践中，NoC 设计者通过以下方式保证 CDG 无环：(1) Turn Model（如 West-First、North-Last、Negative-First）禁止特定转向来剪断可能的环边；(2) 增加 virtual channel 将物理环分离为多个逻辑层（如 dateline routing 用于 torus 拓扑）；(3) escape VC 提供无环的逃生路径；(4) DFBM 的 approach 本质是通过 credit-aware admission control 保证 chiplet-to-interposer 出口方向在最坏情况下仍有足够吸收能力，从而防止 backpressure 形成跨域 CDG 环——不依赖修改 CDG 拓扑或增加 VC，而依赖保证依赖边不会被不可释放地占用（即从根源上消解环形成的条件）。本文通过 coherence transaction flow 与 NoC packet routing dependency 的映射，将 CDG 环的潜在形成路径映射到一致性事务的 expected response 上界，在 admission 点截断。

涉及论文标题：
- Deadlock-Free Bridge Module for Inter-Chiplet Communication in Open Chiplet Ecosystem
