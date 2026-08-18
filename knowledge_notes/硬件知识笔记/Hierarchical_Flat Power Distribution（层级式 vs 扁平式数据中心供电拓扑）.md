## Hierarchical/Flat Power Distribution（层级式 vs 扁平式数据中心供电拓扑）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
数据中心供电拓扑决定电如何从电网分配到服务器。层级式（hierarchical）：ATS（自动转换开关）→ 多台冗余 UPS → 各 UPS 供若干 PDU（配电单元）→ PDU 供服务器行/机架，每级有功率上限与故障域；域预算 X 每机功耗 Y 时只能部署 ⌊X/Y⌋ 台，剩余 X−Y·⌊X/Y⌋ 为 stranded power（功率碎片，已付钱但用不上的容量）。扁平式（flat）：更大范围（如整数据中心）共享功率池，减少碎片但故障隔离差、维护复杂。本论文（Rearchitecting the Datacenter Lifecycle for AI）指出 AI 加速器功率密度暴涨（DGX H100 8×H100 需 10.2kW vs 64 核 Emerald Rapids 385W）使碎片问题恶化，并用 TCO 框架比较两种拓扑（表 V：per-PDU/per-UPS/per-DC 三档的 stranding/复杂度/维护/故障隔离权衡），结论是行业标准的层级式并非 AI 部署的 TCO 最优，per-DC 扁平供电（虽不总是可行）降 TCO 4.2%。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
供电拓扑对硬件部署的影响流程：
```
层级式: 电网 → ATS → UPS1(冗余) → PDU1 → 机架A(上限 X_A)
                             → PDU2 → 机架B(上限 X_B)
   每域按 ⌊X/Y⌋ 部署 GPU server → 域内余量成为 stranded power
扁平式: 电网 → 大功率池(整 DC) → 所有机架共享 → ⌊ΣX/ΣY⌋ 级全局最优
   → 碎片减少，但任一处故障影响面更大，需要更多冗余与更复杂维护
框架评估: flat 相对 hierarchical 生命周期 TCO 降 4.2%（图 6a）
```
论文还把供电拓扑与刷新联动：投更大 powersharing 域虽增 build 成本，但为 IT provisioning 的代际混合部署提供灵活性（不同功耗 GPU 可混放同域），并允许 operation 层 oversubscription/derating（安全降载多部署），这是跨 stage 协同的典型案例。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：层级式是行业标准（多数云厂商沿用 [120]）；扁平式/大域设计（如 Meta/Google 的整楼供电）需更高压母线、更多冗余与精细的功率监控（快速负载波动产生瞬态电流需电气设计与监测）。本论文在 AI Lifecycle Compass（https://github.com/Azure/AI-Lifecycle-Compass）中以 YAML 参数化电源拓扑（供电成本 $7.0/W、每域容量/冗余假设），TCO 模块比较不同拓扑下的碎片损失与 CapEx/OpEx 差异。PowerGrad/Power Sloshing 等论文则关注域内功率在组件间的运行时分配（software 层），与本文的 build 层拓扑选择互补。

涉及论文标题：
- Rearchitecting the Datacenter Lifecycle for AI
