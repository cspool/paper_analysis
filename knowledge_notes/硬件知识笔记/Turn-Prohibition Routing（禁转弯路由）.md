## Turn-Prohibition Routing（禁转弯路由）

术语解释
ConBin 在修复后不规则拓扑上，用"带禁转弯约束的最短路径"路由（Starobinski et al. [50]），保证无环且全局连通，作为所有通信路径 R(v_i,v_j) 的计算基础。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Turn-prohibition（TP）算法（Starobinski, Karpovsky, Zakrevski，IEEE/ACM ToN 2003）：通过禁止节点上特定的"输入-输出链接对"（转弯）而不是整条链接来打破网络所有环，使网络成为 feed-forward（前馈）路由网络；关键保证：任意拓扑下被禁转弯 ≤ 总转弯数的 1/3，且保持全局连通；算法复杂度关于节点数多项式。与 up/down 路由（Autonet）相比：TP 禁转弯数有上界、吞吐高 10–20%。Web 证据：原文及应用页（ipsit.bu.edu）确认算法属性与性能。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
ConBin 中 TP 用于修复后不规则 mesh：故障剔除 + 冗余链旁路会破坏 mesh 规则性，维序路由可能成环/死锁；TP 保证"无环 + 全局连通"，使任何修复拓扑都可用最短路径路由。通信路径 R(v_i,v_j) 在该路由下确定后，直接进入映射代价函数 LCC_exp（沿路径累加 contention 权重，Eqn.6）与 BookSim2 仿真。运转例子：聚簇故障区被长程 R-R 链旁路后，TP 预计算禁转弯集合，BookSim2 按禁转弯最短路径转发 flit，避免非规则拓扑下成环。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
离线计算禁转弯集合（多项式图算法），路由表/路由器按转弯规则转发；NoC 容错设计中常用 TP 替代 up/down 以保留更多可用转弯与更高利用率。使用：作为"不规则拓扑上的无死锁基线路由"，是 WSC 修复后互连可用的前置条件；ConBin 未报告其计算开销（论文未明确说明）。

涉及论文标题：
- ConBin: A Performance-Convergence Framework for Wafer-Scale Chip Binning
