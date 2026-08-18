## 归纳证明、中间状态机（ISM）与精化（Refinement）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
这是 täkōFormal 端到端机器检查 soundness 证明的三块基石。(1) 归纳证明（induction）：对状态机证明性质 P 只需证 Init(s)⟹P(s) 与 P(s)∧Next(s,s')⟹P(s')，但直接证常失败（非可达状态产生虚假反例），故构造归纳不变量 Inv(s)（Inv⟹P 且被 Next 保持）。täkōFormal 的不变量含 119 条附加子句、61K LoC 证明注解。(2) 中间状态机（ISM）：公理是"完整执行图"的性质，而 operational model 逐迁移构建图——ISM 在两者之间：其状态是部分执行图，迁移模拟运行程序如何更新图，公理被强化后对 Init 成立且被 Next 保持（部分执行也满足公理）。(3) 精化（refinement）：证明 operational model 的每条 trace 经抽象函数 abs 映射后是合法 ISM trace——operational 迁移要么映射为合法 ISM 迁移（更新执行图，如结束 OnWB s1→s2、开始 OnMiss s4→s5），要么是内部步骤映射为 NoOp（如 L3 发 OnMiss 请求 s2→s3、Engine 收请求 s3→s4，因 ISA 级 MCM 不建模这些硬件细节）。prior work（[5]）机器检查了"ISM 生成 axiomatically consistent 结果"部分，但"operational↔ISM 对应"部分靠手写证明——täkōFormal 首次把两部分都机器检查（end-to-end）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程例子（图 18）：operational model 执行 s1→s5 四条迁移（结束 OnWB→L3 发请求→Engine 收请求→开始 OnMiss），只有 s1→s2 与 s4→s5 更新执行图；abs 把 s2/s3/s4 映射到同一 ISM 状态（NoOp），s5 映射到新 ISM 状态。归纳不变量保证中间状态（部分执行图）也满足每个公理，避免"公理在完整执行成立、在部分执行不成立"导致无法归纳（图 17 的 Axiom1 失败例子→改造成前缀封闭的 Axiom2/MeInt）。精化证明假设一致性协议正确（文献 [46,48]），聚焦回调/phantom 语义；对 coherence 部分仍显式建模所有 transient 状态验证 dirty bit 保留等 coherence-consistency 接口性质。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：全部在 Dafny 中落地——定义 ISM 状态与迁移、归纳不变量、abs 抽象函数，Dafny/Z3 机器检查每步。应用场景：任何"公理模型 + 实现模型"的 soundness 验证都可套用（prior: [5,25,47,64] 用 ISM 思路但手工连 R 部分）。使用要点：公理必须前缀封闭，归纳不变量通常需大量手工设计（61K LoC 注解是主要工作量），模型需参数化（cache 容量、core 数、地址→L3 bank 映射、prefetching/替换/NoC 细节）以保证证明对任意配置成立。

涉及论文标题：
- täkōFormal: Enabling Robust Software for Programmable Memory Hierarchies
