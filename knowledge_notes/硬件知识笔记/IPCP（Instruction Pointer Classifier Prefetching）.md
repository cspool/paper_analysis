## IPCP（Instruction Pointer Classifier Prefetching）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- IPCP（Bouquet of Instruction Pointers，Pakalapati & Panda，ISCA 2020）是基于指令指针（PC/IP）分类器的硬件空间预取器：用单一硬件表把访存流按 PC 分类为四类——Constant Stride（CS，恒定步长）、Complex Stride（CPLX，复杂步长）、Global Stream（GS，全局流）、Next Line（NL，下一行），类优先级 GS>CS>CPLX>NL，按类选择对应预取策略。论文称其为 DPC3 冠军（注：联网证据显示 DPC3 第 1 名实为 OPCP，IPCP 为高影响力参赛设计/后续常作 SOTA baseline；此处以论文表述为主，存疑处标注）。Moirai 把它与 Berti 一起作为"记忆式预取器"主 baseline：IPCP 存储 16.7KB。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- IPCP 运转流程（作为 Moirai 对比）：LSU 产生地址+PC → 按 PC 查分类表判定类（CS/CPLX/GS/NL）→ 依类预测（CS：恒步长加偏移；GS：跟踪全局流历史；NL：下一缓存行）→ 生成预取注入 L1D。Moirai 评估中：IPCP 单核 12.12% 平均 speedup（vs Moirai 11.48%）、多核与 Moirai 相当；在 0.8KB 存储约束下（Table 与 Moirai 对齐，按比例缩小表项）IPCP 性能大幅下降——容量 miss 与条目 aliasing 暴露记忆式范式的脆弱性。
- 对比：IPCP 靠精确模式匹配，对复杂/噪声/指针追逐模式（omnetpp、GAP）链断裂；Moirai 的 TCN 泛化在这些场景反超（bfs 70.65%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：单张硬件表 + 分类器逻辑（每条目存 PC、类、步长/历史），ChampSim 预取器模块实现；DPC3/DPC4 框架下公开。使用：作为通用 L1D 预取 baseline 评估新设计（Moirai 用它测"泛化 vs 记忆化"）；存储 16.7KB 说明其在 L1D 预算内仍有空间换性能的空间，但复杂模式覆盖有限。局限：需 PC 表（KB 级）、模式必须精确见过。

涉及论文标题：
- From Memorization to Generalization: A Practical Neural Network Prefetching Framework
