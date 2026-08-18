## Emissary（Enhanced Miss Awareness Replacement Policy）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Emissary（Nagendra et al., ISCA'23，Google/Princeton）是首个面向指令缓存的 cost-aware L2 替换策略：识别"miss 会导致 decode starvation（译码饥饿，取指流水线停顿）"的指令行，并抵制淘汰它们（pin 在 L2C），以保住未来命中的可能。Bumper 的评估（最优配置：最多 25% ways 可 pin、pin 概率 1/16，经大量参数搜索）：Emissary 在移动应用上反而损害性能——原因是被它标为 high-priority 并 pin 的行中有 56.7% 是 useless（Definition 1），pin 大量 useless 行增加 L2C 争用；即 Emissary 只关心"miss 代价"、无法区分 useful/useless，且其目标（保留）与 Bumper（加速淘汰 useless）方向相反。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
流程：L2C 指令 miss 发生且引发 decode starvation → Emissary 把对应行标记为高优先级并 pin → 该行可绕过常规替换被保留；移动应用中约 30% L2C 是代码行、其中 >50% useless，Emissary 将多数代码行判定为 high-priority（它们都曾造成饥饿）→ 错误 pin 的 useless 行挤占数据与 useful 行。Bumper 论文 Fig.6b 对比：Emissary 的 high-priority 指标与 Bumper 的 usefulness 指标在所有插入过 L2C 的代码行上分类差异巨大。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：监控取指 miss 事件与饥饿信号，维护 per-line 成本/优先级元数据并约束 pin 行数量（ways 比例与概率阈值）；原论文在带 FDIP 的现代核上获 3.24% geomean 加速（ISCA'23）。后续工作 ICARUS 基于 Emissary 用分支历史与复用信息改进关键代码行检测。Bumper 与之可组合，但论文实测组合无额外收益（Bumper 已移除其 pin 掉的 useless 行）。

涉及论文标题：
- Bumper: Hinting Instruction Usefulness for Robust Unified Caches
