## Branch Target Buffer（BTB，分支目标缓冲，多级 BTB）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
BTB 缓存"最近执行过的分支 PC → 目标地址"映射，在取指最早期（分支尚未译码时）就预测 taken 分支的目标，实现无气泡的取指重定向，是 BPU 的关键部件。Bumper 基线为两级 BTB：1K-entry L1-BTB + 16K-entry L2-BTB（只存 committed taken branch，配合 post-fetch correction），另配 RAS。BTB 容量 miss 是移动应用前端重定向的主因：总 BPU MPKI 8.0 中主要来自 BTB miss。敏感性研究：L2-BTB 从 16K 扩到 64K entries 使 BPU MPKI 显著下降、baseline 性能明显提升（16K→18K 的 12.5% 扩容只带来 1.1% IPC 增益，故直接扩容不划算），但 Bumper 在所有 BTB 尺寸下持续有效（64K 时分支工作集仍超过 BTB+TAGE 容量，错误路径污染仍存在，只是差距收窄）。BTB prefill 技术（在 L1I fill 时预填 BTB）收益有限——代码行进入 L1I 的时机太晚，错过行内分支的时间局部性。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
流程：取指地址并行查 L1-BTB → 命中则下一周期从目标地址取指（0 气泡）；未命中继续顺序取指，直到译码发现分支再修正（post-fetch correction）→ 触发 FTQ 重定向。BTB miss 引发的重定向使 >50% 进入 cache 层次的行来自错误路径，是 Bumper 要解决的根本问题（Bumper 不修 BTB，而是在 L2C 侧清理其后果）。Bumper 的 hint 链不经过 BTB，与 BTB 正交。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：组相联 SRAM 表（键 PC 低位索引 + tag，值目标地址 + 分支类型元数据）；多级结构用小而快 L1 + 大而慢 L2 平衡延迟与容量，仅 committed taken branch 入表以降低污染。相关优化方向（Bumper 引文 [23]–[28]）：替代组织与多级设计扩展有效容量、prefill 机制、替换/索引策略减少别名。

涉及论文标题：
- Bumper: Hinting Instruction Usefulness for Robust Unified Caches
