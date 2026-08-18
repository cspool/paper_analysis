## Matrix Scheduler（矩阵调度器）与 Matrix Scheduler Reloaded（MS-rel）

术语解释
Matrix scheduler 指用 SRAM-like 依赖矩阵实现唤醒逻辑的 IQ 调度器（较 CAM 型可扩展）；MS-rel（matrix scheduler reloaded）通过减少/动态分配矩阵列数进一步提升可扩展性；两者都是 HWL 的直接对比对象。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：(1) 矩阵调度器（Goshima 等 [3]）——唤醒逻辑 SRAM 化（即 wakeup matrix），比 CAM 可扩展，但矩阵是 IQ 尺寸的平方阵，现代大 IQ 下优势被侵蚀；(2) MS-rel（Sassone 等 [9]）——观察到大量 IQ 内指令无目的寄存器（如分支/存储）或在 IQ 中无消费者，逻辑上不需要矩阵列，故减少列数并动态分配给需要列的指令；(3) MS-rel 局限——只减列不减行、且"无消费者"指令比例不高（本论文测 CAM 广播分类：broadcast heard 平均 59%，即多数广播确实被消费，列压缩空间有限），列需求实测 110/200（55%）；(4) 本论文对比——相近 IPC（MS-rel 平均退化 1.0% vs HWL 0.9%）下 MS-rel 周期仅降 18% vs HWL 53%（Fig.18），因为只能压宽度不能压高度。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
论文评估流程（V-G2）：扫 MS-rel 的矩阵列数，找达到 HWL 相近 IPC 的最小列数（110 列）→ 用 HSPICE 测该配置 wakeup+select 延迟 → 相对基线 200-entry 得 18% 周期缩减。CAM 广播分类定义：broadcast heard（有消费者在 IQ）、broadcast wasted（广播但无消费者）、no broadcast（无目的寄存器）——前两类决定矩阵列是否可省。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SRAM 型依赖矩阵 + 动态列分配表（指令需要列时获取、发射后回收）。使用：作为 wakeup matrix 可扩展性的代表方案与 HWL 对比；其"行数不减"与"列压缩率有限"两点为 HWL 的 L1（连行一起变小）提供了差异化论据。本论文的 HWL 基线/对比实现未开源。

涉及论文标题：
- Hierarchical Wakeup Logic of the Issue Queue for High Scalability
