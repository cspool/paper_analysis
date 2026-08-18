## Last-ready Predictor（LRP，最后就绪预测器）

术语解释
LRP 预测一条指令两个未就绪源操作数中哪一个最后就绪的硬件预测器（结构类似 gshare 分支预测器），HWL 用它选择派发的目标 segment——last-ready 源所在的 producer segment，因为该源决定指令最终就绪时刻并主导唤醒延迟。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：(1) 指令有两个未就绪源时，其就绪时间由最后就绪的源决定，把该源的 producer segment 作为目标 segment 才能最大化 L1 单周期唤醒的价值；(2) 结构——pattern history table（PHT）按 PC 与全局分支历史的 hash 值索引，每项一个 2-bit 饱和计数器；第一个源最后就绪则计数器减、第二个则加；预测 = 计数器高位（0→第一源 last-ready，1→第二源）；(3) 训练发生在执行/提交时按实际 last-ready 顺序更新；(4) 准确率——仅双未就绪指令场景平均 89%；扩展到"至少一个未就绪源"的指令平均 97%（17/19 基准 >95%），误预测对 IPC 影响有限。源自 Ernst & Austin 的 tag elimination 前端预测（[22]，论文引用；该类方案原用于 CAM 唤醒把 last-ready 操作数 tag 放左侧以发挥顺序唤醒优势，本文改用于 segment 选择）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程（论文 IV-A）：rename 时对该指令的两个未就绪源各读 RMT 得 producer segment 号；LRP 预测 last-ready 源 → 其 segment 号作为目标 segment（用于 HSD 判定 chunk 归属）；另一个源若其 producer 也在该 segment 则写 L1，否则写 L2。默认配置：8K-entry、4-branch history PHT（Table II），存储 2.0KB。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现与 gshare 分支预测器相同：PC+全局历史 hash 索引 PHT、2-bit 饱和计数器、预测时读高位。使用场景：任何依赖"哪个操作数最后就绪"的调度决策——本论文的 HSD 目标 segment 选择、以及 CAM 唤醒中 last-ready tag 的左侧放置（顺序唤醒）。论文未开源。

涉及论文标题：
- Hierarchical Wakeup Logic of the Issue Queue for High Scalability
