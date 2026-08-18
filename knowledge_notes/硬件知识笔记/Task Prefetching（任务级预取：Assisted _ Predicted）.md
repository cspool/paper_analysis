## Task Prefetching（任务级预取：Assisted / Predicted）

术语解释
ATX 提出的两类任务级预取：不等 CPU 正式发出任务，UTE 就先以 prefetch mode 派发任务、把未来任务的数据提前取进 L2。Assisted 预取针对"NCA 不够用、任务积压在 InTaskQ"，Predicted 预取针对"核产生任务太慢、InTaskQ 常空"。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
动机：提高 MLP 的自然做法是增加 UTE 后端并发处理的任务数，但受两个障碍限制——(1) NCA scratchpad 不够存所有任务取回的数据；(2) 核来不及产生新任务。两个机制各打一个障碍。prefetch mode 的共性：不分配 NCA，只把 leaf 流（依赖树的叶子，即最终数据流）的数据预取到 L2；非 leaf 流数据取进 UTE 中子流的 Stream Units 用于生成地址，但不送到 PAcc Port。之后任务以非 prefetch mode 重发时，数据大概率已在 L2。Predicted 的核心观察："预测任务"比"预测访存地址"容易——同类型任务的流依赖与 bexp 配置相同，只有运行时常量不同，因此任务预测退化为"预测下一组运行时常量"。论文用简单 stride 算法：观察同类型相邻两任务的常量提取步长，当前任务常量加 N×步长得到第 N 个未来任务（N = Prefetch Distance，按平均任务输入大小的运行时启发式设定：>32KB 用 1、16KB 用 2……）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- Assisted：任务已进 InTaskQ 但无空闲 PAcc/scratchpad → 允许其以 prefetch mode 派发：硬件只走流取数路径（非 leaf 流数据供子流算地址、leaf 流数据进 L2），不写任何 NCA 缓冲 → NCA 释放后任务以非 prefetch mode 重发，数据命中 L2。论文显示只有 `ser` 矩阵受益——因为大多数情况下核产生任务不够快，InTaskQ 鲜有积压。
- Predicted：InTaskQ 常空时，Task Predictor/Prefetcher（UTE 前端）由最近两任务的常量提取步长、预测第 N 个未来任务的常量，合成预测任务以 prefetch mode 派发。论文显示 Predicted 是最有效机制（SpMM/SDDMM 主要收益来源；GeMM 因访存规整、预取帮助小）。消融：Predicted 距离启发式不总是最优（图 18），Inf UTE + 静态最优距离可把除 LIV 外所有矩阵顶到 roofline；LIV 的缺口正是 stride 预测本身对不规则任务失准。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要素：任务预测器（stride 提取 + 增量器）、prefetch mode 的派发路径（绕过 PAcc 分配、改写数据目的地为 L2）、预取距离寄存器/启发式。设计考量：距离过深浪费带宽且污染缓存、过浅盖不住延迟；论文将更复杂的运行时距离调节（借鉴硬件预取研究）留作未来工作。适用场景：CPU 侧任务产出率与 NCA 消费率不匹配的流水线型 kernel；Assisted 只在核是瓶颈、Predicted 只在核不是瓶颈时才各自有效。

涉及论文标题：
- ATX: Accelerator Task Extensions
