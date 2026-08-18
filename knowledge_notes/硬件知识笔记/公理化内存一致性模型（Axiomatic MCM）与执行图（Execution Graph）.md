## 公理化内存一致性模型（Axiomatic MCM）与执行图（Execution Graph）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
公理化内存一致性模型把程序的一次执行建模为有向图（执行图）：节点是内存事件（指令的读写等），带标签的边是事件间关系，合法执行由一组公理（axioms）约束这些关系得到。täkōFormal 是典型例子：图 6 定义全部公理，事件包括常规 W/R/RMW 与 phantom 的 Wcb/Rcb/RMWcb、回调起止 Ms/Me/Es/Ee（E 带 dirty bit 区分 OnEvict/OnWB）、Fl；关系包括 sb（sequenced-before，程序序）、rf（reads-from）、fr（from-reads）、mo（modification order，同地址写全序）、sw（synchronizes-with）、hb（happens-before=(I×¬I ∪ sb ∪ sw ∪ vf ∪ eb ∪ cbo 扩展)^+）、cbo（同地址回调全序）、viscb、vf/ef/eb 等；race 也定义为关系表达式。执行满足所有公理才被 MCM 允许。SC 的公理化是 acyclic(rf∪fr∪sb∪mo)。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程例子（禁止 r1=2,r2=0，图 4/7）：程序员把 täkō 程序的可能执行画成执行图——Wcb([x],1) 与 Rcb([x]) 必须被 VfWf 约束（Rcb 前必须有 M_e）；若 Rcb 读到 OnMiss 生成值 2，则图中必须有第二个 OnMiss，OEInt 强制两次 Me 间存在 eviction（Es/Ee），EvDirty/WbDirty 强制 dirty 行必须 OnWB，而 OnWB 写 [y] 的 Me→Es 边经 cbo 进 hb，使 (i5) hb→(i4) 读 [y]，最终 Vis 公理禁止 r1=2,r2=0。执行图即"程序员与模型检查器共用的推理语言"：把图交给公理判定，allowed/forbidden/racy 立刻可得，无需理解微架构。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：公理模型用关系代数表达（täkōFormal 图 6：empty([Ms];cbo;cbo;[Me]∩thd) 表示 Ms 与 Me 间无同线程干扰等），可编码进 Alloy（täkōFormal 把全部公理 + litmus tests 编码为 .als 模型，run_alloy_tests.sh 逐个确认结果）；业界工具族包括 herd7（CAT 模型）、Dartagnan（SMT 编码）、rmem（操作模型）。前缀封闭（prefix-closure）是公理设计的关键性质：若公理对完整执行成立则对构造过程中所有部分执行成立，从而可用归纳法在状态机上证明实现满足公理。

涉及论文标题：
- täkōFormal: Enabling Robust Software for Programmable Memory Hierarchies
