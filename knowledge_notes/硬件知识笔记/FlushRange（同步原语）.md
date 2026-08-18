## FlushRange（同步原语）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FlushRange 是 täkō 提供的同步原语：执行 FlushRange(R) 会把地址范围 R 内所有 cache line 从缓存中逐出，逐出时按 dirty 与否调用 OnEvict 或 OnWB 回调，并阻塞 FlushRange 直到这些回调全部完成。它把"OnEvict/OnWB 可任意时刻异步执行"的不确定性收束为同步点，是消除 phantom 程序 race 的关键工具——对应 MCM 中的 Fl 事件与 eb（evicts-before）关系。论文未说明 FlushRange 的具体实现机制（cache 逐出引擎/指令），细节以 täkōFormal 论文为准。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程例子（wbr vs wbf litmus test，图 11）：Core 0 写 phantom [x]（(i1)），OnWB (i5) 发布 [y]←1，Core 0 随后读 [y]（(i3)）——无 FlushRange 时 (i5) 与 (i3) 无 hb 序，构成 race（wbr racy）；加 FlushRange[x]（(i2)）后，Fl 必须在该地址 OnMiss 之前或 OnWB 之后提交（EbWf 公理），若在 OnWB 之后提交则 eb 边使 (i5) hb→(i3)，race 消除且 r1=0 被 MCM 禁止（wbf）。MCM 语义：Fl 前必须有 E_e（EbWf），Fl 与回调/读写的顺序经 eb⊆hb 参与 Vis 公理判定。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：程序员在需要同步时对 phantom（或普通）地址范围执行 FlushRange，确保该范围内的回调副作用（如发布到普通内存的结果）在其返回前完成；多核场景（phiR/phiNR，图 12）用 RMW+FlushRange 保证"最后一个写者的 OnWB 已发布"。设计要点（täkōFormal 发现）：FlushRange 要能消除 race，回调不得在 FlushRange 提交后再运行会造成 race 的访问（如 phiR 的 OnWB 第二次运行写 [z] 会与 (i6) 竞争，phiNR 用"按被逐出值分支"避免）。

涉及论文标题：
- täkōFormal: Enabling Robust Software for Programmable Memory Hierarchies
