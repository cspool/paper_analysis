## Phantom Address（幻影地址）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Phantom address 是 täkō 可编程内存层级中的一类特殊虚拟地址：它没有主存背衬（not backed by main memory），其 cache line 的内容完全由注册在该地址上的 OnMiss 回调在 cache miss 时计算生成。因此 phantom 地址的数据只存在于缓存中——一旦被逐出且未写回，其内容就被丢弃；再次访问会 miss 并触发 OnMiss 重新生成（每次 OnMiss 可能生成不同值，如 HATS 中每次返回下一条图边）。由于无主存可回写，dirty phantom 行的逐出走 OnEvict（而非 OnWB 写回主存），但 dirty 信息仍决定回调类型（EvDirty/WbDirty 公理）。Leviathan（secs_2025/36，near-data 通用系统）也用"cache 地址与内存地址一一映射"的 phantom address 概念（类似 memory overlay），实现软件无法做的 cache-to-memory 动态填充。Web：täkō 原论文 [55] 为闭源，无官方文档，具体语义以 täkōFormal 论文为准。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程例子（图 2）：core 写 phantom 地址 [x]（指令 [x]←1）→ miss → engine 运行 OnMiss 生成 [x]=2 填 cache → (i1) 把 2 覆盖为 1 → 若 (i1) 与 (i2) r1←[x] 之间发生 eviction，[x] 第二次 miss 触发第二次 OnMiss 生成 2 → (i2) 读到 2 而非 (i1) 写的 1——(i1) 的写被"完全丢弃"，这正是 phantom 地址无主存背衬的后果。täkōFormal 的 MCM 用 Rcb/Wcb/RMWcb 事件 + VfWf 公理（任何 phantom 读/写前在 cbo 中必须有 M_e）强制"读 phantom 前必须先 OnMiss"，用 viscb/CboVal 保证回调值对应，用 OEInt/OMInt 强制 OnMiss 与 eviction 回调交替，用 EvDirty/WbDirty 约束 OnEvict/OnWB 与 dirty bit 对应，从而把上述反直觉行为纳入可形式化推理的框架。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：地址被标记为 phantom（注册回调的地址范围即 phantom 或混合范围）；täkō 硬件在 directory 级 cache 中为 phantom 数据建条目，之后对一致性协议与普通数据不可区分。使用场景（täkō 论文 [55]）：graph traversal（HATS 用 phantom 地址顺序存边）、scatter-updates（phantom 地址作 write-combining buffer，OnWB 发布结果到普通内存）、non-volatile memory transactions；程序员需配合 FlushRange 同步并保证回调正确性（täkōFormal 的 litmus tests 如 mpcb/wbf/phiNR/hatsNR 展示如何写出无 race 的 phantom 程序）。

涉及论文标题：
- täkōFormal: Enabling Robust Software for Programmable Memory Hierarchies
