## Two-Level Homes（TLH，Global/Temporary home 两级目录归属）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TLH 是 Dorado 的核心机制：每个 cache line 除唯一 Global home 目录分片（物理地址哈希决定簇与分片）外，还可以在被引用到的每个簇内拥有一个 Temporary home 目录分片与目录项（同样按地址哈希选簇内分片），Temporary home 条目按需分配、可被驱逐，同一行可在多个簇同时有 Temporary home。Global home 负责最终串行化，Temporary home 把事务留在簇内。与 two-level directory（MGS、Cashmere-2L：簇内硬件目录 + 跨簇页级软件协议，两套协议）不同：TLH 只有一种目录类型（同时记录共享簇与本地共享核）、单一协议；与 hierarchical directory（树形多层目录）也不同：消息仍是"簇→home"的扁平流，扩容只需同构地加簇。（Web 无此术语索引——论文为 ISCA'26 新工作，以论文原文为准。）

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
三种共享者指针（Table II）：LLptr（本行 home 本地、共享者本地，存本地核 ID）、LRptr（home 本地、共享者远端，存远端簇 ID，一个指针覆盖远端簇 32 核）、RLptr（home 远端、共享者本地，Temporary home 条目中存本地核 ID）。读 miss 且行未被修改（Table III）：无本地条目→访问 Global home（无则建条目并加 LRptr），数据回本地并建 Temporary home（RLptr，D=0）；本地条目含 LLptr/LRptr→按"本地 LLC→本地其他 L2→远端共享簇"优先序供数并加 LLptr；本地条目含 RLptr→按"本地 LLC→本地 L2→Global home→远端共享簇"供数并加 RLptr。收益机制：对 32 核簇，91% 以上可本地满足的远端 home 读 miss 因此无需离簇；论文实测 L2-miss 远端访问降 89.6%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现（论文）：在 SST 中周期级建模，TLA+ 规格经 TLC 模型检验（4 簇×4 核×4 行实例验证单写者多读者、脏位一致、home 一致性、共享者健全、读正确性 5 条性质 + 无死锁/活锁）。死锁避免规则：可能改 Global home 状态的事务在成功锁 Global home 前不得锁任何 Temporary home 资源。配套改动：指针 5b ID + 1b 类型位（因簇化后 ID 空间缩小）；MESI 加 MS 状态避免脏行回传 Global home。使用要点：多 home 元数据需保持一致（home consistency 性质），Temporary home 驱逐时不能丢弃在途事务所需元数据。

涉及论文标题：
- Dorado: Clustered Hardware Cache Coherence for 1,000+ Cores
