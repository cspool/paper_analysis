## 可编程内存层级（Programmable Memory Hierarchy，PMH）与 täkō 缓存事件回调（OnMiss/OnEvict/OnWB）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
可编程内存层级（PMH）是一类让软件（而非仅硬件策略）控制数据在内存层级中移动的硬件架构：把 cache miss/eviction/writeback 这些传统上纯硬件处理的事件暴露给程序员定义的回调函数，数据变换/清理可以作为数据移动的一部分在缓存内完成（而非等数据加载到核里再做），从而提升性能与能效。täkō（tiled chip：每 tile 含 core、L1、私有 L2、共享 L3 bank 分片、运行回调的专用 engine）是代表性 PMH 原型：程序员为虚拟地址范围注册 OnMiss（miss 时在 engine 上运行，计算并填充该 cache line 的内容）、OnEvict（clean 行被逐出时运行）、OnWriteback/OnWB（dirty 行写回时运行）三类回调；回调以独立线程语义在专用 engine 上与 core 程序线程并行执行（区别于 IMO/EcMon 等 cache-event trap 的 function call 语义——打断 core 线程执行 handler），同地址回调被 engine 按 FIFO 串行化；无回调注册的地址保持传统 load/store 语义。Vault 证据：本论文 A.-tak-o-Hardware-Overview-.md 与 B.-Callback-Synchronization.md（omnisearch score 181/1062）；Leviathan（secs_2025/36）的 actor 构造/析构函数明确对标 täkō 的 onMiss/onEviction/onWriteback（score 40.6）。Web：täkō 原论文 [55] 无开源实现（论文未给链接，无法确认）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在硬件架构中 PMH 把"何时搬数据/搬什么"的决策权交给回调，prefetching 与替换策略这些以往隐藏的细节直接决定程序结果。运转流程例子（OnMiss workflow，图 3）：core 访问注册了 OnMiss 的地址 [x] → cache miss → 硬件线程把 OnMiss 请求经 Network（engine 请求按地址 FIFO）发给该 tile 的 engine → engine 运行回调计算 cache line 内容并填充 → 后续访问命中；OnEvict/OnWB 在 clean/dirty 行被逐出时运行，可把"处理结果"发布到普通内存（如 scatter-update 的 write-combining buffer 经 OnWB 发布 [y]←1）。反直觉结果：phantom 地址 [x] 被 (i1) 写入 1 后若在 (i2) 读前被 evict，(i2) miss 会触发第二次 OnMiss 生成新值，之前写入的 1 被完全丢弃（因 phantom 无主存背衬）——图 2 的 r1=2 结果。täkōFormal 正是为这种接口建立 ISA 级 MCM（事件 Rcb/Wcb/Ms/Me/Es/Ee/Fl + 关系 cbo/viscb/vf/ef/eb + 前缀封闭公理）使程序员无需理解微架构即可推理回调程序。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现形态：tile 级"core+L1 / Engine+L1 / L2 / L3 slice / Memory / Network"分层状态机（Dafny 实现，含 inclusive cache 与分层目录式 MSI 一致性），回调代码运行在 engine 上；用环境迁移过近似"回调可被预取/替换任意触发"的行为。使用方式：程序员注册回调（[x].OnMiss / [x].OnEvict / [x].OnWB）并用 FlushRange 同步；täkōFormal 提供 Alloy 编码的 MCM + litmus tests 供程序员检查程序是否 racy/某结果是否可达，并提供 Dafny operational model 验证实现。案例：HATS 图遍历用 phantom 地址顺序存边 + OnMiss 遍历 + OnEvict 记日志；Leviathan（通用 near-data 系统）用 data-triggered actions 提供类似能力。

涉及论文标题：
- täkōFormal: Enabling Robust Software for Programmable Memory Hierarchies
