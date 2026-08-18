## 目录式缓存一致性协议（Directory-Based Cache Coherence，home 串行化点）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
目录式一致性协议为每个 cache line 维护一个固定"home"目录分片（与内存/L3 slice 同居一处，home 由物理地址决定），目录项记录该行在哪些 cache 有副本及状态（共享者集合 + Dirty 位）。请求者发生 miss 时先点对点访问 home：home 作为该行的串行化点（serialization point）裁决冲突事务的顺序，再向共享者发无效化/转发或直接供数。与 snooping（广播到所有缓存、依赖总线/NoC 顺序）相比，目录协议点对点消息、可扩展，但代价是：home 可能在远端簇（长延迟/高流量），且目录存储随核数与 cache 容量增长（精确跟踪共享者是二次方开销）。Web 证据（Utah CS7820 讲义 https://users.cs.utah.edu/~rajeev/cs7820/pres/7820-08.pdf、Edinburgh PA 讲义 https://www.inf.ed.ac.uk/teaching/courses/pa/Notes/lecture06-directory.pdf）：home 需用 busy 状态 + NACK/缓冲处理 in-flight 冲突，请求者与 home 两侧都要串行化。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Dorado 的 1024 核组织中，每个簇内有多个目录-LLC 分片（每分片 = Traditional Directory + Extended Directory），行地址哈希决定其 home 簇与簇内分片。一次读 miss 的流程：core L2 miss → 先查本地目录分片（Temporary home，若有）→ 无本地条目则跨簇（2D mesh，60 cycles RT）访问 Global home 串行化 → home 从 DRAM/远端共享者供数 → 本地建目录项并登记指针。写事务（Table V 的 C1–C6）在 home 串行化后向本地共享者（LLptr）、远端簇（LRptr→该簇 RLptr）分层发无效化并回收目录项。Dorado 对串行化的两个处理：(1) Global home 是最终串行化点——事务必须先成功锁住 Global home 条目才能锁 Temporary home，避免不一致与死锁；(2) Temporary home 承担本地事务，减少到 Global home 的往返。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现形态：扁平目录（内存式全量目录、缓存式链表目录如 SCI/Sequent NUMA-Q）、层次目录（树形）、粗粒度向量（SGI Origin 2000 每 bit 覆盖 64 节点）；共享者表示见"目录共享者跟踪表示"条目。商用近似：Intel 非包容 L3 的 CHA + snoop filter、DRAM 中的 memory directory bits（过滤远端 snoop，见 Stack Overflow DC PMM 讨论 https://stackoverflow.com/questions/65316397）；AMD EPYC 跨 CCD 一致性有限（LLC 每 chiplet 私有）。模拟工具：gem5 Ruby（SLICC 协议描述语言）、SST（Dorado 所用）。使用要点：目录项溢出策略（broadcast 位 / 强制无效化 / 溢出到 LLC 或 PointerSpace）决定大规模系统的流量行为。QED 对一致性的假设视角补充：QED 把一致性正确性（写串行化 write serialization、以及 MCM 要求的"多副本写原子性"multi-copy write atomicity）当作已由标准一致性验证覆盖的既定前提（引用大量一致性验证工作），从而把验证问题聚焦到 LSQ 的访存指令排序——因为一致性 bug 与排序 bug 可能交互/不可区分，QED 必须显式分离这两者；QED 验证的仍是本核 LSQ 与外部一致性事件的交互（invalidation、外部读请求等作为排序事件进入探索树）。

täkōFormal 的层次化目录式 MSI 视角补充：täkō 的 cache 为 inclusive，一致性用两层目录式 MSI 协议（hierarchical directory-based MSI）——上层协议以 L2 为 directory、Core 与 Engine 的 L1 为子；下层协议以 L3 shard 为 directory、L2 为子；两层之间用 HieraGen 的 proxy L2 cache 桥接通信。täkōFormal 在 Dafny operational model 中显式建模所有协议 transient 状态及其与 täkō 回调的交互（验证 coherence-consistency 接口问题，如 directory 级 dirty bit 被协议准确保留以保证 OnEvict/OnWB 正确触发），但假设一致性协议本身正确（引用文献 [46,48]）以聚焦于"回调/phantom 地址/FlushRange 语义与 ISA 级 MCM 的对应"——phantom 数据从 engine 收到并填充 directory 级 cache 后，对一致性协议而言与从主存取回的数据不可区分；Network 中 coherence 消息用无序集合建模（过近似各种 NoC），engine 回调请求按地址 FIFO（täkō 要求）。

涉及论文标题：
- Dorado: Clustered Hardware Cache Coherence for 1,000+ Cores
- QED Scalable Consistency Verification of Memory Instruction Reordering in Hardware
- täkōFormal: Enabling Robust Software for Programmable Memory Hierarchies
