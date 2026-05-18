## Synchronization Variable Directory (Sync-Val Directory) / 同步变量目录

术语是什么？

Sync-Val Directory是LRM-GPU在LLC中实现的一种轻量级目录结构，专门追踪同步变量（而非所有数据）的owner chiplet。与传统GPU coherence protocol中追踪所有cache line sharer的完整directory（如HMG每chiplet 12K entries）不同，sync-val directory只跟踪explicit synchronization操作中涉及的同步变量地址，因此容量极小（4-chiplet系统仅64 entries、0.4 KB，约一个L1 cache容量的0.3%）。每entry包含：valid bit (1 bit) + tag address (48 bits) + owner chiplet ID (2 bits for 4-chiplet) = 51 bits/entry。Directory在acquire/release同步操作进入LLC时被查询和更新，决定是否需要触发L1.5 cache的coherence action (invalidate/flush)。

从硬件架构角度拆解：

Sync-Val Directory的硬件工作原理：
1. **查询阶段**：当SM发出acquire/release同步请求到达LLC时，以同步变量地址为key查询directory → 三种命中结果：Invalid（无记录）、命中（owner=本地或远端）、满（无空entry需evict）。
2. **分配阶段**：Invalid或Evicted场景下分配entry → 写入tag address + owner chiplet ID + set valid bit → 若因满而evict，按LRU策略选择victim entry → 若victim entry的owner L1.5为write-back policy则需先flush其L1.5。
3. **更新阶段**：Remote chiplet场景下更新owner field为当前chiplet ID。
4. **Coherence action触发**：仅在owner跨chiplet迁移或首次分配时触发L1.5 invalidate/flush → 同chiplet内重复同步无directory触发的coherence action。
5. **存储开销**：4-chiplet系统：64 entries × 51 bits = 3264 bits ≈ 0.4 KB。8-chiplet系统：owner field需要3 bits (支持8 chiplet)，每entry 52 bits。

与传统完整coherence directory的对比：HMG[43]的LLC coherence directory每chiplet 12K entries，每entry覆盖4条cache line，追踪所有数据的sharer而不只是同步变量。但HMG需要为所有写入数据向sharer发送invalidation request，在同步密集workload上产生大量coherence message traffic。LRM-GPU的sync-val directory只追踪同步变量owner，目录本身容量小但cover了同步性能的关键瓶颈。

术语一般如何实现？如何使用？

Sync-Val Directory的硬件实现方式：(1) 存储结构——小的CAM或直接索引的寄存器文件，因为entry数少（64-256 entries级别）无需复杂组织；(2) 访问延迟——通常1 cycle lookup（远快于LLC data access的10-20 cycles），可与LLC access流水线overlap；(3) Eviction policy——LRU或FIFO均可，主要处理directory容量不足时驱逐冷同步变量的情况；(4) 与LLC的集成——directory逻辑位于LLC controller中，与LLC tag lookup并行执行（同步变量地址同时查询directory和LLC tag）。在LRM-GPU实现中，directory随LLC在memory-side，每个chiplet有自己的LLC slice，directory entries可以按chiplet划分（16 entries/chiplet）或全局共享。Sync-val directory的正确性依赖所有同步操作bypass L1/L1.5直接到LLC的设计（与传统GPU一致），确保同步变量本身不在多级cache中产生不一致副本。

涉及论文标题：
- LRM-GPU: Alleviating Synchronization Overhead for Multi-Chiplet GPU Architecture

