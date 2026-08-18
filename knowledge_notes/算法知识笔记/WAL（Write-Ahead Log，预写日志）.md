## WAL（Write-Ahead Log，预写日志）

术语解释
- WAL 是持久化存储系统的经典机制：写操作先追加到日志（顺序写、落盘）再更新主数据结构，崩溃后可重放日志恢复一致性。论文的 SSD 原生 KV 用它做持久化与写摊销——合并同桶更新、超阈值后批量提交进 blocked-Cuckoo 块并回收日志空间。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- WAL 的核心是把"随机小写"转化为"顺序追加"，并以日志作为崩溃一致性的权威来源（主流存储引擎如 RocksDB/WiredTiger/SQLite 均采用）。论文把 WAL 放在 SSD 上（SSD-resident WAL）：PUT 更新先追加到 WAL 合并目标同桶的更新，WAL 超阈值时把合并后的更新提交（commit）进 blocked-Cuckoo 哈希块、再回收被重用的日志空间。这样既保证持久性（不丢条目，区别于丢弃溢出的 CacheLib），又通过合并把多个 KV 更新聚合成块级读改写、摊销写成本。
- 从算法pipeline角度拆解术语：一次 update 的 pipeline：主机收到 PUT → 追加到 SSD WAL（顺序写）→ WAL 达到阈值 → 读目标桶块（read-modify-write）→ 合并更新写回桶块 → 回收日志。GET:PUT 混合下，写比例越高、read-modify-write 越多、I/O 流量越大、吞吐越低（论文 Fig. 8：读写比从 100:0 到 50:50 吞吐显著下降）；强局部性（σ=1.2）使同桶更新更集中、每次 WAL flush 的 read-modify-write 更少。DRAM 只缓存热 KV 对，索引/日志全在 SSD。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：WAL 区域管理（追加指针、阈值触发、回收）、崩溃恢复（重放）、与主结构的批量提交事务；论文未给出 WAL 具体实现细节（块大小、刷盘策略等），仅描述其合并-提交-回收流程。论文用途：作为"flash 主动层"下持久 KV 的写路径设计示范——把 DRAM 从索引/日志中完全解放。信息缺口：WAL 的具体刷盘/组提交机制论文未明确说明。

涉及论文标题：
- Five-Minute Rule 40 Years Later A First-Principles Revisit for Modern Memory Hierarchy

涉及论文标题：
- FEnc2: Unifying Data Packing for Efficient Private Inference via Convolution and Architecture-Aware Fragment Encoding
