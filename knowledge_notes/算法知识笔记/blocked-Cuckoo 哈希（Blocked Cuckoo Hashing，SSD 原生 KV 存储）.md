## blocked-Cuckoo 哈希（Blocked Cuckoo Hashing，SSD 原生 KV 存储）

术语解释
- blocked-Cuckoo 哈希是把 Cuckoo 哈希与"桶=块"结合的哈希结构：每个 key 映射到两个候选桶（各对应一个 SSD 块，块内多个槽位），桶满时用重定位（relocation）而非丢弃处理溢出；论文用它构建完全无 DRAM 驻留索引/元数据的 SSD 原生持久 KV 存储。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 标准 Cuckoo 哈希（Pagh & Rodler 2004）：每个 key 有两个候选位置，插入冲突时把旧 key "踢"到其另一候选位，形成位移链；无 stash 时负载因子实用上限约 50%，加入多槽桶（blocked）或 stash 后可到 ~90-95%（网络来源：EMOMA 用 stash+CBBF 达 ~95% 负载、每次查找至多一次片外访问）。论文的 KV 设计：key 映射到两个 SSD 驻留候选桶，每桶=一个 SSD 块（l_blk=512B on Storage-Next、4KB on Normal SSD），桶大小 B=l_blk/l_KV（l_KV≈64B），每次查找需 1-2 次 SSD 块读（平均 1.5）；负载因子需低于临界值 α_critical（B≥4 时通常 >0.95），插入位移链期望长度 E[L]=α^(2B)/(1−α^B)，运行在远低于临界处使 E[L]≪1、插入延迟近常数。DRAM 全部用于缓存热 KV 对（个体粒度），SSD 驻留 WAL 合并更新后批量提交回桶块。
- 从算法pipeline角度拆解术语：一次 GET 的 pipeline：主机算两个桶哈希 → 查 DRAM 缓存（命中即返回）→ 未命中发 1-2 次 SSD 块读（平均 1.5）→ SSD 返回后返回客户端；一次 PUT（insert 或 update）的 pipeline：更新先追加到 WAL 合并同桶更新 → WAL 超阈值 → 提交合并更新进 blocked-Cuckoo 块 → 回收日志空间；更新负载下每次 WAL flush 把分散的 KV 更新聚合成块级读改写。相比 CacheLib（桶溢出即丢弃条目）与内存 KV（DRAM 索引随 key 基数线性增长），本设计把索引完全放到 SSD 块内，DRAM 成本与 key 数无关。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现要点：桶哈希函数（两个独立哈希）、桶内线性扫描、插入时沿候选链重定位、负载因子控制与 stash/重哈希兜底；SSD 版本额外要求块对齐、WAL 持久化与更新合并。论文评估：5TB KV、800 亿 64B 条目、负载因子 0.7、GET:PUT 100:0/90:10/70:30/50:50、lognormal 强弱局部性（σ=1.2/0.4），GPU+SN 在读重混合下达 100+ Mops/s（FASTER 内存级水平），CPU+SN 则受 host IOPS 限制。论文为模型驱动评估（分析框架+MQSim-Next 模拟），无开源实现。

涉及论文标题：
- Five-Minute Rule 40 Years Later A First-Principles Revisit for Modern Memory Hierarchy
