## Cuckoo filter（布谷鸟过滤器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cuckoo filter 是一种支持删除的近似集合成员查询数据结构（Fan et al., CoNEXT'14 [22]），用 fingerprint 而非完整 key 表示成员：每个 key 经哈希得到 11-bit fingerprint 与两个候选桶位置，插入冲突时用"踢出并重定位"（cuckoo 式位移）处理，查找无假阴性、只有假阳性。ShadowUpdate（ISCA'26）把 Cuckoo filter 用作 IfMT 的快速预筛器：64 buckets × 每桶 4 slots × 11-bit fingerprint（xxHash 对 VA 哈希），两个候选 bucket 并行查、1 cycle 完成；命中（可能假阳性）才查 UMPT 确认，未命中（保证无假阴性）直接旁路 UMPT 去 GMMU。平均假阳性率 0.94%，把"每次 L2 TLB miss 都要查 256 项 UMPT"的开销降到极低。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
硬件中的运转流程：L2 TLB miss 后请求先送 IfMT → 用请求 VA 经 xxHash 生成 fingerprint → 按哈希查两个候选 bucket（每 bucket 4 个 11-bit 槽，并行比较）→ 两个桶都无该 fingerprint → 确认非迁移页，1 cycle 内旁路，请求直进 GMMU 走查（路径 B）；任一桶命中 → 可能假阳性，继续查 UMPT 确认（路径 C）。插入：invalidation 广播到达时把新迁移页的 fingerprint 写入其候选桶（桶满则踢旧 fingerprint 到另一候选桶）；删除：拷贝完成广播到达时从桶中删掉 fingerprint。存储仅 352B（11bit × 256 / 8），Synopsys Design Compiler + FreePDK 45nm 估面积 0.0243mm²、功耗 1.5852mW（折算 28nm 仅占 AMD Fiji 596mm² 的 0.0016%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
通用实现：为每个候选位置选 fingerprint 的 Cuckoo 哈希表；关键性质是无假阴性（没查到=一定不在集合）、支持 O(1) 删除、负载因子高（多槽桶）。使用场景包括网络/存储系统去重、GPU 侧"快速判定某地址是否在某个小集合"的预筛（本论文）；区别于 blocked-Cuckoo hashing（vault 证据：repos/repo_2025/knowledge_repo/知识库_硬件架构.md 的 SSD 原生 KV 条目——后者是"桶=SSD 块"的哈希表组织，用于 KV 存储索引，不是过滤器）。设计上"用 0.94% 假阳性换 1 cycle 全速旁路"体现了近似结构在翻译关键路径上的典型用法。

涉及论文标题：
- Reducing Page Faults via Invalidation-based Mapping Propagation in Multi-GPU Systems
