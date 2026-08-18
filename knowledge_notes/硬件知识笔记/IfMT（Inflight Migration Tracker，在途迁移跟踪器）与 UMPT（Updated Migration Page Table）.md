## IfMT（Inflight Migration Tracker，在途迁移跟踪器）与 UMPT（Updated Migration Page Table）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
IfMT 是 ShadowUpdate（ISCA'26）新增的轻量 GPU 硬件表，放在 L2 TLB 之后、GMMU 之前，用于保证"新映射提前装进页表但数据拷贝尚未完成"期间的翻译正确性：跟踪正处于迁移（in-flight）的页，阻塞对这些页的翻译请求，直到拷贝完成才放行。IfMT 由两部分组成：① **UMPT（Updated Migration Page Table）**——256 项全相联表，每项存 36-bit 虚拟页号 + 40-bit 新物理页号 + 1-bit pending(P) 位，8 个并行比较器，最坏 32 cycle 查找；② **Cuckoo filter**——64 buckets × 4 slots × 11-bit fingerprint（xxHash 对 VA 哈希），1 cycle 并行查两个候选 bucket，平均假阳性率 0.94%、无假阴性，用于快速筛选避免每次 L2 TLB miss 都查 UMPT。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
翻译查找流程（Figure 12）：CU 访存 0xA（正在迁移）→ TLB shootdown 后 L1/L2 TLB 均 miss → 请求存 L2 TLB MSHR（合并重复翻译请求）→ 先经 IfMT：Cuckoo filter 查 0xA 的 fingerprint → 未命中（无假阴性，确认非迁移页）直接旁路 UMPT 去 GMMU 走查；命中则查 UMPT 确认 → 置 P 位挂起翻译（不发 ATS、不产生 re-fault）→ 拷贝完成后 completion 广播 → Cuckoo filter 删 fingerprint、UMPT 驱逐条目，若 P 位置位则把存储的新 PA 直接返回 L2 TLB MSHR 完成翻译（不重放 page table walk）。插入时机：invalidation 广播携带新映射到达时，IfMT 同时把 VA→新PA 写入 UMPT 并加 fingerprint；UMPT 满则暂停迁移直到有空位（因此 256 项容量是设计参数，见敏感性实验）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现为 L2 TLB 与 GMMU 之间的小型硬件结构（UMPT 2464B + Cuckoo filter 352B，@22nm/45nm 面积功耗详见 Cuckoo filter 条目），由 invalidation/completion 广播驱动插入与删除，无需软件参与。使用上体现"跟踪粒度=页、阻塞只发生在迁移窗口"的取舍：把"早装映射的收益"与"拷贝未完成误访新位置的正确性风险"解耦——这是迁移类硬件设计中"预发布映射 + 短暂阻塞"模式的实例。容量敏感性：64/128/256 项分别 0.87×/1.19×/饱和（256 最优），全相联优于 16-way（0.63×，因冲突导致插入失败会暂停迁移）。

涉及论文标题：
- Reducing Page Faults via Invalidation-based Mapping Propagation in Multi-GPU Systems
