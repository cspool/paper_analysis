## DDU 与 CDB（Duplication and Deduplication Unit / Candidate Deduplication Buffer）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DDU 是 CDFD 的复制/去重决策引擎：接收远端读 far-fault、判断 duplication ratio、发起 32MB 复制或"先细粒度去重再复制"，并把去重请求传播到其他 GPU（更新其 sharer table 与页表）；同时处理来自其他 GPU 的复制/去重请求。CDB 是 DDU 内的候选去重缓冲（256 项：36-bit VPN + 12-bit 总远端更新计数 + 16×17-bit 子项，共 10,240 B）：记录收到远端更新频繁的重复页，每子项 8-bit 本地访问计数 + 8-bit 远端更新计数 + valid bit，子项粒度 2MB（32MB 页）或 64KB（1MB 页）；CDB 满时换出总远端更新计数最低的项。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
DDU runtime 收远端读 fault → 若复制 32MB 页后 current ratio ≤ target：直接发复制请求（必要时 LRU 常规页换出到 CPU 内存）；否则向 CDB 索取去重信息 → 按收益 = 本地访问数 − 远端更新数，对最低收益子页去重直至腾够空间 → 再复制 32MB 页。CDB 侧：远端更新按 VPN 匹配项（未命中则从页表构造项）→ 对应子项与总远端更新计数各 +1；DDU 查询时返回访问数据。收益最低选择用 256×12-bit 最小值归约实现（20K–38K 门）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
片上缓冲 + DDU runtime 状态机（管理本地 fault 与跨 GPU 复制/去重请求的转达）；sub-entry 组织对齐 NVIDIA TLB 页大小，直接复用页表信息构造 CDB 项。面积：tag 比较 36.9K 门（256×36-bit XNOR）+ eviction 20K–38K 门 + 计数器逻辑 10K–40K 门。

涉及论文标题：
- Coarse-Grained Duplication First, Fine-Grained Deduplication Later: Duplication-Centric Multi-GPU Memory Management
