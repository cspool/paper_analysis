## Duplicated TLB 与 Duplicated Page Sharer Table

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CDFD 为"写重复页 → 广播远端更新"引入的翻译结构。页表用 1-bit 标志标记页是否重复；duplicated TLB（32 项，每项 36-bit VPN + 7×36-bit PFN，共 1,152 B）缓存"哪些 GPU 持有该重复页副本"（8 GPU 内最多 7 个共享者）；miss 时对内存中的 duplicated page sharer table 做页走查（仿 GPS 的 sharer 表实现）。写重复页流程：L1 TLB 识别重复页写 → 转发 duplicated TLB → miss 则走查 sharer table → 命中装回 duplicated TLB → 把共享者物理地址交 GMMU → GMMU 发远端写请求。复制/去重事件同步更新两结构。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
把"多副本归属关系"做成独立的查找结构，使远端更新 off critical path：普通访存不经过它，只有写重复页才查；走查与填充和常规 TLB 同构，可用同样硬件机制维护。去重后本地页表与各 GPU 的 sharer table 都要删项，保证后续写不再误广播。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
片上 32 项小 TLB + 内存 sharer table（VPN 为索引，每项存至多 7 个 36-bit PFN）；由复制/去重事件驱动表项增删。面积 1,152 B，tag 比较计入 CDFD 总开销 66.9K–114.9K NAND2 门。

涉及论文标题：
- Coarse-Grained Duplication First, Fine-Grained Deduplication Later: Duplication-Centric Multi-GPU Memory Management
