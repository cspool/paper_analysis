## 优先级队列（top-K hammered-victim 跟踪，用于 RFM/主动缓解选行）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 优先级队列是 RowHammer 缓解中维护"最需要被刷新的行"的硬件队列：每 bank 一个，按计数器值（hammered count，HC）排序保存 top-K 个 victim/aggressor 行及其计数，供 Alert→RFM 与 proactive mitigation 直接取出刷新目标，避免触发 Alert 时全 bank 扫描计数器。逻辑链：计数型缓解（PRAC/QPRAC/PVAC 等）在每行计数器超阈值时需要知道"刷哪几行"，而 per-row 计数器数组无法快速排序/查询；用一个按计数排序的小队列即可在 O(K) 内维护最热门行。QPRAC（HPCA 2025，[91]）在 PRAC 上引入按计数器排序的服务队列；PVAC（ISCA 2026，arXiv:2604.20576）沿用并按 victim 语义扩展。Web 证据：PVAC arXiv 引用 [91] 的 priority queue；PrISM 论文对 QPRAC 的描述（"每个 Alert 触发 1 个 RFM + 5 项优先级服务队列，Back-Off 阈值按目标 TRH-D 调优"）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- PVAC 中的运转流程：每 bank 维护一个按 hammered count 排序的 top-K 队列（K=20）。每次 ACT 后，PVAC 的 CSA 更新 5 个计数器（4 victim + 1 aggressor），同时检查队列：若行已在队列中则更新其计数；若不在且计数超过队列中最小计数，则逐出最小项、插入新行。这样队列始终保存全 bank 被锤最狠的 20 行。使用时机有二：(1) proactive mitigation——每 tREFI 检查队列中是否有行计数超过阈值 NBO/2，有则在 tRFC 内提前刷新队列中 4 个最高 HC 的 victim 行（同时把这些行计数清零），减少后续 Alert；(2) RFM——某 victim 计数超 NBO 触发 Alert 后，MC 在 tABO_Recovery=NM it×350ns 内发 NMit 个 RFM，每次 RFM 直接从队列取出 4 个最高优先级的 victim 行刷新。队列容量推导（§V-B）：最坏情况下 RFM 刷新行数 = NMit_max(4)×每 RFM 行数(4)=16，proactive 再加 4，共 20——比 QPRAC 的队列大 4 倍，但面积开销仅 0.2%/chip。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：硬件排序队列（比较器网络/二叉堆结构），存行地址 + 计数器值，支持插入/更新/逐出最小项；面积远小于 SRAM/CAM 全表（0.2%/chip）。使用：与 per-row 计数器协同——计数器给出"某行被锤了多少次"，队列给出"全 bank 被锤最狠的 K 行是谁"，二者配合决定刷新顺序与 RFM 目标；QPRAC 用 5 项服务队列、PVAC 用 top-K=20。论文未明确说明队列的具体硬件结构（堆/排序网络）与每 bank 的实现细节，但给出容量依据与面积开销。同类：ProTRR 用 Misra-Gries（ProMG）频繁项近似替代精确 top-K 表，以更低面积换取近似最优。

涉及论文标题：
- PVAC: A RowHammer Mitigation Architecture Exploiting Per-victim-row Counting
