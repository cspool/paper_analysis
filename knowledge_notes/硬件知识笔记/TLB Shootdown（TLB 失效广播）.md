## TLB Shootdown（TLB 失效广播）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
当页映射被修改或删除时，必须使所有持有该页旧翻译条目的 TLB 失效，否则会继续使用过期映射访问错误物理位置——这一跨处理器/跨 GPU 的 TLB 失效过程称为 TLB shootdown（TLB 广播失效）。在 multi-GPU UVM 页迁移中，host UVM 驱动把 invalidation 请求广播到所有 GPU，各 GPU 对自己的 L1/L2 TLB 执行 shootdown，同时经 GMMU 页表走查清除对应 PTE，以保证翻译一致性。ShadowUpdate（ISCA'26）的关键观察正是：这个 invalidation 广播已经遍历了每个 GPU 的页表（GMMU 走查），因此可以把"新映射更新"piggyback 到 invalidation 消息上，让 TLB shootdown + PTE 清除 + 新 PTE 写入在一次遍历中完成，不产生额外 page table walk。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
ShadowUpdate 中的运转流程：GPU 0 访问 GPU 1 上的远端页、访问计数达 256 → host 在 GPU 0 分配新物理页 → host 广播 invalidation 消息（ShadowUpdate 中携带新 PA）到所有 GPU → 每个 GPU 的 GMMU 做 page table walk 清除旧 PTE 并写新 PTE，同时 L1/L2 TLB 做 shootdown（旧条目失效）→ 拷贝完成后 completion 广播清除 IfMT 条目。baseline 中 shootdown 后其他 GPU 的 PTE 保持无效（新映射只装 destination），导致后续 re-fault；ShadowUpdate 用同一次遍历装好新映射，消除了 shootdown 带来的"翻译空窗"。相关机制：ConServe（ISCA'26）在 CUDA VMM 下把 cuMemUnmap 后的 TLB invalidation 延迟到空闲窗口执行，把翻译失效移出关键路径（vault 证据：repos/repo_paper_isca26_full/knowledge_repo/知识库_硬件架构.md 的 CUDA VMM 条目）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
通用实现（CPU 侧，Web 证据：Amit, USENIX ATC'17）：OS 向每个相关核发送 IPI 中断，各核在返回内核态时清 TLB，全部确认后才允许页表修改生效；GPU 侧由 host UVM 驱动把 invalidation 经互联广播给各 GPU 的 GMMU，GMMU 清除 PTE 并触发本地 TLB 失效。使用上，TLB shootdown 的代价是迁移关键路径上的串行同步开销；ShadowUpdate 论文把"invalidation（30.35%）+ mapping（26.02%）"两步合并，迁移处理总延迟降 22.79%，是"用已有失效路径承载更多工作"的典型设计。

涉及论文标题：
- Reducing Page Faults via Invalidation-based Mapping Propagation in Multi-GPU Systems
