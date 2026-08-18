## 跨页指令预取（Page-Cross / Page-Crossing Prefetch）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
跨页预取是指预取器发出跨越指令页边界（虚拟页）的预取请求：目标指令块与当前取指位置不在同一虚拟页。由于 L1I 是 VIPT（虚拟索引物理标记）结构、预取器在虚拟地址域工作，预取器可以选择丢弃（conservative，No Page Cross）或放行（aggressive，Permit Page Cross）跨页请求。放行跨页预取的代价是预取请求需要地址翻译——翻译 miss 在 TLB 时要走查页表，走查延迟破坏预取及时性；收益是覆盖率高、能预取指令翻译到 TLB 层次（间接 TLB 预取），且商业处理器（如 [11] 引用的 Intel/AMD 现代产品）越来越多地采用。论文实验（图 2）：三场景对比显示 Permit Page Cross 一致优于 No Page Cross，而"Free Translation"（跨页预取翻译零成本）又显著高于 Permit——证明翻译延迟是剩余瓶颈。与数据跨页预取不同（数据跨页预取按负载可好可坏），指令跨页预取几乎总是有益，因为指令流是顺序+循环控制流、高度可预测，而数据访问常为指针追逐等不可预测模式。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
跨页预取在硬件中的流程：EPI 在虚拟地址域预测未来指令块（可能跨页）→ 预取请求查 iTLB→sTLB→miss 时查 tPB/页表走查获取翻译 → 翻译完成后以物理地址访问 L2C/L1I。论文的 "Free Translation L1I Prefetching" 理想场景（跨页预取 sTLB miss 瞬时转为 hit）量化了翻译延迟的上限收益；IP-CaT 的 tPB 正是为实现该收益的实用近似。跨页预取翻译结果（PTE）也是 TLB 污染的来源之一——tPB 专门收纳这些翻译、不污染 sTLB。相关：Morrigan（MICRO 2021，Vavouliotis 等）是专门的指令 TLB 预取器，论文评估 IP-CaT 优于它。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：在 L1I 预取器配置中启用跨页放行（permit page-cross），配合 TLB 层次（iTLB/sTLB）与页表走查器；IP-CaT 用 tPB 缓存跨页预取翻译、用 sTLB MSHR 的 cb bit 标记跨页预取来源。使用方式：对 server workload（指令足迹大）启用跨页预取以获得覆盖增益；配合 IP-CaT 抵消翻译延迟。局限：跨页预取翻译延迟若不被隐藏（无 tPB），收益被部分抵消（图 2 中 Free Translation 与 Permit 的 gap 即此损失）。

涉及论文标题：
- Enhancing Instruction Prefetching via Cache and TLB Management
