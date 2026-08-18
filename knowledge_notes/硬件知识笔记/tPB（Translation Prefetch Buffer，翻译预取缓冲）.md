## tPB（Translation Prefetch Buffer，翻译预取缓冲）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
tPB 是 IP-CaT 的 TLB 侧组件：位于末级 TLB（sTLB）旁的小容量 set-associative 结构，专门收纳由 L1I 跨页预取请求触发的页表走查取回的指令页表项（PTE，含 vpn 索引 + ppn + 属性位）。逻辑链：L1I 预取器在虚拟地址域工作（L1I 为 VIPT），跨页预取请求需要地址翻译；若翻译 miss 在 iTLB/sTLB 就要走查页表，走查延迟破坏预取及时性，且直接把翻译塞进 sTLB 会污染 TLB（挤占 demand 条目）。tPB 把"L1I 预取带来的翻译"单独缓存：①L1I 预取请求在 iTLB/sTLB 都 miss 后先查 tPB，命中则按 sTLB 插入策略把翻译移入 sTLB 并失效 tPB 条目（免走查）；②demand 指令翻译 miss 在 sTLB 后也查 tPB，命中同样移入 sTLB 服务 demand 访问——论文显示大量 demand sTLB miss 由 tPB 命中服务（sTLB MPKI 降 31.6%/18.2%/32.3%）。只由 L1I 跨页预取翻译请求填充（cb bit 标记来源），不触发额外 TLB 预取，与 TLB 预取器（如 Morrigan）正交。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
tPB 在硬件中的运转（图 5）：L1I 预取请求查 iTLB→sTLB→（miss）→tPB：命中→按 sTLB 插入策略插入 sTLB、tPB 条目失效；miss→触发页表走查→cb=1 的走查结果存入 iTLB+tPB（不进 sTLB）。demand 指令访问 miss 在 iTLB/sTLB→查 tPB→命中→插入 sTLB+失效 tPB。为区分翻译请求来源，sTLB MSHR 每项新增 1-bit cb（cross-bit，来自 L1I 跨页预取为 1）。实现选择：tPB 默认 64 项全相联（命中率 36.2%）；敏感性实验显示 8→128 项时 EPI 命中率 3.1%→48.3%；全相联/32-way/16-way 命中率差异小（37.2% vs 28.0% direct-mapped），4 sets×16 ways 是更实用的配置；也可按 sTLB 相同 associativity 集成（sTLB 增 4/8 sets×12 ways，命中率 25.6%/41.6%）。ISO-storage 对比：tPB 胜过给 sTLB 加 1 way（+128 项）的方案。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：tPB 为与 sTLB 并列的硬件缓冲，存储 vpn/ppn/属性位，LRU 管理；需 sTLB MSHR 的 cb bit 区分翻译请求来源；集成进 sTLB 时翻译一致性（TLB shootdown）开销最小。使用方式：作为 L1I 跨页预取的翻译缓存，隐藏页表走查延迟；对 TLB 不密集的负载无害（论文对比 tPB+SRRIP 与 IP-CaT 于 788 个全 workload）。相关 prior art：POM-TLB（die-stacked TLB）、Victima（用 L2C 部分作 L3 TLB）、DVMT、以及用缓冲存预取翻译的 TLB 预取器——tPB 只存 L1I 跨页预取翻译，与其正交。局限：2MB 大页场景下跨页预取 miss 少，tPB 收益小（100% 2MB 时无收益）。

涉及论文标题：
- Enhancing Instruction Prefetching via Cache and TLB Management
