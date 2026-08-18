## IP-CaT（Instruction Prefetch Centric Cache and TLB Management，指令预取为中心的缓存与 TLB 管理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
IP-CaT 是论文提出的微架构方案，第一个通过协调 TLB 与缓存管理来最大化 L1I 预取收益的机制。逻辑链：现代服务器负载指令足迹巨大且年增长达 30%，L1I 预取器（EPI/FNL+MMA/Barça）能缓解前端瓶颈，但有两个限制其收益的因素——①L1I 跨页预取需要地址翻译，翻译延迟（sTLB miss 后的页表走查）破坏预取及时性；②L1I 预取取回的代码行在 L2C 中复用行为高度可变（大量 dead-on-arrival、少数高复用）。IP-CaT 用两个模块分别解决：tPB（Translation Prefetch Buffer）在 sTLB 旁缓存 L1I 跨页预取取回的翻译以减少翻译成本；TIPRP（Trimodal Instruction Prefetch Replacement Policy）用决策树 L2C 替换策略按复用潜力管理预取代码行。协同收益：tPB 减少页表走查进而降低 L2C 争用，使 TIPRP 更有效（IP-CaT 6.1% 超过 tPB 2.9% + TIPRP 2.9% 之和）。总存储开销仅 0.79KB（占 L2C 容量 0.08%）。评估：与三种 SOTA L1I 预取器组合在 105 个单核 server workload 与 160 个 4 核 mix 上，对 EPI/Barça/FNL+MMA 分别 +6.1%/+8.3%/+7.9% geomean speedup，全面超过 CHiRP、Morrigan、CLIP、EMISSARY、PACIPV、PACMAN、SHiP++、Mockingjay 等 TLB/缓存管理策略。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
IP-CaT 在标准微架构中的完整数据路径（图 7）：L1I 预取请求（页内或跨页）先查 iTLB（1cc），miss 再查 sTLB（8cc），sTLB miss 时查 tPB——tPB 命中则将翻译按 sTLB 插入策略写入 sTLB 并失效 tPB 条目；tPB miss 才触发页表走查（4 级 radix tree + MMU PSC），cb=1（来自 L1I 跨页预取）的走查结果写入 iTLB+tPB 而非 sTLB 以免疫 TLB 污染。翻译完成后物理地址访问 L1I/L1D，miss 后查 L2C，TIPRP 依据 pb bit 与 PSEL1/PSEL2 决策树在 PIP/NPIP/BIP 中选择驱动 L2C 插入/提升/替换。demand 指令翻译 miss 在 sTLB 后同样查 tPB，命中移入 sTLB 服务 demand。要求硬件改造：sTLB MSHR 每项 1-bit cb、L2C 每块 1-bit pb（pb 在多数 L2C 设计已存在），tPB 可按同 associativity 集成进 sTLB（额外 set）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：基于开源 ChampSim（trace-based 乱序处理器 + 三级缓存模拟器）在 ChampSim 中新增 tPB 结构与 TIPRP 替换策略，并扩展 SMT 支持；baseline 配置类 Intel Cascade Lake（4GHz、6-wide、352-entry ROB、TAGE-SC-L、L1I 32KB、sTLB 1536 项 12-way、L2 1MB、LLC 1.375MB/core）。使用方式：作为任意 L1I 预取器的"附加件"——论文对 EPI/Barça/FNL+MMA 三种预取器组合评估均提升性能；tPB 默认 64 项全相联（也可 4 sets×16 ways 或集成进 sTLB），TIPRP 只作用于 L1I 预取行（对 demand 行应用反而 -10.1%）。开源：IP-CaT 源码未见公开仓库（联网无法确认）；论文 arXiv:2605.12433（ISCA 2026）。

涉及论文标题：
- Enhancing Instruction Prefetching via Cache and TLB Management
