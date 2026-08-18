## TIPRP（Trimodal Instruction Prefetch Replacement Policy，三模指令预取替换策略）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TIPRP 是 IP-CaT 的 L2C 侧组件：决策树驱动的 L2C 替换策略，专门管理由 L1I 预取取回的代码行，通过预判其未来是否会被访问来减少 L2C 污染并保留高复用行。逻辑链：L1I 预取取回的行复用高度可变——EPI 下平均 36.1% 的行 dead-on-arrival（零访问）、51.6% 服务 1-8 次、0.8% 服务 >128 次；单策略（LRU/SRRIP）无法兼顾"快速淘汰死行"与"保护高复用行"。TIPRP 组合三个 RRPV 基策略：PIP（Prioritize Instruction Prefetch，eviction 时保护预取行，优先逐出非预取行）、NPIP（Non-Prioritize，插入时置 RRPV=3 底部，易被淘汰）、BIP（Bypass，插入时完全 bypass 不占 L2C）；两级决策树由两个饱和计数器 PSEL1/PSEL2 驱动，静态把 L2C set 分为 PIP/NPIP/BIP 的 Leader Sets + Follower Sets（32/16/16 组），Follower 按 PSEL1/PSEL2 选策略（PSEL1≥T1 选 PIP，否则 PSEL2<T2 选 BIP，否则 NPIP）。训练在 L2C hit 与 eviction 上按 pb 不对称更新（PIP Leader 只在 pb=1 时更新、NPIP/BIP Leader 只在 pb=0 时更新），比 set-dueling 的单计数器更细粒度、响应更快。只作用于 L1I 预取行，demand 行仍走 SRRIP。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
TIPRP 运转流程（图 6，伪代码）：L2C 访问到达 → 判断所在 set 属于哪类 Leader/Follower：Leader Set 直接用其策略；Follower Set 查 PSEL1/PSEL2：if PSEL1≥T1 → PIP（eviction 时找 set 内最久未用且非预取行，无则逐 RRPV=3 行；预取行受保护）；elif PSEL2<T2 → BIP（预取行不插入）；else → NPIP（预取行插入置 RRPV=3）。训练：PIP Leader Set 上 demand hit 命中 pb=1 行 → PSEL1++、PSEL2--；PIP Leader 逐出 pb=1 行 → PSEL1--、PSEL2--；NPIP/BIP Leader Set 上 hit 命中 pb=0 行 → PSEL1--、PSEL2++；逐出 pb=0 行 → PSEL1++、PSEL2--。不对称训练（只在强相关事件更新）比全事件更新 IPC 高 5%。决策树节点位置（PIP→NPIP→BIP 转移方向）为经验最优。参数：PSEL1/PSEL2 各 10-bit，Leader Sets 32/16/16。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：基于 RRPV（Re-Reference Prediction Value，SRRIP [19] 的核心）的三种插入/提升/替换变体 + 决策树选择逻辑；需要 L2C 每块 1-bit pb（prefetch bit）标记行是否由 L1I 预取取回（多数 L2C 已有此 bit，无则从 L1I/L2C MSHR 传播）。使用方式：插入到 ChampSim 的 L2C 替换模块评估（对比 CLIP/EMISSARY/Mockingjay/PACIPV/PACMAN/SHiP++/DRRIP/SRRIP）；TIPRP 单独对 EPI/Barça/FNL+MMA 分别 +2.9%/+4.8%/+5.0%。局限：只优化预取行，对 demand 行应用 TIPRP 反而降低性能（IP-CaT D+P 消融 -10.1%），因为 demand 行与预取行复用模式不同；LLC 越大 TIPRP 相对贡献越小（1MB→4MB LLC 时 IP-CaT speedup 12.7%→2.6%）。

涉及论文标题：
- Enhancing Instruction Prefetching via Cache and TLB Management
