## RRPV / SRRIP 与 set-dueling（集合决斗自适应替换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RRPV（Re-Reference Prediction Value）是 Jaleel 等提出的 SRRIP（Static Re-Reference Interval Prediction）替换策略的核心：每块维护一个 2-bit 预测值（0=高复用、3=LRU 底部，插入位置由策略决定），按预测的再引用间隔驱动替换，比 LRU 更鲁棒（对扫描型访问友好）。set-dueling（集合决斗，如 DIP/BIP）是自适应策略选择机制：把缓存 set 静态分为 Leader Sets（固定跑某策略）与 Follower Sets（跟随 Leader 中表现好的策略），用一个饱和计数器（如 PSEL）在 eviction 时更新，根据 Leader 表现动态切换。IP-CaT 的 TIPRP 扩展了 set-dueling：①两级决策树（PSEL1/PSEL2 两个计数器）在 PIP/NPIP/BIP 三策略间选择（超单计数器的二选一）；②在 L2C hit 与 eviction 双事件上更新（而非仅 eviction），响应更快；③按 pb bit 区分预取行与 demand 行更新，捕捉行类型间的效用差异。论文明确对比：TIPRP 超越 BIP/DIP 类单计数器 set-dueling 设计。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
set-dueling 在硬件中的运转（以 TIPRP 为例）：L2C set 静态分为 PIP/NPIP/BIP 的 Leader Sets（32/16/16）与 Follower Sets → Leader Set 访问直接应用其策略（其行为作为"投票"）→ Follower Set 访问查 PSEL1/PSEL2 决策树选策略 → Leader Set 上的 hit/eviction 按图 6(c) 规则更新 PSEL1/PSEL2（PIP Leader 只在 pb=1 行上更新，NPIP/BIP Leader 只在 pb=0 行上更新）→ 计数器越界即切换 Follower 的策略。SRRIP 基（RRPV=0..3）：NPIP 把预取行插 RRPV=3，PIP 在 eviction 时保护预取行，BIP 直接 bypass。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SRRIP/RRPV 是 ChampSim 默认替换策略之一（jaleel 模块），set-dueling 广泛用于 DRRIP（动态 RRIP，论文也作为对比策略评估）；TIPRP 在 ChampSim L2C 替换模块实现（需要 pb bit 与 PSEL1/PSEL2）。使用方式：作为 L2C/LLC 替换策略评估基线（SRRIP/DRRIP/SHiP++ 均为常见对比）；自适应策略适合程序阶段变化的负载。局限：传统 set-dueling 用单计数器、只在 eviction 更新、不区分行类型——TIPRP 在这些维度改进。

涉及论文标题：
- Enhancing Instruction Prefetching via Cache and TLB Management
