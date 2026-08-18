## MOP（Minimalist Open-Page）地址映射与 Rubix 随机行映射（RowHammer 缓解视角）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MOP（Minimalist Open-Page，MICRO 2011，Kaseridis 等 [38]）是面向多核时代的 DRAM 页模式调度/地址映射策略：尽量让连续 cache line 落在同一 DRAM 行（open row 保持打开），提升行缓冲命中率，同时用极简逻辑避免复杂 close-page 决策；PrISM 论文将其作为默认地址映射（Table IV）。Rubix（ASPLOS 2024，Saxena/Mathur/Qureshi [81]）是随机化 line-to-row 映射：用密钥化哈希把逻辑 cache line 随机映射到 DRAM 物理行，打破 aggressor-victim 行的空间相关性、把单个行的高频访问摊薄到多行，从而降低 RowHammer 缓解开销。PrISM 只在 ultra-low TRH-D（≤250）启用 Rubix 风格随机映射，更高阈值保留 MOP。
- 关键定量（Table I，平均每通道 Alert 率 /1K tREFI）：MOP vs Random——TRH-D=1000：2.4 vs 0.04（60×）；500：32.2 vs 3.5（9.2×）；250：312.3 vs 50.3（6.2×）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 PrISM 中的运转（为何需要随机映射）：MOP 保留空间局部性 → 同一高局部性行在多个 mitigation window 被反复采样 → 反复出现在 SHQ 历史 → 良性 workload 也产生大量交集 → 触发不必要的 Alert/RFM。启用 Rubix 随机映射后：相邻 cache line 被哈希到不同 DRAM 行 → 单行跨窗口重复出现概率大降 → SHQ 交集骤减 → Alert 率下降。取舍：随机映射牺牲行缓冲局部性（行命中率下降），故仅在低阈值下"减少良性交集"的收益显著时启用；TRH-D≥500 时 Alert 率已低（2.4/32.2 per 1K tREFI），保留 MOP 的行缓冲局部性更优。完整链路：内存控制器按 MOP（或 Rubix）把地址映射到 channel/bank/row → 每次 ACT 触发 PrISM 采样逻辑 → 采样行入 SSQ 并与 SHQ 比对 → 交集频率直接由映射的空间局部性决定 → Alert→RFM 影响性能。
- 与 RowHammer 安全的联系：随机化映射是一类独立的 RowHammer 缓解思路（Randomized Row-Swap、Rubix、DAPPER 的 secure hashing），PrISM 用它作为辅助手段而非主缓解，主缓解仍是采样历史交集。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：MOP 在内存控制器/模拟器（Ramulator 2.0）的地址映射模块配置（开放页策略，连续地址→同行）；Rubix 风格随机化按论文 [81] 在模拟器中实现（密钥化哈希把行号随机化，PrISM 仅 TRH-D≤250 时启用）。评估软件：Ramulator 2.0 + 57 个开源 workload（SPEC2006/2017、TPC、Hadoop、MediaBench、YCSB，按 RBMPKI 分三档，Table V），8 核同构混合、FR-FCFS 调度。使用场景：研究 RowHammer 缓解时，地址映射是决定"同一行被采样的频率"与"行缓冲命中率"的关键系统参数，MOP vs 随机映射的开关直接影响概率缓解的良性交集开销。

涉及论文标题：
- Loaded Dice: Solving the Non-Selection Problem for Scalable Probabilistic RowHammer Defense
