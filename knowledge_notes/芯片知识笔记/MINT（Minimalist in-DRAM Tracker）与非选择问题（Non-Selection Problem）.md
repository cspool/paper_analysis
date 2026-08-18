## MINT（Minimalist in-DRAM Tracker）与非选择问题（Non-Selection Problem）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MINT（MICRO 2024，Qureshi/Qazi/Jaleel，[72]）是固定速率概率性 in-DRAM RowHammer 缓解的 SOTA：不维护 per-row 计数器，而是把激活流划分为固定大小的 mitigation window（两次缓解之间最大允许激活数），每个窗口内随机采样 1 个激活槽，窗口结束时缓解该槽对应的行；并带一个延迟缓解队列以支持 refresh 与 RFM 推迟（兼容 JEDEC DDR5 的缓解时机）。它代表"概率采样类 in-DRAM 防御"的最低成本路线（极小 in-DRAM 状态，无 counter cells）。在较高阈值（TRH-D ≥ 1000）下仅需每 48 次激活缓解一次，高内存强度 workload 只有 1.4% slowdown，极具吸引力。非选择问题（Non-Selection Problem）是这类固定速率概率缓解在低阈值下的统计瓶颈：每个窗口只随机选 1 行，重度 hammer 的 aggressor 行可能连续多个窗口都未被采样、长期不被缓解直至诱发位翻转；为维持安全，MINT 必须静态提高缓解速率（TRH-D=500 时每 24 激活一次 RFM、250 时每 11 次），即使无攻击也在烧带宽（有效带宽降约 23%/40%），高内存强度 workload 下 slowdown 升至 7.1%/17.5%。Web 佐证：MINT 论文 "Securely Mitigating RowHammer with a Minimalist In-DRAM Tracker"（MICRO 2024）；PVAC（ISCA 2026）与 ColumnKeeper 论文在相关工作/REFERENCES 中引用了 MINT；PrISM 论文（arXiv 2605.17358，ISCA 2026）将 MINT 作为主要对比 baseline。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 芯片内运转：DRAM 芯片内部维护极小的采样逻辑（随机数源 + 1 个采样槽 + 延迟缓解队列），每 W 次激活为 1 个窗口，窗口内随机采样 1 次（aggressor 被采中概率 1/W，W=72 时约 1.4%），窗口末对采样行做 TRR 或 RFM 缓解。非选择问题的芯片级体现：W 窗口只有 1 个采样槽时，persistent aggressor 在 L 个连续窗口全部错过的概率为 (1−1/W)^L；在低 TRH-D 下，这个漏检概率足以让 aggressor 在 32ms tREFW 内积累超过阈值的未缓解激活而翻位。芯片只能通过提高固定缓解速率（缩小 W）来补偿——每次 RFM 停通道 350ns，直接损耗有效带宽。PrISM 的解法是芯片内加 Sampled History Queue（SHQ）把"已采样未选中"的行跨窗口保留，用交集检测重复活跃行（见 PrISM 条目），从而在低 TRH-D 下不必全局提高缓解速率。
- 与 PRAC 的芯片设计取舍对比：PRAC 用 per-row 计数器精确追踪（每次激活更新计数器、tRP/tRC 膨胀、core 面积 +9%）；MINT 用无状态随机采样（零 counter 面积但低阈值下固定速率升级）。两者在"芯片内状态 vs 每激活逻辑开销"上各执一端，PrISM 用"跨窗口采样历史（几百字节 SRAM）+ 按需 ABO"在中间取平衡。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：MINT 在 DRAM 芯片内实现为极简逻辑（TRNG/PRNG 随机采样 + 采样槽 + 延迟缓解队列），评估时在 Ramulator 2.0 中按窗口大小 11/24/48（对应 TRH-D 250/500/1000）配置，作为概率性缓解的 baseline；PrISM 论文用它做对比，展示"固定速率提升"vs"交集按需提升"在低阈值下的性能差距（TRH-D=500：7.1% vs 0.2%；250：10.7% vs 1.5%）。典型使用场景：作为低硬件成本的 DDR5 近中期 RowHammer 缓解候选（PRAC 是 DDR5 optional 特性、商用采纳不确定），尤其适用于阈值不低于 1000 的场景。

涉及论文标题：
- Loaded Dice: Solving the Non-Selection Problem for Scalable Probabilistic RowHammer Defense
