## Counter Subarray（CSA，计数器子阵列）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- CSA 是在 DRAM 芯片内部、与数据子阵列（DSA）并行的专用"计数器子阵列"：把 RowHammer 缓解所需的 per-row 计数器从数据行内迁出，单独组织为一个子阵列来存储和更新。背景逻辑链：PRAC（DDR5 JEDEC 标准）把计数器位直接嵌在 DRAM 数据行内，每次计数更新必须走完整 ACT-PRE 序列并受数据通路时序（tRCD/tRP/tRC）约束，重置一个 aggressor 计数器（BR=2）甚至需要至少 5 次串行激活（≈208ns）——计数更新成为关键时序瓶颈。CSA 通过 subarray-level parallelism（如 SALP/MASA 思想）把计数器读写从数据通路中解耦：因为 CSA 与 DSA 不共享数据通路，二者可并发激活，计数器更新不延长关键时序路径。Chronus（HPCA 2025, CMU-SAFARI, arXiv:2502.12650）首次提出将计数器与数据分离的 counter subarray 并配 Concurrent Counter Update（CCU），使计数更新与正常访存并行、可按默认 DDR5 时序工作；CnC-PRAC（arXiv:2506.11970）在 Chronus 的 counter sub-array 上进一步合并（coalesce）计数器行激活降能耗；PVAC（arXiv:2604.20576, ISCA 2026）用 CSA 承载 victim 计数所需的"每次 ACT 更新 5 个计数器"的并发读写。Web 证据：Chronus arXiv 与 talk slides、CnC-PRAC arXiv。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 PVAC 中 CSA 的具体组织与运转流程：每 DSA 512 行、每行 8-bit 计数器（常用配置，同 Chronus）；由于 RH 扰动被限制在单个 subarray 内（受行译码/感知放大的物理隔离），PVAC 把每个 DSA 的全部计数器映射到一行 CSA 上，一次 CSA 激活即可完成该 DSA 内全部计数更新；64K 行 bank 只需 64 行 CSA（64KB），容量开销 <0.1%。CSA 行之间插入 guard rows（BR=2 时每 CSA 行 2 条，容量开销 0.29%）防止计数器位翻转；CSA 行随每 tREFW 的 normal refresh 自动刷新，无需额外刷新逻辑。运转例子（一次 DSA ACT）：MC 发 ACT 到 DSA 行 A → DSA 与 CSA 行译码器同时激活对应行 → tRCD_CSA=7.6ns 后 CSA 内置计数器更新逻辑串行处理 5 个 8-bit 计数器（4 个 victim + 1 个 aggressor，每个需 tUP=0.83ns 读-改-写）→ tWR_CSA=19.2ns 写回 → tRP_CSA=4.1ns 预充电；总延迟 tRCD_CSA+5×tUP+tWR_CSA+tRP_CSA=35.1ns，完整藏在 DSA 的 tRC=48ns 内。CSA 时序参数因行数少（64 行 vs 64K 行）而远小于 DSA：来自 SPICE 仿真 [22],[55] 的缩减比 47.4%/52.0%/25.3%/63.8% 得 tRCD_CSA/tRAS_CSA/tRP_CSA/tWR_CSA=7.6/16.7/4.1/19.2ns。节能优化：REFab 并行刷新 8 个 DSA 时，若每 DSA 计数器都在一行 CSA 上，需串行激活 8 行 CSA（8×35.1=280.8ns，超出 tRFC=295ns 预算且能耗大）；PVAC 把 64 行 CSA 拆成两个并行的 32 行子阵列，每 DSA 计数器切成 4 个 128 行 chunk 交错存放，使 8 个 DSA 的刷新计数更新只需 1 次 CSA 激活（仅 3/128 概率需双 CSA 并行激活且仍收在 tRC 内），REF 能耗从 +161% 降到 +19.3%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：CSA 是 DRAM 芯片内的独立子阵列（含本地行译码、感知放大器、专门的计数器更新逻辑——PVAC 用 Synopsys Design Compiler + TSMC 40nm 综合该逻辑，tUP=0.83ns），与数据阵列共用 bank 级外围但独占读写通路。面积开销 <0.01%/bank（更新逻辑）、容量 0.29%（含 guard rows）；拆分双 CSA 增加的感知放大器等外围电路在芯片级可忽略。使用：任何"计数器更新频繁、不能拖慢数据访问"的 RowHammer 缓解都需要 CSA——Chronus（CCU）、CnC-PRAC（coalescing）、PVAC（victim 计数）均以 CSA 为实现基座，评估都在 Ramulator 2.0 中为 CSA 建模独立时序参数并叠加 SPICE 标定的每访问能耗。论文未明确说明 CSA 的物理工艺细节（如 cell 类型），但指出 sense amplifier 比 DRAM cell 大约两个数量级，故拆分 CSA 的外围面积代价在芯片级可忽略。

涉及论文标题：
- PVAC: A RowHammer Mitigation Architecture Exploiting Per-victim-row Counting
